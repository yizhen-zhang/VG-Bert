import argparse
import os
import pickle
import sys
import time
import numpy as np

import torch
import models
import dataloaders
from steps import train, validate
from collections import OrderedDict

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

print("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--stage", type=str, default='relational_grounding', help="training stage", choices=["two_stream_pretraining", "relational_grounding"])
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument("--resume", action="store_true", dest="resume", help="load from exp_dir if True")
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('--seed', default=0, type=int, metavar='N', help='random seed (default: 0)')
parser.add_argument('-b', '--batch-size', default=100, type=int, metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay', default=15, type=int, metavar='LRDECAY', help='Divide the learning rate every lr_decay epochs')
parser.add_argument('--lr-ratio', default=0.5, type=float, metavar='LRDECAY', help='Divide the learning rate by lr_ratio every lr_decay epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-7, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--n_epochs", type=int, default=100, help="number of maximum training epochs")
parser.add_argument("--n_print_steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument("--language-model", type=str, default="Bert_base", help="language model architecture", choices=["Bert_base", "Bert_object"])
parser.add_argument('--freezeQK', dest='freezeQK', action='store_true')
parser.add_argument('--no-freezeQK', dest='freezeQK', action='store_false')
parser.set_defaults(freezeQK=True)
parser.add_argument("--image-model", type=str, default="VGG16_Attention", help="image model architecture", choices=["VGG16", "VGG16_Attention"])
parser.add_argument("--use-position", type=str, default="learn", help="Use positional encoding for the image model", choices=["no", "predefine", "learn"])
parser.add_argument("--pretrained-vgg", action="store_true", dest="pretrained_vgg", help="Use an image network pretrained on ImageNet")
parser.add_argument("--load-pretrained", type=str, default="", help="directory to model pretrained on ImageNet/COCO")
parser.add_argument("--num-heads", type=int, default=8, dest="num_heads", help="number of heads in cross attention module")
parser.add_argument("--subspace-dim", type=int, default=32, dest="subspace_dim", help="bilinear relation matrix dimension")
parser.add_argument("--embedding-dim", default=768, type=int, help="embedding dimension for output visual/language features")
parser.add_argument("--relation-num", type=int, default=115, dest="relation_num", help="number of relation synsets to train: for curriculum learning")
parser.add_argument("--sigma", type=float, default=0.1, help="sigma paramater for contrastive loss")
parser.add_argument("--temperature", type=float, default=1.0, help="temperature parameter for NCE-based contrastive loss")
parser.add_argument("--dropout-rate", type=float, default=0.3, dest="dropout_rate", help="dropout rates for base models")
parser.add_argument("--base-model-learnable-layers", type=int, default=0, dest="base_model_learnable_layers", help="allow base model to be learnable")

args = parser.parse_args()
resume = args.resume
torch.manual_seed(args.seed)

if args.resume:
    assert(bool(args.exp_dir))
    with open("%s/args.pkl" % args.exp_dir, "rb") as f:
        args = pickle.load(f)
args.resume = resume

print(args)

if args.stage=='two_stream_pretraining':
    train_loader = torch.utils.data.DataLoader(
        dataloaders.LoadCoCoDataset(args.data_train, image_conf={'center_crop':True}),
        batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        dataloaders.LoadCoCoDataset(args.data_val, image_conf={'center_crop':True}),
        batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
elif args.stage=='relational_grounding':
    train_loader = torch.utils.data.DataLoader(
        dataloaders.LoadVisualGenomeDataset(args.data_train, image_conf={'center_crop':False}, relation_num = args.relation_num),
        batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        dataloaders.LoadVisualGenomeDataset(args.data_val, image_conf={'center_crop':False}, relation_num = args.relation_num),
        batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

# load language model
if args.language_model=='Bert_base':
    language_model = models.Bert_base()
elif args.language_model=='Bert_object':
    language_model = models.Bert_object(embedding_dim=args.embedding_dim, dropout_rate=args.dropout_rate)

# load image model
if args.image_model=='VGG16':
    image_model = models.VGG16(embedding_dim=args.embedding_dim, pretrained=args.pretrained_vgg)
elif args.image_model=='VGG16_Attention':
    image_model = models.VGG16_Attention(embedding_dim=args.embedding_dim, pretrained=args.pretrained_vgg, use_position=args.use_position, dropout_rate=args.dropout_rate)

if args.stage=='two_stream_pretraining':
    # load model pretrained on ImageNet
    state_dict = torch.load(args.load_pretrained + "best_model.pth", map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "module" in k:
            name = k.replace('module.', '')
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    print(new_state_dict.keys())
    image_model.load_state_dict(new_state_dict,strict=False)
    print("loaded pretrained parameters from "+ args.load_pretrained)
    # Define full model
    full_model = models.TwoStreamPretrain(image_model, language_model, embedding_dim=args.embedding_dim, sigma=args.sigma)

elif args.stage=='relational_grounding':
    cross_attention_model = models.CrossAttention(embedding_dim=args.embedding_dim, num_heads=args.num_heads, subspace_dim=args.subspace_dim, relation_base=args.relation_num, dropout_rate=args.dropout_rate)
    full_model = models.RelationalGrounding(image_model, language_model, cross_attention_model, relation_size=args.relation_num, temperature=args.temperature)
    # load pretrained model on COCO
    model_path = args.load_pretrained+'best_model.pth'
    state_dict = torch.load(model_path, map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "module" in k:
            name = k.replace('module.', '')
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    # load params
    full_model.load_state_dict(new_state_dict,strict=False)

# freeze weights
for p in full_model.image_model.parameters():
    p.requires_grad = False
# learnable parameters in the visual stream
# makes last few layers learnable
for p in full_model.image_model.embedder.parameters():
    p.requires_grad = True
for p in full_model.image_model.attention.parameters():
    p.requires_grad = True
# learnbale parameters in the language stream
for p in full_model.language_model.parameters():
    p.requires_grad = False
if args.base_model_learnable_layers != 0:
    # for bert base model
    ll = 12-args.base_model_learnable_layers
    for p in full_model.language_model.language_model.encoder.layer[ll:12].parameters():
        p.requires_grad = True
print('Freeze query and key in bert encoder: '+str(args.freezeQK))
if args.freezeQK == True:
    # freeze query and key to keep contexual information
    for ll in range(12):
        for p in full_model.language_model.language_model.encoder.layer[ll].attention.self.query.parameters():
            p.requires_grad = False
        for p in full_model.language_model.language_model.encoder.layer[ll].attention.self.key.parameters():
            p.requires_grad = False

if not args.resume:
    print("\nexp_dir: %s" % args.exp_dir)
    os.makedirs("%s/models" % args.exp_dir)
    with open("%s/args.pkl" % args.exp_dir, "wb") as f:
        pickle.dump(args, f)

train(full_model, train_loader, val_loader, args)
