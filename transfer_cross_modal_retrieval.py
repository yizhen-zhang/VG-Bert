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

import shutil
import torch.nn as nn

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


print("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument("--resume", action="store_true", dest="resume", help="load from exp_dir if True")
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=100, type=int, metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay', default=30, type=int, metavar='LRDECAY', help='Divide the learning rate every lr_decay epochs')
parser.add_argument('--lr-ratio', default=0.5, type=float, metavar='LRDECAY', help='Divide the learning rate by lr_ratio every lr_decay epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-7, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--n_epochs", type=int, default=100, help="number of maximum training epochs")
parser.add_argument("--n_print_steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument("--language-model", type=str, default="Bert_base", help="language model architecture", choices=["Bert_base", "Bert_object"])
parser.add_argument("--bert-num-layers", type=int, default=5, help="number of bert encoder layers to keep")
parser.add_argument("--image-model", type=str, default="VGG16_Attention", help="image model architecture", choices=["VGG16", "VGG16_Attention"])
parser.add_argument("--use-position", type=str, default="learn", help="Use positional encoding for the image model", choices=["no", "predefine", "learn"])
parser.add_argument("--pretrained-vgg", action="store_true", dest="pretrained_vgg", help="Use an image network pretrained on ImageNet")
parser.add_argument("--load-pretrained", type=str, default="", help="directory to model pretrained on ImageNet/COCO")
parser.add_argument("--num-heads", type=int, default=8, dest="num_heads", help="number of heads in cross attention module")
parser.add_argument("--subspace-dim", type=int, default=32, dest="subspace_dim", help="bilinear relation matrix dimension")
parser.add_argument("--embedding-dim", default=768, type=int, help="embedding dimension for output visual/language features")
parser.add_argument("--relation-num", type=int, default=115, dest="relation_num", help="number of relation synsets to train: for curriculum learning")
parser.add_argument("--sigma", type=float, default=0.1, help="sigma paramater for contrastive loss")
parser.add_argument("--dropout-rate", type=float, default=0.3, dest="dropout_rate", help="dropout rates for base models")

args = parser.parse_args()

resume = args.resume

if args.resume:
    assert(bool(args.exp_dir))
    with open("%s/args.pkl" % args.exp_dir, "rb") as f:
        args = pickle.load(f)
args.resume = resume

print(args)

train_loader = torch.utils.data.DataLoader(
    dataloaders.LoadCoCoDataset(args.data_train),
    batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    dataloaders.LoadCoCoDataset(args.data_val, image_conf={'center_crop':True}),
    batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

if args.language_model=='Bert_base':
    language_model = models.Bert_base()
elif args.language_model=='Bert_object':
    language_model = models.Bert_object(embedding_dim=args.embedding_dim, dropout_rate=args.dropout_rate)

if args.image_model=='VGG16':
    image_model = models.VGG16(embedding_dim=args.embedding_dim, pretrained=args.pretrained_vgg)
elif args.image_model=='VGG16_Attention':
    image_model = models.VGG16_Attention(embedding_dim=args.embedding_dim, pretrained=args.pretrained_vgg, use_position=args.use_position, dropout_rate=args.dropout_rate)

cross_attention_model = models.CrossAttention(embedding_dim=args.embedding_dim, num_heads=args.num_heads, subspace_dim=args.subspace_dim, relation_base=args.relation_num, dropout_rate=args.dropout_rate)

# Define full model
transfer_model = models.TransferCrossModalRetrieval(image_model, language_model, cross_attention_model)

# load pretrained model
model_path = args.load_pretrained
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
transfer_model.load_state_dict(new_state_dict,strict=False)

# freeze weights
for p in transfer_model.parameters():
    p.requires_grad = False
for p in transfer_model.image_projection_head.parameters():
    p.requires_grad = True
for p in transfer_model.language_projection_head.parameters():
    p.requires_grad = True

if not args.resume:
    print("\nexp_dir: %s" % args.exp_dir)
    os.makedirs("%s/models" % args.exp_dir)
    with open("%s/args.pkl" % args.exp_dir, "wb") as f:
        pickle.dump(args, f)

def train(transfer_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)
    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    progress = []
    best_epoch, best_loss = 0, np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_acc,
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    # create/load exp
    if args.resume:
        progress_pkl = "%s/progress.pkl" % exp_dir
        progress, epoch, global_step, best_epoch, best_acc = load_progress(progress_pkl)
        print("\nResume training from:")
        print("  epoch = %s" % epoch)
        print("  global_step = %s" % global_step)
        print("  best_epoch = %s" % best_epoch)
        print("  best_acc = %.4f" % best_acc)

    if epoch != 0:
        transfer_model.load_state_dict(torch.load("%s/models/transfer_model.%d.pth" % (exp_dir, epoch)))
        print("loaded parameters from epoch %d" % epoch)

    transfer_model = transfer_model.to(device)

    # Set up the optimizer
    trainables = [p for p in transfer_model.parameters() if p.requires_grad]
    if args.optim == 'sgd':
       optimizer = torch.optim.SGD(trainables, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(trainables, args.lr,
                                weight_decay=args.weight_decay,
                                betas=(0.95, 0.999))
    else:
        raise ValueError('Optimizer %s is not supported' % args.optim)

    if epoch != 0:
        optimizer.load_state_dict(torch.load("%s/models/optim_state.%d.pth" % (exp_dir, epoch)))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("loaded state dict from epoch %d" % epoch)

    epoch += 1

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")

    if not isinstance(transfer_model, torch.nn.DataParallel):
        transfer_model = nn.DataParallel(transfer_model)

    transfer_model.train()

    while epoch <= args.n_epochs:
        adjust_learning_rate(args.lr, args.lr_decay, args.lr_ratio, optimizer, epoch)
        end_time = time.time()
        for i, (_, image_input, language_input, token_type, input_mask, nwords) in enumerate(train_loader):
            B = image_input.size(0)

            language_input = language_input.to(device)
            image_input = image_input.to(device)
            token_type = token_type.to(device)
            input_mask = input_mask.to(device)
            nwords = nwords.to(device)

            optimizer.zero_grad()

            (image_metric, language_metric, loss_I2L, loss_L2I) = TransferCrossModalRetrieval(image_input, language_input, token_type, input_mask, nwords, args.sigma)

            loss_I2L = torch.mean(loss_I2L)
            loss_L2I = torch.mean(loss_L2I)
            loss = loss_I2L + loss_L2I

            loss.backward()
            optimizer.step()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)

            if global_step % args.n_print_steps == 0 and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.3f}\t'
                  'loss_I2L {loss_I2L:.3f}\t'
                  'loss_L2I {loss_L2I:.3f}\t'.format(
                   epoch, i, len(train_loader), # batch_time=batch_time, data_time=data_time, loss_meter=loss_meter,
                   loss=loss, loss_I2L=loss_I2L, loss_L2I=loss_L2I), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        Loss_test = validate(transfer_model, test_loader, args)
        avg_loss = Loss_test

        torch.save(transfer_model.state_dict(),
                "%s/models/transfer_model.pth" % (exp_dir))
        torch.save(optimizer.state_dict(), "%s/models/optim_state.pth" % (exp_dir))

        if avg_acc < best_loss:
            best_epoch = epoch
            best_loss = avg_loss
            torch.save(transfer_model.state_dict(),"%s/models/best_model.pth" % (exp_dir))
            torch.save(transfer_model.language_model.state_dict(),"%s/models/best_language_model.pth" % (exp_dir))
        _save_progress()
        epoch += 1

def validate(transfer_model, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(transfer_model, torch.nn.DataParallel):
        transfer_model = nn.DataParallel(transfer_model)

    transfer_model = transfer_model.to(device)
    # switch to evaluate mode
    transfer_model.eval()

    end = time.time()
    N_examples = val_loader.dataset.__len__()
    I_embeddings = []
    L_embeddings = []
    word_counts = []
    Loss_I2L = 0
    Loss_L2I = 0
    with torch.no_grad():
        for i, (_, image_input, language_input, token_type, input_mask, nwords) in enumerate(val_loader):
            image_input = image_input.to(device)
            language_input = language_input.to(device)
            token_type = token_type.to(device)
            input_mask = input_mask.to(device)
            nwords = nwords.to(device)

            # compute output
            (image_metric, language_metric, loss_I2L, loss_L2I) = TransferCrossModalRetrieval(image_input, language_input, token_type, input_mask, nwords, args.sigma)

            Loss_I2L += torch.sum(loss_I2L)
            Loss_L2I += torch.sum(loss_L2I)

    Loss_I2L = Loss_I2L
    Loss_L2I = Loss_L2I
    Loss_test = Loss_I2L + Loss_L2I

    print(' * Loss {Loss_test:.3f} Loss_I2L {Loss_I2L:.3f} Loss_L2I {Loss_L2I:.3f}'.format(Loss_test=Loss_test, Loss_I2L=Loss_I2L, Loss_L2I=Loss_L2I))
    return Loss_test

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(base_lr, lr_decay, lr_ratio, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by half every lr_decay epochs"""
    lr = base_lr * (lr_ratio ** (epoch // lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def load_progress(prog_pkl, quiet=False):
    """
    load progress pkl file
    Args:
        prog_pkl(str): path to progress pkl file
    Return:
        progress(list):
        epoch(int):
        global_step(int):
        best_epoch(int):
        best_avg_r10(float):
    """
    def _print(msg):
        if not quiet:
            print(msg)

    with open(prog_pkl, "rb") as f:
        prog = pickle.load(f)
        epoch, global_step, best_epoch, best_avg_r10, _ = prog[-1]

    _print("\nPrevious Progress:")
    msg =  "[%5s %7s %5s %7s %6s]" % ("epoch", "step", "best_epoch", "best_avg_r10", "time")
    _print(msg)
    return prog, epoch, global_step, best_epoch, best_avg_r10

train(transfer_model, train_loader, val_loader, args)
