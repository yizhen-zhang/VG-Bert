import time
import shutil
import numpy as np
import pickle
from .util import *
from collections import OrderedDict

import torch
import torch.nn as nn

def train(full_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)
    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()
    top10 = AverageMeter()

    top1_obj = AverageMeter()
    top5_obj = AverageMeter()
    top10_obj = AverageMeter()

    progress = []
    if args.stage=='two_stream_pretraining':
        best_epoch, best_acc = 0, np.inf
    elif args.stage=='relational_grounding':
        best_epoch, best_acc = 0, 0
    else:
        raise ValueError('Training stage %s is not recognized' % args.stage)

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
        full_model.load_state_dict(torch.load("%s/models/full_model.%d.pth" % (exp_dir, epoch)))
        print("loaded parameters from epoch %d" % epoch)

    full_model = full_model.to(device)

    # Set up the optimizer
    trainables = [p for p in full_model.parameters() if p.requires_grad]
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(trainables, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=args.weight_decay, betas=(0.95, 0.999))
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

    if not isinstance(full_model, torch.nn.DataParallel):
        full_model = nn.DataParallel(full_model)

    full_model.train()

    while epoch <= args.n_epochs:
        adjust_learning_rate(args.lr, args.lr_decay, args.lr_ratio, optimizer, epoch)
        end_time = time.time()
        full_model.train()
        # stage 2: two stream pretraining
        if args.stage=='two_stream_pretraining':
            for i, (_, image_input, language_input, token_type, input_mask, nwords) in enumerate(train_loader):
                B = image_input.size(0)

                language_input = language_input.to(device)
                image_input = image_input.to(device)
                token_type = token_type.to(device)

                optimizer.zero_grad()

                (loss_I2L, loss_L2I) = full_model(image_input, language_input, token_type, input_mask, nwords)
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
                      'Loss_I2L {loss_I2L:.3f}\t'
                      'Loss_L2I {loss_L2I:.3f}\t'.format(
                       epoch, i, len(train_loader),
                       loss=loss, loss_I2L=loss_I2L, loss_L2I=loss_L2I), flush=True)
                    if np.isnan(loss_meter.avg):
                        print("training diverged...")
                        return

                end_time = time.time()
                global_step += 1

            Loss_test = validate(full_model, test_loader, args)
            avg_acc = Loss_test

            torch.save(full_model.state_dict(),"%s/models/full_model.%d.pth" % (exp_dir, epoch))
            torch.save(optimizer.state_dict(), "%s/models/optim_state.pth" % (exp_dir))

            if avg_acc < best_acc:
                best_epoch = epoch
                best_acc = avg_acc
                torch.save(full_model.state_dict(),"%s/models/best_model.pth" % (exp_dir))
                # torch.save(full_model.language_model.state_dict(),"%s/models/best_language_model.pth" % (exp_dir))

            _save_progress()
            epoch += 1

        # stage 3: finetuning for relation prediction
        elif args.stage=='relational_grounding':
            for i, (_, image_input, object_input_all, segments_tensors_all, input_token_mask_all, input_mask, n_object, object_target, relation_target) in enumerate(train_loader):
                B = image_input.size(0)

                image_input = image_input.to(device)
                object_input_all = object_input_all.to(device)
                segments_tensors_all = segments_tensors_all.to(device)
                input_token_mask_all = input_token_mask_all.to(device)
                input_mask = input_mask.to(device)
                n_object = n_object.to(device)
                object_target = object_target.to(device)
                relation_target = relation_target.to(device)

                optimizer.zero_grad()

                (loss_cls_object, loss_pair, loss_relation, relation_target, relation_prediction, object_target, object_prediction) = full_model(image_input, object_input_all, segments_tensors_all, input_token_mask_all, input_mask, n_object, object_target, relation_target)

                loss_cls_object = torch.sum(torch.mul(loss_cls_object, object_target>0)) / B  # grounded object classification: auxilary loss
                loss_pair = torch.mean(loss_pair)  # contrastive loss_obj
                loss_relation = torch.mean(loss_relation)  # contrastive loss_rel
                loss = loss_cls_object + loss_relation + loss_pair

                loss.backward()
                optimizer.step()

                # measure object classification accuracy
                for b in range(B):
                    output = object_prediction[b].clone().cpu()
                    target = object_target[b].clone().cpu()
                    object_mask = (object_target[b]>0).cpu() # filter out NULL objects
                    if torch.sum(object_mask)>0:
                        output = output[:,object_mask].t()
                        target = target[object_mask]
                        n_objects = len(target)
                        if n_objects>0:
                            acc1_obj, acc5_obj, acc10_obj = accuracy(output, target, topk=(1, 5, 10))
                            top1_obj.update(acc1_obj[0], n_objects)
                            top5_obj.update(acc5_obj[0], n_objects)
                            top10_obj.update(acc10_obj[0], n_objects)

                # measure relation classification accuracy
                for b in range(B):
                    output = torch.flatten(relation_prediction[b].clone(),1).cpu()
                    target = torch.flatten(relation_target[b].clone()).cpu()
                    relation_mask = torch.flatten(relation_target[b]>0).cpu() # filter out NULL relations
                    if torch.sum(relation_mask)>0:
                        output = output[:,relation_mask].t()
                        target = target[relation_mask]
                        n_relations = len(target)
                        if n_relations>0:
                            acc1, acc5, acc10 = accuracy(output, target, topk=(1, 5, 10))
                            top1.update(acc1[0], n_relations)
                            top5.update(acc5[0], n_relations)
                            top10.update(acc10[0], n_relations)

                # record loss
                loss_meter.update(loss.item(), B)
                batch_time.update(time.time() - end_time)

                if global_step % args.n_print_steps == 0 and global_step != 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss:.3f}\t'
                      'Loss_CLS_object {loss_cls_object:.3f}\t'
                      'Loss_pair {loss_pair:.3f}\t'
                      'Loss_relation {loss_relation:.3f}\t'.format(
                       epoch, i, len(train_loader),
                       loss=loss, loss_cls_object=loss_cls_object, loss_pair=loss_pair, loss_relation=loss_relation), flush=True)
                    if np.isnan(loss_meter.avg):
                        print("training diverged...")
                        return

                end_time = time.time()
                global_step += 1

            print('Training: Relational Prediction Acc: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Acc@10 {top10.avg:.3f}'
                .format(top1=top1, top5=top5, top10=top10))
            print('Training: Object Classification Acc: Acc@1 {top1_obj.avg:.3f} Acc@5 {top5_obj.avg:.3f} Acc@10 {top10_obj.avg:.3f}'
                .format(top1_obj=top1_obj, top5_obj=top5_obj, top10_obj=top10_obj))
            top1.reset()
            top5.reset()
            top10.reset()
            top1_obj.reset()
            top5_obj.reset()
            top10_obj.reset()

            loss_meter.reset()
            avg_acc = validate(full_model, test_loader, args)

            torch.save(full_model.state_dict(),"%s/models/full_model.%d.pth" % (exp_dir, epoch))
            torch.save(optimizer.state_dict(), "%s/models/optim_state.%d.pth" % (exp_dir, epoch))

            if avg_acc > best_acc:
                best_epoch = epoch
                best_acc = avg_acc
                torch.save(full_model.state_dict(),"%s/models/best_model.pth" % (exp_dir))
                # torch.save(full_model.language_model.state_dict(),"%s/models/best_language_model.pth" % (exp_dir))
            _save_progress()
            epoch += 1
        else:
            raise ValueError('Training stage %s is not recognized' % args.stage)

def validate(full_model, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(full_model, torch.nn.DataParallel):
        full_model = nn.DataParallel(full_model)
    full_model = full_model.to(device)
    # switch to evaluate mode
    full_model.eval()
    end = time.time()
    N_examples = val_loader.dataset.__len__()

    # stage 2: two stream pretraining
    if args.stage=='two_stream_pretraining':
        I_embeddings = []
        L_embeddings = []
        word_counts = []

        Loss_I2L = 0
        Loss_L2I = 0
        Loss_test = 0
        with torch.no_grad():
            for i, (_, image_input, language_input, token_type, input_mask, nwords) in enumerate(val_loader):
                image_input = image_input.to(device)
                language_input = language_input.to(device)
                token_type = token_type.to(device)
                # compute output
                (loss_I2L, loss_L2I) = full_model(image_input, language_input, token_type, input_mask, nwords)

                Loss_I2L += torch.sum(loss_I2L)
                Loss_L2I += torch.sum(loss_L2I)

        Loss_I2L = Loss_I2L
        Loss_L2I = Loss_L2I
        Loss_test = Loss_I2L + Loss_L2I

        print(' * Loss {Loss_test:.3f} Loss_I2L {Loss_I2L:.3f} Loss_L2I {Loss_L2I:.3f}'.format(Loss_test=Loss_test, Loss_I2L=Loss_I2L, Loss_L2I=Loss_L2I))
        return Loss_test
    # stage 3: finetuning for relation prediction
    elif args.stage=='relational_grounding':
        top1 = AverageMeter()
        top5 = AverageMeter()
        top10 = AverageMeter()

        top1_obj = AverageMeter()
        top5_obj = AverageMeter()
        top10_obj = AverageMeter()

        Loss_cls_object = 0
        Loss_pair = 0
        Loss_relation = 0

        with torch.no_grad():
            for i, (_, image_input, object_input_all, segments_tensors_all, input_token_mask_all, input_mask, n_object, object_target, relation_target) in enumerate(val_loader):
                image_input = image_input.to(device)
                object_input_all = object_input_all.to(device)
                segments_tensors_all = segments_tensors_all.to(device)
                input_token_mask_all = input_token_mask_all.to(device)
                input_mask = input_mask.to(device)
                n_object = n_object.to(device)
                object_target = object_target.to(device)
                relation_target = relation_target.to(device)

                B = image_input.size(0)

                (loss_cls_object, loss_pair, loss_relation, relation_target, relation_prediction, object_target, object_prediction) = full_model(image_input, object_input_all, segments_tensors_all, input_token_mask_all, input_mask, n_object, object_target, relation_target)

                Loss_cls_object += torch.sum(torch.mul(loss_cls_object, object_target>0)) / B
                Loss_pair += torch.mean(loss_pair)
                Loss_relation += torch.mean(loss_relation)

                # measure object classification accuracy
                for b in range(B):
                    output = object_prediction[b].clone().cpu()
                    target = object_target[b].clone().cpu()
                    object_mask = (object_target[b]>0).cpu() # filter out NULL objects
                    if torch.sum(object_mask)>0:
                        output = output[:,object_mask].t()
                        target = target[object_mask]
                        n_objects = len(target)
                        if n_objects>0:
                            acc1_obj, acc5_obj, acc10_obj = accuracy(output, target, topk=(1, 5, 10))
                            top1_obj.update(acc1_obj[0], n_objects)
                            top5_obj.update(acc5_obj[0], n_objects)
                            top10_obj.update(acc10_obj[0], n_objects)

                # measure classification accuracy
                for b in range(B):
                    output = torch.flatten(relation_prediction[b].clone(),1).cpu()
                    target = torch.flatten(relation_target[b].clone()).cpu()
                    relation_mask = torch.flatten(relation_target[b]>0).cpu() # filter out NULL relations
                    if torch.sum(relation_mask)>0:
                        output = output[:,relation_mask].t()
                        target = target[relation_mask]
                        n_relations = len(target)
                        if n_relations>0:
                            acc1, acc5, acc10 = accuracy(output, target, topk=(1, 5, 10))
                            top1.update(acc1[0], n_relations)
                            top5.update(acc5[0], n_relations)
                            top10.update(acc10[0], n_relations)

                batch_time.update(time.time() - end)
                end = time.time()

        Loss_cls_object = Loss_cls_object # / N_examples
        Loss_pair = Loss_pair # / N_examples
        Loss_relation = Loss_relation # / N_examples

        print(' * Testing: Loss_CLS_object {Loss_cls_object:.2f} Loss_pair {Loss_pair:.2f} Loss_relation {Loss_relation:.2f}'
            .format(Loss_cls_object=Loss_cls_object, Loss_pair=Loss_pair, Loss_relation=Loss_relation))
        print(' * Testing: Relational Prediction Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Acc@10 {top10.avg:.3f}'
            .format(top1=top1, top5=top5, top10=top10))
        print(' * Testing: Object Classification Acc: Acc@1 {top1_obj.avg:.3f} Acc@5 {top5_obj.avg:.3f} Acc@10 {top10_obj.avg:.3f}'
            .format(top1_obj=top1_obj, top5_obj=top5_obj, top10_obj=top10_obj))
        return (top1.avg)
    else:
        raise ValueError('Training stage %s is not recognized' % args.stage)
