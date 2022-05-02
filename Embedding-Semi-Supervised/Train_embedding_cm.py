'''
 * Copyright (c) 2018, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
from __future__ import print_function
import random

import time
import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter

from LinearModel import LinearNN
import datasets.cifar as cifar
import datasets.nih as nih
from utils import accuracy, setup_default_logging, AverageMeter, WarmupCosineLrScheduler
from utils import load_from_checkpoint
from Expert import CIFAR100Expert, NIHExpert
from feature_extractor.embedding_model import EmbeddingModel


def set_model(args):
    """Initialize models

    :param args: training arguments
    :return: tuple
        - model: Initialized model
        - criteria_x: Supervised loss function
        - ema_model: Initialized ema model
    """
    if args.dataset.lower() == 'cifar100':
        feature_dim = 1280
    elif args.dataset.lower() == 'nih':
        feature_dim = 512
    else:
        print(f'Dataset {args.dataset} not defined')
        sys.exit()
    model = LinearNN(num_classes=args.n_classes, feature_dim=feature_dim, proj=True)

    model.train()
    model.cuda()  
    
    if args.eval_ema:
        ema_model = LinearNN(num_classes=args.n_classes, feature_dim=feature_dim, proj=True)
        for param_q, param_k in zip(model.parameters(), ema_model.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net
        ema_model.cuda()  
        ema_model.eval()
    else:
        ema_model = None
        
    criteria_x = nn.CrossEntropyLoss().cuda()
    return model,  criteria_x, ema_model

@torch.no_grad()
def ema_model_update(model, ema_model, ema_m):
    """Momentum update of evaluation model (exponential moving average)

    :param model: Model
    :param ema_model: EMA-Model
    :param ema_m: Ema parameter
    :return:
    """
    for param_train, param_eval in zip(model.parameters(), ema_model.parameters()):
        param_eval.copy_(param_eval * ema_m + param_train.detach() * (1-ema_m))

    for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
        buffer_eval.copy_(buffer_train)  
        
def train_one_epoch(epoch,
                    model,
                    ema_model,
                    emb_model,
                    prob_list,
                    criteria_x,
                    optim,
                    lr_schdlr,
                    dltrain_x,
                    dltrain_u,
                    args,
                    n_iters,
                    logger,
                    queue_feats,
                    queue_probs,
                    queue_ptr,
                    ):
    """Train one epoch on the train set

    :param epoch: Current epoch
    :param model: Model
    :param ema_model: EMA-Model
    :param emb_model: Embedding model
    :param prob_list: List of probabilities
    :param criteria_x: Supervised loss function
    :param optim: Optimizer
    :param lr_schdlr: Learning rate scheduler
    :param dltrain_x: Data loader for the labeled training instances
    :param dltrain_u: Data loader for the unlabeled training instances
    :param args: Training arguments
    :param n_iters: Number of iterations per epoch
    :param logger: Logger
    :param queue_feats: Memory bank feature vectors
    :param queue_probs: Memory bank probabilities
    :param queue_ptr: Memory bank ptr
    :return: tuple
        - Average supervised loss
        - Average unsupervised loss
        - Average contrastive loss
        - Average mask
        - Average number of edges in the pseudo label graph
        - Percentage of correct pseudo labels
        - Memory bank feature vectors
        - Memory bank probabilities
        - Memory bank ptr
        - List of probabilities
    """

    model.train()
    loss_x_meter = AverageMeter()
    loss_u_meter = AverageMeter()
    loss_contrast_meter = AverageMeter()
    # the number of correct pseudo-labels
    n_correct_u_lbs_meter = AverageMeter()
    # the number of confident unlabeled data
    n_strong_aug_meter = AverageMeter()
    mask_meter = AverageMeter()
    # the number of edges in the pseudo-label graph
    pos_meter = AverageMeter()

    
    epoch_start = time.time()  # start time
    dl_x, dl_u = iter(dltrain_x), iter(dltrain_u)
    for it in range(n_iters):
        ims_x_weak, lbs_x, im_id = next(dl_x)
        (ims_u_weak, ims_u_strong0, ims_u_strong1), lbs_u_real, im_id = next(dl_u)

        lbs_x = lbs_x.cuda()
        lbs_u_real = lbs_u_real.cuda()

        # --------------------------------------
        bt = ims_x_weak.size(0)
        btu = ims_u_weak.size(0)

        imgs = torch.cat([ims_x_weak, ims_u_weak, ims_u_strong0, ims_u_strong1], dim=0).cuda()
        embedding = emb_model.get_embedding(batch=imgs)
        logits, features = model(embedding)

        logits_x = logits[:bt]
        logits_u_w, logits_u_s0, logits_u_s1 = torch.split(logits[bt:], btu)
        
        feats_x = features[:bt]
        feats_u_w, feats_u_s0, feats_u_s1 = torch.split(features[bt:], btu)
  
        loss_x = criteria_x(logits_x, lbs_x)

        with torch.no_grad():
            logits_u_w = logits_u_w.detach()
            feats_x = feats_x.detach()
            feats_u_w = feats_u_w.detach()
            
            probs = torch.softmax(logits_u_w, dim=1)            
            # DA
            prob_list.append(probs.mean(0))
            if len(prob_list)>32:
                prob_list.pop(0)
            prob_avg = torch.stack(prob_list, dim=0).mean(0)
            probs = probs / prob_avg
            probs = probs / probs.sum(dim=1, keepdim=True)   

            probs_orig = probs.clone()
            
            if epoch>0 or it>args.queue_batch: # memory-smoothing 
                A = torch.exp(torch.mm(feats_u_w, queue_feats.t())/args.temperature)       
                A = A/A.sum(1,keepdim=True)                    
                probs = args.alpha*probs + (1-args.alpha)*torch.mm(A, queue_probs)               
            
            scores, lbs_u_guess = torch.max(probs, dim=1)
            mask = scores.ge(args.thr).float() 
                   
            feats_w = torch.cat([feats_u_w,feats_x],dim=0)   
            onehot = torch.zeros(bt,args.n_classes).cuda().scatter(1,lbs_x.view(-1,1),1)
            probs_w = torch.cat([probs_orig,onehot],dim=0)
            
            # update memory bank
            n = bt+btu   
            queue_feats[queue_ptr:queue_ptr + n,:] = feats_w
            queue_probs[queue_ptr:queue_ptr + n,:] = probs_w      
            queue_ptr = (queue_ptr+n)%args.queue_size

            
        # embedding similarity
        sim = torch.exp(torch.mm(feats_u_s0, feats_u_s1.t())/args.temperature) 
        sim_probs = sim / sim.sum(1, keepdim=True)
        
        # pseudo-label graph with self-loop
        Q = torch.mm(probs, probs.t())       
        Q.fill_diagonal_(1)    
        pos_mask = (Q>=args.contrast_th).float()
            
        Q = Q * pos_mask
        Q = Q / Q.sum(1, keepdim=True)
        
        # contrastive loss
        loss_contrast = - (torch.log(sim_probs + 1e-7) * Q).sum(1)
        loss_contrast = loss_contrast.mean()  
        
        # unsupervised classification loss
        loss_u = - torch.sum((F.log_softmax(logits_u_s0,dim=1) * probs),dim=1) * mask                
        loss_u = loss_u.mean()
        
        loss = loss_x + args.lam_u * loss_u + args.lam_c * loss_contrast
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_schdlr.step()

        if args.eval_ema:
            with torch.no_grad():
                ema_model_update(model, ema_model, args.ema_m)
                
        loss_x_meter.update(loss_x.item())
        loss_u_meter.update(loss_u.item())
        loss_contrast_meter.update(loss_contrast.item())
        mask_meter.update(mask.mean().item())       
        pos_meter.update(pos_mask.sum(1).float().mean().item())
        
        corr_u_lb = (lbs_u_guess == lbs_u_real).float() * mask
        n_correct_u_lbs_meter.update(corr_u_lb.sum().item())
        n_strong_aug_meter.update(mask.sum().item())

        if (it + 1) % 64 == 0:
            t = time.time() - epoch_start

            lr_log = [pg['lr'] for pg in optim.param_groups]
            lr_log = sum(lr_log) / len(lr_log)

            logger.info("{}-x{}-s{}, {} | epoch:{}, iter: {}. loss_u: {:.3f}. loss_x: {:.3f}. loss_c: {:.3f}. "
                        "n_correct_u: {:.2f}/{:.2f}. Mask:{:.3f}. num_pos: {:.1f}. LR: {:.3f}. Time: {:.2f}".format(
                args.dataset, args.n_labeled, args.seed, args.exp_dir, epoch, it + 1, loss_u_meter.avg, loss_x_meter.avg, loss_contrast_meter.avg, n_correct_u_lbs_meter.avg, n_strong_aug_meter.avg, mask_meter.avg, pos_meter.avg, lr_log, t))
            epoch_start = time.time()

    return loss_x_meter.avg, loss_u_meter.avg, loss_contrast_meter.avg, mask_meter.avg, pos_meter.avg, n_correct_u_lbs_meter.avg/n_strong_aug_meter.avg, queue_feats, queue_probs, queue_ptr, prob_list


def evaluate(model, ema_model, emb_model, dataloader):
    """Evaluate model on train or validation set

    :param model: Model
    :param ema_model: EMA-Model
    :param emb_model: Embedding model
    :param dataloader: Data loader for the evaluation set
    :return: tuple
        - Accuracy of the model
        - Accuracy of the ema_model
    """
    
    model.eval()
    preds = []
    targets = []
    top1_meter = AverageMeter()
    ema_top1_meter = AverageMeter()

    with torch.no_grad():
        for ims, lbs, im_id in dataloader:
            ims = ims.cuda()
            lbs = lbs.cuda()

            embedding = emb_model.get_embedding(batch=ims)
            logits, _ = model(embedding)
            scores = torch.softmax(logits, dim=1)
            preds += torch.argmax(scores, dim=1).cpu().tolist()
            targets += lbs.cpu().tolist()
            top1 = accuracy(scores, lbs, (1, ))
            top1_meter.update(top1.item())
            
            if ema_model is not None:
                embedding = emb_model.get_embedding(batch=ims)
                logits, _ = ema_model(embedding)
                scores = torch.softmax(logits, dim=1)
                top1 = accuracy(scores, lbs, (1, ))
                ema_top1_meter.update(top1.item())
    return top1_meter.avg, ema_top1_meter.avg


def predict_cifar(model, ema_model, emb_model, trainloader_x, trainloader_u, testloader):
    """Generate predictions for the cifar dataset

    :param model: Model
    :param ema_model: EMA-Model
    :param emb_model: Embedding model
    :param trainloader_x: Dataloader for the train set
    :param trainloader_u: Dataloader for the train set
    :param testloader: Dataloader for the test set
    :return: Dict of artificial expert annotations
    """
    model.eval()

    predictions = {'train': np.zeros(50000, dtype=int), 'test': np.zeros(10000, dtype=int)}
    with torch.no_grad():
        # use expert labels as predictions for the labeled set
        for ims, lbs, im_id in trainloader_x:
            for j in range(len(lbs)):
                predictions['train'][im_id[j]] = lbs[j]
        # generate predictions for the unlabeled set
        for ims, lbs, im_id in trainloader_u:
            ims = ims.cuda()
            lbs = lbs.cuda()
            embedding = emb_model.get_embedding(batch=ims)
            if ema_model is not None:
                logits, _ = ema_model(embedding)
            else:
                logits, _ = model(embedding)

            output = torch.softmax(logits, dim=1)
            for j in range(len(lbs)):
                predictions['train'][im_id[j]] = int(output[j].cpu().tolist())
        # generate predictions for the test set
        for ims, lbs, im_id in testloader:
            ims = ims.cuda()
            lbs = lbs.cuda()
            embedding = emb_model.get_embedding(batch=ims)
            if ema_model is not None:
                logits, _ = ema_model(embedding)
            else:
                logits, _ = model(embedding)
            output = torch.softmax(logits, dim=1)
            for j in range(len(lbs)):
                predictions['test'][im_id[j]] = int(output[j].cpu().tolist())
    return predictions


def predict_nih(model, ema_model, emb_model, trainloader_x, trainloader_u, testloader):
    """Generate predictions for the nih dataset

    :param model: Model
    :param ema_model: EMA-Model
    :param emb_model: Embedding model
    :param trainloader_x: Dataloader for the train set
    :param trainloader_u: Dataloader for the train set
    :param testloader: Dataloader for the test set
    :return: Dict of artificial expert annotations
    """
    model.eval()

    predictions = {}
    with torch.no_grad():
        # use expert labels as predictions for the labeled set
        for ims, lbs, im_id in trainloader_x:
            for j in range(len(lbs)):
                predictions[im_id[j]] = int(lbs[j])
        # generate predictions for the unlabeled set
        for ims, lbs, im_id in trainloader_u:
            ims = ims.cuda()
            lbs = lbs.cuda()
            embedding = emb_model.get_embedding(batch=ims)
            if ema_model is not None:
                logits, _ = ema_model(embedding)
            else:
                logits, _ = model(embedding)
            output = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(output, dim=1).cpu().numpy()
            for j in range(len(lbs)):
                predictions[im_id[j]] = int(predicted_class[j])
        # generate predictions for the test set
        for ims, lbs, im_id in testloader:
            ims = ims.cuda()
            lbs = lbs.cuda()
            embedding = emb_model.get_embedding(batch=ims)
            if ema_model is not None:
                logits, _ = ema_model(embedding)
            else:
                logits, _ = model(embedding)
            output = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(output, dim=1).cpu().numpy()
            for j in range(len(lbs)):
                predictions[im_id[j]] = int(predicted_class[j])

    return predictions


def main():
    parser = argparse.ArgumentParser(description='CoMatch Cifar Training')
    parser.add_argument('--root', default='./data', type=str, help='dataset directory')
    parser.add_argument('--wresnet-k', default=2, type=int,
                        help='width factor of wide resnet')
    parser.add_argument('--wresnet-n', default=28, type=int,
                        help='depth of wide resnet')    
    parser.add_argument('--dataset', type=str, default='CIFAR100',
                        help='number of classes in dataset')
    parser.add_argument('--n-classes', type=int, default=2,
                         help='number of classes in dataset')
    parser.add_argument('--n-labeled', type=int, default=40,
                        help='number of labeled samples for training')
    parser.add_argument('--n-epoches', type=int, default=512,
                        help='number of training epoches')
    parser.add_argument('--batchsize', type=int, default=64,
                        help='train batch size of labeled samples')
    parser.add_argument('--mu', type=int, default=7,
                        help='factor of train batch size of unlabeled samples')
    parser.add_argument('--n-imgs-per-epoch', type=int, default=64 * 1024,
                        help='number of training images for each epoch')
    
    parser.add_argument('--eval-ema', default=True, help='whether to use ema model for evaluation')
    parser.add_argument('--ema-m', type=float, default=0.999)
    
    parser.add_argument('--lam-u', type=float, default=1.,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--lr', type=float, default=0.03,
                        help='learning rate for training')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for random behaviors, no seed if negtive')
    
    parser.add_argument('--temperature', default=0.2, type=float, help='softmax temperature')
    parser.add_argument('--low-dim', type=int, default=64)
    parser.add_argument('--lam-c', type=float, default=1,
                        help='coefficient of contrastive loss')    
    parser.add_argument('--contrast-th', default=0.8, type=float,
                        help='pseudo label graph threshold')   
    parser.add_argument('--thr', type=float, default=0.95,
                        help='pseudo label threshold')   
    parser.add_argument('--alpha', type=float, default=0.9)   
    parser.add_argument('--queue-batch', type=float, default=5, 
                        help='number of batches stored in memory bank')    
    parser.add_argument('--exp-dir', default='EmbeddingCM_bin', type=str, help='experiment id')
    parser.add_argument('--checkpoint', default='', type=str, help='use pretrained model')
    parser.add_argument('--ex_strength', default=60, help='Strength of the expert')
    
    args = parser.parse_args()
    
    logger, output_dir = setup_default_logging(args)
    logger.info(dict(args._get_kwargs()))
    
    tb_logger = SummaryWriter(output_dir)

    if args.seed >= 0:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    n_iters_per_epoch = args.n_imgs_per_epoch // args.batchsize  # 1024
    n_iters_all = n_iters_per_epoch * args.n_epoches  # 1024 * 200

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.n_labeled}")
    
    model, criteria_x, ema_model = set_model(args)
    emb_model = EmbeddingModel(os.getcwd(), args.dataset)
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))

    if 'cifar' in args.dataset.lower():
        expert = CIFAR100Expert(20, int(args.ex_strength), 1, 0, 123)
        dltrain_x, dltrain_u = cifar.get_train_loader(
            args.dataset, expert, args.batchsize, args.mu, n_iters_per_epoch, L=args.n_labeled, root=args.root,
            method='comatch')
        dlval = cifar.get_val_loader(args.dataset, expert, batch_size=64, num_workers=2)
    elif 'nih' in args.dataset.lower():
        expert = NIHExpert(int(args.ex_strength), 2)
        dltrain_x, dltrain_u = nih.get_train_loader(
            expert, args.batchsize, args.mu, n_iters_per_epoch, L=args.n_labeled, root=args.root,
            method='comatch')
        dlval = nih.get_val_loader(expert, batch_size=64, num_workers=2)

    wd_params, non_wd_params = [], []
    for name, param in model.named_parameters():
        if 'bn' in name:
            non_wd_params.append(param)  
        else:
            wd_params.append(param)
    param_list = [
        {'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
    optim = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay,
                            momentum=args.momentum, nesterov=True)

    lr_schdlr = WarmupCosineLrScheduler(optim, n_iters_all, warmup_iter=0)
    
    model, ema_model, optim, lr_schdlr, start_epoch, metrics, prob_list, queue = \
        load_from_checkpoint(output_dir, model, ema_model, optim, lr_schdlr)

    # memory bank
    args.queue_size = args.queue_batch*(args.mu+1)*args.batchsize
    if queue is not None:
        queue_feats = queue['queue_feats']
        queue_probs = queue['queue_probs']
        queue_ptr = queue['queue_ptr']
    else:
        queue_feats = torch.zeros(args.queue_size, args.low_dim).cuda()
        queue_probs = torch.zeros(args.queue_size, args.n_classes).cuda()
        queue_ptr = 0

    train_args = dict(
        model=model,
        ema_model=ema_model,
        emb_model=emb_model,
        prob_list=prob_list,
        criteria_x=criteria_x,
        optim=optim,
        lr_schdlr=lr_schdlr,
        dltrain_x=dltrain_x,
        dltrain_u=dltrain_u,
        args=args,
        n_iters=n_iters_per_epoch,
        logger=logger
    )
    
    best_acc = -1
    best_epoch = 0

    if metrics is not None:
        best_acc = metrics['best_acc']
        best_epoch = metrics['best_epoch']
    logger.info('-----------start training--------------')
    for epoch in range(start_epoch, args.n_epoches):
        
        loss_x, loss_u, loss_c, mask_mean, num_pos, guess_label_acc, queue_feats, queue_probs, queue_ptr, prob_list = \
        train_one_epoch(epoch, **train_args, queue_feats=queue_feats,queue_probs=queue_probs,queue_ptr=queue_ptr)

        top1, ema_top1 = evaluate(model, ema_model, emb_model, dlval)


        tb_logger.add_scalar('loss_x', loss_x, epoch)
        tb_logger.add_scalar('loss_u', loss_u, epoch)
        tb_logger.add_scalar('loss_c', loss_c, epoch)
        tb_logger.add_scalar('guess_label_acc', guess_label_acc, epoch)
        tb_logger.add_scalar('test_acc', top1, epoch)
        tb_logger.add_scalar('test_ema_acc', ema_top1, epoch)
        tb_logger.add_scalar('mask', mask_mean, epoch)
        tb_logger.add_scalar('num_pos', num_pos, epoch)

        if best_acc < top1:
            best_acc = top1
            best_epoch = epoch

        logger.info("Epoch {}. Acc: {:.4f}. Ema-Acc: {:.4f}. best_acc: {:.4f} in epoch{}".
                    format(epoch, top1, ema_top1, best_acc, best_epoch))
        
        save_obj = {
            'model': model.state_dict(),
            'ema_model': ema_model.state_dict(),
            'optimizer': optim.state_dict(),
            'lr_scheduler': lr_schdlr.state_dict(),
            'prob_list': prob_list,
            'queue': {'queue_feats':queue_feats, 'queue_probs':queue_probs, 'queue_ptr':queue_ptr},
            'metrics': {'best_acc': best_acc, 'best_epoch': best_epoch},
            'epoch': epoch,
        }
        torch.save(save_obj, os.path.join(output_dir, 'ckp.latest'))
    _, _ = evaluate(model, ema_model, emb_model, dlval)
    if 'cifar' in args.dataset.lower():
        predictions = predict_cifar(model, ema_model, emb_model, dltrain_x, dltrain_u, dlval)
    elif 'nih' in args.dataset.lower():
        predictions = predict_nih(model, ema_model, emb_model, dltrain_x, dltrain_u , dlval)
    logger.info("***** Generate Predictions *****")
    with open(f'predictions/{args.exp_dir}_{args.dataset.lower()}_expert{args.ex_strength}.{args.seed}@{args.n_labeled}_predictions.json', 'w') as f:
        json.dump(predictions, f)


        
if __name__ == '__main__':
    main()
