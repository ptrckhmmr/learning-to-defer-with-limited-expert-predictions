import copy
import sys

import torch
import torch.nn as nn
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import os
from torch.utils.tensorboard.writer import SummaryWriter
import math

from src.data_loading import CIFAR100_Dataloader, NIH_Dataloader
from src.utils import load_from_checkpoint, save_to_checkpoint, log_test_metrics, get_train_dir
from src.metrics import accuracy, AverageMeter, metrics_print, metrics_print_2step
from src.models import WideResNet, Resnet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def my_CrossEntropyLoss(outputs, labels):
    """Get custom cross entropy loss

    :param outputs: Model outputs
    :param labels: Labels
    :return: Loss
    """
    batch_size = outputs.size()[0]  # batch_size
    outputs = - torch.log2(outputs[range(batch_size), labels]+1e-12) # pick the values corresponding to the labels
    return torch.sum(outputs) / batch_size

def train_classifier(args, train_loader, model, optimizer, scheduler, epoch):
    """Train classifier for one epoch on the training set

    :param args: Training arguments
    :param train_loader: Dataloader for the training set
    :param model: Classifier model
    :param optimizer: Optimizer
    :param scheduler: Learning rate scheduler
    :param epoch: Current epoch

    :return: Loss
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, indices) in enumerate(train_loader):
        target = target.to(device)
        input = input.to(device)

        # compute output
        output = model(input)
        # compute loss
        if 'cifar100' in args['approach']:
            loss = my_CrossEntropyLoss(output, target)
        elif 'nih' in args['approach']:
            loss = nn.CrossEntropyLoss()(output, target)
        else:
            print('Classifier model not defined')
            sys.exit()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))

    return losses.avg


def validate_classifier(args, val_loader, model):
    """Perform validation for the classifier

    :param args: Training arguments
    :param val_loader: Dataloader for the validation set
    :param model: Classifier model
    :return: Validation accuracy
    """

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, indices) in enumerate(val_loader):
        target = target.to(device)
        input = input.to(device)

        # compute output
        with torch.no_grad():
            output = model(input)
        # compute loss
        if 'cifar100' in args['approach']:
            loss = my_CrossEntropyLoss(output, target)
        elif 'nih' in args['approach']:
            loss = nn.CrossEntropyLoss()(output, target)
        else:
            print('Classifier model not defined')
            sys.exit()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def run_classifier(args, expert_fn, epochs, train_batch_size, test_batch_size, seed, num_classes):
    """Run classifier training

    :param args: Training arguments
    :param expert_fn: Expert labels
    :param epochs: Number of epochs to train
    :param train_batch_size: Batch size for train set
    :param test_batch_size: Batch size for the test set
    :param seed: Random seed
    :param num_classes: Number of classes
    :return: Classifier model
    """
    global best_prec1
    # get train directory
    train_dir = get_train_dir(os.getcwd(), args, 'confidence-classifier')

    # Data loading and model initiation
    if 'cifar100' in args['approach']:
        model_classifier = WideResNet(28, num_classes, 4, dropRate=0)
        cifar_dl = CIFAR100_Dataloader(train_batch_size=train_batch_size, test_batch_size=test_batch_size,
                                               seed=seed, small_version=False)
        train_loader, val_loader, test_loader = cifar_dl.get_data_loader()
        lr = 0.1
    elif 'nih' in args['approach']:
        model_classifier = Resnet(num_classes)
        nih_dl = NIH_Dataloader(labeler_id=args['ex_strength'], train_batch_size=train_batch_size, test_batch_size=test_batch_size, seed=seed)
        train_loader, test_loader = nih_dl.get_data_loader()
        lr = 0.001
    else:
        print('Classifier model not defined')
        sys.exit()
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model_classifier.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model_classifier = model_classifier.to(device)

    # optionally resume from a checkpoint
    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    optimizer = torch.optim.SGD(model_classifier.parameters(), lr,
                                momentum=0.9, nesterov=True,
                                weight_decay=5e-4)

    # cosine learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * 200)

    # try to load previous training from checkpoint
    model_classifier, optimizer, \
    scheduler, start_epoch, best_loss = load_from_checkpoint(train_dir, model_classifier, optimizer, scheduler, seed)
    best_model = model_classifier
    if best_loss is None:
        best_loss = 100
    for epoch in range(start_epoch+1, epochs):
        # train for one epoch
        loss = train_classifier(args, train_loader, model_classifier, optimizer, scheduler, epoch)
        if loss < best_loss:
            print(f'save model with new best loss {loss} at epoch {epoch}')
            best_model = model_classifier
            save_to_checkpoint(train_dir, epoch, best_model, optimizer, scheduler, loss, seed)
            best_loss = loss
    validate_classifier(args, test_loader, model_classifier)
    save_to_checkpoint(train_dir, 200, best_model, optimizer, scheduler, best_loss, seed)
    return model_classifier


def train_expert(args, train_loader, model, optimizer, scheduler, epoch, expert_fn):
    """Train expert model for one epoch on the train set

    :param args: Training arguments
    :param train_loader: Dataloader for the train set
    :param model: Expert model
    :param optimizer: Optimizer
    :param scheduler: Leraning rate scheduler
    :param epoch: Current epoch
    :param expert_fn: Expert labels
    :return: Loss
    """

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    loss_log = []
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, indices) in enumerate(train_loader):
        target = target.to(device)
        input = input.to(device)

        # compute output
        output = model(input)

        # compute new target
        batch_size = output.size()[0]  # batch_size

        m = expert_fn(indices)
        for j in range(0, batch_size):
            m[j] = 1 - (m[j] == target[j].item())

        m = torch.tensor(m, dtype=torch.long)
        m = m.to(device)
        # compute loss
        if 'cifar100' in args['approach']:
            loss = my_CrossEntropyLoss(output, m)
        elif 'nih' in args['approach']:
            loss = nn.CrossEntropyLoss()(output, m)
        else:
            print('Classifier model not defined')
            sys.exit()

        if math.isnan(loss.data.item()) or loss.data.item() == float('inf'):
            print(f'Loss at {i}: {loss.data.item()}, output: {output[:5]}, m: {m[:5]}')
            print(f'loss_log: {loss_log}')
            sys.exit()
        loss_log.append(loss)
        # measure accuracy and record loss
        prec1 = accuracy(output.data, m, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))

    return losses.avg


def validate_expert(args, val_loader, model, expert_fn, test=False):
    """Perform validation with dhe expert model

    :param args: Training arguments
    :param val_loader: Dataloader for the validation set
    :param model: Expert model
    :param expert_fn: Expert labels
    :param test: True if validation on the test set
    :return: Accuracy
    """

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, indices) in enumerate(val_loader):
        target = target.to(device)
        input = input.to(device)

        # compute output
        with torch.no_grad():
            output = model(input)
        # expert prediction
        batch_size = output.size()[0]  # batch_size
        m = expert_fn(indices, test=test)
        for j in range(0, batch_size):
            m[j] = 1 - (m[j] == target[j].item())
        m = torch.tensor(m, dtype=torch.long)
        m = m.to(device)
        # compute loss
        if 'cifar100' in args['approach']:
            loss = my_CrossEntropyLoss(output, m)
        elif 'nih' in args['approach']:
            loss = nn.CrossEntropyLoss()(output, m)
        else:
            print('Classifier model not defined')
            sys.exit()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, m, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def run_expert(args, model_classifier, expert_fn, epochs, train_batch_size, test_batch_size, seed, num_classes):
    """Run training of the expert model

    :param args: Training arguments
    :param model_classifier: Classifier model
    :param expert_fn: Expert label
    :param epochs: Number of epochs to train
    :param train_batch_size: Batch size for the train set
    :param test_batch_size: Batch size for the test set
    :param seed: Random seed
    :param num_classes: Number of classes
    :return: Best metrics
    """
    global best_prec1
    # get train directory
    train_dir = get_train_dir(os.getcwd(), args, 'confidence-expert')
    # initialize tensorbaord writer
    writer = SummaryWriter(train_dir + 'logs/')
    # Data loading and model initiation
    if 'cifar100' in args['approach']:
        model_expert = WideResNet(10, 2, 4, dropRate=0)
        cifar_dl = CIFAR100_Dataloader(train_batch_size=train_batch_size, test_batch_size=test_batch_size,
                                               seed=seed, small_version=False)
        train_loader, val_loader, test_loader = cifar_dl.get_data_loader()
        lr = 0.1
    elif 'nih' in args['approach']:
        nih_dl = NIH_Dataloader(labeler_id=args['ex_strength'], train_batch_size=train_batch_size, test_batch_size=test_batch_size, seed=seed)
        train_loader, test_loader = nih_dl.get_data_loader()
        model_expert = Resnet(2)
        lr = 0.00005
    else:
        print('Classifier model not defined')
        sys.exit()

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model_expert.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()

    model_expert = model_expert.to(device)

    # optionally resume from a checkpoint
    cudnn.benchmark = True

    # define optimizer
    optimizer = torch.optim.SGD(model_expert.parameters(), lr,
                                momentum=0.9, nesterov=True,
                                weight_decay=5e-4)

    # cosine learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * 200)

    # try to load previous training from checkpoint
    model_expert, optimizer, \
    scheduler, start_epoch, best_metrics = load_from_checkpoint(train_dir, model_expert, optimizer, scheduler, seed)
    if best_metrics is None:
        best_loss = 100
    else:
        best_loss = best_metrics['system loss']
    best_model = model_expert
    for epoch in range(start_epoch, epochs):
        # train for one epoch
        loss = train_expert(args, train_loader, model_expert, optimizer, scheduler, epoch, expert_fn)
        #_ = validate_expert(args, test_loader, model_expert, epoch, expert_fn, num_classes, test=True)
        test_metrics = metrics_print_2step(model_classifier, model_expert, expert_fn, num_classes, test_loader, test=True)
        test_metrics['system loss'] = loss
        # log metrics and save framework to checkpoint
        log_test_metrics(writer, epoch, test_metrics, num_classes)
        if loss < best_loss:
            best_loss = loss
            print(f'save model with new best loss {loss} at epoch {epoch}')
            best_metrics = test_metrics
            best_model = copy.deepcopy(model_expert)
            save_to_checkpoint(train_dir, epoch, model_expert, optimizer, scheduler, best_metrics, seed)
    best_metrics = metrics_print_2step(model_classifier, best_model, expert_fn, num_classes, test_loader, test=True)
    best_metrics['system loss'] = best_loss
    log_test_metrics(writer, epochs, best_metrics, num_classes)
    validate_expert(args, test_loader, best_model, expert_fn, test=True)
    save_to_checkpoint(train_dir, epochs, best_model, optimizer, scheduler, best_metrics, seed)
    return best_metrics



