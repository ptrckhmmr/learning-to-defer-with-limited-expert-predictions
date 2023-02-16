import copy
import torch
import torch.nn as nn
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import os, sys
import math
from torch.utils.tensorboard.writer import SummaryWriter

from src.data_loading import CIFAR100_Dataloader, NIH_Dataloader
from src.utils import load_from_checkpoint, save_to_checkpoint, log_test_metrics, get_train_dir
from src.metrics import accuracy, AverageMeter, metrics_print, fairness_print
from src.models import WideResNet, Resnet, NIH_Network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def reject_CrossEntropyLoss(outputs, m, labels, m2, n_classes):
    """The L_{CE} loss implementation

    :param outputs: Model output
    :param m: Cost of deferring to expert cost of classifier predicting (I_{m =y})
    :param labels: Targets
    :param m2: Cost of classifier predicting (alpha* I_{m\neq y} + I_{m =y})
    :param n_classes: Number of classes
    :return: L_{CE} loss
    """
    batch_size = outputs.size()[0]  # batch_size
    rc = [n_classes] * batch_size
    loss_defer = -m * torch.log2(outputs[range(batch_size), rc]+1e-6)
    loss_class = -m2 * torch.log2(outputs[range(batch_size), labels]+1e-6)
    loss_total = torch.sum(loss_class + loss_defer) /batch_size
    return loss_total


def my_CrossEntropyLoss(outputs, labels):
    """Regular cross entropy loss

    :param outputs: Model outputs
    :param labels: Targets
    :return: Loss
    """
    batch_size = outputs.size()[0]  # batch_size
    outputs = - torch.log2(outputs[range(batch_size), labels]+1e-12)  # regular CE
    return torch.sum(outputs) / batch_size


def train_reject(args, train_loader, model, optimizer, scheduler, epoch, expert_fn, n_classes, alpha):
    """Train one epoch on the training set

    :param args: Training arguments
    :param train_loader: Dataloader for the training set
    :param model: Model
    :param optimizer: Optimizer
    :param scheduler: Leraning Rate Scheduler
    :param epoch: Epoch
    :param expert_fn: Expert labels
    :param n_classes: Number of classes
    :param alpha: Alpha parameter in L_{CE}^{\alpha}
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

        # get expert  artificial_expert_labels and costs
        batch_size = output.size()[0]  # batch_size
        m = expert_fn(indices)
        m2 = [alpha] * batch_size
        if 'cifar' in args['approach'] or True:
            for j in range(0, batch_size):
                if m[j] == target[j].item():
                    m[j] = 1
                    m2[j] = alpha
                else:
                    m[j] = 0
                    m2[j] = 1
        m = torch.tensor(m)
        m2 = torch.tensor(m2)
        m = m.to(device)
        m2 = m2.to(device)
        # done getting expert artificial_expert_labels and costs
        # compute loss
        criterion = nn.CrossEntropyLoss()

        loss = reject_CrossEntropyLoss(output, m, target, m2, n_classes)

        if math.isnan(loss.data.item()) or loss.data.item() == float('inf'):
            print(f'Loss at {i}: {loss.data.item()}, output: {output[:5]}, m: {m[:5]}')
            print(f'loss_log: {loss_log}')
            sys.exit()
        loss_log.append(loss.data.item())
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



def train_reject_class(train_loader, model, optimizer, scheduler, epoch):
    """Train for one epoch on the training set without deferral

    :param train_loader: Dataloader for the training set
    :param model: Model
    :param optimizer: Optimizer
    :param scheduler: Learning rate scheduler
    :param epoch: Epoch
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
        loss = my_CrossEntropyLoss(output, target)

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


def validate_reject(val_loader, model, expert_fn, n_classes, test=False):
    """Perform validation on the validation set with deferral

    :param val_loader: Dataloader fo the validation set
    :param model: Model
    :param expert_fn: Expert labels
    :param n_classes: Number of classes
    :param test: True if validation on test set
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
        alpha = 1
        m2 = [0] * batch_size
        for j in range(0, batch_size):
            if m[j] == target[j].item():
                m[j] = 1
                m2[j] = alpha
            else:
                m[j] = 0
                m2[j] = 1
        m = torch.tensor(m)
        m2 = torch.tensor(m2)
        m = m.to(device)
        m2 = m2.to(device)
        # compute loss
        loss = reject_CrossEntropyLoss(output, m, target, m2, n_classes)

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


def run_reject(args, expert_fn, epochs, alpha, train_batch_size=128, test_batch_size=128, seed=0, num_classes=20):
    """Run training of the classifier with deferral

    :param args: Training arguments
    :param expert_fn: Expert labels
    :param epochs: Number of epochs to train
    :param alpha: Alpha parameter in L_{CE}^{\alpha}
    :param train_batch_size: Batch size for the train set
    :param test_batch_size: Batch size for the test set
    :param seed: Random seed
    :param num_classes: Number of classes
    :return: Best system metrics
    """
    # get train directory
    train_dir = get_train_dir(os.getcwd(), args, 'monzannar_sontag')
    # initialize tensorbaord writer
    writer = SummaryWriter(train_dir + 'logs/')
    # initialize model
    if 'cifar100' in args['approach']:
        model = WideResNet(28, num_classes + 1, 4, dropRate=0)
        # get cifer 100 data loader
        cifar_dl = CIFAR100_Dataloader(train_batch_size=train_batch_size, test_batch_size=test_batch_size,
                                               seed=seed, small_version=False)
        train_loader, val_loader, test_loader = cifar_dl.get_data_loader()
        lr = 0.1
    elif 'nih' in args['approach']:
        model = Resnet(num_classes+1)
        nih_dl = NIH_Dataloader(labeler_id=args['ex_strength'], train_batch_size=train_batch_size, test_batch_size=test_batch_size, seed=seed)
        train_loader, test_loader = nih_dl.get_data_loader()
        lr = 0.0001
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.to(device)


    # optionally resume from a checkpoint
    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=0.9, nesterov=True,
                                weight_decay=5e-4)

    # cosine learning rate
    if 'cifar100' in args['approach']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * epochs)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * 200)

    # try to load previous training from checkpoint
    model, optimizer, \
    scheduler, start_epoch, best_metrics = load_from_checkpoint(train_dir, model, optimizer, scheduler, seed)
    if best_metrics is None:
        best_loss = 100
    else:
        best_loss = best_metrics['system loss']
    best_model = model
    for epoch in range(start_epoch, epochs):
        # train for one epoch
        loss = train_reject(args, train_loader, model, optimizer, scheduler, epoch, expert_fn, num_classes, alpha)

        test_metrics = metrics_print(model, expert_fn, num_classes, test_loader, test=True)
        test_metrics['system loss'] = loss
        # log metrics and save framework to checkpoint
        log_test_metrics(writer, epoch, test_metrics, num_classes)
        if loss < best_loss:
            best_loss = loss
            print(f'save new best loss {loss} at epoch {epoch}')
            best_metrics = test_metrics
            best_model = copy.deepcopy(model)
            save_to_checkpoint(train_dir, epoch, model, optimizer, scheduler, best_metrics, seed)
    best_metrics = metrics_print(best_model, expert_fn, num_classes, test_loader, test=True)
    #fairness_print(best_model, expert_fn, num_classes, test_loader, args, test=True, seed=seed)
    best_metrics['system loss'] = best_loss
    log_test_metrics(writer, epochs, best_metrics, num_classes)
    save_to_checkpoint(train_dir, epochs, best_model, optimizer, scheduler, best_metrics, seed)
    return best_metrics