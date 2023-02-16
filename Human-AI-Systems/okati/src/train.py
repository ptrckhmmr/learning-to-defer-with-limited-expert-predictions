import sys
import torch
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import os
from torch.utils.tensorboard.writer import SummaryWriter
import copy
import numpy as np
import pandas as pd

from src.data_loading import CIFAR100_Dataloader, NIH_Dataloader
from src.utils import load_from_checkpoint, save_to_checkpoint, log_test_metrics, get_train_dir, find_machine_samples
from src.metrics import get_system_metrics, AverageMeter, accuracy, metrics_print_2step
from src.models import WideResNet, Resnet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def one_hot(a, num_classes):
    """Get one hot encoding for class labels

    :param a: Class labels
    :param num_classes: Number of classes
    :return: One hot label
    """
    a = np.array(a).astype(int)
    one_hot = np.squeeze(np.eye(num_classes)[a.reshape(-1)])
    epsilon = 0.0005
    for i in range(one_hot.shape[0]):
        p = np.argmax(one_hot[i])
        for c in range(num_classes):
            if c == p:
                one_hot[i][c] = np.log(one_hot[i][c] - epsilon)
            else:
                one_hot[i][c] = np.log(one_hot[i][c] + epsilon/(num_classes-1))
    return one_hot


def my_CrossEntropyLoss(outputs, labels):
    """Cross entropy loss

    :param outputs: Model outputs
    :param labels: Labels
    :return: Loss
    """
    # m: expert costs, labels: ground truth, n_classes: number of classes
    batch_size = outputs.size()[0]  # batch_size
    outputs = - torch.log2(outputs[range(batch_size), labels]+1e-12) # pick the values corresponding to the labels
    return torch.sum(outputs) / batch_size


def train_classifier(args, train_loader, model, optimizer, scheduler, epoch, expert_fn, n_classes):
    """Train classifier model for one epoch on the training set

    :param args: Training arguments
    :param train_loader: Dataloader of the train set
    :param model: Classifier model
    :param optimizer: Optimizer
    :param scheduler: Learning rate scheduler
    :param epoch: Epoch
    :param expert_fn: Expert labels
    :param n_classes: Number of classes
    :return: Loss
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    loss_func = torch.nn.NLLLoss(reduction='none')
    # switch to train mode
    model.train()
    machine_loss = []

    end = time.time()
    with torch.no_grad():
        mprim = copy.deepcopy(model)
    for i, (X_batch, Y_batch, indices) in enumerate(train_loader):
        batch_size = Y_batch.size()[0]
        Y_batch = Y_batch.to(device)
        X_batch = X_batch.to(device)
        E_batch = expert_fn(indices)
        E_batch = torch.tensor(one_hot(E_batch, num_classes=n_classes), dtype=torch.float).to(device)

        with torch.no_grad():
            machine_scores_batch = mprim(X_batch)
            machine_loss_batch = loss_func(machine_scores_batch, Y_batch)
            machine_loss.extend(machine_loss_batch.detach())

            human_scores_batch = E_batch
            human_loss_batch = loss_func(human_scores_batch, Y_batch)

        machine_indices = find_machine_samples(machine_loss_batch, human_loss_batch)
        human_indices = []
        for j in range(batch_size):
            if j not in machine_indices:
                human_indices.append(j)
        assert len(machine_indices)+len(human_indices) == batch_size
        X_machine = X_batch[machine_indices]
        Y_machine = Y_batch[machine_indices]
        optimizer.zero_grad()
        loss = loss_func(model(X_machine), Y_machine)
        if 'cifar100' in args['approach']:
            total_loss = loss.sum()
        elif 'nih' in args['approach']:
            loss_u = loss_func(model(X_batch), Y_batch)
            total_loss = loss.sum() + 0.1*loss_u.sum()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        losses.update(loss.mean().data.item(), batch_size)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses))
    return losses.avg


def validate_classifier(args, val_loader, model, expert_fn, n_classes, test=False):
    """Perform validation on the validation set

    :param args: Training arguments
    :param val_loader: Dataloader of hte validation set
    :param model: Classifier model
    :param expert_fn: Expert labels
    :param n_classes: Number of classes
    :param test: True if validation on test set
    :return: Accuracy
    """
    loss_func = torch.nn.NLLLoss(reduction='none')

    machine_loss = []
    human_loss = []
    machine_scores = []
    human_scores = []
    targets = []
    # switch to evaluate mode
    model.eval()

    for i, (X_batch, Y_batch, indices) in enumerate(val_loader):
        batch_size = Y_batch.size()[0]
        Y_batch = Y_batch.to(device)
        X_batch = X_batch.to(device)
        E_batch = expert_fn(indices, test=test)
        E_batch = torch.tensor(one_hot(E_batch, num_classes=n_classes), dtype=torch.float).to(device)
        with torch.no_grad():
            machine_scores_batch = model(X_batch)
            machine_loss_batch = loss_func(machine_scores_batch, Y_batch)
            human_scores_batch = E_batch # expert_model(X_batch)
            human_loss_batch = loss_func(human_scores_batch, Y_batch)
        machine_loss += machine_loss_batch.cpu().tolist()
        human_loss += human_loss_batch.cpu().tolist()
        machine_scores += machine_scores_batch.cpu().tolist()
        human_scores += human_scores_batch.cpu().tolist()
        targets += Y_batch.cpu().tolist()

    machine_indices = find_machine_samples(torch.tensor(machine_loss), torch.tensor(human_loss))
    human_indices = []
    for j in range(len(targets)):
        if j not in machine_indices:
            human_indices.append(j)
    machine_preds = np.argmax(machine_scores, axis=1)
    human_preds = np.argmax(human_scores, axis=1)
    targets = np.array(targets)
    metrics = get_system_metrics(machine_preds, human_preds, machine_indices.cpu().tolist(), human_indices, targets, n_classes)
    return metrics


def run_classifier(args, expert_fn, epochs, train_batch_size, test_batch_size, seed, num_classes):
    """Run training of the classifier model

    :param args: Training arguments
    :param expert_fn: Expert labels
    :param epochs: Number of epochs to train
    :param train_batch_size: Batch size of train set
    :param test_batch_size: Batch size of test set
    :param seed: Random seed
    :param num_classes: Number of classes
    :return: Classifier model
    """
    global best_prec1
    # get train directory
    train_dir = get_train_dir(os.getcwd(), args, 'triage-classifier')

    # Data loading and model initiation
    if 'cifar100' in args['approach']:
        model_classifier = WideResNet(28, num_classes, 4, dropRate=0)
        cifar_dl = CIFAR100_Dataloader(train_batch_size=train_batch_size, test_batch_size=test_batch_size,
                                               seed=seed, small_version=False)
        train_loader, val_loader, test_loader = cifar_dl.get_data_loader()
        lr = 0.001
    elif 'nih' in args['approach']:
        model_classifier = Resnet(num_classes)
        nih_dl = NIH_Dataloader(labeler_id=args['ex_strength'], train_batch_size=train_batch_size, test_batch_size=test_batch_size, seed=seed)
        train_loader, test_loader = nih_dl.get_data_loader()
        lr = 0.00001
    else:
        print('Dataset not defined')
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
    scheduler, start_epoch, best_metrics = load_from_checkpoint(train_dir, model_classifier, optimizer, scheduler, seed)
    best_model = model_classifier
    if best_metrics is None:
        best_metrics = {'classifier_loss': 1e6}
    for epoch in range(start_epoch+1, epochs):
        # train for one epoch
        loss = train_classifier(args, train_loader, model_classifier, optimizer, scheduler, epoch, expert_fn, num_classes)
        print(f'loss@epoc{epoch}: {loss}')
        test_metrics = validate_classifier(args, test_loader, model_classifier, expert_fn,
                                           num_classes, test=True)
        if loss < best_metrics['classifier_loss']:
            print(f'save model with new best loss {loss} at epoch {epoch}')
            best_model = model_classifier
            best_metrics = test_metrics
            best_metrics['classifier_loss'] = loss
            save_to_checkpoint(train_dir, epoch, best_model, optimizer, scheduler, best_metrics, seed)
    _ = validate_classifier(args, test_loader, best_model, expert_fn, num_classes, test=True)
    save_to_checkpoint(train_dir, epochs, best_model, optimizer, scheduler, best_metrics, seed)
    return best_model


def train_expert(args, classifier_model, train_loader, model, optimizer, scheduler, epoch, expert_fn, n_classes):
    """Train expert model for one epoch on the training set

    :param args: Training arguments
    :param classifier_model: Classification model
    :param train_loader: Dataloader for the train set
    :param model: Expert model
    :param optimizer: Optimizer
    :param scheduler: Scheduler
    :param epoch: Epoch
    :param expert_fn: Expert labels
    :param n_classes: Number of classes
    :return: Loss
    """
    loss_func = torch.nn.NLLLoss(reduction='none')

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (X_batch, Y_batch, indices) in enumerate(train_loader):
        batch_size = Y_batch.size()[0]
        Y_batch = Y_batch.to(device)
        X_batch = X_batch.to(device)
        E_batch = expert_fn(indices)
        E_batch = torch.tensor(one_hot(E_batch, num_classes=n_classes), dtype=torch.float).to(device)

        with torch.no_grad():
            machine_scores_batch = classifier_model(X_batch)
            machine_loss_batch = loss_func(machine_scores_batch, Y_batch)
            human_scores_batch = E_batch
            human_loss_batch = loss_func(human_scores_batch, Y_batch)

        machine_indices = find_machine_samples(machine_loss_batch, human_loss_batch)
        g_labels_batch = torch.tensor([0 if j in machine_indices else 1 for j in range(batch_size)]).to(device)

        optimizer.zero_grad()
        gpred = model(X_batch)
        g_loss = loss_func(gpred, g_labels_batch)
        g_loss.mean().backward()
        optimizer.step()
        scheduler.step()

        # measure accuracy and record loss
        losses.update(g_loss.mean().data.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses))

    return losses.avg


def validate_expert(args, classifier_model, val_loader, model, expert_fn, n_classes, test=True):
    """Perform validation

    :param args: Training arguments
    :param classifier_model: Classifier model
    :param val_loader: Dataloader
    :param model: Expert model
    :param expert_fn: Expert labels
    :param n_classes: Number of classes
    :param test: True if validation on the test set
    :return: System metrics
    """
    loss_func = torch.nn.NLLLoss(reduction='none')

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    targets = []
    g_predictions = []
    machine_scores = []
    human_scores = []
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (X_batch, Y_batch, indices) in enumerate(val_loader):
        batch_size = Y_batch.size()[0]
        Y_batch = Y_batch.to(device)
        X_batch = X_batch.to(device)
        E_batch = expert_fn(indices, test=test)

        E_batch = torch.tensor(one_hot(E_batch, num_classes=n_classes), dtype=torch.float).to(device)

        with torch.no_grad():
            machine_scores_batch = classifier_model(X_batch)
            machine_loss_batch = loss_func(machine_scores_batch, Y_batch)
            human_scores_batch = E_batch
            human_loss_batch = loss_func(human_scores_batch, Y_batch)

        train_dir = get_train_dir(os.getcwd(), args, 'triage-classifier')
        machine_indices = find_machine_samples(machine_loss_batch, human_loss_batch)
        g_labels_batch = torch.tensor([0 if j in machine_indices else 1 for j in range(batch_size)]).to(device)

        gpred = model(X_batch)
        g_loss = loss_func(gpred, g_labels_batch)

        g_predictions += torch.argmax(gpred, dim=1).cpu().tolist()

        targets += Y_batch.cpu().tolist()
        machine_scores += machine_scores_batch.cpu().tolist()
        human_scores += human_scores_batch.cpu().tolist()

        # measure accuracy and record loss
        prec1 = accuracy(gpred.data, g_labels_batch, topk=(1,))[0]
        losses.update(g_loss.mean().data.item(), batch_size)
        top1.update(prec1.item(), batch_size)

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
    machine_indices = []
    human_indices = []
    for j, g_p in enumerate(g_predictions):
        if g_p == 0:
            machine_indices.append(j)
        elif g_p == 1:
            human_indices.append(j)
        else:
            print('error')
            sys.exit()
    machine_preds = np.argmax(machine_scores, axis=1)
    human_preds = np.argmax(human_scores, axis=1)

    metrics = get_system_metrics(machine_preds, human_preds, machine_indices, human_indices, targets, n_classes)
    metrics['gmodel'] = top1.avg
    return metrics


def fairness_test(args, classifier_model, val_loader, model, expert_fn, n_classes, test=True, seed=123):
    """Analyze fairness for the NIH dataset w.r.t. the patients age and gender

    :param args: Training arguments
    :param classifier_model: Classifier model
    :param val_loader: Dataloader
    :param model: Expert model
    :param expert_fn: Expert labels
    :param n_classes: Number of classes
    :param test: True if validation on the test set
    :param seed: Random seed

    :return:
    """
    loss_func = torch.nn.NLLLoss(reduction='none')

    targets = []
    indices = []
    g_predictions = []
    machine_scores = []
    human_scores = []
    # switch to evaluate mode
    model.eval()

    for i, (X_batch, Y_batch, idxs) in enumerate(val_loader):
        batch_size = Y_batch.size()[0]
        Y_batch = Y_batch.to(device)
        X_batch = X_batch.to(device)
        E_batch = expert_fn(idxs, test=test)

        E_batch = torch.tensor(one_hot(E_batch, num_classes=n_classes), dtype=torch.float).to(device)


        with torch.no_grad():
            machine_scores_batch = classifier_model(X_batch)
            machine_loss_batch = loss_func(machine_scores_batch, Y_batch)
            human_scores_batch = E_batch
            human_loss_batch = loss_func(human_scores_batch, Y_batch)

        train_dir = get_train_dir(os.getcwd(), args, 'triage-classifier')
        machine_indices = find_machine_samples(machine_loss_batch, human_loss_batch)
        g_labels_batch = torch.tensor([0 if j in machine_indices else 1 for j in range(batch_size)]).to(device)

        gpred = model(X_batch)
        g_loss = loss_func(gpred, g_labels_batch)

        g_predictions += torch.argmax(gpred, dim=1).cpu().tolist()

        targets += Y_batch.cpu().tolist()
        indices += idxs
        machine_scores += machine_scores_batch.cpu().tolist()
        human_scores += human_scores_batch.cpu().tolist()

    img_dir = os.getcwd()[:-len('human-AI-systems/okati')] + 'nih_images/'
    metadata = pd.read_csv(img_dir + 'nih_meta_data.csv')

    machine_preds = np.argmax(machine_scores, axis=1)
    human_preds = np.argmax(human_scores, axis=1)
    correct = []
    fpr = []
    fnr = []
    gender = []
    age = []
    for j, g_p in enumerate(g_predictions):
        gender.append(metadata.loc[metadata['Image Index'] == indices[j]]['Patient Gender'].values[0])
        age.append(metadata.loc[metadata['Image Index'] == indices[j]]['Patient Age'].values[0])
        if g_p == 0:
            correct.append((machine_preds[j] == targets[j]))
            fpr.append((machine_preds[j] == 1) and (targets[j] == 0))
            fnr.append((machine_preds[j] == 0) and (targets[j] == 1))
        elif g_p == 1:
            correct.append((human_preds[j] == targets[j]))
            fpr.append((human_preds[j] == 1) and (targets[j] == 0))
            fnr.append((human_preds[j] == 0) and (targets[j] == 1))
        else:
            print('error')
            sys.exit()

    os.makedirs(os.path.join(os.getcwd(), 'fairness'), exist_ok=True)
    fairness_results = pd.DataFrame({'correct': correct, 'fpr': fpr, 'fnr': fnr, 'gender': gender, 'age': age})
    fairness_results.to_csv(f'fairness/fair_{args["approach"]}_{args["ex_strength"]}ex_{seed}.csv')


def run_expert(args, model_classifier, expert_fn, epochs, train_batch_size, test_batch_size, seed, num_classes):
    """Run training for the expert model

    :param args: Training arguments
    :param model_classifier: Classifier model
    :param expert_fn: Expert labels
    :param epochs: Number of epochs to train
    :param train_batch_size: Batch size of the train set
    :param test_batch_size: Batch size of the test set
    :param seed: Random seed
    :param num_classes: Number of classes
    :return: System metrics
    """
    global best_prec1
    # get train directory
    train_dir = get_train_dir(os.getcwd(), args, 'triage-expert')
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
        val_loader = None
        model_expert = Resnet(2)
        lr = 0.001
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

    # define loss function (criterion) and optimizer
    # define loss function (criterion) and optimizer
    optimizer = torch.optim.SGD(model_expert.parameters(), lr,
                                momentum=0.9, nesterov=True,
                                weight_decay=5e-4)

    # cosine learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * 200)

    # try to load previous training from checkpoint
    model_expert, optimizer, \
    scheduler, start_epoch, best_metrics = load_from_checkpoint(train_dir, model_expert, optimizer, scheduler, seed)

    if 'cifar100' in args['approach']:
        if best_metrics is None:
            best_acc = 0
        else:
            best_acc = best_metrics['system accuracy']
        best_expert = model_expert
        for epoch in range(start_epoch, epochs):
            # train for one epoch
            loss = train_expert(args, model_classifier, train_loader, model_expert, optimizer, scheduler,
                                epoch, expert_fn, num_classes)
            val_metrics = validate_expert(args, model_classifier, val_loader, model_expert,
                                          expert_fn, num_classes, test=False)
            test_metrics = validate_expert(args, model_classifier, test_loader, model_expert,
                                           expert_fn, num_classes)
            val_acc = val_metrics['system accuracy']
            test_metrics['system loss'] = loss
            # log metrics and save framework to checkpoint
            log_test_metrics(writer, epoch, test_metrics, num_classes)
            if val_acc > best_acc:
                best_loss = loss
                print(f'save model with new best val acc {val_acc} at epoch {epoch}')
                best_metrics = test_metrics
                best_expert = copy.deepcopy(model_expert)
                save_to_checkpoint(train_dir, epoch, model_expert, optimizer, scheduler, best_metrics, seed)

    elif 'nih' in args['approach']:
        if best_metrics is None:
            best_loss = 100
        else:
            best_loss = best_metrics['system loss']
        best_expert = model_expert
        for epoch in range(start_epoch, epochs):
            # train for one epoch
            loss = train_expert(args, model_classifier, train_loader, model_expert, optimizer, scheduler,
                                epoch, expert_fn, num_classes)
            test_metrics = validate_expert(args, model_classifier, test_loader, model_expert,
                                           expert_fn, num_classes)
            test_metrics['system loss'] = loss
            # log metrics and save framework to checkpoint
            log_test_metrics(writer, epoch, test_metrics, num_classes)
            if loss < best_loss:
                best_loss = loss
                print(f'save model with new best loss {loss} at epoch {epoch}')
                best_metrics = test_metrics
                best_expert = copy.deepcopy(model_expert)
                save_to_checkpoint(train_dir, epoch, model_expert, optimizer, scheduler, best_metrics, seed)
    save_to_checkpoint(train_dir, epochs, best_expert, optimizer, scheduler, best_metrics, seed)
    best_metrics = validate_expert(args, model_classifier, test_loader, best_expert, expert_fn, num_classes)
    #fairness_test(args, model_classifier, test_loader, best_expert, expert_fn, num_classes, seed=seed)
    best_metrics['system loss'] = best_loss
    log_test_metrics(writer, epochs, best_metrics, num_classes)

    return best_metrics
