import os
import sys
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from src.metrics import get_metrics, get_accuracy
from src.losses import joint_loss, joint_loss_binary
from src.models import Resnet, Network, WideResNet
from src.models import get_effnet_feature_extractor, get_effnet_classifier
from src.utils import get_train_dir, load_from_checkpoint, save_to_checkpoint, log_test_metrics
from src.data_loading import CIFAR100_Dataloader, NIH_Dataloader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_one_epoch(epoch, feature_extractor, classifier, deferral_system, train_loader, optimizer, scheduler,
                    expert_fns, loss_fn, num_classes, use_lr_scheduler=False):
    """Train one epoch of the human-AI collaboration framework

    :param epoch: Epoch
    :param feature_extractor: Feature extractor
    :param classifier: Classifier
    :param deferral_system: Deferral system
    :param train_loader: Train loader
    :param optimizer: Optimizer
    :param scheduler: Scheduler
    :param expert_fns: Expert prediction functions
    :param num_classes: Number of classes
    :param loss_fn: Loss function
    :param use_lr_scheduler: Whether to use lr-scheduling (default: False)
    :return:
    """
    losses = []
    feature_extractor.train()
    classifier.train()
    deferral_system.train()
    for i, (batch_input, batch_targets, batch_indices) in enumerate(train_loader):
        batch_input = batch_input.to(device)
        batch_targets = batch_targets.to(device)
        expert_batch_preds = np.array(expert_fns(batch_indices))
        batch_features = feature_extractor(batch_input)
        batch_outputs_classifier = classifier(batch_features)
        batch_outputs_deferral_system = deferral_system(batch_features)

        batch_loss = loss_fn(epoch=epoch, classifier_output=batch_outputs_classifier,
                             deferral_system_output=batch_outputs_deferral_system,
                             expert_preds=expert_batch_preds, targets=batch_targets, num_classes=num_classes)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if use_lr_scheduler:
            scheduler.step()
        losses.append(batch_loss.item())
    return np.mean(losses)


def evaluate_one_epoch(epoch, feature_extractor, classifier, deferral_system, data_loader, expert_fns, loss_fn, binary,
                       test=False, num_classes=20):
    """Evaluate one epoch of the human-AI collaboration framework

    :param epoch: Epoch
    :param feature_extractor: Feature extractor
    :param classifier: Classifier
    :param deferral_system: Deferral system
    :param data_loader: Dat loader
    :param expert_fns: Expert prediction functions
    :param loss_fn: Loss function
    :param binary: Whether to use binary expert labels
    :param test: Whether to evaluate the test data (default: False)
    :param num_classes: Number of classes (default: 20)
    :return: tuple
        - system_accuracy - System accuracy
        - system_loss - System loss
        - metrics - Metrics
    """
    feature_extractor.eval()
    classifier.eval()
    deferral_system.eval()

    classifier_outputs = torch.tensor([]).to(device)
    deferral_system_outputs = torch.tensor([]).to(device)
    targets = torch.tensor([]).to(device)
    indices = []

    with torch.no_grad():
        for i, (batch_input, batch_targets, batch_indices) in enumerate(data_loader):
            batch_input = batch_input.to(device)
            batch_targets = batch_targets.to(device)

            batch_features = feature_extractor(batch_input)
            batch_classifier_outputs = classifier(batch_features)
            batch_deferral_system_outputs = deferral_system(batch_features)

            classifier_outputs = torch.cat((classifier_outputs, batch_classifier_outputs))
            deferral_system_outputs = torch.cat((deferral_system_outputs, batch_deferral_system_outputs))
            targets = torch.cat((targets, batch_targets))
            indices.extend(batch_indices)

    expert_preds = np.array(expert_fns(indices, test))

    classifier_outputs = classifier_outputs.cpu().numpy()
    deferral_system_outputs = deferral_system_outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    system_accuracy, system_loss, metrics = get_metrics(epoch, deferral_system_outputs, classifier_outputs,
                                                        expert_preds, targets, loss_fn, binary, num_classes)

    return system_accuracy, system_loss, metrics


def run_team_performance_optimization(method, args, seed, expert_fns, epochs, num_hidden_units, binary, num_classes=2,
                                      num_experts=1, train_batch_size=512, test_batch_size=512, lr=5e-3, dropout=0.00):
    """Run team performance optimization with the human-AI collaboration framework

    :param method: Method
    :param args: Args
    :param seed: Seed
    :param expert_fns: Expert prediction functions
    :param epochs: Epochs to train
    :param num_hidden_units: Number of hidden units for the classifier/ deferral system
    :param binary: Whether to use binary expert labels
    :param num_classes: Number of classes (default: 20)
    :param num_experts: Number of experts (default: 1)
    :param train_batch_size: Train batch size (default: 512)
    :param test_batch_size: Test batch size (default: 512)
    :param lr: Learning Rate (default: 5e-3)
    :param dropout: Dropout (default: 0)
    :return: tuple
        - system metrics
        - classifier metrics
    """
    print(f'Team Performance Optimization with {method}')
    if binary:
        loss_fct = joint_loss_binary
    else:
        loss_fct = joint_loss

    # get train directory
    train_dir = get_train_dir(os.getcwd(), args, method)
    # initialize tensorbaord writer
    writer = SummaryWriter(train_dir + 'logs/')
    # get feature extractor model
    #
    if 'cifar100' in args['approach']:
        feature_extractor = WideResNet(28, num_classes, 4, dropRate=0).to(device)
        feature_size = 256
        # get cifer 100 data loader
        cifar_dl = CIFAR100_Dataloader(train_batch_size=train_batch_size, test_batch_size=test_batch_size,
                                               seed=seed, small_version=False)
        train_loader, val_loader, test_loader = cifar_dl.get_data_loader()
        train_indices = cifar_dl.train_indices
    elif 'nih' in args['approach']:
        feature_extractor = Resnet(num_classes, 'nih').to(device)
        feature_size = 512
        nih_dl = NIH_Dataloader(labeler_id=args['ex_strength'], train_batch_size=train_batch_size, test_batch_size=test_batch_size, seed=seed)
        train_loader, test_loader = nih_dl.get_data_loader()
        train_indices = nih_dl.train_indices
        lr = 0.0001
    else:
        print('Classifier model not defined')
        sys.exit()
    # get classifier
    classifier = Network(input_size=feature_size,
                         output_size=num_classes,
                         softmax_sigmoid="softmax",
                         num_hidden_units=num_hidden_units,
                         dropout=dropout).to(device)
    # get deferral system
    deferral_system = Network(input_size=feature_size,
                              output_size=num_experts + 1,
                              dropout=dropout,
                              num_hidden_units=num_hidden_units,
                              softmax_sigmoid="softmax").to(device)
    # initialize optimizer and lr-scheduler
    parameters = list(classifier.parameters()) + list(deferral_system.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(train_loader))

    best_loss = 100
    # try to load previous training from checkpoint
    classifier, deferral_system, optimizer, \
    scheduler, start_epoch, best_metrics = load_from_checkpoint(train_dir, classifier, deferral_system,
                                                                optimizer, scheduler, seed)
    # train framework
    for epoch in tqdm(range(start_epoch, epochs + 1)):
        # train one epoch
        loss = train_one_epoch(epoch, feature_extractor, classifier, deferral_system, train_loader, optimizer, scheduler,
                        expert_fns, loss_fct, num_classes)
        # evaluate one epoch
        _, _, test_metrics = evaluate_one_epoch(epoch, feature_extractor, classifier, deferral_system, test_loader,
                                                expert_fns, loss_fct, binary, num_classes=num_classes, test=True)
        # save best metrics
        if loss < best_loss:
            best_loss = loss
            best_metrics = test_metrics
        # log metrics and save framework to checkpoint
        log_test_metrics(writer, epoch, test_metrics, num_classes)
        save_to_checkpoint(train_dir, epoch, classifier, deferral_system, optimizer, scheduler, best_metrics, seed)

    print(f'\n Earlystopping Results for {method}:')
    system_metrics_keys = [key for key in best_metrics.keys() if "System" in key]
    for k in system_metrics_keys:
        print(f'\t {k}: {best_metrics[k]}')
    print()

    classifier_metrics_keys = [key for key in best_metrics.keys() if "Classifier" in key]
    for k in classifier_metrics_keys:
        print(f'\t {k}: {best_metrics[k]}')
    print()

    """for exp_idx in range(num_experts):
      expert_metrics_keys = [key for key in best_metrics.keys() if f'Expert {exp_idx+1} ' in key]
      for k in expert_metrics_keys:
          print(f'\t {k}: {best_metrics[k]}')
    print()"""

    return best_metrics["System Accuracy"], best_metrics["Classifier Coverage"]


def train_full_automation_one_epoch(feature_extractor, classifier, train_loader, optimizer, scheduler,
                                    use_lr_scheduler=False):
    """Train one epoch of the full automation baseline

    :param feature_extractor: Feature extractor
    :param classifier: Classifier
    :param train_loader: Train loader
    :param optimizer: Optimizer
    :param scheduler: Scheduler
    :param use_lr_scheduler: Whether to use lr-scheduling (default: False)
    :return:
    """
    # switch to train mode
    feature_extractor.eval()
    classifier.train()
    losses = []
    for i, (batch_input, batch_targets, batch_filenames, batch_indices) in enumerate(train_loader):
        batch_input = batch_input.to(device)
        batch_targets = batch_targets.to(device)

        batch_features = feature_extractor(batch_input)
        batch_outputs_classifier = classifier(batch_features)

        log_output = torch.log(batch_outputs_classifier + 1e-7)
        batch_loss = nn.NLLLoss()(log_output, batch_targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if use_lr_scheduler:
            scheduler.step()
        losses.append(batch_loss)
    return np.mean(losses)


def evaluate_full_automation_one_epoch(feature_extractor, classifier, data_loader):
    """Evaluate oen epoch of the full automation baseline

    :param feature_extractor: Feature extractor
    :param classifier: Classifier
    :param data_loader: Data loader
    :return: tuple
        - full_automation_accuracy - Acurracy
        - full_automation_loss - Loss
    """
    feature_extractor.eval()
    classifier.eval()

    classifier_outputs = torch.tensor([]).to(device)
    targets = torch.tensor([]).to(device)
    filenames = []

    with torch.no_grad():
        for i, (batch_input, batch_targets, batch_filenames, batch_indices) in enumerate(data_loader):
            batch_input = batch_input.to(device)
            batch_targets = batch_targets.to(device)

            batch_features = feature_extractor(batch_input)
            batch_classifier_outputs = classifier(batch_features)

            classifier_outputs = torch.cat((classifier_outputs, batch_classifier_outputs))
            targets = torch.cat((targets, batch_targets))
            filenames.extend(batch_filenames)

    log_output = torch.log(classifier_outputs + 1e-7)
    full_automation_loss = nn.NLLLoss()(log_output, targets.long())

    classifier_outputs = classifier_outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    classifier_preds = np.argmax(classifier_outputs, 1)
    full_automation_accuracy = get_accuracy(classifier_preds, targets)

    return full_automation_accuracy, full_automation_loss


def run_full_automation(args, seed, epochs, num_hidden_units, train_batch_size=512, test_batch_size=512,
                        num_classes=20, lr=5e-3, dropout=0.00):
    """Run full automation baseline

    :param args: Args
    :param seed: Seed
    :param epochs: Epoch
    :param num_hidden_units: Number of hidden units
    :param train_batch_size: Train batch size (default: 512)
    :param test_batch_size: Test batch size (default 512)
    :param num_classes: Number of classes (default: 20)
    :param lr: Learning Rate (default: 5e-3)
    :param dropout: Dropout (default: 0)
    :return: Full automation accuracy
    """
    print(f'Training full automation baseline')
    # get train dir and tensorboard writer
    train_dir = get_train_dir(os.getcwd(), args, 'full_automation')
    writer = SummaryWriter(train_dir + 'logs/')
    # get feature extractor
    if 'cifar100' in args['approach']:
        feature_extractor = get_effnet_feature_extractor(num_classes).to(device)
        feature_size = 1280
        # get dataloader
        cifar_dl = CIFAR100_Dataloader(train_batch_size=train_batch_size, test_batch_size=test_batch_size,
                                               seed=seed, small_version=False)
        train_loader, val_loader, test_loader = cifar_dl.get_data_loader()
    elif 'nih' in args['approach']:
        feature_extractor = Resnet(num_classes).to(device)
        feature_size = 512
        nih_dl = NIH_Dataloader(labeler_id=args['ex_strength'], train_batch_size=train_batch_size, test_batch_size=test_batch_size, seed=seed)
        train_loader, test_loader = nih_dl.get_data_loader()
        lr = 0.001
        train_indices = nih_dl.train_indices
    else:
        print('Classifier model not defined')
        sys.exit()
    # get classifier
    classifier = Network(input_size=feature_size,
                         output_size=num_classes,
                         softmax_sigmoid="softmax",
                         num_hidden_units=num_hidden_units,
                         dropout=dropout).to(device)
    # get optimizer and lr-scheduler
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(train_loader))

    best_loss = 100
    # try to load previous training from checkpoint
    classifier, _, optimizer, scheduler, \
    start_epoch, best_test_system_accuracy = load_from_checkpoint(train_dir, classifier, None, optimizer, scheduler)
    # train full automation
    for epoch in tqdm(range(start_epoch, epochs + 1)):
        # train one epoch
        loss = train_full_automation_one_epoch(epoch, feature_extractor, classifier, train_loader, optimizer, scheduler)
        # evaluate one epoch
        test_system_accuracy, test_system_loss, = evaluate_full_automation_one_epoch(feature_extractor,
                                                                                     classifier, test_loader)
        # save best metrics
        if loss < best_loss:
            best_val_system_loss = loss
            best_test_system_accuracy = test_system_accuracy
        # log metrics to tensorboard
        writer.add_scalar('system/accuracy/test', test_system_accuracy, epoch)
        writer.add_scalar('system/loss/test', test_system_loss, epoch)
        save_to_checkpoint(train_dir, epoch, classifier, None, optimizer, scheduler, test_system_accuracy)

    print(f'Full Automation Accuracy: {best_test_system_accuracy}\n')
    return best_test_system_accuracy
