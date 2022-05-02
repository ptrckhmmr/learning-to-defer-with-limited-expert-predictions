import numpy as np
import torch
from sklearn.metrics import accuracy_score
from collections import Counter

from src.data_loading import CIFAR100_3_Split_Dataloader
from src.experts import Cifar100AverageExpert
from src.utils import get_nonbin_target


def get_accuracy(preds, targets):
    """Get accuracy

    :param preds: Predictions
    :param targets: Targets
    :return: Accuracy
    """
    if len(targets) > 0:
        acc = accuracy_score(targets, preds)
    else:
        acc = 0
    return acc


def get_coverage(task_subset_targets, targets, num_classes):
    """Get coverage

    :param task_subset_targets: Targets of the task subset
    :param targets: All targets
    :param num_classes: Number of classes
    :return: tuple
        - coverage - Coverage
        - cover_classes - Coverage by class
    """
    num_images = len(targets)
    num_images_in_task_subset = len(task_subset_targets)
    coverage = num_images_in_task_subset / num_images
    c = Counter()
    targets_counter = Counter(targets)
    task_subset_counter = Counter(task_subset_targets)
    cover_classes = []
    for c in range(num_classes):
        cover_classes.append(task_subset_counter[c]/targets_counter[c])
    return coverage, cover_classes


def get_classifier_metrics(classifier_preds, deferral_system_decisions, targets, num_classes):
    """Get metrics for the classifier

    :param classifier_preds: Classifier artificial_expert_labels
    :param deferral_system_decisions: Deferral system decisions
    :param targets: Targets
    :param num_classes: Number of classes
    :return: tuple
        - classifier_accuracy - Classifier accuracy
        - classifier_task_subset_accuracy - Accuracy on classifier task subset
        - classifier_coverage - Classifier coverage
    """
    # classifier performance on all tasks
    classifier_accuracy = get_accuracy(classifier_preds, targets)
    # filter for subset of tasks that are allocated to the classifier
    task_subset = (deferral_system_decisions == 0)
    # classifier performance on those tasks
    task_subset_classifier_preds = classifier_preds[task_subset]
    task_subset_targets = targets[task_subset]
    classifier_task_subset_accuracy = get_accuracy(task_subset_classifier_preds, task_subset_targets)
    # coverage
    classifier_coverage = get_coverage(task_subset_targets, targets, num_classes)
    return classifier_accuracy, classifier_task_subset_accuracy, classifier_coverage


def get_experts_metrics(expert_preds, deferral_system_decisions, targets, num_classes):
    """Get expert metrics

    :param expert_preds: Expert artificial_expert_labels
    :param deferral_system_decisions: Deferral system decisions
    :param targets: Targets
    :param num_classes: Number of classes
    :return: tuple
        - expert_accuracies - Expert accuracies
        - expert_task_subset_accuracies - Expert accuracies on task subset
        - expert_coverages - Coverages of the expert
    """
    expert_accuracies = []
    expert_task_subset_accuracies = []
    expert_coverages = []
    # calculate metrics for the expert
    # expert performance on all tasks
    preds = expert_preds
    expert_accuracy = get_accuracy(preds, targets)
    # filter for subset of tasks that are allocated to the expert with number "idx"
    task_subset = (deferral_system_decisions == 1)
    # expert performance on tasks assigned by allocation system
    task_subset_expert_preds = preds[task_subset]
    task_subset_targets = targets[task_subset]
    expert_task_subset_accuracy = get_accuracy(task_subset_expert_preds, task_subset_targets)
    # coverage
    expert_coverage = get_coverage(task_subset_targets, targets, num_classes)
    # append accuracies and coverages
    expert_accuracies.append(expert_accuracy)
    expert_task_subset_accuracies.append(expert_task_subset_accuracy)
    expert_coverages.append(expert_coverage)
    return expert_accuracies, expert_task_subset_accuracies, expert_coverages


def get_metrics(epoch, deferral_system_outputs, classifier_outputs, expert_preds, targets, loss_fn, binary, num_classes):
    """Get metrics for human-AI collaboration framework

    :param epoch: Epoch
    :param deferral_system_outputs: Outputs of the deferral system
    :param classifier_outputs: Outputs of the classifier
    :param expert_preds: Expert artificial_expert_labels
    :param targets: Targets
    :param loss_fn: Loss function
    :param binary: Whether to use binary expert labels
    :return: tuple
        - system_accuracy - System accuracy
        - system_loss - System loss
        - metrics - Metrics
    """
    metrics = {}
    # get system loss
    system_loss = loss_fn(epoch=epoch,
                          classifier_output=torch.tensor(classifier_outputs).float(),
                          deferral_system_output=torch.tensor(deferral_system_outputs).float(),
                          expert_preds=expert_preds,
                          targets=torch.tensor(targets).long(),
                          num_classes=num_classes)
    if binary:
        expert_preds = [get_nonbin_target(preds, targets, num_classes=20) for preds in expert_preds]
    # Metrics for system
    deferral_system_decisions = np.argmax(deferral_system_outputs, 1)
    classifier_preds = np.argmax(classifier_outputs, 1)
    preds = np.vstack((classifier_preds, expert_preds)).T
    system_preds = preds[range(len(preds)), deferral_system_decisions.astype(int)]
    system_accuracy = get_accuracy(system_preds, targets)
    metrics["System Accuracy"] = system_accuracy
    metrics["System Loss"] = system_loss
    # Metrics for classifier
    classifier_accuracy, classifier_task_subset_accuracy, classifier_coverage = \
        get_classifier_metrics(classifier_preds, deferral_system_decisions, targets, num_classes)
    metrics["Classifier Accuracy"] = classifier_accuracy
    metrics["Classifier Task Subset Accuracy"] = classifier_task_subset_accuracy
    metrics["Classifier Coverage"] = classifier_coverage
    # Metrics for experts
    expert_accuracies, experts_task_subset_accuracies, experts_coverages = \
        get_experts_metrics(expert_preds, deferral_system_decisions, targets, num_classes)
    for expert_idx, (expert_accuracy, expert_task_subset_accuracy, expert_coverage) in \
            enumerate(zip(expert_accuracies, experts_task_subset_accuracies, experts_coverages)):
        metrics[f'Expert {expert_idx+1} Accuracy'] = expert_accuracy
        metrics[f'Expert {expert_idx+1} Task Subset Accuracy'] = expert_task_subset_accuracy
        metrics[f'Expert {expert_idx+1} Coverage'] = expert_coverage
    return system_accuracy, system_loss, metrics


def get_accuracy_of_best_expert(seed, expert_fns, train_batch_size=512, test_batch_size=512, num_experts=1):
    """Get accuracy of the best exper alone

    :param seed:  Seed
    :param expert_fns: Expert prediction functions
    :param train_batch_size: Train batch size
    :param test_batch_size: Test batch size
    :param num_experts: Number of experts
    :return: Best expert accuracy
    """
    cifar_dl = CIFAR100_3_Split_Dataloader(train_batch_size=train_batch_size, test_batch_size=test_batch_size,
                                           seed=seed, small_version=False)
    _, _, test_loader = cifar_dl.get_data_loader()
    targets = torch.tensor([]).long()
    indices = []
    with torch.no_grad():
        for i, (_, batch_targets, batch_subclass_idxs, batch_indices) in enumerate(test_loader):
            targets = torch.cat((targets, batch_targets))
            indices.extend(batch_indices)
    expert_preds = np.empty((num_experts, len(targets)))
    for idx, expert_fn in enumerate(expert_fns):
        expert_preds[idx] = np.array(expert_fn(targets, [], indices, test = True))
    expert_accuracies = []
    for idx in range(num_experts):
        preds = expert_preds[idx]
        acc = accuracy_score(targets, preds)
        expert_accuracies.append(acc)
    print(f'Best Expert Accuracy: {max(expert_accuracies)}\n')
    return max(expert_accuracies)


def get_accuracy_of_average_expert(seed, expert_fns, train_batch_size=512, test_batch_size=512):
    """Get accuracy of average expert

    :param seed: Seed
    :param expert_fns: Expert prediction function
    :param train_batch_size: Train batch size
    :param test_batch_size: Test batch size
    :return: Average expert accuracy
    """
    cifar_dl = CIFAR100_3_Split_Dataloader(train_batch_size=train_batch_size, test_batch_size=test_batch_size,
                                           seed=seed, small_version=False)
    _, _, test_loader = cifar_dl.get_data_loader()
    targets = torch.tensor([]).long()
    subclass_idxs = []
    indices = []
    with torch.no_grad():
        for i, (_, batch_targets, batch_subclass_idxs, batch_indices) in enumerate(test_loader):
            targets = torch.cat((targets, batch_targets))
            subclass_idxs.extend(batch_subclass_idxs)
            indices.extend(batch_indices)
    avg_expert = Cifar100AverageExpert(expert_fns)
    avg_expert_preds = avg_expert.predict(targets, subclass_idxs, indices, test=True)
    avg_expert_acc = accuracy_score(targets, avg_expert_preds)
    print(f'Average Expert Accuracy: {avg_expert_acc}\n')
    return avg_expert_acc


