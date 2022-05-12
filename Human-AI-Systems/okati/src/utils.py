import os
import pickle
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
import pandas as pd
import copy


def concat_args(args, mode):
    """Concatenate arguments to string

    :param args: Arguments
    :param mode: Mode
    :return: Concatenated arguments
    """
    model_args = copy.deepcopy(args)
    if mode == 'confidence-classifier':
        if 'cifar100' in model_args['approach']:
            model_args['dataset'] = 'cifar100'
        elif 'nih' in model_args['approach']:
            model_args['dataset'] = 'nih'
        try:
            del model_args['approach']
            del model_args['ex_strength']
        except KeyError:
            pass
    args_string = mode+'@'
    for key in model_args:
        args_string += str(key) + '-' + str(model_args[key]) + '-'
    return args_string[:-1]


def get_train_dir(wkdir, args, mode):
    """Get or create training directory

    :param wkdir: Working directory
    :param args: Args
    :param mode: Mode
    :return: Training directory
    """
    path = wkdir + '/experiments/' + concat_args(args, mode) + '/'
    if not os.path.exists(wkdir + '/experiments'):
        os.makedirs(wkdir + '/experiments')
    if not os.path.exists(path):
        os.makedirs(path)
    try:
        os.mkdir(path + 'logs/')
        os.mkdir(path + 'args/')
        os.mkdir(path + 'checkpoints/')
    except:
        pass
    return path


def load_from_checkpoint(train_dir, model, optimizer, scheduler, seed):
    """Load from checkpoint

    :param train_dir: Training directory
    :param model: Model
    :param scheduler: Learning rate scheduler
    :param optimizer: Optimizer
    :param seed: Random seed
    :return: tuple
        - model: Model
        - optimizer: Optimizer
        - scheduler: Learning rate scheduler
        - epoch: Current epoch
        - test_metrics: Test metrics
    """
    if seed == 123:
        seed = ''
    cp_dir = f'{train_dir}/checkpoints/ckp{seed}.latest'

    try:
        checkpoint = torch.load(cp_dir)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']+1
        test_metrics = checkpoint['test_metrics']
        print('Found latest checkpoint at', cp_dir)
        print('Continuing in epoch', epoch)
    except FileNotFoundError:
        epoch = 0
        test_metrics = None
        print(f'No Checkpoint found at {cp_dir}')
        print('Starting new from epoch', epoch)

    return model, optimizer, scheduler, epoch, test_metrics


def save_to_checkpoint(train_dir, epoch, model, optimizer, scheduler, test_metrics, seed):
    """Save to checkpoint

    :param train_dir: Training directory
    :param epoch: Epoch
    :param model: Model
    :param optimizer: Optimizer
    :param scheduler: Scheduler
    :param test_metrics: Test metrics
    :param seed: Random seed
    :return:
    """
    if seed == 123:
        seed = ''
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'test_metrics': test_metrics}, f'{train_dir}/checkpoints/ckp{seed}.latest')


def get_tabeled_list(experiment_data, predictions):
    """Get tabeled list of results

    :param experiment_data: Experiment data
    :param predictions: Predictions
    :return:
    """
    table_list = []
    table_list.append(['Full Automation', experiment_data['acc_full_aut'], '--------'])
    table_list.append(['--------', '--------', '--------'])
    table_list.append(['Best Expert', experiment_data['acc_best_ex'], '--------'])
    table_list.append(['True Expert', experiment_data['acc_true_ex'], experiment_data['cov_true_ex'][0]])
    for p in predictions:
        table_list.append(['Pred Ex ' + p, experiment_data['acc_'+p], experiment_data['cov_'+p][0]])
        table_list.append(['--------', '--------', '--------'])
    print(tabulate(table_list, headers=['Method', 'Accuracy', 'Coverage']))


def unpickle(file):
    """Function to open the files using pickle

    :param file: File to be loaded
    :return: Loaded file as dictionary
    """
    with open(file, 'rb') as fo:
        myDict = pickle.load(fo, encoding='latin1')
    return myDict


def load_targets(wkdir, mode):
    """Load CIFAR100 fine targets

    :param wkdir: Working directory
    :param mode: Mode
    :return: tuple (trainData, testData, metaData)
        - trainData['fine_labels'] - fine labels for training data
        - testData['fine_labels'] - fine labels for test data
    """
    trainData = unpickle(wkdir + '/data/cifar-100-python/train')
    testData = unpickle(wkdir + '/data/cifar-100-python/test')

    return trainData[mode], testData[mode]


def get_accuracy_by_cat(y_true, y_expert, y_pred, n_classes, exp_strengths):
    """Get accuracy by superclass

    :param y_true: Ground truth labels
    :param y_expert: True expert labels
    :param y_pred: Predicted expert labels
    :param int n_classes: Number of classes
    :param exp_strengths: Expert strengths
    :return: Accuracy dataframe
    """
    cm_true = get_confusion_matrix(y_true, y_expert)
    cm_pred = get_confusion_matrix(y_true, y_pred)
    strength_super = []
    for i in exp_strengths:
        for fine, coarse in fine_id_coarse_id().items():
            if fine == i:
                strength_super.append(coarse)
    from collections import Counter
    sw_class = np.array([Counter(strength_super)[i] / 5 for i in range(n_classes)])

    cat_acc = cm_true.diagonal() / cm_true.sum(axis=1)
    pred_cat_acc = cm_pred.diagonal() / cm_pred.sum(axis=1)
    target = ["Category {}".format(i) for i in range(n_classes)]
    acc_df = pd.DataFrame({'Expert_Acc.': cat_acc, 'Pred_Ex_Acc.': pred_cat_acc})
    acc_df['Diff'] = acc_df['Expert_Acc.'] - acc_df['Pred_Ex_Acc.']

    acc_df['Streng_%'] = sw_class
    acc_df.index = target
    acc_df = acc_df.round(2)

    return acc_df


def get_confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix

    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    return cm


def fine_id_coarse_id():
    """Mapping between fine and coarse labels

    :return: Mapping as dictionary
    """
    return {0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3, 10: 3, 11: 14, 12: 9, 13: 18, 14: 7,
            15: 11, 16: 3, 17: 9, 18: 7, 19: 11, 20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3,
            29: 15, 30: 0,
            31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16, 37: 9, 38: 11, 39: 5, 40: 5, 41: 19, 42: 8, 43: 8, 44: 15,
            45: 13, 46: 14,
            47: 17, 48: 18, 49: 10, 50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0, 56: 17, 57: 4, 58: 18, 59: 17, 60: 10,
            61: 3, 62: 2,
            63: 12, 64: 12, 65: 16, 66: 12, 67: 1, 68: 9, 69: 19, 70: 2, 71: 10, 72: 0, 73: 1, 74: 16, 75: 12, 76: 9,
            77: 13, 78: 15,
            79: 13, 80: 16, 81: 19, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19, 90: 18, 91: 1, 92: 2,
            93: 15, 94: 6, 95: 0,
            96: 17, 97: 8, 98: 14, 99: 13}


def get_nonbin_target(bin, y, num_classes, seed=123):
    """Get multiclass targets from binary targets

    :param bin: Binary targets
    :param y: Ground truth targets
    :param num_classes: Number of classes
    :param seed: Seed (default: 123)
    :return: Multiclass targets
    """
    np.random.seed(seed)
    # create empty arrays for the nonbinary targets
    nonbin = np.zeros(len(y))

    for i in range(len(y)):
        # multiclass target = ground truth target if binary target == 1
        if bin[i] == 1:
            nonbin[i] = y[i]
        # otherwise draw class from uniform distribution
        else:
            nonbin[i] = int(np.random.uniform(0, num_classes))

    return nonbin


def log_test_metrics(writer, epoch, test_metrics, num_classes):
    """Log metrics to tensorboard

    :param writer: Tensorboard writer
    :param epoch: Epoch
    :param test_metrics: Test metrics
    :param num_classes: Number of classes
    :return:
    """
    writer.add_scalar('system/accuracy', test_metrics['system accuracy'], epoch)
    writer.add_scalar('system/loss', test_metrics['system loss'], epoch)
    writer.add_scalar('classifier/accuracy', test_metrics['alone classifier'], epoch)
    writer.add_scalar('classifier/task_subset_accuracy', test_metrics['classifier accuracy'], epoch)
    writer.add_scalar('expert/accuracy', test_metrics['expert accuracy'], epoch)
    writer.add_scalar('expert/coverage', test_metrics['coverage'], epoch)
    for c in range(num_classes):
        writer.add_scalar('classifier/coverage/class' + str(c), test_metrics['cov per class'][c], epoch)


def find_machine_samples(machine_loss, hloss):
    """Get indices of instances assigned to the classifier

    :param machine_loss: Loss of classifier
    :param hloss: Loss of expert
    :return: Machine indices
    """
    diff = machine_loss - hloss
    diff = diff.cpu().numpy()
    machine_list = np.where(diff < 0)[0]

    return torch.tensor(machine_list, device='cuda')