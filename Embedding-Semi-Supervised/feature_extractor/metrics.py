import torch
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score

import feature_extractor.data_loading as prep


def accuracy(outputs, labels):
    """Get accuracy from model prediction

    :param outputs: Classifier output
    :param labels: True labels
    :return: Accuracy
    """
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def get_confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix

    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    return cm


def plot_confusion_matrix(y_true, y_pred):
    """Cplot confusion matrix

    :param y_true: True labels
    :param y_pred: Predicted labels
    :return:
    """
    cm = get_confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(cm, annot=True, cmap='Blues')
    ax.set_title('Predicted Expert Labels Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')


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
        for fine, coarse in prep.fine_id_coarse_id().items():
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


def get_accuracy_by_stength(y_true, y_expert, y_pred, n_classes, exp_strengths, print_df=True):
    """Calculate accurracy by class strength

    :param y_true: Ground truth label
    :param y_expert: Expert label
    :param y_pred: Predicted expert label
    :param int n_classes: Number of classes
    :param exp_strengths: Experts strengths [0,100]
    :param bool print_df: Whether dfs should be printed
    :return:
    """

    df = get_accuracy_by_cat(y_true, y_expert, y_pred, n_classes, exp_strengths)
    acc_df_mean = df.groupby('Streng_%').mean()
    acc_df_sd = df.groupby('Streng_%').std()
    if print_df:
        print(acc_df_mean)
        print(acc_df_sd)
    return acc_df_mean, acc_df_sd

def expert_accuracy_score(gt_targets, ex_targets, pred, binary=False):
    """Calculate custom expert accuracy, expert strength accuracy and expert weakness accuracy

    :param gt_targets: Ground truth targets
    :param ex_targets: Expert targets
    :param pred: Predicted targets
    :param binary: Boolean flag for binary classification
    :return: dict
        - Ex_Acc - Custom expert accuracy
        - Streng_Acc - Strength accuracy
        - Weak_Acc - Weakness accuracy
    """
    correct_pred = 0
    strength_acc = 0
    weakness_acc = 0
    strength = 0
    if binary:
        for i in range(len(gt_targets)):
            if ex_targets[i] == pred[i]:
                correct_pred += 1
            # expert strength if ex target equals the gt target
            if ex_targets[i] == 1:
                strength += 1
                if ex_targets[i] == pred[i]:
                    strength_acc += 1
            # expert weakness if ex target is unequal to the gt target
            else:
                if ex_targets[i] == pred[i]:
                    weakness_acc += 1
    else:
        for i in range(len(gt_targets)):
            # correct prediction if and only if the predicted class equals the ex target or the predicted class
            # does not equal neither the gt target nor the ex target if the ex target in unequal to the gt target
            if ex_targets[i] == pred[i] or (ex_targets[i] != gt_targets[i] and pred[i] != gt_targets[i]):
                correct_pred += 1
            # expert strength if ex target equals the gt target
            if ex_targets[i] == gt_targets[i]:
                strength += 1
                # correct prediction in expert strength if and only if the predicted class equals the ex target
                if ex_targets[i] == pred[i]:
                    strength_acc += 1
            # expert weakness if ex target is unequal to the gt target
            else:
                # correct prediction in expert weakness if and only if the predicted class is unequal to the gt target
                if pred[i] != gt_targets[i]:
                    weakness_acc += 1

    return {'Ex_Acc': correct_pred / len(pred), 'Streng_Acc': strength_acc / strength,
            'Weak_Acc': weakness_acc / (len(pred) - strength)}



