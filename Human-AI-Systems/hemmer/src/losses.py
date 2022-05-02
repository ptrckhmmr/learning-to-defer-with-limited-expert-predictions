import numpy as np
import torch
import torch.nn as nn


def joint_loss(classifier_output, deferral_system_output, expert_preds, targets, num_classes=2):
    """Get loss for the joint approach

    :param classifier_output:  Output of the classifier
    :param deferral_system_output: Output of the deferral system
    :param expert_preds: Expert artificial_expert_labels
    :param targets: Targets
    :param num_classes: Number od classes
    :return: System Loss
    """

    batch_size = len(targets)
    team_probs = torch.zeros((batch_size, num_classes)).to(
        classifier_output.device)  # set up zero-initialized tensor to store team artificial_expert_labels
    team_probs = team_probs + deferral_system_output[:, 0].reshape(-1,1) * classifier_output  # add the weighted classifier prediction to the team prediction
    one_hot_expert_preds = torch.tensor(np.eye(num_classes)[expert_preds.astype(int)]).to(
        classifier_output.device)
    team_probs = team_probs + deferral_system_output[:, 1].reshape(-1, 1) * one_hot_expert_preds

    log_output = torch.log(team_probs + 1e-7)
    system_loss = nn.NLLLoss()(log_output, targets)

    return system_loss
