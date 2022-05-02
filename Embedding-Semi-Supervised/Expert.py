import random
import numpy as np
import sys
from sklearn.metrics import accuracy_score
import pandas as pd


class CIFAR100Expert:
    """Class for the synthetic cifar100 expert

    :param num_classes: Number of classes
    :param n_strengths: Number of strengths
    :param per_s: Percentage of correct annotations for images of the expert's strengths
    :param per_w: Percentage of correct annotations for images of the expert's weaknesses
    :param seed: Random seed

    :ivar num_classes: Number of classes
    :ivar n_strengths: Number of strengths
    :ivar per_s: Percentage of correct annotations for images of the expert's strengths
    :ivar per_w: Percentage of correct annotations for images of the expert's weaknesses
    :ivar seed: Random seed
    :ivar fine_id_coarse_id: Mapping from the fine labels to the coarse labels
    :ivar probs: Mistake probabilities (similarities between the coarse classes)
    :ivar strengths_ind: Class indices of fine classes selected as the expert's strengths
    """
    def __init__(self, num_classes, n_strengths, per_s, per_w, seed=123):
        self.num_classes = num_classes
        self.n_strengths = n_strengths
        self.per_s = per_s
        self.per_w = per_w
        self.fine_id_coarse_id = {0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3, 10: 3, 11: 14, 12: 9,
                                  13: 18, 14: 7, 15: 11, 16: 3, 17: 9, 18: 7, 19: 11, 20: 6, 21: 11, 22: 5, 23: 10,
                                  24: 7, 25: 6, 26: 13, 27: 15, 28: 3, 29: 15, 30: 0, 31: 11, 32: 1, 33: 10, 34: 12,
                                  35: 14, 36: 16, 37: 9, 38: 11, 39: 5, 40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13,
                                  46: 14, 47: 17, 48: 18, 49: 10, 50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0, 56: 17,
                                  57: 4, 58: 18, 59: 17, 60: 10, 61: 3, 62: 2, 63: 12, 64: 12, 65: 16, 66: 12, 67: 1,
                                  68: 9, 69: 19, 70: 2, 71: 10, 72: 0, 73: 1, 74: 16, 75: 12, 76: 9, 77: 13, 78: 15,
                                  79: 13, 80: 16, 81: 19, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19,
                                  90: 18, 91: 1, 92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13}
        try:
            self.probs = np.load('data/mistake_probs.npy')
        except FileNotFoundError:
            print('Probabilities not found -> generating probabilities')
            sys.exit()

        self.seed = seed
        self.strengths_ind = self.draw_expert_strengths()

    def draw_expert_strengths(self):
        """Draw expert strengths

        :return: expert strengths
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        # draw the first strength subclass randomly
        strength_base = random.randint(0, 100)
        # draw the remaining strength subclasses with probabilities based on their similarity to the base strength
        strengths_drawn = np.random.choice(range(100), self.n_strengths-1, replace=False, p=self.probs[strength_base])
        return np.append(strength_base, strengths_drawn)

    def generate_expert_labels(self, y_true, binary=False):
        """Generate expert labels

        :param y_true: Ground truth labels (fine-labels)
        :param binary: Boolean flag to generate binary labels
        :returns: Expert labels
        """
        np.random.seed(self.seed)
        # generate expert labels for the training data
        y_expert = np.zeros(len(y_true), dtype=int)
        for i in range(len(y_true)):
            # check if observation is experts strength
            if y_true[i] in self.strengths_ind:
                if np.random.uniform(0, 1) < self.per_s:
                    # if obs is in strenghts -> per_s chance of perfect prediction of the superclass
                    if binary:
                        y_expert[i] = 1
                    else:
                        y_expert[i] = self.fine_id_coarse_id[y_true[i]]
                else:
                    # if oby is not in strengths -> draw false label according to probabilities
                    if binary:
                        y_expert[i] = 0
                    else:
                        y_expert[i] = self.fine_id_coarse_id[
                            np.random.choice(range(100), 1, p=self.probs[y_true[i]])[0]]
            else:
                if np.random.uniform(0, 1) < self.per_w:
                    # if obs is in weakness -> per_w chance perfect prediction of the superclass
                    if binary:
                        y_expert[i] = 1
                    else:
                        y_expert[i] = self.fine_id_coarse_id[y_true[i]]
                else:
                    # if oby is not in strengths -> draw false label according to probabilities
                    if binary:
                        y_expert[i] = 0
                    else:
                        y_expert[i] = self.fine_id_coarse_id[
                            np.random.choice(range(100), 1, p=self.probs[y_true[i]])[0]]
        return y_expert


class NIHExpert:
    """Class for the NIH expert

    :param id: Labeler id of the expert
    :param n_classes: Number of classes
    :param target: Targets

    :ivar labeler_id: Labeler id of the expert
    :ivar n_classes: Number of classes
    :ivar target: Targets
    :strengths_ind: Class wise accuracy of the experts annotations
    """
    def __init__(self, id, n_classes, target='Airspace_Opacity'):
        self.labeler_id = id
        self.n_classes = n_classes
        self.target = target
        self.strengths_ind = self.get_strengths()

    def get_strengths(self):
        """Get class wise accuracy (strength) of the experts annotation

        :return: Class wise accuracy
        """

        individual_labels = pd.read_csv("data/nih_labels.csv")
        data = individual_labels[individual_labels['Reader ID'] == self.labeler_id]

        y_ex_data = np.array(data[self.target + '_Expert_Label'])
        y_gt_data = np.array(data[self.target + '_GT_Label'])

        class_idx = {c: np.where(y_gt_data == c) for c in range(self.n_classes)}
        class_acc = [accuracy_score(y_gt_data[class_idx[c]], y_ex_data[class_idx[c]]) for c in range(self.n_classes)]
        return class_acc
