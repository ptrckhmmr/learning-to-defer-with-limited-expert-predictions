import random
import numpy as np

from src.generate_probabilities import generate_error_probs


class CIFAR100_Expert():
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
            generate_error_probs()
            self.probs = np.load('data/mistake_probs.npy')

        self.seed = seed
        self.strengths_ind = self.draw_expert_strengths()

    def draw_expert_strengths(self):
        """ Draw expert strengths

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
        :returns: list
            - y_expert - Expert labels
        """
        np.random.seed(self.seed)
        # generate expert labels for the training data
        y_expert = np.zeros(len(y_true))
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
