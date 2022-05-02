import torch.utils.data
import json
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Cifar100Expert:
    """A class used to represent an Expert on CIFAR100 data.

    :param pred_dir: Directory containing the prediction files
    :param pred: Filename of the file containing the artificial expert artificial_expert_labels
    :param true_pred: Filename of the file containing the true expert artificial_expert_labels

    :ivar pred_dir: Directory of the prediction files
    :ivar artificial_expert_labels: Artificial expert labels
    :ivar true_expert_labels: True expert labels
    :ivar subclass_idx_to_superclass_idx: Dictionary mapping the 100 subclass indices of CIFAR100 to their 20 superclass indices
    """

    def __init__(self, pred_dir, pred: str = None, true_pred: str = None):
        self.pred_dir = pred_dir[:-len('okati')]
        if pred is not None:
            with open(self.pred_dir + '/artificial_expert_labels/' + pred + '.json') as json_file:
                self.artificial_expert_labels = json.load(json_file)
        if true_pred is not None:
            with open(self.pred_dir + '/artificial_expert_labels/' + true_pred + '.json') as json_file:
                self.true_expert_labels = json.load(json_file)

        self.subclass_idx_to_superclass_idx = {0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3, 10: 3,
                                               11: 14, 12: 9, 13: 18, 14: 7,
                                               15: 11, 16: 3, 17: 9, 18: 7, 19: 11,
                                               20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3,
                                               29: 15, 30: 0, 31: 11, 32: 1,
                                               33: 10, 34: 12, 35: 14, 36: 16, 37: 9,
                                               38: 11, 39: 5, 40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 46: 14,
                                               47: 17, 48: 18, 49: 10, 50: 16,
                                               51: 4, 52: 17, 53: 4, 54: 2, 55: 0,
                                               56: 17, 57: 4, 58: 18, 59: 17, 60: 10, 61: 3, 62: 2, 63: 12, 64: 12,
                                               65: 16, 66: 12, 67: 1, 68: 9,
                                               69: 19, 70: 2, 71: 10, 72: 0, 73: 1,
                                               74: 16, 75: 12, 76: 9, 77: 13, 78: 15, 79: 13, 80: 16, 81: 19, 82: 2,
                                               83: 4, 84: 6, 85: 19, 86: 5,
                                               87: 5, 88: 8, 89: 19, 90: 18, 91: 1,
                                               92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13}

    def predict(self, batch_indices, test=False) -> list:
        """Get expert labels for a given batch of images

        :param batch_indices: Indices of the current batch
        :param test: True if current batch is from the test dataset
        :return: Expert labels for the batch
        """
        if test:
            # for images from the test set always get labels from the true expert
            # -> evaluation only with the true expert
            all_ex_labels = np.array(self.true_expert_labels['test'])
        else:
            all_ex_labels = np.array(self.artificial_expert_labels['train'])

        return all_ex_labels[batch_indices]


class NihExpert:
    """A class used to represent an Expert on NIH ChestX-ray data.

    :param pred_dir: Working directory containing the prediction files
    :param pred: Filename of the file containing the artificial expert artificial_expert_labels
    :param true_pred: Filename of the file containing the true expert artificial_expert_labels

    :ivar pred_dir: Directory of the prediction files
    :ivar image_id_to_artificialex_label: Directory mapping the image ids to the artificial expert labels
    :ivar image_id_to_trueex_label: Directory mapping the image ids to the true expert labels
    """

    def __init__(self, pred_dir, pred: str = None, true_pred: str = None):
        self.pred_dir = pred_dir[:-len('okati')] + '/artificial_expert_labels/'
        if pred is not None:
            with open(self.pred_dir + pred + '.json') as json_file:
                self.image_id_to_artificialex_label = json.load(json_file)
        if true_pred is not None:
            with open(self.pred_dir + true_pred + '.json') as json_file:
                self.image_id_to_trueex_label = json.load(json_file)
        imgs = list(self.image_id_to_trueex_label.keys())
        imgs_pred = list(self.image_id_to_artificialex_label.keys())
        assert imgs == imgs_pred

    def predict(self, image_ids, test=False):
        """Returns the experts artificial_expert_labels for the given image ids. Works only for image ids that are labeled by the expert

        :param image_ids: List of image ids
        :param test: True if current batch is from the test dataset
        :return: Expert labels for image ids
        """
        if test:
            ex_labels = [self.image_id_to_trueex_label[image_id] for image_id in image_ids]
        else:
            ex_labels = [self.image_id_to_artificialex_label[image_id] for image_id in image_ids]
        return ex_labels
