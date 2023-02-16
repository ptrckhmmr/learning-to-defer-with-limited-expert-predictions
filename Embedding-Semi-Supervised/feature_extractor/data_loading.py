import os
import sys
import numpy as np
import pandas as pd
import pickle
import torch
import cv2
import torchvision.transforms as tt

from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.datasets import CIFAR100, CIFAR10
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image


def unpickle(file):
    """Function to open the files using pickle

    :param file: File to be loaded
    :return: Loaded file as dictionary
    """
    with open(file, 'rb') as fo:
        myDict = pickle.load(fo, encoding='latin1')
    return myDict


def load_coarse_targets(wkdir):
    """Load CIFAR100 fine targets

    :param wkdir: Working directory
    :return: tuple (trainData, testData, metaData)
        - trainData['fine_labels'] - fine labels for training data
        - testData['fine_labels'] - fine labels for test data
    """
    trainData = unpickle(wkdir + '/data/cifar-100-python/train')
    testData = unpickle(wkdir + '/data/cifar-100-python/test')

    return trainData['coarse_labels'], testData['coarse_labels']


def load_fine_targets(wkdir):
    """Load CIFAR100 fine targets

    :param wkdir: Working directory
    :return: tuple (trainData, testData, metaData)
        - trainData['fine_labels'] - fine labels for training data
        - testData['fine_labels'] - fine labels for test data
    """
    trainData = unpickle(wkdir + '/data/cifar-100-python/train')
    testData = unpickle(wkdir + '/data/cifar-100-python/test')

    return np.array(trainData['fine_labels']), np.array(testData['fine_labels'])


def transform_to_coarse(targets):
    """Transforms fine targets into coarse targets

    :param targets: Fine targets
    :return: Coarse targets
    """
    coarse = np.array([fine_id_coarse_id()[t] for t in targets])
    return coarse


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


def get_device():
    """Get active device

    :return: device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(data, device):
    """Load to device

    :param data: Data
    :param device: Device
    :return: Data loaded to device
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class ToDeviceLoader:
    """Class for the toDeviceLoader

    :param data: Data
    :param device: Active device

    :ivar data: Data
    :ivar device: device
    """
    def __init__(self, data, device):
        self.data = data
        self.device = device

    def __iter__(self):
        for batch in self.data:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.data)


def get_train_val_test_data(expert, binary=False, model='efficientnet_b1', valid=True, L=None, gt_targets=False,
                            seed=123, dataset='cifar100'):
    """Generate the train, validation and test set

    :param expert: Expert
    :param binary: Boolean flag to generate binary labels
    :param model: Model name
    :param valid: Boolean flag for generating a validation dataset
    :param L: Number of labeled instances
    :param gt_targets: Boolean flag for returning ground-truth targets
    :param seed: Random seed
    :param dataset: Name of the dataset
    :return: tuple
        - train_data: Data set containing the training data
        - test_data: Data set containing the test data
        - valid_data: Data set containing the validation data (optional)
        - train_gt_data: Ground-truth label for the training data (optional)
        - test_gt_data: Ground-truth label for the test data (optional)
        - valid_gt_data: Ground-truth label for the val data (optional)
    """
    if dataset == 'cifar100':
        return get_cifar100_data(expert, binary, model, valid, L, gt_targets, seed)
    elif dataset == 'nih':
        return get_nih_data(expert=expert, valid=valid, L=L, gt_targets=gt_targets, seed=seed, binary=binary)
    else:
        print(f'Dataset {dataset} not defined')
        sys.exit()


def get_cifar100_data(expert, binary=False, model='efficientnet_b1', valid=True, L=None, gt_targets=False, seed=123):
    """Generate the train, validation and test set for the cifar100 dataset

    :param expert: CIFAR100 Expert
    :param binary: Boolean flag to generate binary labels
    :param model: Embedding-Model name
    :param valid: Boolean flag for generating a validation dataset
    :param L: Number of instances with expert labels
    :param gt_targets: Boolean flag for returning ground truth targets
    :param seed: Random seed
    :return: tuple (train_data, test_data)
        - train_data: Data set containing the training data
        - test_data: Data set containing the test data
        - valid_data: Data set containing the validation data (optional)
        - train_gt_data: Ground-truth label for the training data (optional)
        - test_gt_data: Ground-truth label for the test data (optional)
        - valid_gt_data: Ground-truth label for the val data (optional)
    """
    if model != 'efficientnet_b1':
        img_size = (32, 32)
    else:
        img_size = (224, 224)
    stats = ((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))
    # specify transforms
    train_transform = tt.Compose([
        tt.RandomHorizontalFlip(),
        tt.RandomCrop(32, padding=4, padding_mode="reflect"),
        tt.Resize(img_size),
        tt.ToTensor(),
        tt.Normalize(*stats)
    ])

    test_transform = tt.Compose([
        tt.Resize(img_size),
        tt.ToTensor(),
        tt.Normalize(*stats)
    ])

    # load train, test and validation data
    train_data = CIFAR100_Dataset(download=True, root="./data", transform=train_transform)
    unlabeled_data = CIFAR100_Dataset(root="./data", transform=train_transform)
    val_data = CIFAR100_Dataset(root="./data", transform=test_transform)
    test_data = CIFAR100_Dataset(root="./data", train=False, transform=test_transform)

    X_train_data = train_data.data
    y_train_data = train_data.targets
    train_index = None
    if valid:
        # split train and validation data
        sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=seed)
        for train_index, val_index in sss.split(train_data.data, train_data.targets):
            X_train_data, val_data.data = train_data.data[train_index], train_data.data[val_index]
            y_train_data, val_data.targets = np.array(train_data.targets)[train_index], \
                                             np.array(train_data.targets)[val_index]

        train_data.data = X_train_data
        train_data.targets = y_train_data

    if L is not None:
        sss = StratifiedShuffleSplit(n_splits=2, test_size=L / len(y_train_data), random_state=seed)
        for exclude_index, include_index in sss.split(X_train_data, transform_to_coarse(y_train_data)):
            train_data.data, unlabeled_data.data = X_train_data[include_index], X_train_data[exclude_index]
            train_data.targets, unlabeled_data.targets = np.array(y_train_data)[include_index], np.empty(len(exclude_index))

    train_gt_data = transform_to_coarse(train_data.targets)
    test_gt_data = transform_to_coarse(test_data.targets)

    # generate expert labels
    if expert is not None:
        train_data.targets = expert.generate_expert_labels(train_data.targets, binary=binary)
        test_data.targets = expert.generate_expert_labels(test_data.targets, binary=binary)
        if valid:
            val_gt_data = transform_to_coarse(val_data.targets)
            val_data.targets = expert.generate_expert_labels(val_data.targets, binary=binary)
    else:
        train_data.targets = transform_to_coarse(train_data.targets)
        test_data.targets = transform_to_coarse(test_data.targets)
        if valid:
            val_gt_data = transform_to_coarse(val_data.targets)
            val_data.targets = transform_to_coarse(val_data.targets)

    if valid and gt_targets:
        return train_data, test_data, val_data, train_gt_data, test_gt_data, val_gt_data
    elif valid:
        return train_data, test_data, val_data
    elif gt_targets:
        return train_data, test_data, train_gt_data, test_gt_data
    else:
        return train_data, test_data


def get_nih_data(expert, seed=123, valid=True, L=None, gt_targets=True, binary=True):
    """Generate the train, validation and test set for the NIH dataset

    :param expert: NIH Expert
    :param seed: Random seed
    :param valid: Boolean flag for generating a validation dataset
    :param L: Number of instances with expert labels
    :param gt_targets: Boolean flag for returning ground truth targets
    :param binary: Boolean flag for binary expert labels
    :return: tuple (train_data, test_data)
        - train_data: Data set containing the training data
        - test_data: Data set containing the test data
        - valid_data: Data set containing the validation data (optional)
        - train_gt_data: Ground-truth label for the training data (optional)
        - test_gt_data: Ground-truth label for the test data (optional)
        - valid_gt_data: Ground-truth label for the val data (optional)
    """
    if expert is not None:
        target = expert.target
    else:
        target = "Airspace_Opacity"

    individual_labels = pd.read_csv("data/nih_labels.csv")
    img_dir = os.getcwd()[:-len('Embedding-Semi-Supervised')]+'nih_images/'
    if expert is not None:
        labeler_id = expert.labeler_id
        data = individual_labels[individual_labels['Reader ID'] == labeler_id]
    else:
        data = individual_labels
    x_data = np.array(data['Image ID'])

    y_ex_data = np.array(data[target+'_Expert_Label'])
    y_gt_data = np.array(data[target+'_GT_Label'])

    if binary:
        y_ex_data = 1*(y_gt_data == y_ex_data)

    # split train and test data
    train_index, test_index = generate_patient_train_test_split(data, 1234)
    x_train_data, x_test_data = x_data[train_index], x_data[test_index]
    y_gt_train_data, y_gt_test_data = y_gt_data[train_index], y_gt_data[test_index]
    y_ex_train_data, y_ex_test_data = y_ex_data[train_index], y_ex_data[test_index]

    if valid:
        # split train and validation data
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        for train_index, val_index in sss.split(x_train_data, y_gt_train_data):
            x_train_data, x_val_data = x_train_data[train_index], x_train_data[val_index]
            y_gt_train_data, y_gt_val_data = y_gt_train_data[train_index], y_gt_train_data[val_index]
            y_ex_train_data, y_ex_val_data = y_ex_train_data[train_index], y_ex_train_data[val_index]

    if L is not None:
        # generate labeled subset
        sss = StratifiedShuffleSplit(n_splits=1, test_size=L / len(x_train_data), random_state=seed)
        for _, train_index in sss.split(x_train_data, y_gt_train_data):
            x_train_subset_data = x_train_data[train_index]
            y_gt_train_subset_data = y_gt_train_data[train_index]
            y_ex_train_subset_data = y_ex_train_data[train_index]
    else:
        x_train_subset_data = x_train_data
        y_gt_train_subset_data = y_gt_train_data
        y_ex_train_subset_data = y_ex_train_data

    train_data = NIH_Dataset({'img': x_train_subset_data, 'label': y_ex_train_subset_data}, img_dir)
    test_data = NIH_Dataset({'img': x_test_data, 'label': y_ex_test_data}, img_dir)
    if valid:
        val_data = NIH_Dataset({'img': x_val_data, 'label': y_ex_val_data}, img_dir)

    if valid and gt_targets:
        return train_data, test_data, val_data, y_gt_train_subset_data, y_gt_test_data, y_gt_val_data
    elif valid:
        return train_data, test_data, val_data
    elif gt_targets:
        return train_data, test_data, y_gt_train_subset_data, y_gt_test_data
    else:
        return train_data, test_data


def get_data_loader(train_data, test_data, val_data=None, batch_size=64, shuffle_train=True):
    """Get data loaders

    :param train_data: Train data
    :param test_data: Test data
    :param val_data: Validation data
    :param batch_size: Batchsize
    :param shuffle_train: Shuffle the training data set
    :return: tuple
        - train_loader: Dataloader for the train set
        - test_loader: Dataloader for the test set
        - val_loader: Dataloader for the val set (optional)
        - device: Active device
    """
    # initiate data loaders for training and test data
    train_loader = DataLoader(train_data, batch_size, num_workers=4, pin_memory=True, shuffle=shuffle_train)
    test_loader = DataLoader(test_data, batch_size, num_workers=4, pin_memory=True)

    # get active device
    device = get_device()

    # get validation data loader if required
    if val_data is not None:
        val_loader = DataLoader(val_data, batch_size, num_workers=4, pin_memory=True)
        val_loader = ToDeviceLoader(val_loader, device)

        return train_loader, test_loader, val_loader, device
    else:
        return train_loader, test_loader, device


def resize_img(img, shape):
    """Resize image to specific shape

    :param img: Image
    :param shape: Tuple describing the shape of the image (height, width, channels)
    :return: Resized image
    """
    return cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


class CIFAR100_Dataset(CIFAR100):
    """Class representing the cifar100 dataset

    """
    def __getitem__(self, index: int):
        """Get item from cifar100 dataset
        :param index: Index
        :return: tuple
            - img: Images
            - target: Targets
            - index: Indices
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class NIH_Dataset(Dataset):
    """Class representing the NIH dataset

    :param data: Dataframe containing the image-ids, targets and individual expert labels
    :param img_dir: Directory of the images

    :ivar image_ids: Image ids
    :ivar targets: Ground-truth targets
    :ivar device: Device
    :ivar tfms: Image transformations
    :ivar images: Images
    """
    def __init__(self, data: pd.DataFrame, img_dir) -> None:
        self.image_ids = data['img']
        self.targets = data['label']
        # get active device
        self.device = get_device()

        self.tfms = tt.Compose(
            [
              tt.ToTensor(),
              tt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.images = []
        for filename in self.image_ids:
            img = Image.open(img_dir + filename)
            img = img.convert("RGB")
            img = img.resize((224, 224))
            img = self.tfms(img)
            self.images.append(img)

    def __getitem__(self, index: int):
        """Get item of the NIH dataset

        :param index: Index of the item to be returned
        :return: tuple
            - img: Image
            - target: Target
            - filename: Image id
        """
        filename, target = self.image_ids[index], self.targets[index]
        img = self.images[index]
        return img, target, filename

    def __len__(self) -> int:
        """Get length of NIH dataset

        :return: length
        """
        return len(self.images)


def get_nonbin_target(bin, y, num_classes):
    """Get multiclass targets from binary targets

    :param bin: Binary targets
    :param y: Ground truth targets
    :param num_classes: Number of classes
    :return: Multiclass targets
    """
    np.random.seed(123)
    # create empty arrays for the nonbinary targets
    nonbin = np.zeros(len(y))

    for i in range(len(y)):
        # multiclass target = ground truth target if binary target == 1
        if bin[i] == 1:
            nonbin[i] = y[i]
        # otherwise draw class from uniform distribution
        else:
            if num_classes == 2:
                nonbin[i] = 1-y[i]
            else:
                nonbin[i] = int(np.random.uniform(0, num_classes))
    return nonbin


def generate_patient_train_test_split(data, seed=1234):
    """Generate train test split based on patient ids

    :param data: Dataframe containing the image ids and the patient ids
    :param seed: Random seed
    :return: tuple
        - train_idx: Train indices
        - test_idx: Test indices
    """
    patient_ids = np.unique(data['Patient ID'])
    np.random.seed(seed)
    test_ids = np.random.choice(patient_ids, int(len(patient_ids)*0.2))
    test_idx = []
    train_idx = []
    for i, id in enumerate(data['Patient ID']):
        if id in test_ids:
            test_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, test_idx

