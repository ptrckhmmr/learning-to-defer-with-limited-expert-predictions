import pickle

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as tt
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def get_train_test_data():
    """ Generate the train and test set

    :return: tuple (train_data, test_data)
        - train_data - Data set containing the training data
        - test_data - Data set containing the test data
    """
    stats = ((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))
    transform = tt.Compose([
        tt.Resize((224, 224)),
        tt.ToTensor(),
        tt.Normalize(*stats)
    ])

    train_data = CIFAR100(download=True, root="./data", transform=transform)
    test_data = CIFAR100(root="./data", train=False, transform=transform)

    return train_data, test_data


def get_device():
    """ Get active device

    :return: device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(data, device):
    """ Load data to device

    :param data: Data
    :param device: Device
    :return: data (loaded to device)
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class ToDeviceLoader:
    def __init__(self, data, device):
        self.data = data
        self.device = device

    def __iter__(self):
        for batch in self.data:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.data)


def get_data_loader(train_data, test_data, batch_size=64):
    """ Get train and test data loader

    :param train_data: Trainings data
    :param test_data:  Test data
    :param batch_size: Batchsize
    :return: tuple (train_loader, test_loader, device)
        - train_loader - Data loader for the trainings data
        - test_loader - Data loader for the test data
        - device - Device used
    """
    train_loader = DataLoader(train_data, batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size, num_workers=4, pin_memory=True)

    device = get_device()

    train_loader = ToDeviceLoader(train_loader, device)
    test_loader = ToDeviceLoader(test_loader, device)

    return train_loader, test_loader, device


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


def unpickle(file):
    """Function to open the files using pickle

    :param file: File to be loaded
    :return: Loaded file as dictionary
    """
    with open(file, 'rb') as fo:
        myDict = pickle.load(fo, encoding='latin1')
    return myDict


def load_meta_data(wkdir):
    """Load CIFAR100 fine targets

    :param wkdir: Working directory
    :return: tuple (trainData, testData, metaData)
        - trainData['fine_labels'] - fine labels for training data
        - testData['fine_labels'] - fine labels for test data
    """
    metaData = unpickle(wkdir + '/data/meta')

    return metaData


def load_test_data(wkdir):
    """Load CIFAR100 fine targets

    :param wkdir: Working directory
    :return: tuple (trainData, testData, metaData)
        - trainData['fine_labels'] - fine labels for training data
        - testData['fine_labels'] - fine labels for test data
    """
    testData = unpickle(wkdir + '/data/test')

    return testData


def get_confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix

    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    return cm


def plot_confusion_matrix(y_true, y_pred, plot_name=None):
    """Cplot confusion matrix

    :param y_true: True labels
    :param y_pred: Predicted labels
    :param plot_name: Filename saved plot
    :return:
    """
    cm = get_confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(cm, annot=False, cmap='Blues')
    ax.set_title('Predicted Expert Labels Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    if plot_name is not None:
        plt.savefig('plots/' + plot_name + '.png')
    plt.show()
