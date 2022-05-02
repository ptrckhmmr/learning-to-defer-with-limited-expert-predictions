import os.path as osp
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import transform as T
from datasets.randaugment import RandomAugment
from datasets.sampler import RandomSampler, BatchSampler


class TwoCropsTransform:
    """Take 2 random augmentations of one image

    :param trans_weak: Transform for the weak augmentation
    :param trans_strong: Transform for the strong augmentation

    :ivar trans_weak: Transform for the weak augmentation
    :ivar trans_strong: Transform for the strong augmentation
    """

    def __init__(self, trans_weak, trans_strong):
        self.trans_weak = trans_weak
        self.trans_strong = trans_strong

    def __call__(self, x):
        x1 = self.trans_weak(x)
        x2 = self.trans_strong(x)
        return [x1, x2]


class ThreeCropsTransform:
    """Take 3 random augmentations of one image

    :param trans_weak: Transform for the weak augmentation
    :param trans_strong0: Transform for the first strong augmentation
    :param trans_strong1: Transform for the second strong augmentation

    :ivar trans_weak: Transform for the weak augmentation
    :ivar trans_strong0: Transform for the first strong augmentation
    :ivar trans_strong1: Transform for the second strong augmentation
    """

    def __init__(self, trans_weak, trans_strong0, trans_strong1):
        self.trans_weak = trans_weak
        self.trans_strong0 = trans_strong0
        self.trans_strong1 = trans_strong1

    def __call__(self, x):
        x1 = self.trans_weak(x)
        x2 = self.trans_strong0(x)
        x3 = self.trans_strong1(x)
        return [x1, x2, x3]


def load_data_train(L=250, dataset='CIFAR10', dspth='./data'):
    """Load the train dataset

    :param L: Number of labeled instances
    :param dataset: Name of the dataset
    :param dspth: Path of the dataset

    :return: tuple
        - data_x: Images of the labeled set
        - label_x: Label of the labeled set
        - data_u: Images of the unlabeled set
        - label_u: Label of the unlabeled set
    """

    if dataset == 'CIFAR10':
        datalist = [
            osp.join(dspth, 'cifar-10-batches-py', 'data_batch_{}'.format(i + 1))
            for i in range(5)
        ]
        n_class = 10
        assert L in [10, 20, 40, 80, 250, 4000]
    elif dataset == 'CIFAR100':
        datalist = [
            osp.join(dspth, 'cifar-100-python', 'train')]
        n_class = 20
        assert L in [None, 40, 80, 120, 200, 400, 1000, 5000]
    # load images and labels
    data, labels = [], []
    for data_batch in datalist:
        with open(data_batch, 'rb') as fr:
            entry = pickle.load(fr, encoding='latin1')
            lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
            data.append(entry['data'])
            labels.append(lbs)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    labels_coarse = transform_to_coarse(labels)
    # generate the labeled and unlabeled datasets
    if L is None:
        data = [
            el.reshape(3, 32, 32).transpose(1, 2, 0)
            for el in data
        ]
        return data, labels, None, None
    else:
        n_labels = L // n_class
        data_x, label_x, data_u, label_u = [], [], [], []
        for i in range(n_class):
            indices = np.where(labels_coarse == i)[0]
            np.random.shuffle(indices)
            inds_x, inds_u = indices[:n_labels], indices[n_labels:]
            data_x += [
                data[i].reshape(3, 32, 32).transpose(1, 2, 0)
                for i in inds_x
            ]
            label_x += [labels[i] for i in inds_x]
            data_u += [
                data[i].reshape(3, 32, 32).transpose(1, 2, 0)
                for i in inds_u
            ]
            label_u += [labels[i] for i in inds_u]
        return data_x, label_x, data_u, label_u


def load_data_val(dataset, dspth='./data'):
    """Load data for the validation set

    :param dataset: Name of the dataset
    :param dspth: Path of the dataset

    :return: tuple
        - data: Images
        - labels: Labels
    """
    if dataset == 'CIFAR10':
        datalist = [
            osp.join(dspth, 'cifar-10-batches-py', 'test_batch')
        ]
    elif dataset == 'CIFAR100':
        datalist = [
            osp.join(dspth, 'cifar-100-python', 'test')
        ]
    # load data
    data, labels = [], []
    for data_batch in datalist:
        with open(data_batch, 'rb') as fr:
            entry = pickle.load(fr, encoding='latin1')
            lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
            data.append(entry['data'])
            labels.append(lbs)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    data = [
        el.reshape(3, 32, 32).transpose(1, 2, 0)
        for el in data
    ]
    return data, labels


def compute_mean_var():
    """Compute mean and variance of the images from the train set

    :return:
    """
    data_x, label_x, data_u, label_u = load_data_train()
    data = data_x + data_u
    data = np.concatenate([el[None, ...] for el in data], axis=0)

    mean, var = [], []
    for i in range(3):
        channel = (data[:, :, :, i].ravel() / 127.5) - 1
        #  channel = (data[:, :, :, i].ravel() / 255)
        mean.append(np.mean(channel))
        var.append(np.std(channel))

    print('mean: ', mean)
    print('var: ', var)


class Cifar(Dataset):
    """Class representing the CIFAR dataset

    :param dataset: Name of the dataset
    :param data: Images
    :param labels: Labels
    :param mode: Mode
    :param imsize: Image size

    :ivar data: Images
    :ivar labels: Labels
    :ivar mode: Mode
    """
    def __init__(self, dataset, data, labels, mode, imsize):
        super(Cifar, self).__init__()
        self.data, self.labels = data, labels
        self.mode = mode
        assert len(self.data) == len(self.labels)
        if dataset == 'CIFAR10':
            mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        elif dataset == 'CIFAR100':
            mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

        trans_weak = T.Compose([
            T.Resize((imsize, imsize)),
            T.PadandRandomCrop(border=4, cropsize=(imsize, imsize)),
            T.RandomHorizontalFlip(p=0.5),
            T.Normalize(mean, std),
            T.ToTensor(),
        ])
        trans_strong0 = T.Compose([
            T.Resize((imsize, imsize)),
            T.PadandRandomCrop(border=4, cropsize=(imsize, imsize)),
            T.RandomHorizontalFlip(p=0.5),
            RandomAugment(2, 10),
            T.Normalize(mean, std),
            T.ToTensor(),
        ])
        trans_strong1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(imsize, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        if self.mode == 'train_x':
            self.trans = trans_weak
        elif self.mode == 'train_u_comatch':
            self.trans = ThreeCropsTransform(trans_weak, trans_strong0, trans_strong1)
        elif self.mode == 'train_u_fixmatch':
            self.trans = TwoCropsTransform(trans_weak, trans_strong0)
        else:
            self.trans = T.Compose([
                T.Resize((imsize, imsize)),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])

    def __getitem__(self, idx):
        im, lb = self.data[idx], self.labels[idx]
        return self.trans(im), lb, idx

    def __len__(self):
        leng = len(self.data)
        return leng


def get_train_loader(dataset, expert, batch_size, mu, n_iters_per_epoch, L, root='data', method='comatch', imsize=32):
    """Get data loader for the train set

    :param dataset: Name of the dataset
    :param expert: Synthetic cifar expert
    :param batch_size: Batch size
    :param mu: Factor of train batch size of unlabeled samples
    :param n_iters_per_epoch: Number of iteration per epoch
    :param L: Number of labeled instances
    :param root: Path of the dataset
    :param method: Training algorithm (either comatch or fixmatch)
    :param imsize: Size of images

    :return: tuple
        - dl_x: Dataloader for the labeled instances
        - dl_u: Dataloader for the unlabeled instances
    """
    data_x, label_x, data_u, label_u = load_data_train(L=L, dataset=dataset, dspth=root)
    if expert is not None:
        label_x = expert.generate_expert_labels(label_x, binary=True)
        if label_u is not None:
            label_u = expert.generate_expert_labels(label_u, binary=True)
    ds_x = Cifar(
        dataset=dataset,
        data=data_x,
        labels=label_x,
        mode='train_x',
        imsize=imsize
    )  # return an iter of num_samples length (all indices of samples)
    sampler_x = RandomSampler(ds_x, replacement=True, num_samples=n_iters_per_epoch * batch_size)
    batch_sampler_x = BatchSampler(sampler_x, batch_size, drop_last=True)  # yield a batch of samples one time
    dl_x = torch.utils.data.DataLoader(
        ds_x,
        batch_sampler=batch_sampler_x,
        num_workers=2,
        pin_memory=True
    )
    if data_u is None:
        return dl_x
    else:
        ds_u = Cifar(
            dataset=dataset,
            data=data_u,
            labels=label_u,
            mode='train_u_%s' % method,
            imsize=imsize
        )
        sampler_u = RandomSampler(ds_u, replacement=True, num_samples=mu * n_iters_per_epoch * batch_size)
        # sampler_u = RandomSampler(ds_u, replacement=False)
        batch_sampler_u = BatchSampler(sampler_u, batch_size * mu, drop_last=True)
        dl_u = torch.utils.data.DataLoader(
            ds_u,
            batch_sampler=batch_sampler_u,
            num_workers=2,
            pin_memory=True
        )
        return dl_x, dl_u


def get_val_loader(dataset, expert, batch_size, num_workers, pin_memory=True, root='data', imsize=32):
    """Get data loader for the validation set

    :param dataset: Name of the dataset
    :param expert: Synthetic cifar expert
    :param batch_size: Batch size
    :param num_workers: Number of workers
    :param pin_memory: Pin memory
    :param root: Path of the dataset
    :param imsize: Size of images

    :return: Dataloader
    """
    data, labels = load_data_val(dataset=dataset, dspth=root)
    if expert is not None:
        labels = expert.generate_expert_labels(labels, binary=True)

    ds = Cifar(
        dataset=dataset,
        data=data,
        labels=labels,
        mode='test',
        imsize=imsize
    )
    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dl


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
