import numpy as np
import pandas as pd
import os

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from datasets import transform as T
from datasets.randaugment import RandomAugment
from datasets.sampler import RandomSampler, BatchSampler
from PIL import Image
import torch
from collections import Counter

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


def generate_patient_train_test_split(data, seed=1234):
    """Generate train test split from the patient ids

    :param data: Dataset including the patient ids and image ids
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


def load_data_train(L, expert):
    """Load the train dataset

    :param L: Number of labeled instances
    :param expert: NIH expert

    :return: tuple
        - data_x: Images of the labeled set
        - label_x: Label of the labeled set
        - data_u: Images of the unlabeled set
        - label_u: Label of the unlabeled set
    """
    n_class = 2
    if expert is not None:
        target = expert.target
    else:
        target = 'Airspace_Opacity'
    individual_labels = pd.read_csv("data/nih_labels.csv")

    if expert is not None:
        labeler_id = expert.labeler_id
        data = individual_labels[individual_labels['Reader ID'] == labeler_id]
    else:
        data = individual_labels
    x_data = np.array(data['Image ID'])

    y_ex_data = np.array(data[target+'_Expert_Label'])
    y_gt_data = np.array(data[target + '_GT_Label'])
    y_data = y_ex_data

    # split train and test data
    train_index, _ = generate_patient_train_test_split(data, 12345)
    x_train_data = x_data[train_index]
    y_train_data = y_data[train_index]
    y_gt_train_data = y_gt_data[train_index]

    if L is not None:
        n_labels = L // n_class
        data_x, label_x, data_u, label_u = [], [], [], []
        for i in range(n_class):
            indices = np.where(y_gt_train_data == i)[0]
            np.random.shuffle(indices)
            inds_x, inds_u = indices[:n_labels], indices[n_labels:]
            data_x += [x_train_data[i] for i in inds_x]
            label_x += [y_train_data[i] for i in inds_x]
            data_u += [x_train_data[i] for i in inds_u]
            label_u += [y_train_data[i] for i in inds_u]
    else:
        data_x = x_train_data
        label_x = y_train_data
        data_u = None
        label_u = None

    return data_x, label_x, data_u,  label_u


def load_data_val(expert):
    """Load data for the validation set

    :param expert: NIH expert

    :return: tuple
        - data: Images
        - labels: Labels
    """
    if expert is not None:
        target = expert.target
    else:
        target = 'Airspace_Opacity'
    individual_labels = pd.read_csv("data/nih_labels.csv")

    if expert is not None:
        labeler_id = expert.labeler_id
        data = individual_labels[individual_labels['Reader ID'] == labeler_id]
    else:
        data = individual_labels
    x_data = np.array(data['Image ID'])

    y_ex_data = np.array(data[target + '_Expert_Label'])
    y_gt_data = np.array(data[target + '_GT_Label'])
    y_data = y_ex_data

    # split train and test data
    _, test_index = generate_patient_train_test_split(data, 12345)
    x_test_data = x_data[test_index]
    y_test_data = y_data[test_index]

    data = x_test_data
    label = y_test_data

    return data, label


class NIH_Dataset(Dataset):
    """Class representing the NIH dataset

    :param data: Images
    :param labels: Labels
    :param mode: Mode
    :param imsize: Image size

    :ivar data: Images
    :ivar labels: Labels
    :ivar mode: Mode
    """
    def __init__(self, data, labels, mode, imsize=224) -> None:
        self.image_ids = data
        self.labels = labels
        self.mode = mode
        # directory of the images fro mthe NIH dataset
        img_dir = os.getcwd()[:-len('Semi-Supervised')] + 'nih_images/'
        images = []
        for filename in self.image_ids:
            img = Image.open(img_dir + filename)
            img = img.convert("RGB")
            img = img.resize((imsize, imsize))
            images.append(np.array(img))
        self.images = images

        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

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

    def __getitem__(self, index: int):
        filename, label = self.image_ids[index], self.labels[index]
        im = self.images[index]
        return self.trans(im), label, filename

    def __len__(self) -> int:
        return len(self.images)


def get_train_loader(expert, batch_size, mu, n_iters_per_epoch, L, method='comatch', imsize=224):
    """Get data loader for the train set

    :param expert: Synthetic cifar expert
    :param batch_size: Batch size
    :param mu: Factor of train batch size of unlabeled samples
    :param n_iters_per_epoch: Number of iteration per epoch
    :param L: Number of labeled instances
    :param method: Training algorithm (either comatch or fixmatch)
    :param imsize: Size of images

    :return: tuple
        - dl_x: Dataloader for the labeled instances
        - dl_u: Dataloader for the unlabeled instances
    """
    data_x, label_x, data_u, label_u = load_data_train(L=L, expert=expert)
    print(f'Label check: {Counter(label_x)}')
    ds_x = NIH_Dataset(
        data=data_x,
        labels=label_x,
        mode='train_x',
        imsize=imsize
    )
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
        ds_u = NIH_Dataset(
            data=data_u,
            labels=label_u,
            mode='train_u_%s'%method,
            imsize=imsize
        )
        sampler_u = RandomSampler(ds_u, replacement=True, num_samples=mu * n_iters_per_epoch * batch_size)
        batch_sampler_u = BatchSampler(sampler_u, batch_size * mu, drop_last=True)
        dl_u = torch.utils.data.DataLoader(
            ds_u,
            batch_sampler=batch_sampler_u,
            num_workers=2,
            pin_memory=True
        )
        return dl_x, dl_u


def get_val_loader(expert, batch_size, num_workers, pin_memory=True, imsize=224):
    """Get data loader for the validation set

    :param expert: Synthetic cifar expert
    :param batch_size: Batch size
    :param num_workers: Number of workers
    :param pin_memory: Pin memory
    :param imsize: Size of images

    :return: Dataloader
    """
    data, labels = load_data_val(expert=expert)

    ds = NIH_Dataset(
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


