import numpy as np
import torchvision
from torchvision import transforms
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import pandas as pd
import os


class CIFAR100_Dataset(torchvision.datasets.CIFAR100):
    """Class representing the CIFAR100 dataset
    """
    def __getitem__(self, index: int):
        """Get item from dataset

        :param index: Index of image to get
        :return: tuple
            - img: Image
            - target: Label
            - index: Index
        """
        img, fine_target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(fine_target)
        else:
            target = fine_target

        return img, target, index


class CIFAR100_Dataloader:
    """Dataloader for the CIFAR100 dataset

    :param train_batch_size: Batch size for the training set
    :param test_batch_size: Batch size for the test set
    :param seed: Random seed
    :param small_version: True if small version of CIFAR100 should be loaded

    :ivar train_batch_size: Batch size for the training set
    :ivar test_batch_size: Batch size for the test set
    :ivar seed: Random seed
    :ivar small_version: True if small version of CIFAR100 should be loaded
    :ivar trainset: Training set
    :ivar valset: Validation set
    :ivar testset: Test set
    :ivar train_indices: Indices of train set
    :ivar val_indices: Indices of validation set
    :ivar test_indices: Indices of test set
    """

    def __init__(self, train_batch_size=128, test_batch_size=128, seed=42, small_version=True):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.seed = seed
        self.small_version = small_version

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                 [0.2675, 0.2565, 0.2761])])

        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                 [0.2675, 0.2565, 0.2761])])

        coarse_labels = np.array([
            4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
            3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
            6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
            0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
            5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
            16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
            10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
            2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
            16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
            18, 1, 2, 15, 6, 0, 17, 8, 14, 13])

        target_transform = lambda x: coarse_labels[x]

        np.random.seed(self.seed)
        train_val_set = CIFAR100_Dataset(root='./data', train=True, download=True, transform=transform_train,
                                         target_transform=target_transform)
        all_indices = np.arange(0, 50000, 1)
        train_indices = np.random.choice(all_indices, 40000, replace=False)
        val_indices = np.setdiff1d(all_indices, train_indices)
        self.trainset = torch.utils.data.Subset(train_val_set, train_indices)
        self.valset = torch.utils.data.Subset(train_val_set, val_indices)

        self.testset = CIFAR100_Dataset(root='./data', train=False, download=True, transform=transform_test,
                                        target_transform=target_transform)

        if self.small_version:
            np.random.seed(self.seed)
            train_indices = np.random.choice(np.arange(0, 40000, 1), 4000, replace=False)
            val_indices = np.random.choice(np.arange(0, 10000, 1), 1000, replace=False)
            test_indices = np.random.choice(np.arange(0, 10000, 1), 1000, replace=False)

            self.trainset = torch.utils.data.Subset(self.trainset, train_indices)
            self.valset = torch.utils.data.Subset(self.valset, val_indices)
            self.testset = torch.utils.data.Subset(self.testset, test_indices)

        self.train_indices = train_indices
        self.val_indices = val_indices
        if self.small_version:
            self.test_indices = test_indices
        else:
            self.test_indices = range(10000)

    def get_data_loader(self):
        """Get dataloader for train, validation and test set

        :return: tuple
            - train_loader: Dataloader for the train set
            - val_loader: Dataloader for the validation set
            - test_loader: Dataloader for the test set
        """
        train_loader = self._get_data_loader(dataset=self.trainset, batch_size=self.train_batch_size, drop_last=True)
        val_loader = self._get_data_loader(dataset=self.valset, batch_size=self.test_batch_size, drop_last=False)
        test_loader = self._get_data_loader(dataset=self.testset, batch_size=self.test_batch_size, drop_last=False)
        return train_loader, val_loader, test_loader

    def _get_data_loader(self, dataset, batch_size, drop_last, shuffle=True):
        """Get data loader from dataset

        :param dataset: Dataset
        :param batch_size: Batch size
        :param drop_last: Drop last
        :param shuffle: Shuffle instances
        :return: dataloader
        """
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2,
                                           drop_last=drop_last, pin_memory=True)


class NIH_Dataset(Dataset):
    """Class representing the NIH dataset

    :param data: Dict containing the image ids and labels
    :param img_dir: Directory of the images

    :ivar image_ids: Ids of the images
    :ivar targets: Targets
    :ivar tfms: Tramsform
    :ivar images: Images
    """
    def __init__(self, data: dict, img_dir) -> None:
        self.image_ids = data['img']
        self.targets = data['label']

        self.tfms = transforms.Compose(
            [
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.images = []
        for filename in self.image_ids:
            img = Image.open(img_dir + filename)
            img = img.convert("RGB")
            img = img.resize((224, 224))
            img = self.tfms(img)
            self.images.append(img)

    def __getitem__(self, index: int):
        """Get item from NIH dataset

        :param index: Index of the image to get
        :return: tuple
            - img: Image
            - target: Target
            - filename: Image ID
        """
        filename, target = self.image_ids[index], self.targets[index]
        img = self.images[index]
        return img, target, filename

    def __len__(self) -> int:
        """Get length of dataset

        :return: length
        """
        return len(self.images)


class NIH_Dataloader():
    """Dataloader for the NIH dataset

    :param labeler_id: ID of the expert labeler
    :param target: Target name
    :param seed: Random seed
    :param train_batch_size: Batch size for the training set
    :param test_batch_size: Batch size for the test set

    :ivar train_batch_size: Batch size for the training set
    :ivar test_batch_size: Batch size for the test set
    :ivar train_indices: Indices of the training set
    :ivar test_indices: Indices of the test set
    :ivar trainset: Train set
    :ivar testset: Test set
    """

    def __init__(self, labeler_id=4323195249, target="Airspace_Opacity", seed=1234, train_batch_size=128, test_batch_size=128):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        img_dir = os.getcwd()[:-len('human-AI-systems/hemmer')]+ 'nih_images/images_indlabels/'
        individual_labels = pd.read_csv(img_dir[:-len('images_indlabels/')]+'nih_labels.csv')
        data = individual_labels[individual_labels['Reader ID'] == labeler_id]
        x_data = np.array(data['Image ID'])
        y_data = np.array(data[target+'_GT_Label'])

        # split train and test data
        train_index, test_index = self.generate_patient_train_test_split(data, seed=seed)
        x_train_data, x_test_data = x_data[train_index], x_data[test_index]
        y_train_data, y_test_data = y_data[train_index], y_data[test_index]

        self.train_indices = train_index
        self.test_indices = test_index

        self.trainset = NIH_Dataset({'img': x_train_data, 'label': y_train_data}, img_dir)
        self.testset = NIH_Dataset({'img': x_test_data, 'label': y_test_data}, img_dir)

    def get_data_loader(self):
        """Get train and test dataloader

        :return: tuple
            - train_loader: Data loader for the train set
            - test_loader: Data loader for the test set
        """
        train_loader = self._get_data_loader(dataset=self.trainset, batch_size=self.train_batch_size, drop_last=True)
        test_loader = self._get_data_loader(dataset=self.testset, batch_size=self.test_batch_size, drop_last=False)
        return train_loader, test_loader

    def _get_data_loader(self, dataset, batch_size, drop_last, shuffle=True):
        """Get dataloader from dataset

        :param dataset: Dataset
        :param batch_size: Batch size
        :param drop_last: Drop last
        :param shuffle: Shuffle dataset
        :return: Dataloader
        """
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2,
                                           drop_last=drop_last, pin_memory=True)

    def generate_patient_train_test_split(self, data, seed=123):
        """Generate train test split based on patient ids

        :param data: Dataframe containing image ids and patient ids
        :param seed: Random seed
        :return: tuple
            - train_idx: Indices of the train set
            - test_ids: Indices of the test set
        """
        patient_ids = np.unique(data['Patient ID'])
        np.random.seed(seed)
        test_ids = np.random.choice(patient_ids, int(len(patient_ids) * 0.2))
        test_idx = []
        train_idx = []
        for i, id in enumerate(data['Patient ID']):
            if id in test_ids:
                test_idx.append(i)
            else:
                train_idx.append(i)
        return train_idx, test_idx
