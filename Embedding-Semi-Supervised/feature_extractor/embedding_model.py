import os
import sys
import torch
import timm
import torchvision.transforms as T
from torchvision.models.resnet import resnet18
import torch.nn as nn


class EmbeddingModel:
    """Class representing the embedding model

    :param train_dir: Training directory containing the saved weights of the embedding model
    :param dataset: Name of the dataset

    :ivar train_dir: Training directory
    :ivar dataset: Name of the dataset
    :ivar args: Training arguments
    :ivar device: Device
    :ivar emb_model: Embedding model
    """
    def __init__(self, train_dir, dataset):
        self.train_dir = train_dir
        self.dataset = dataset.lower()
        if self.dataset == 'cifar100':
            self.args = {'dataset': self.dataset, 'fe_model': 'efficientnet_b1', 'num_classes': 20, 'batch': 800}
        elif self.dataset == 'nih':
            self.args = {'dataset': self.dataset, 'fe_model': 'resnet18', 'num_classes': 2, 'batch': 800}
        else:
            print(f'Dataset {self.dataset} not implemented')
            sys.exit()
        self.device = get_device()
        self.emb_model = self.get_emb_model(os.getcwd())

    def get_emb_model(self, wkdir):
        """Initialize base model

        :param wkdir:
        :return: model
        """
        if self.dataset == 'cifar100':
            # load model
            model = timm.create_model(self.args['fe_model'], pretrained=True, num_classes=self.args['num_classes'])
            model = self.load_emb_net_from_checkpoint(model, wkdir)
            model = torch.nn.Sequential(*list(model.children())[:-1])
        elif self.dataset == 'nih':
            model = Resnet(self.args['num_classes'], self.train_dir)
        print('Loaded Model', self.args['fe_model'])
        model = to_device(model, self.device)
        return model

    def get_embedding(self, batch):
        """Get embedding from images

        :param batch: Batch of images
        :return: Feature vectors
        """
        if self.dataset == 'cifar100':
            batch = T.Resize((224, 224))(batch)
        self.emb_model.eval()
        batch_features = self.emb_model(batch)
        return batch_features

    def get_emb_net_dir(self, wkdir):
        """Get training directory of the embedding net

        :param wkdir: Working directory
        :return: base_cnn_dir
        """
        args_base = {'model': self.args['fe_model'],
                     'num_classes': self.args['num_classes'],
                     'batch': 64}

        base_cnn_dir = get_train_dir(wkdir, args_base, 'base_net')
        return base_cnn_dir

    def load_emb_net_from_checkpoint(self, emb_model, wkdir, mode='best'):
        """Load base model weights from checkpoint

        :param emb_model: Initialized base model
        :param wkdir: Working directory
        :param mode: Checkpoint to load (best or latest)
        :return: base model
        """
        # get checkpoint
        cp_dir = self.get_emb_net_dir(wkdir) + 'checkpoints/checkpoint.' + mode
        try:
            # load state dict from checkpoint
            checkpoint = torch.load(cp_dir)
            emb_model.load_state_dict(checkpoint['model_state_dict'])
            print('Found base net checkpoint at', cp_dir)
        except FileNotFoundError:
            print('No base net Checkpoint found at', cp_dir)
            sys.exit()

        # freeze base model layers
        for param in emb_model.parameters():
            param.requires_grad = False

        return emb_model


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


def concat_args(args, mode):
    """Concatenate args to string

    :param args: Args
    :param mode: Mode
    :return: String
    """
    args_string = mode + '@'
    for key in args:
        if key not in ['batch', 'epochs', 'input_shape']:
            args_string += str(key) + '-' + str(args[key]) + '-'
    return args_string[:-1]


def get_train_dir(wkdir, args, mode):
    """Get or create training directory

    :param wkdir: Working directory
    :param mode: Mode
    :param args: Args
    """

    path = wkdir + '/CIFAR100/' + concat_args(args, mode) + '/'
    try:
        os.mkdir(path)
    except:
        pass
    return path


class Resnet(torch.nn.Module):
    def __init__(self, num_classes, train_dir):
        super().__init__()
        self.num_classes = num_classes
        self.resnet = resnet18(pretrained=True)

        try:
            print('load Resnet-18 checkpoint')
            print(self.load_my_state_dict(
                torch.load(
                    train_dir + "/NIH/emb_net@dataset-nih-model-resnet18-num_classes-2/checkpoints/checkpoint.pretrain"),
                strict=False))
        except KeyError:
            print('load Resnet-18 pretrained on ImageNet')

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)


    def load_my_state_dict(self, state_dict, strict=True):
        pretrained_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        self.resnet.load_state_dict(pretrained_dict, strict=strict)

    def forward(self, x, return_features=True):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        features = torch.flatten(x, 1)
        if return_features:
            return features
        else:
            out = self.resnet.fc(features)
            out = nn.Softmax(dim=1)(out)
            return out

