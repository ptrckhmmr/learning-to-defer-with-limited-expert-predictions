import sys
import torch
import timm
import torch.nn as nn
import torch.optim as optim
import math
import warnings

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score

import scripts.data_loading as prep

from scripts.metrics import expert_accuracy_score, get_accuracy_by_cat
from scripts.utils import printProgressBar, get_train_dir
from scripts.emb_model_lib import EmbeddingModel, Resnet
from scripts.Expert import CIFAR100Expert, NIHExpert

warnings.filterwarnings("ignore", category=UserWarning)


class NNExpertModel(EmbeddingModel):
    """Class representing the NN expert model

    :param args: Training arguments for the embedding model
    :param wkdir: Working directory
    :param writer: Tensorboard writer

    :ivar global_step: Global step
    :ivar args: Training arguments
    :ivar device: Active device
    :ivar train_dir: Training directory
    :ivar writer: Tensorboard writer
    :ivar emb_model: Embedding model
    :ivar model: Expert model
    :ivar optimizer: Optimizer
    :ivar scheduler: Learning rate scheduler
    :ivar loss_function: Loss function
    :ivar expert: Expert
    :ivar train_data: Train dataset
    :ivar test_data: Test dataset
    :ivar val_data: Validation dataset
    :ivar train_gt_data: Ground-truth label for the train dataset
    :ivar test_gt_data: Ground-truth label for the test dataset
    :ivar val_gt_data: Ground-truth label for the validation dataset
    """
    def __init__(self, args, wkdir, writer):
        self.global_step = 0
        self.args = args
        self.device = prep.get_device()
        self.train_dir = get_train_dir(wkdir, args, 'expert_net')
        self.writer = writer
        self.emb_model = self.get_emb_model(wkdir)
        self.model = self.get_expert_model()
        if self.args['opt'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args['lr'],
                                             dampening=0, momentum=0.9, nesterov=True)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args['lr'])
        if self.args['sched'] == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [50, 70, 90], gamma=0.2)
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=40)
        if self.args['binary']:
            self.loss_function = nn.BCELoss()
        else:
            self.loss_function = nn.CrossEntropyLoss()

        if self.args['n_strengths'] is not None:
            if self.args['dataset'] == 'nih':
                self.expert = NIHExpert(id=self.args['n_strengths'], n_classes=args['num_classes'],
                                        target='Airspace_Opacity')
            else:
                self.expert = CIFAR100Expert(self.args['num_classes'], self.args['n_strengths'], 1, 0,
                                             seed=args['ex_seed'])
        else:
            self.expert = None
        if 'cifar' in self.args['dataset']:
            self.train_data, self.test_data, self.val_data, self.train_gt_data, self.test_gt_data, self.val_gt_data = \
                prep.get_train_val_test_data(dataset=args['dataset'], expert=self.expert, binary=args['binary'],
                                             gt_targets=True,
                                             model=args['emb_model'], L=args['labels'], seed=args['seed'])
            self.train_loader, self.test_loader, self.val_loader, device = prep.get_data_loader(self.train_data,
                                                                                                self.test_data,
                                                                                                self.val_data,
                                                                                                args['batch'])
        elif self.args['dataset'] == 'nih':
            self.train_data, self.test_data, self.train_gt_data, self.test_gt_data = \
                prep.get_train_val_test_data(dataset=args['dataset'], expert=self.expert, binary=args['binary'],
                                             gt_targets=True,
                                             model=args['emb_model'], L=args['labels'], seed=args['seed'],
                                             valid=False)
            self.train_loader, self.test_loader, device = prep.get_data_loader(self.train_data,
                                                                               self.test_data,
                                                                               batch_size=args['batch'])

        self.save_model_args()

    def get_emb_model(self, wkdir):
        """Initialize base model

        :param wkdir: working directory
        :return: model
        """
        # load model
        if self.args['dataset'] == 'nih':
            model = Resnet(num_classes=self.args['num_classes'])
            model = self.load_emb_net_from_checkpoint(model, wkdir, strict=False)
        else:
            model = timm.create_model(self.args['emb_model'], pretrained=True, num_classes=self.args['num_classes'])
            model = self.load_emb_net_from_checkpoint(model, wkdir)
            model = torch.nn.Sequential(*list(model.children())[:-1])
        print('Loaded Model', self.args['emb_model'])
        model = prep.to_device(model, self.device)
        return model

    def get_expert_model(self):
        """Initialize expert model

        :return: model
        """
        if self.args['binary']:
            num_classes = 1
        else:
            num_classes = self.args['num_classes']

        if self.args['emb_model'] == 'resnet18':
            feature_dim = 512
        else:
            feature_dim = 1280

        if self.args['ex_model'][:8] == 'LinearNN':
            model = LinearNN(num_classes, size=self.args['ex_model'][9:], feature_dim=feature_dim)
        else:
            print('No class fround for expert model', self.args['ex_model'])
            sys.exit()

        model = prep.to_device(model, self.device)
        print('Initialized Expert Model:')
        print(model)
        return model

    def train_one_epoch(self, epoch):
        """Train one epoch of the expert model

        :param epoch: Epoch
        :return: loss
        """
        self.emb_model.eval()
        self.model.train()
        for ii, (data, target, index) in enumerate(self.train_loader):
            data = data.to(self.device)
            if self.args['binary']:
                target = target.float().to(self.device)
            else:
                target = target.long().to(self.device)
            self.optimizer.zero_grad()

            if self.args['dataset'] == 'nih':
                features = self.emb_model(data, return_features=True)
            else:
                features = self.emb_model(data)
            pred = self.model(features)

            if self.args['binary']:
                m = nn.Sigmoid()
                pred = m(pred)
                pred = torch.reshape(pred, [pred.shape[0]])

            loss = self.loss_function(pred, target)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += len(index)
            printProgressBar(ii + 1, math.ceil((len(self.train_data.targets) / self.args['batch'])),
                             prefix='Train Epoch ' + str(epoch + 1) + ':',
                             suffix='Complete', length=40)

            self.writer.add_scalar('Loss/loss', loss, self.global_step)
        return loss.item()

    def get_metrics(self, mode="Test"):
        """Get metrics

        :param mode: Mode of evaluation (Val, Test)
        :return: Metrics dict
        """
        if mode == 'Test':
            data_loader = self.test_loader
        else:
            data_loader = self.val_loader
        preds = []
        self.emb_model.eval()
        self.model.eval()
        for i, (data, target, indices) in enumerate(data_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            # get model artificial_expert_labels
            with torch.no_grad():
                if self.args['dataset'] == 'nih':
                    emb = self.emb_model(data, return_features=True)
                else:
                    emb = self.emb_model(data)
                output = self.model(emb)

            # get predicted classes from model output
            if self.args['binary']:
                m = nn.Sigmoid()
                output = m(output)
                predicted_class = torch.round(output).cpu().numpy()
            else:
                predicted_class = torch.argmax(output, dim=1).cpu().numpy()

            for j in range(len(target)):
                preds.append(predicted_class[j])

        # calculate metrics
        if mode == 'Test':
            data = self.test_data
            gt_data = self.test_gt_data
        else:
            data = self.val_data
            gt_data = self.val_gt_data
        if self.args['binary']:
            # calculate accuracy, f1-score and auc-score for binary artificial expert annotations
            acc = accuracy_score(data.targets, preds)
            f1 = f1_score(data.targets, preds)
            auc = roc_auc_score(data.targets, preds)
            print(f'Accruacy: {acc}, F1 Score: {f1}, AUC: {auc}')
            return {'acc': acc, 'f1': f1, 'auc': auc}
        else:
            # calculate accuracy and accuracy by strength for multiclass artificial expert annotations
            accuracies = expert_accuracy_score(gt_data, data.targets, preds)
            print(f'Accuracy: {accuracies["Ex_Acc"]}')
            print(f'Strength Acc: {accuracies["Streng_Acc"]}')
            print(f'Weakness Acc: {accuracies["Weak_Acc"]}')
            return {'acc': accuracies["Ex_Acc"], 'sacc': accuracies["Streng_Acc"],
                    'wacc': accuracies["Weak_Acc"]}

    def get_emb_net_dir(self, wkdir):
        """Get training directory of the embedding net

        :param wkdir: Working directory
        :return: Training directory of the embedding net
        """
        args_base = {'dataset': self.args['dataset'],
                     'model': self.args['emb_model'],
                     'num_classes': self.args['num_classes']}


        base_cnn_dir = get_train_dir(wkdir, args_base, 'emb_net')
        return base_cnn_dir

    def load_emb_net_from_checkpoint(self, emb_model, wkdir, mode='best', strict=True):
        """Load embedding model weights from checkpoint

        :param emb_model: Initialized embedding model
        :param wkdir: Working directory
        :param mode: Checkpoint to load (best or latest)
        :param strict: Strict parameter for loading the model state dict
        :return: embedding model
        """
        # get checkpoint
        cp_dir = self.get_emb_net_dir(wkdir) + 'checkpoints/checkpoint.' + mode
        try:
            # load state dict from checkpoint
            checkpoint = torch.load(cp_dir)
            try:
                emb_model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            except KeyError:
                emb_model.load_state_dict(checkpoint, strict=strict)
            print('Found base net checkpoint at', cp_dir)
        except FileNotFoundError:
            print('No base net Checkpoint found at', cp_dir)
            sys.exit()

        # freeze base model layers
        for param in emb_model.parameters():
            param.requires_grad = False

        return emb_model

    def predict(self):
        """Generate artificial expert labels

        :return: Artificial expert labels
        """
        predictions = {'train': [], 'test': []}
        # load data
        train_data, test_data, train_gt_targets, test_gt_targets = \
            prep.get_train_val_test_data(dataset=self.args['dataset'], expert=self.expert, model=self.args['emb_model'],
                                         valid=False, gt_targets=True, binary=self.args['binary'])
        train_loader, test_loader, _ = prep.get_data_loader(train_data, test_data,
                                                            batch_size=self.args['batch'], shuffle_train=False)
        # generate artificial_expert_labels
        self.model.eval()
        loader = {'train': train_loader, 'test': test_loader}
        for mode in ['train', 'test']:
            print(f'predict {mode} data')
            for i, (data, target, batch_indices) in enumerate(loader[mode]):
                data = data.to(self.device)
                target = target.to(self.device)
                # get model artificial_expert_labels
                with torch.no_grad():
                    if self.args['dataset'] == 'nih':
                        emb = self.emb_model(data, return_features=True)
                    else:
                        emb = self.emb_model(data)
                    output = self.model(emb)
                # get predicted classes from model output
                if self.args['binary']:
                    m = nn.Sigmoid()
                    output = m(output)
                    predicted_class = torch.round(output).cpu().numpy()
                else:
                    predicted_class = torch.argmax(output, dim=1).cpu().numpy()
                for j in range(len(target)):
                    predictions[mode].append(int(predicted_class[j]))
        if self.args['dataset'] == 'cifar':
            # replace artificial expert labels with true expert labels for the labeled dataset
            for ii, (data, target, index) in enumerate(self.train_loader):
                for j in range(len(index)):
                    predictions['train'][index[j]] = int(target[j].cpu().numpy())
            # print accuracy check
            print('train check:', accuracy_score(train_data.targets, predictions['train']))
            print('test check:', accuracy_score(test_data.targets, predictions['test']))
        elif self.args['dataset'] == 'nih':
            pred_train = predictions['train']
            pred_test = predictions['test']
            predictions = {}
            for i, id in enumerate(train_data.image_ids):
                predictions[id] = pred_train[i]
            for i, id in enumerate(test_data.image_ids):
                predictions[id] = pred_test[i]
            # replace artificial expert labels with true expert labels for the labeled dataset
            for ii, (data, target, index) in enumerate(self.train_loader):
                for j in range(len(index)):
                    predictions[index[j]] = int(target[j].cpu().numpy())
            # print accuracy check
            print('train check:', accuracy_score(train_data.targets, pred_train))
            print('test check:', accuracy_score(test_data.targets, pred_test))
        return predictions


class LinearNN(nn.Module):
    """Class representing the linear expert model

    :param num_classes: Number of classes
    :param feature_dim: Dimension of feature vectors
    :param size: Size of the linear model

    :ivar self.linear_layers: Linear model
    """
    def __init__(self, num_classes, feature_dim=1280, size='s'):
        super().__init__()
        if size == 's':
            self.linear_layers = nn.Sequential(nn.Linear(feature_dim, num_classes))
        elif size == 'm':
            self.linear_layers = nn.Sequential(nn.Linear(feature_dim, 100),
                                               nn.ReLU(),
                                               nn.Linear(100, num_classes))
        elif size == 'l':
            self.linear_layers = nn.Sequential(nn.Linear(feature_dim, 300),
                                               nn.ReLU(),
                                               nn.Linear(300, 100),
                                               nn.ReLU(),
                                               nn.Linear(100, num_classes))
        else:
            print('Size of LinearNN not correctly specified')
            sys.exit()

    def forward(self, x):
        x = self.linear_layers(x)
        return x
