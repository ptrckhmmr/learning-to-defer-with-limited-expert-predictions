import sys
import torch
import timm
import numpy as np
import warnings

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from torch.utils.data.dataloader import DataLoader

import scripts.data_loading as prep

from scripts.utils import printProgressBar, get_train_dir
from scripts.metrics import expert_accuracy_score
from scripts.emb_model_lib import EmbeddingModel, Resnet
from scripts.Expert import CIFAR100Expert, NIHExpert
warnings.filterwarnings("ignore", category=UserWarning)


class SVMExpertModel(EmbeddingModel):
    """Class representing the SVM expert model

    :param args: Training arguments for the embedding model
    :param wkdir: Working directory

    :ivar global_step: Global step
    :ivar args: Training arguments
    :ivar device: Active device
    :ivar emb_model: Embedding model
    :ivar svc: SVM model
    :ivar expert: Expert
    :ivar train_data: Train dataset
    :ivar test_data: Test dataset
    :ivar val_data: Validation dataset
    :ivar train_gt_data: Ground-truth label for the train dataset
    :ivar test_gt_data: Ground-truth label for the test dataset
    :ivar val_gt_data: Ground-truth label for the validation dataset
    """
    def __init__(self, args, wkdir):
        self.global_step = 0
        self.args = args
        self.device = prep.get_device()
        self.emb_model = self.get_emb_model(wkdir)
        self.svc = None
        if self.args['n_strengths'] is not None or self.args['dataset'] == 'cifar10h':
            if self.args['dataset'] == 'nih':
                self.expert = NIHExpert(id=4295342357, n_classes=args['num_classes'], target='Airspace_Opacity')
            else:
                self.expert = CIFAR100Expert(self.args['num_classes'], self.args['n_strengths'], 1, 0, seed=args['ex_seed'])
        else:
            self.expert = None
        if 'cifar' in self.args['dataset']:
            self.train_data, self.test_data, self.val_data, self.train_gt_data, self.test_gt_data, self.val_gt_data = \
                prep.get_train_val_test_data(dataset=args['dataset'], expert=self.expert, binary=args['binary'], 
                                             gt_targets=True,model=args['emb_model'], L=args['labels'], seed=args['seed'])
        elif self.args['dataset'] == 'nih':
            self.train_data, self.test_data, self.train_gt_data, self.test_gt_data = \
                prep.get_train_val_test_data(dataset=args['dataset'], expert=self.expert, binary=args['binary'], 
                                             gt_targets=True,model=args['emb_model'], L=args['labels'], 
                                             seed=args['seed'], valid=False)

        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

    def get_emb_model(self, wkdir):
        """
        Initialize base model

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

    def get_features(self, data_points):
        """Get features of images as embedding

        :param data_points: Images
        :return: feature vectors
        """
        # get data loader of images
        data_loader = DataLoader(data_points, self.args['batch'], num_workers=0, pin_memory=False)
        # get active device
        device = prep.get_device()
        if self.args['dataset'] == 'cifar100':
            data_loader = prep.ToDeviceLoader(data_loader, device)
        # get feature vectors from base model
        feature_vecs = []
        self.emb_model.eval()
        for ii, (data, target, index) in enumerate(data_loader):
            data = data.to(device)
            with torch.no_grad():
                if self.args['dataset'] == 'nih':
                    batch_features = self.emb_model(data, return_features=True)
                else:
                    batch_features = self.emb_model(data)
            for feature in batch_features:
                feature_vecs.append(feature.cpu().numpy())
            printProgressBar(ii, int(len(data_points.targets)/self.args['batch']))
        return np.array(feature_vecs)

    def do_gridsearch_svc(self, parameters):
        """Perform gridsearch for svc

        :param parameters: Parameter grid for gridsearch
        :return: Validation metrics
        """
        # get features for expert and validation data
        print('Get features from embedding')
        ex_features = self.get_features(self.train_data)
        try:
            test_features = np.load(f'data/test_feature_vecs_{self.args["emb_model"]}_{self.args["dataset"]}.npy')
        except FileNotFoundError:
            test_features = self.get_features(self.test_data)
            np.save(f'data/test_feature_vecs_{self.args["emb_model"]}_{self.args["dataset"]}.npy', test_features)
        # perform gridsearch
        gs = GridSearchCV(SVC(), parameters, scoring=roc_auc_score)
        gs.fit(ex_features, self.train_data.targets)
        # get artificial_expert_labels for the test data
        preds = gs.predict(test_features)
        # return metrics
        return self.get_gs_metrics(gs, preds)

    def train_svc(self, kernel='rbf', C=5, class_weight=None):
        """Train support vector classifier

        :param kernel: Kernel (default: rbf)
        :param C: Regularization parameter (default: 5)
        :param class_weight: Class weighting method (default: None)
        """
        # get feature vectors for expert data
        print('Get features from embedding')
        ex_features = self.get_features(self.train_data)
        # initiate and train svc model
        self.svc = SVC(kernel=kernel, C=C, class_weight=class_weight)
        self.svc.fit(ex_features, self.train_data.targets)

    def get_metrics(self, mode="Test"):
        """Get metrics

        :param mode: Mode of evaluation (Val, Test)
        :return: Metrics dict
        """
        # get feature vectors for eval data
        if mode=='Test':
            try:
                eval_features = np.load(f'data/test_feature_vecs_{self.args["emb_model"]}_{self.args["dataset"]}.npy')
            except FileNotFoundError:
                eval_features = self.get_features(self.test_data)
                np.save(f'data/test_feature_vecs_{self.args["emb_model"]}_{self.args["dataset"]}.npy', eval_features)
        else:
            try:
                eval_features = np.load(f'data/val_feature_vecs_split{self.args["split_seed"]}_{self.args["emb_model"]}_'
                                        f'{self.args["dataset"]}.npy')
            except FileNotFoundError:
                eval_features = self.get_features(self.val_data)
                np.save(f'data/val_feature_vecs_split{self.args["split_seed"]}_{self.args["emb_model"]}_'
                        f'{self.args["dataset"]}.npy', eval_features)
        # predict eval data
        preds = self.svc.predict(eval_features)
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

    def get_gs_metrics(self, gs, preds):
        """Get metrics for gridsearch evaluation

        :param gs: Gridsearch model
        :param preds: Predictions for the validation set
        :return: Metrics dictionary
        """
        if self.args['binary']:
            # calculate accuracy, f1-score and auc-score for binary artificial expert annotations
            acc = accuracy_score(self.test_data.targets, preds)
            f1 = f1_score(self.test_data.targets, preds)
            auc = roc_auc_score(self.test_data.targets, preds)
            print(f'Test Accuracy: {acc}')
            print(f'Test F1-Score: {f1}')
            print(f'Test AUC: {auc}')
            return {'best_params': gs.best_params_, 'acc': acc, 'f1': f1, 'auc': auc}

        else:
            # calculate accuracy and accuracy by strength for multiclass artificial expert annotations
            acc = accuracy_score(self.test_data.targets, preds)
            accuracies = expert_accuracy_score(self.test_gt_data, self.test_data.targets, preds)
            print(f'Accuracy: {accuracies["Ex_Acc"]}')
            print(f'Strength Acc: {accuracies["Streng_Acc"]}')
            print(f'Weakness Acc: {accuracies["Weak_Acc"]}')
            return {'best_params': gs.best_params_, 'acc': acc, 'sacc': accuracies["seak_Acc"], 'wacc': accuracies["Weak_Acc"]}

    def get_emb_net_dir(self, wkdir):
        """Get training directory of the embedding net

        :param wkdir: Working directory
        :return: Training directory of the embedding net
        """
        args_base = {'dataset': self.args['dataset'],
                     'model': self.args['emb_model'],
                     'num_classes': self.args['num_classes']}

        emb_cnn_dir = get_train_dir(wkdir, args_base, 'emb_net')
        return emb_cnn_dir

    def load_emb_net_from_checkpoint(self, emb_model, wkdir, mode='best', strict=True):
        """Load base model weights from checkpoint

        :param emb_model: Initialized base model
        :param wkdir: Working directory
        :param mode: Checkpoint to load (best or latest)
        :param strict: Strict parameter for loading the model state dict
        :return: base model
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

        # freeze emb model layers
        for param in emb_model.parameters():
            param.requires_grad = False

        return emb_model

    def predict(self):
        """Generate artificial expert labels

        :return: Artificial expert labels
        """
        predictions = {'train': [], 'test': []}
        # load train and test data
        train_data, test_data, train_gt_targets, test_gt_targets = \
            prep.get_train_val_test_data(dataset=self.args['dataset'], expert=self.expert, model=self.args['emb_model'],
                                         valid=False, gt_targets=True, binary=True)
        # get feature vectors for the train and test data
        try:
            train_features = np.load(f'data/train_feature_vecs_{self.args["emb_model"]}_{self.args["dataset"]}.npy')
            test_features = np.load(f'data/test_feature_vecs_{self.args["emb_model"]}_{self.args["dataset"]}.npy')
        except FileNotFoundError:
            train_features = self.get_features(train_data)
            test_features = self.get_features(test_data)
            np.save(f'data/train_feature_vecs_{self.args["emb_model"]}_{self.args["dataset"]}.npy', train_features)
            np.save(f'data/test_feature_vecs_{self.args["emb_model"]}_{self.args["dataset"]}.npy', test_features)

        if self.args['dataset'] == 'cifar100':
            # generate artificial_expert_labels for the train and test data
            predictions['train'] = self.svc.predict(train_features).tolist()
            predictions['test'] = self.svc.predict(test_features).tolist()
            # print check
            print('bin train check:', accuracy_score(train_data.targets, predictions['train']))
            print('bin test check:', accuracy_score(test_data.targets, predictions['test']))
            # replace artificial expert labels with true expert labels for the labeled dataset
            data_loader = DataLoader(self.train_data, self.args['batch'], num_workers=0, pin_memory=False)
            for ii, (data, target, index) in enumerate(data_loader):
                for j in range(len(index)):
                    predictions['train'][index[j]] = int(target[j].cpu().numpy())
        elif self.args['dataset'] == 'nih':
            # generate artificial expert labels for the train and test set
            pred_train = self.svc.predict(train_features).tolist()
            pred_test = self.svc.predict(test_features).tolist()
            predictions = {}
            for i, id in enumerate(train_data.image_ids):
                predictions[id] = pred_train[i]
            for i, id in enumerate(test_data.image_ids):
                predictions[id] = pred_test[i]
            # replace artificial expert labels with true expert labels for the labeled dataset
            data_loader = DataLoader(self.train_data, self.args['batch'], num_workers=0, pin_memory=False)
            for ii, (data, target, index) in enumerate(data_loader):
                for j in range(len(index)):
                    predictions[index[j]] = int(target[j].cpu().numpy())
            print('train check:', accuracy_score(train_data.targets, pred_train))
            print('test check:', accuracy_score(test_data.targets, pred_test))
        return predictions