from datetime import datetime
import logging
import os
import sys
import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


def setup_default_logging(args, default_level=logging.INFO,
                          format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s"):
    """Setup default logging

    :param args: Training arguments
    :param default_level: Default level
    :param format: Logging format
    :return: tuple
        - Logger
        - Output directory
    """
    
    if 'CIFAR' in args.dataset or 'NIH' in args.dataset:
        output_dir = os.path.join(args.dataset, args.exp_dir, f'ex{args.ex_strength}_x{args.n_labeled}_seed{args.seed}')
    else:
        output_dir = os.path.join(args.dataset, f'f{args.folds}', args.exp_dir)
        
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger('train')

    logging.basicConfig(  # unlike the root logger, a custom logger can’t be configured using basicConfig()
        filename=os.path.join(output_dir, f'experiment.log'),
        format=format,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=default_level)

    # print
    # file_handler = logging.FileHandler()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(default_level)
    console_handler.setFormatter(logging.Formatter(format))
    logger.addHandler(console_handler)

    return logger, output_dir


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k

    :param output: Model output
    :param target: Targets
    :param topk: k
    :return: Accuracy
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, largest=True, sorted=True)  # return value, indices
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    if len(res) < 2:
        res = res[0]
    return res


class AverageMeter(object):
    """
    Computes and stores the average and current value

    :ivar val: Value
    :ivar avg: Average
    :ivar sum: Sum
    :ivar count: Count
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        # self.avg = self.sum / (self.count + 1e-20)
        self.avg = self.sum / self.count


def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'

    #     time.strftime(format[, t])
    return datetime.today().strftime(fmt)


class WarmupCosineLrScheduler(_LRScheduler):
    """Class for the warmup cosine learning rate scheduler

    :param optimizer: Optimizer
    :param max_iter: Maximum iterations
    :param warmup_iter: Number of warmup iterations
    :param warmup_ratio: Warmup ratio
    :param warmup: Warmup mode
    :param last_epoch: Last epoch

    :ivar max_iter: Maximum iterations
    :ivar warmup_iter: Number of warmup iterations
    :ivar warmup_ratio: Warmup ratio
    :ivar warmup: Warmup mode
    """

    def __init__(
            self,
            optimizer,
            max_iter,
            warmup_iter,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupCosineLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get learning rate

        :return: Learning Rate
        """
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        """Get learning rate ratio

        :return: Ratio
        """
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            real_iter = self.last_epoch - self.warmup_iter
            real_max_iter = self.max_iter - self.warmup_iter
            ratio = np.cos((7 * np.pi * real_iter) / (16 * real_max_iter))            
            #ratio = 0.5 * (1. + np.cos(np.pi * real_iter / real_max_iter))
        return ratio

    def get_warmup_ratio(self):
        """Get warmup ratio

        :return: Ratio
        """
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio


def load_from_checkpoint(train_dir, model, ema_model, optimizer, scheduler, mode='comatch'):
    """
    Load from checkpoint

    :param train_dir: Training directory
    :param model: Model
    :param ema_model: EMA-Model
    :param optimizer: Optimizer
    :param scheduler: Learning rate scheduler
    :param mode: Mode
    :return: tuple
        - Model
        - EMA-Model
        - Optimizer
        - Scheduler
        - Epoch
        - Test metrics
        - List of probabilities
        - Queue (comatch only)
    """
    cp_dir = f'{train_dir}/ckp.latest'

    try:
        checkpoint = torch.load(cp_dir)
        model.load_state_dict(checkpoint['model'])
        ema_model.load_state_dict(checkpoint['ema_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        epoch = checkpoint['epoch'] + 1
        test_metrics = checkpoint['metrics']
        prob_list = checkpoint['prob_list']
        if mode == 'comatch':
            queue = checkpoint['queue']
        print('Found latest checkpoint at', cp_dir)
        print('Continuing in epoch', epoch + 1)
    except FileNotFoundError:
        epoch = 0
        test_metrics = None
        prob_list = []
        if mode == 'comatch':
            queue = None
        print(f'No Checkpoint found at {cp_dir}')
        print('Starting new from epoch', epoch + 1)

    if mode == 'comatch':
        return model, ema_model, optimizer, scheduler, epoch, test_metrics, prob_list, queue
    else:
        return model, ema_model, optimizer, scheduler, epoch, test_metrics, prob_list