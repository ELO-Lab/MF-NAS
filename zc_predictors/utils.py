from __future__ import print_function

from collections import namedtuple

import sys
import logging
import argparse
import torchvision.datasets as dset
from torch.utils.data import Dataset
from sklearn import metrics
from scipy import stats

from collections import OrderedDict

import random
import os.path
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms

from fvcore.common.checkpoint import Checkpointer as fvCheckpointer
from fvcore.common.config import CfgNode

from typing import List, Optional
from fvcore.common.file_io import PathManager
import os

cat_channels = partial(torch.cat, dim=1)

logger = logging.getLogger(__name__)

Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")

def get_train_val_loaders(config, mode):
    """
    Constructs the dataloaders and transforms for training, validation and test data.
    """
    root_data = config['root_data']
    dataset = config['dataset']
    seed = config['search']['seed']
    config = config['search'] if mode == 'train' else config['evaluation']
    if dataset == 'CIFAR-10':
        train_transform, valid_transform = _data_transforms_cifar10(config)
        train_data = dset.CIFAR10(
            root=root_data, train=True, download=True, transform=train_transform
        )
        test_data = dset.CIFAR10(
            root=root_data, train=False, download=True, transform=valid_transform
        )
    elif dataset == 'CIFAR-100':
        train_transform, valid_transform = _data_transforms_cifar100(config)
        train_data = dset.CIFAR100(
            root=root_data, train=True, download=True, transform=train_transform
        )
        test_data = dset.CIFAR100(
            root=root_data, train=False, download=True, transform=valid_transform
        )
    elif dataset == 'svhn':
        train_transform, valid_transform = _data_transforms_svhn(config)
        train_data = dset.SVHN(
            root=root_data, split="train", download=True, transform=train_transform
        )
        test_data = dset.SVHN(
            root=root_data, split="test", download=True, transform=valid_transform
        )
    elif dataset == 'ImageNet16-120':
        from zc_predictor.DownsampledImageNet import ImageNet16

        train_transform, valid_transform = _data_transforms_ImageNet_16_120(config)
        data_folder = root_data + "/ImageNet16-120"
        train_data = ImageNet16(
            root=data_folder,
            train=True,
            transform=train_transform,
            use_num_of_class_only=120,
        )
        test_data = ImageNet16(
            root=data_folder,
            train=False,
            transform=valid_transform,
            use_num_of_class_only=120,
        )
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(config['train_portion'] * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=config['batch_size'],
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True,
        num_workers=0,
        worker_init_fn=np.random.seed(seed),
    )

    valid_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=config['batch_size'],
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True,
        num_workers=0,
        worker_init_fn=np.random.seed(seed),
    )

    test_queue = torch.utils.data.DataLoader(
        test_data,
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        worker_init_fn=np.random.seed(seed),
    )

    return train_queue, valid_queue, test_queue, train_transform, valid_transform


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    if args['cutout']:
        train_transform.transforms.append(Cutout(args['cutout_length'], args['cutout_prob']))
    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return train_transform, valid_transform


def _data_transforms_svhn(args):
    SVHN_MEAN = [0.4377, 0.4438, 0.4728]
    SVHN_STD = [0.1980, 0.2010, 0.1970]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ]
    )
    if args['cutout']:
        train_transform.transforms.append(Cutout(args['cutout_length'], args['cutout_prob']))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ]
    )
    return train_transform, valid_transform


def _data_transforms_cifar100(args):
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    if args['cutout']:
        train_transform.transforms.append(Cutout(args['cutout_length'], args['cutout_prob']))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return train_transform, valid_transform


def _data_transforms_ImageNet_16_120(args):
    IMAGENET16_MEAN = [x / 255 for x in [122.68, 116.66, 104.01]]
    IMAGENET16_STD = [x / 255 for x in [63.22, 61.26, 65.09]]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(16, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET16_MEAN, IMAGENET16_STD),
        ]
    )
    if args['cutout']:
        train_transform.transforms.append(Cutout(args['cutout_length'], args['cutout_prob']))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET16_MEAN, IMAGENET16_STD),
        ]
    )
    return train_transform, valid_transform


class TensorDatasetWithTrans(Dataset):
    """
    TensorDataset with support of transforms.
    """

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def set_seed(seed):
    """
    Set the seeds for all used libraries.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)


def get_last_checkpoint(config, search=True):
    """
    Finds the latest checkpoint in the experiment directory.

    Args:
        config (AttrDict): The config from config file.
        search (bool): Search or evaluation checkpoint

    Returns:
        (str): The path to the latest checkpoint file.
    """
    try:
        path = os.path.join(
            config.save, "search" if search else "eval", "last_checkpoint"
        )
        with open(path, "r") as f:
            checkpoint_name = f.readline()
        return os.path.join(
            config.save, "search" if search else "eval", checkpoint_name
        )
    except:
        return ""


def accuracy(output, target, topk=(1,)):
    """
    Calculate the accuracy given the softmax output and the target.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def count_parameters_in_MB(model):
    """
    Returns the model parameters in mega byte.
    """
    return (
        np.sum(
            np.prod(v.size())
            for name, v in model.named_parameters()
            if "auxiliary" not in name
        )
        / 1e6
    )


def log_args(args):
    """
    Log the args in a nice way.
    """
    for arg, val in args.items():
        logger.info(arg + "." * (50 - len(arg) - len(str(val))) + str(val))


def create_exp_dir(path):
    """
    Create the experiment directories.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    logger.info("Experiment dir : {}".format(path))


def cross_validation(
    xtrain, ytrain, predictor, split_indices, score_metric="kendalltau"
):
    validation_score = []

    for train_indices, validation_indices in split_indices:
        xtrain_i = [xtrain[j] for j in train_indices]
        ytrain_i = [ytrain[j] for j in train_indices]
        xval_i = [xtrain[j] for j in train_indices]
        yval_i = [ytrain[j] for j in train_indices]

        predictor.fit(xtrain_i, ytrain_i)
        ypred_i = predictor.query(xval_i)

        # If the predictor is an ensemble, take the mean
        if len(ypred_i.shape) > 1:
            ypred_i = np.mean(ypred_i, axis=0)

        # use Pearson correlation to be the metric -> maximise Pearson correlation
        if score_metric == "pearson":
            score_i = np.abs(np.corrcoef(yval_i, ypred_i)[1, 0])
        elif score_metric == "mae":
            score_i = np.mean(abs(ypred_i - yval_i))
        elif score_metric == "rmse":
            score_i = metrics.mean_squared_error(yval_i, ypred_i, squared=False)
        elif score_metric == "spearman":
            score_i = stats.spearmanr(yval_i, ypred_i)[0]
        elif score_metric == "kendalltau":
            score_i = stats.kendalltau(yval_i, ypred_i)[0]
        elif score_metric == "kt_2dec":
            score_i = stats.kendalltau(yval_i, np.round(ypred_i, decimals=2))[0]
        elif score_metric == "kt_1dec":
            score_i = stats.kendalltau(yval_i, np.round(ypred_i, decimals=1))[0]

        validation_score.append(score_i)

    return np.mean(validation_score)


def generate_kfold(n, k):
    """
    Input:
        n: number of training examples
        k: number of folds
    Returns:
        kfold_indices: a list of len k. Each entry takes the form
        (training indices, validation indices)
    """
    assert k >= 2
    kfold_indices = []

    indices = np.array(range(n))
    fold_size = n // k

    fold_indices = [indices[i * fold_size : (i + 1) * fold_size] for i in range(k - 1)]
    fold_indices.append(indices[(k - 1) * fold_size :])

    for i in range(k):
        training_indices = [fold_indices[j] for j in range(k) if j != i]
        validation_indices = fold_indices[i]
        kfold_indices.append((np.concatenate(training_indices), validation_indices))

    return kfold_indices


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class AverageMeterGroup:
    """Average meter group for multiple average meters, ported from Naszilla repo."""

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, data, n=1):
        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = NamedAverageMeter(k, ":4f")
            self.meters[k].update(v, n=n)

    def __getattr__(self, item):
        return self.meters[item]

    def __getitem__(self, item):
        return self.meters[item]

    def __str__(self):
        return "  ".join(str(v) for v in self.meters.values())

    def summary(self):
        return "  ".join(v.summary() for v in self.meters.values())


class NamedAverageMeter:
    """Computes and stores the average and current value, ported from naszilla repo"""

    def __init__(self, name, fmt=":f"):
        """
        Initialization of AverageMeter
        Parameters
        ----------
        name : str
            Name to display.
        fmt : str
            Format string to print the values.
        """
        self.name = name
        self.fmt = fmt
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
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = "{name}: {avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class Cutout(object):
    def __init__(self, length, prob=1.0):
        self.length = length
        self.prob = prob

    def __call__(self, img):
        if np.random.binomial(1, self.prob):
            h, w = img.size(1), img.size(2)
            mask = np.ones((h, w), np.float32)
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img *= mask
        return img

def get_config_for_zc_predictor(ss, dataset, database_path, seed):
    all_configs = {
        'darts':
        {
            'search_space': ss,
            'dataset': dataset,
            'root_data': database_path + '/dataset',
            'search':
            {
                'batch_size': 64,
                'cutout': False,
                'cutout_length': 16,
                'cutout_prob': 1.0,
                'train_portion': 0.5,
                'seed': seed
            }
        }
    }
    return all_configs[ss]

def get_zc_predictor(config, method_type):
    from zc_predictors.core.zerocost import ZeroCost
    zc_predictor = ZeroCost(config, method_type=method_type)
    # zc_predictor.pre_process()
    return zc_predictor