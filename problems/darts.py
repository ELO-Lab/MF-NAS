from problems import Problem
from search_spaces import SS_DARTS
from search_spaces.darts.utils import data_transforms_cifar10, AverageMeter, accuracy
import torch.nn as nn
import shutil

import numpy as np
import os
import time
import tensorwatch as tw
import torchvision.datasets as dset
import torch
from torch.autograd import Variable

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')
ROOT_DIR = '/'.join(ROOT_DIR[:-1])

drop_path_prob = 0.3
epochs = 600
learning_rate = 0.025
momentum = 0.9
weight_decay = 3e-4
batch_size = 64
report_freq = 50

criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()

def get_model_stats(model, input_tensor_shape, clone_model=True) -> tw.ModelStats:
    model_stats = tw.ModelStats(model, input_tensor_shape,
                                clone_model=clone_model)
    return model_stats

class DARTS(Problem):
    def __init__(self, max_eval, max_time, dataset, **kwargs):
        super().__init__(SS_DARTS(), max_eval, max_time)
        self.dataset = dataset
        self.save_path = kwargs['save_path']
        train_transform, valid_transform = data_transforms_cifar10(cutout=True, cutout_length=16)

        train_data = dset.CIFAR10(root='./datasets/cifar10', train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root='./datasets/cifar10', train=False, download=True, transform=valid_transform)

        self.train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

        self.valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=batch_size // 2, shuffle=False, pin_memory=True, num_workers=2)

    def evaluate(self, network, **kwargs):
        using_zc_metric = bool(kwargs['using_zc_metric'])

        network.model = self.search_space.get_model(network.genotype)
        if using_zc_metric:
            metric = kwargs['metric']
            cost_time = self.zc_evaluate(network, metric=metric)
        else:
            metric = '_'.join(kwargs['metric'].split('_')[:-1])
            iepoch = int(kwargs['metric'].split('_')[-1])
            cost_time = self.train(network, metric=metric, iepoch=iepoch)
        return cost_time

    @staticmethod
    def zc_evaluate(network, **kwargs):
        # TODO: Re-write
        metric = kwargs['metric']

        if metric in ['flops', 'params']:
            s = time.time()
            model_stats = get_model_stats(network.model, input_tensor_shape=(1, 3, 32, 32), clone_model=True)
            flops = float(model_stats.Flops) / 1e6  # mega flops
            params = float(model_stats.parameters) / 1e6  # mega params
            cost_time = time.time() - s
            if metric == 'flops':
                score = flops
            else:
                score = params
        else:
            raise ValueError()
        network.score = score
        return cost_time

    def train(self, network, **kwargs):
        # TODO: Re-write
        metric = kwargs['metric']
        iepoch = kwargs['iepoch']

        optimizer = torch.optim.SGD(
            network.model.parameters(),
            learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        network_id = ''.join(list(map(str, network.genotype)))

        if network.info['cur_iepoch'][-1] != 0:
            checkpoint = torch.load(f'{self.save_path}/{network_id}.checkpoints.pth.tar')
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            network.load_state_dict(checkpoint['model_state_dict'])

        score = -np.inf
        s = time.time()
        for epoch in range(network.info['cur_iepoch'][-1], iepoch+1):
            scheduler.step()
            print('epoch %d lr %e', epoch, scheduler.get_lr()[0])
            network.model.drop_path_prob = drop_path_prob * epoch / epochs

            train_acc, train_objs = _train(self.train_queue, network.model, criterion, optimizer)
            print('train_acc %f', train_acc)
            valid_acc, valid_objs = infer(self.valid_queue, network.model, criterion)
            is_best = valid_acc > score
            if valid_acc > score:
                score = valid_acc
            print('valid_acc %f, best_acc %f', valid_acc, score)

            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': network.model.state_dict(),
                'best_acc': score,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, filename=f'{self.save_path}/{network_id}_epoch{epoch}')
        cost_time = time.time() - s
        if 'loss' in metric:
            score *= -1
        network.score = score

        network.info['cur_iepoch'].append(iepoch)
        network.info['train_time'].append(cost_time)
        return cost_time

    def get_test_performance(self, network, **kwargs):
        return network.score

def _train(train_queue, model, criterion, optimizer):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()

    auxiliary = True
    auxiliary_weight = 0.4
    grad_clip = 5

    for step, (inputs, targets) in enumerate(train_queue):
        inputs = Variable(inputs).cuda()
        targets = Variable(targets).cuda()

        optimizer.zero_grad()
        logits, logits_aux = model(inputs)
        loss = criterion(logits, targets)

        if auxiliary:
            loss_aux = criterion(logits_aux, targets)
            loss += auxiliary_weight * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()

        prec1, prec5 = accuracy(logits, targets, topk=(1, 5))
        n = inputs.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % report_freq == 0:
            print('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

def infer(valid_queue, model, criterion):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    for step, (inputs, targets) in enumerate(valid_queue):
        inputs = Variable(inputs, volatile=True).cuda()
        targets = Variable(targets, volatile=True).cuda()

        logits, _ = model(inputs)
        loss = criterion(logits, targets)

        prec1, prec5 = accuracy(logits, targets, topk=(1, 5))
        n = inputs.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % report_freq == 0:
            print('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

def save_checkpoint(state, is_best, filename):
    if not os.path.isdir(filename):
        os.makedirs(filename)

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(os.path.join(filename, 'checkpoints.pth.tar'), os.path.join(filename + '_best_model.pth.tar'))
