from problems import Problem
from search_spaces import SS_DARTS
from search_spaces.darts.utils import data_transforms_cifar10, AverageMeter, accuracy
import torch.nn as nn

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
            raise NotImplementedError()
        network.score = score
        return cost_time

    def train(self, network, **kwargs):
        metric = kwargs['metric']
        iepoch = kwargs['iepoch']
        network.model.to('cuda')

        optimizer = torch.optim.SGD(
            network.model.parameters(),
            learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        network_id = ''.join(list(map(str, network.genotype)))
        print('- Network:', self.search_space.decode(network.genotype))
        score = -np.inf
        if network.info['cur_iepoch'][-1] != 0:
            checkpoint = torch.load(f'{self.save_path}/{network_id}/checkpoint.pth.tar')
            best_checkpoint = torch.load(f'{self.save_path}/{network_id}/best_model.pth.tar')
            network.model.load_state_dict(checkpoint['model_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to('cuda')
            optimizer.load_state_dict(checkpoint['optimizer'])
            score = best_checkpoint['score']
            print('  + Load weighted - Done!')
        s = time.time()
        for epoch in range(network.info['cur_iepoch'][-1], iepoch+1):
            network.model.drop_path_prob = drop_path_prob * epoch / epochs
            train_acc, train_objs = _train(self.train_queue, network.model, criterion, optimizer)

            valid_acc, valid_objs = infer(self.valid_queue, network.model, criterion)
            is_best = valid_acc > score
            if valid_acc > score:
                score = valid_acc

            print(f'  + Epoch: {epoch}  -  LR: {round(scheduler.get_last_lr()[0], 6)}  -  Train Acc: {round(train_acc, 2)}  -  Valid Acc: {round(valid_acc, 2)}  -  Best: {round(score, 2)}')
            if not os.path.isdir(f'{self.save_path}/{network_id}'):
                os.makedirs(f'{self.save_path}/{network_id}')
            scheduler.step()
            state = {
                'epoch': epoch,
                'score': score,
                'model_state_dict': network.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            if is_best:
                torch.save(state, f'{self.save_path}/{network_id}/best_model.pth.tar')
            if epoch == iepoch:
                torch.save(state, f'{self.save_path}/{network_id}/checkpoint.pth.tar')
        cost_time = time.time() - s
        if 'loss' in metric:
            score *= -1
        network.score = score

        network.info['cur_iepoch'].append(iepoch)
        network.info['train_time'].append(cost_time)
        print('-'*100)
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
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        prec1, prec5 = accuracy(logits, targets, topk=(1, 5))
        n = inputs.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

    return top1.avg, objs.avg

def infer(valid_queue, model, criterion):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    for step, (inputs, targets) in enumerate(valid_queue):
        with torch.no_grad():
            inputs = Variable(inputs).cuda()
            targets = Variable(targets).cuda()

            logits, _ = model(inputs)
            loss = criterion(logits, targets)

            prec1, prec5 = accuracy(logits, targets, topk=(1, 5))
            n = inputs.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    return top1.avg, objs.avg
