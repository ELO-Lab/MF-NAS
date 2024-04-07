import argparse
import logging
import sys
import os
import json

import torch.nn as nn

from torch.autograd import Variable
from search_spaces.darts.utils import data_transforms_cifar10, AverageMeter, accuracy

import torch
import torchvision.datasets as dataset
from search_spaces import SS_DARTS
import numpy as np
import gc
from utils import set_seed
import yaml
from tqdm import tqdm

with open('configs/problem.yaml', 'r') as file:
    all_configs = yaml.safe_load(file)
configs = all_configs['darts']

drop_path_prob = configs['drop_path_prob']
max_epochs = configs['max_epochs']
learning_rate = configs['learning_rate']
momentum = configs['momentum']
weight_decay = configs['weight_decay']
batch_size = configs['batch_size']
train_portion = configs['train_portion']

auxiliary = configs['auxiliary']
auxiliary_weight = configs['auxiliary_weight']
grad_clip = configs['grad_clip']

SEARCH_SPACE = SS_DARTS()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run(kwargs):
    set_seed(kwargs.seed)
    if kwargs.dataset == 'cifar10':
        train_transform, valid_transform = data_transforms_cifar10(cutout=True, cutout_length=16)
        train_data = dataset.CIFAR10(root='./datasets/cifar10', train=True, download=True, transform=train_transform)
        # valid_data = dataset.CIFAR10(root='./datasets/cifar10', train=False, download=True, transform=test_transform)
    else:
        raise NotImplementedError

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(train_portion * num_train))

    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    # valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                                               pin_memory=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
                                               pin_memory=True, num_workers=2)

    network_id = kwargs.network_id
    genotype = list(map(int, list(network_id)))
    model = SEARCH_SPACE.get_model(genotype)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs)

    start_iepoch = kwargs.start_iepoch
    end_iepoch = kwargs.end_iepoch

    save_path = kwargs.save_path
    best_score = -np.inf
    logging.info(f'- Network: {genotype}')
    if start_iepoch != 0:
        checkpoint = torch.load(f'{save_path}/{network_id}/last_checkpoint.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to('cuda')
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_score = checkpoint['best_score']
        logging.info('  + Load weighted - Done!')

    criterion = nn.CrossEntropyLoss()

    for iepoch in range(start_iepoch + 1, end_iepoch + 1):
        model.train()
        objs = AverageMeter()
        top1 = AverageMeter()

        model.drop_path_prob = drop_path_prob * iepoch / max_epochs

        for step, (inputs, targets) in enumerate(tqdm(train_loader)):
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

            prec1 = accuracy(logits, targets, topk=(1,))[0]
            n = inputs.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)

        train_acc, train_objs = top1.avg, objs.avg

        objs = AverageMeter()
        top1 = AverageMeter()
        model.eval()

        for step, (inputs, targets) in enumerate(tqdm(valid_loader)):
            with torch.no_grad():
                inputs = Variable(inputs).cuda()
                targets = Variable(targets).cuda()

                logits, _ = model(inputs)
                loss = criterion(logits, targets)

                prec1 = accuracy(logits, targets, topk=(1,))[0]
                n = inputs.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)

        valid_acc, valid_objs = top1.avg, objs.avg

        is_best = valid_acc > best_score
        best_score = max(valid_acc, best_score)
        scheduler.step()

        logging.info(
            f'  + Epoch: {iepoch}  -  LR: {round(scheduler.get_last_lr()[0], 6)}  -  Train Acc: {round(train_acc, 2)}  -  Valid Acc: {round(valid_acc, 2)}  -  Best: {round(best_score, 2)}')
        if not os.path.isdir(f'{save_path}/{network_id}'):
            os.makedirs(f'{save_path}/{network_id}')
        state = {
            'epoch': iepoch,
            'best_score': best_score,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        if is_best:
            torch.save(state, f'{save_path}/{network_id}/best_model.pth.tar')
        if iepoch == end_iepoch:
            torch.save(state, f'{save_path}/{network_id}/last_checkpoint.pth.tar')
    train_status = {'score': best_score}
    fp = open(f'{save_path}/{network_id}/status.json', 'w')
    json.dump(train_status, fp, indent=4)
    fp.close()

    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--start_iepoch', type=int)
    parser.add_argument('--end_iepoch', type=int)
    parser.add_argument('--network_id', type=str)

    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

    run(args)
