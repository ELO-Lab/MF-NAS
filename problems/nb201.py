from problems import Problem
from search_spaces import SS_201
import json
import numpy as np
import pickle as p
import os
from pymoo.indicators.hv import HV
from problems.utils import get_ref_point

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')
ROOT_DIR = '/'.join(ROOT_DIR[:-1])

list_supported_zc_metrics = ['jacov', 'plain', 'grasp', 'fisher', 'epe_nas', 'zen', 'grad_norm', 'snip', 'l2_norm', 'synflow', 'nwot', 'flops', 'params']
list_supported_training_based_metrics = ['train_acc', 'val_acc', 'train_loss', 'val_loss']

OP_NAMES_NB201 = ['skip_connect', 'none', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3']
EDGE_LIST = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))

def convert_str_to_op_indices(str_encoding):
    """
    Converts NB201 string representation to op_indices
    """
    nodes = str_encoding.split('+')

    def get_op(x):
        return x.split('~')[0]

    node_ops = [list(map(get_op, n.strip()[1:-1].split('|'))) for n in nodes]

    enc = []
    for u, v in EDGE_LIST:
        enc.append(OP_NAMES_NB201.index(node_ops[v - 2][u - 1]))

    return tuple(enc)

class NB_201(Problem):
    def __init__(self, max_eval, max_time, dataset, **kwargs):
        super().__init__(SS_201(), max_eval, max_time)
        self.dataset = dataset
        self.zc_database = json.load(open(ROOT_DIR + f'/database/nb201/[NB201]_zc_data.json'))
        self.benchmark_database = p.load(open(ROOT_DIR + f'/database/nb201/[NB201][{self.dataset}]_data.p', 'rb'))
        self.list_pof = json.load(open(ROOT_DIR + f'/database/nb201/[NB201][{self.dataset}]_pof.json'))
        self.mo_objective = kwargs['mo_objective']

    def mo_evaluate(self, list_networks, list_metrics, **kwargs):
        max_time = kwargs['max_time']
        need_trained = kwargs['need_trained']
        cur_total_time = kwargs['cur_total_time']
        TOTAL_TIME, TOTAL_EPOCHS = 0.0, 0
        if not isinstance(list_networks, list) and not isinstance(list_networks, np.ndarray):
            list_networks = [list_networks]
        for network in list_networks:
            scores = {}
            train_time, train_epoch = 0.0, 0.0
            for i, metric in enumerate(list_metrics):
                score, _train_time, _train_epoch = self._evaluate(network, not need_trained[i], metric, inplace=False)
                train_time += _train_time
                train_epoch += _train_epoch

                if 'flops' in metric or 'params' in metric or 'loss' in metric:
                    scores[metric] = score
                else:
                    if 'acc' in metric:
                        scores[metric] = 1 - score
                    else:
                        scores[metric] = -score
                    # print(metric, scores[metric])
            network.score = np.round([scores[metric] for metric in list_metrics], 4)
            if cur_total_time + TOTAL_TIME + train_time > max_time:
                return TOTAL_TIME, TOTAL_EPOCHS, True
            TOTAL_TIME += train_time
            TOTAL_EPOCHS += train_epoch
        return TOTAL_TIME, TOTAL_EPOCHS, False

    def evaluate(self, networks, inplace=True, **kwargs):
        max_time = kwargs['max_time']
        cur_total_time = kwargs['cur_total_time']
        TOTAL_TIME, TOTAL_EPOCHS = 0.0, 0
        if not isinstance(networks, list) and not isinstance(networks, np.ndarray):
            networks = [networks]
        for _network in networks:
            cur_score = _network.score
            if inplace:
                train_time, train_epoch = self._evaluate(_network, bool(kwargs['using_zc_metric']), kwargs['metric'], inplace=True)
            else:
                score, train_time, train_epoch = self._evaluate(_network, bool(kwargs['using_zc_metric']), kwargs['metric'], inplace=False)
                _network.score = score
            if cur_total_time + TOTAL_TIME + train_time > max_time:
                _network.score = cur_score
                return TOTAL_TIME, TOTAL_EPOCHS, True
            TOTAL_TIME += train_time
            TOTAL_EPOCHS += train_epoch
        return TOTAL_TIME, TOTAL_EPOCHS, False

    def _evaluate(self, network, using_zc_metric, metric, inplace):
        total_epoch = 0
        if using_zc_metric:
            _metric = metric
            if inplace:
                total_time = self.zc_evaluate(network, inplace=True, metric=_metric)
            else:
                score, total_time = self.zc_evaluate(network, inplace=False, metric=_metric)
        else:
            _metric = '_'.join(metric.split('_')[:-1])
            iepoch = int(metric.split('_')[-1])
            if inplace:
                total_time, total_epoch = self.train(network, inplace=True, metric=_metric, iepoch=iepoch)
            else:
                score, total_time, total_epoch = self.train(network, inplace=False, metric=_metric, iepoch=iepoch)
        if inplace:
            return total_time, total_epoch
        else:
            return score, total_time, total_epoch

    def zc_evaluate(self, network, inplace, **kwargs):
        metric = kwargs['metric']
        genotype = network.genotype
        phenotype = self.search_space.decode(genotype)
        op_indices = str(convert_str_to_op_indices(phenotype))

        if metric in ['flops', 'params']:
            h = ''.join(map(str, genotype))
            if metric == 'flops':
                score, time = self.benchmark_database['200'][h]['FLOPs'], self.zc_database[self.dataset][op_indices]['flops']['time']
            else:
                score, time = self.benchmark_database['200'][h]['params'], self.zc_database[self.dataset][op_indices]['params']['time']
        else:
            score, time = self.zc_database[self.dataset][op_indices][metric]['score'], self.zc_database[self.dataset][op_indices][metric]['time']

        if inplace:
            network.score = score
            return time
        else:
            return score, time

    def train(self, network, inplace, **kwargs):
        metric = kwargs['metric']
        iepoch = kwargs['iepoch']
        genotype = network.genotype
        dif_epoch = iepoch - network.info['cur_iepoch'][-1]

        h = ''.join(map(str, genotype))
        info = self.benchmark_database['200'][h]
        score = info[metric][iepoch - 1]
        if 'loss' in metric:
            score *= -1
        time = info['train_time'] * dif_epoch
        if self.dataset == 'cifar10':
            time /= 2
        network.info['cur_iepoch'].append(iepoch)
        network.info['train_time'].append(time)
        if inplace:
            network.score = score
            return time, dif_epoch
        else:
            return score, time, dif_epoch

    def get_test_performance(self, network, **kwargs):
        genotype = network.genotype
        h = ''.join(map(str, genotype))
        time = self.benchmark_database['200'][h]['train_time'] * 200
        if self.dataset == 'cifar10':
            time /= 2
        test_acc = np.round(self.benchmark_database['200'][h]['test_acc'][-1] * 100, 2)
        return test_acc, time

    def get_hv_value(self, list_networks, list_metrics):
        try:
            pof = self.list_pof[list_metrics]
        except KeyError:
            raise KeyError('Not supports these metrics:', list_metrics)
        list_metrics = list_metrics.split('&')
        F = []
        for network in list_networks:
            # network()
            test_acc, _ = self.get_test_performance(network)
            test_err = 100 - test_acc
            _F = [test_err]
            for metric in list_metrics[1:]:
                score, _ = self.zc_evaluate(network, inplace=False, metric=metric)
                _F.append(score)
            F.append(_F)
        F = np.array(F)

        nadir_point = np.max(pof, axis=0)
        utopian_point = np.min(pof, axis=0)
        pof = (pof - utopian_point) / (nadir_point - utopian_point)
        F = (F - utopian_point) / (nadir_point - utopian_point)

        ref_point = get_ref_point(len(list_metrics), pof)
        hv_cal = HV(ref_point)
        hv_value = hv_cal(F)
        return hv_value