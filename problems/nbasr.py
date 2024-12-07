from problems import Problem
from search_spaces import SS_ASR
from search_spaces.nbasr.utils import get_model_graph, graph_hash
import numpy as np
import pickle as p
import json

import pathlib

ROOT_DIR = str(pathlib.Path.cwd())

class NB_ASR(Problem):
    def __init__(self, max_eval, max_time, dataset, **kwargs):
        super().__init__(SS_ASR(), max_eval, max_time)
        self.dataset = dataset
        self.zc_database = p.load(open(ROOT_DIR + f'/database/nbasr/[NBASR]_zc_data.p', 'rb'))
        self.benchmark_database = p.load(open(ROOT_DIR + f'/database/nbasr/[NBASR]_data.p', 'rb'))
        self.pof = json.load(open(ROOT_DIR + f'/database/nbasr/[NBASR]_pof.json'))

        # self.mo_objective = kwargs['mo_objective']
        self.mo_objective = None

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

                if 'flops' in metric or 'params' in metric:
                    scores[metric] = score
                else:
                    if 'per' in metric:
                        scores[metric] = -score
                    else:
                        scores[metric] = -score
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
        time = 0.0

        h = self.get_h(network=genotype)
        if metric in ['flops', 'params']:
            if metric == 'flops':
                score = self.benchmark_database[h]['FLOPs']
            else:
                score = self.benchmark_database[h]['params']
        else:
            score = self.zc_database[h][metric]
        if np.isnan(score) or np.isinf(score):
            score = np.inf
        if inplace:
            network.score = score
            return time
        else:
            return score, time

    def train(self, network, inplace, **kwargs):
        metric = kwargs['metric']
        iepoch = kwargs['iepoch']
        dif_epoch = iepoch - network.info['cur_iepoch'][-1]

        genotype = network.genotype

        h = self.get_h(network=genotype)
        info = self.benchmark_database[h]
        score = info[metric][iepoch - 1]
        if 'per' in metric:
            score *= -1
        time = 0
        network.info['cur_iepoch'].append(iepoch)
        network.info['train_time'].append(time)
        if inplace:
            network.score = score
            return time, dif_epoch
        else:
            return score, time, dif_epoch

    def get_test_performance(self, network, **kwargs):
        time = 0.0
        genotype = network.genotype
        h = self.get_h(network=genotype)
        test_PER = np.round(self.benchmark_database[h]['test_per'] * 100, 2)
        return test_PER, time

    def get_h(self, network):
        if isinstance(network, list):
            network = np.array(network)
        if isinstance(network, np.ndarray):
            network = self.search_space.decode(network)
        g, _ = get_model_graph(network, ops=None, minimize=True)
        h = graph_hash(g)
        return h

    def get_hv_value(self, list_networks, list_metrics):
        try:
            pof = self.list_pof[list_metrics]
        except KeyError:
            raise KeyError('Not supports these metrics:', list_metrics)
        list_metrics = list_metrics.split('&')
        F = []
        for network in list_networks:
            test_per, _ = self.get_test_performance(network)
            _F = [test_per]
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
