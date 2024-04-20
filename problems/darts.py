from problems import Problem
from search_spaces import SS_DARTS
import numpy as np
import time
import json
import os
import torchvision
from utils.params_flops_counters import get_model_infos
import pathlib
from zc_predictors.core import ZeroCost
from zc_predictors.utils import get_config_for_zc_predictor
import pickle as p
root = pathlib.Path(__file__).parent.parent

METRICS = ['epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'params', 'plain', 'snip',
           'synflow', 'zen']

class DARTS(Problem):
    def __init__(self, max_eval, max_time, dataset='cifar10', **kwargs):
        super().__init__(SS_DARTS(), max_eval, max_time)
        self.dataset = dataset
        self.save_path = kwargs['save_path']
        self.n_models_per_train = kwargs['n_models_per_train']
        self.using_ray = kwargs['using_ray']

        if self.using_ray:
            import ray
            ray.init(_temp_dir=f'{root}/exp')

        _ = torchvision.datasets.CIFAR10(root=f'{root}/dataset/cifar10', train=True, download=True)
        _ = torchvision.datasets.CIFAR10(root=f'{root}/dataset/cifar10', train=False, download=True)

        config = get_config_for_zc_predictor('darts', 'CIFAR-10', f'{root}/dataset/cifar10', 42)
        self.zc_predictor = ZeroCost(config)
        self.GP_model = p.load(open(f'{root}/database/model_all-multiple-0.7-v2_9.p', 'rb'))

    def evaluate(self, networks, **kwargs):
        TOTAL_TIME, TOTAL_EPOCHS = 0.0, 0
        if not isinstance(networks, list) and not isinstance(networks, np.ndarray):
            networks = [networks]
        if bool(kwargs['using_zc_metric']):
            metric = kwargs['metric']
            for _network in networks:
                total_time = self.zc_evaluate(_network, metric=metric)
                TOTAL_TIME += total_time
        else:
            end_iepoch = int(kwargs['metric'].split('_')[-1])
            total_time, total_epoch = self._parallel_train(list_network=networks, end_iepoch=end_iepoch, **kwargs)
            TOTAL_TIME += total_time
            TOTAL_EPOCHS += total_epoch
        return TOTAL_TIME, TOTAL_EPOCHS, False

    def zc_evaluate(self, network, **kwargs):
        metric = kwargs['metric']
        model = self.search_space.get_model(network.genotype, auxiliary=False)

        s = time.time()
        if isinstance(metric, list):
            list_scores = {metric: -100000 for metric in METRICS}
            predicted_scores = self.zc_predictor.query(model, metric)
            for metric, value in predicted_scores.items():
                list_scores[metric] = value
            flops, params = get_model_infos(model, shape=(1, 3, 32, 32))  # Params in MB
            list_scores['flops'] = flops
            list_scores['params'] = params
            X = np.array([list(list_scores.values())])
            score = self.GP_model.predict(X)
        else:
            if metric in ['flops', 'params']:
                flops, params = get_model_infos(model, shape=(1, 3, 32, 32))  # Params in MB
                if metric == 'flops':
                    score = flops
                else:
                    score = params
            else:
                scores = self.zc_predictor.query(model, metric)
                score = scores[metric]
        cost_time = time.time() - s
        network.score = score
        return cost_time

    def _parallel_train(self, list_network, **kwargs):
        start_iepoch = list_network[-1].info['cur_iepoch'][-1]
        end_iepoch = kwargs['end_iepoch']
        total_epoch = 0
        s = time.time()

        for i in range(0, len(list_network), self.n_models_per_train):
            _list_network = list_network[i:i + self.n_models_per_train]
            for network in _list_network:
                total_epoch += (end_iepoch - start_iepoch)
                network_id = ''.join(list(map(str, network.genotype)))
                if self.using_ray:
                    script = f"python train_darts_ray.py --seed {kwargs['algo'].seed} --save_path {self.save_path} --network_id {network_id} --dataset {self.dataset} --start_iepoch {start_iepoch} --end_iepoch {end_iepoch} | tee -a {self.save_path}/log_{network_id}_{start_iepoch}_{end_iepoch}.txt &"
                else:
                    script = f"python train_darts.py --seed {kwargs['algo'].seed} --save_path {self.save_path} --network_id {network_id} --dataset {self.dataset} --start_iepoch {start_iepoch} --end_iepoch {end_iepoch} | tee -a {self.save_path}/log_{network_id}_{start_iepoch}_{end_iepoch}.txt &"
                os.system(script)
            while True:
                done = True
                for network in _list_network:
                    network_id = ''.join(list(map(str, network.genotype)))
                    if not os.path.exists(f'{self.save_path}/{network_id}/status.json'):
                        done = False
                        break
                if done:  # All candidate networks are trained
                    break
                else:  # Wait 30 seconds
                    time.sleep(30)
            for network in _list_network:
                network_id = ''.join(list(map(str, network.genotype)))
                train_status = json.load(open(f'{self.save_path}/{network_id}/status.json', 'r'))
                network.score = train_status['score']
                network.info['cur_iepoch'].append(end_iepoch)
                os.remove(f'{self.save_path}/{network_id}/status.json')
        total_time = time.time() - s
        return total_time, total_epoch

    def get_test_performance(self, network, **kwargs):
        print('This is the validation performance. To achieve the test performance, you need to train from scratch.')
        return network.score

if __name__ == '__main__':
    problem = DARTS(100, 100, 'cifar10', save_path='None', n_models_per_train=1, using_ray=False)
    ss = SS_DARTS()
    genotype = ss.sample(True)
    model = ss.get_model(genotype=genotype, search=True)
    _, _, _ = problem.evaluate(networks=[model], using_zc_metric=True, metric=['flops', 'zen', 'synflow'])
    print(model.score)