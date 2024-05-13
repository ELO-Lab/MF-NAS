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
from utils import get_gpu_memory

root = pathlib.Path(__file__).parent.parent

METRICS = ['epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'params', 'plain', 'snip',
           'synflow', 'zen']

class DARTS(Problem):
    def __init__(self, max_eval, max_time, dataset='cifar10', **kwargs):
        super().__init__(SS_DARTS(), max_eval, max_time)
        self.dataset = dataset
        self.save_path = kwargs['save_path']
        self.n_models_per_train = kwargs['n_models_per_train']
        # self.using_ray = kwargs['using_ray']
        self.max_epoch = kwargs['max_epoch']

        # if self.using_ray:
        #     import ray
        #     ray.init(_temp_dir=f'{root}/exp')

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
        # model = self.search_space.get_model(network.genotype, auxiliary=False)
        model = self.search_space.get_model(network.genotype, search=False)

        s = time.time()
        if isinstance(metric, list):
            _metric = metric.copy()
            try:
                metric.remove('flops')
                metric.remove('params')
            except ValueError:
                pass
            list_scores = {metric: -100000 for metric in METRICS}
            predicted_scores = self.zc_predictor.query(model, metric)
            for metric, value in predicted_scores.items():
                list_scores[metric] = value
            flops, params = -100000, -100000
            if 'flops' in _metric or 'params' in _metric:
                flops, params = get_model_infos(model, shape=(1, 3, 32, 32))  # Params in MB
            list_scores['flops'] = flops
            list_scores['params'] = params
            X = np.array([list(list_scores.values())])
            score = self.GP_model.predict(X)[0]
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
        max_epoch = self.max_epoch
        total_epoch = 0
        s = time.time()

        total_network = len(list_network)
        i = 0
        while total_network > 0:
            free_memory = get_gpu_memory()
            estimated_gpu_each_subprocess = 3096
            n_available_process_each_gpu = free_memory // estimated_gpu_each_subprocess
            n_models_per_train = min(sum(n_available_process_each_gpu), self.n_models_per_train)
            print(free_memory, n_available_process_each_gpu)
            print('# Training Models:', n_models_per_train)

            # for i in range(0, len(list_network), self.n_models_per_train):
            #     _list_network = list_network[i:i + self.n_models_per_train]
            _list_network = list_network[i:i + n_models_per_train]
            for network in _list_network:
                total_epoch += (end_iepoch - start_iepoch)
                network_id = ''.join(list(map(str, network.genotype)))
                # if self.using_ray:
                #     script = f"python train_darts_ray.py --seed {kwargs['algo'].seed} --save_path {self.save_path} --network_id {network_id} --dataset {self.dataset} --start_iepoch {start_iepoch} --end_iepoch {end_iepoch} | tee -a {self.save_path}/log_{network_id}_{start_iepoch}_{end_iepoch}.txt &"
                # else:
                # Select GPU for training
                gpu_id = select_gpu(n_available_process_each_gpu)
                script = f"python train_darts.py --seed {kwargs['algo'].seed} --max_epoch {max_epoch} --device cuda:{gpu_id} --save_path {self.save_path} --network_id {network_id} --dataset {self.dataset} --start_iepoch {start_iepoch} --end_iepoch {end_iepoch} | tee -a {self.save_path}/log_{network_id}_{start_iepoch}_{end_iepoch}.txt &"
                n_available_process_each_gpu[gpu_id] -= 1
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
            i += n_models_per_train
            total_network -= n_models_per_train
        total_time = time.time() - s
        return total_time, total_epoch

    def get_test_performance(self, network, **kwargs):
        print('This is the validation performance. To achieve the test performance, you need to train from scratch.')
        return network.score

def select_gpu(n_available_process_each_gpu):
    return np.argmax(n_available_process_each_gpu)