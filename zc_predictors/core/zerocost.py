"""
Original source code: https://github.com/automl/NASLib
"""

import math
import torch
from .predictor import Predictor
from zc_predictors.utils import get_train_val_loaders
from zc_predictors.core.pruners import predictive


class ZeroCost(Predictor):
    def __init__(self, config, method_type='synflow'):
        super().__init__()
        # available zero-cost method types: 'jacov', 'snip', 'synflow', 'grad_norm', 'fisher', 'grasp'

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.method_type = method_type

        self.dataload = 'random'
        self.num_imgs_or_batches = 1
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            print('Using GPU')
        else:
            self.device = torch.device('cpu')
            print('Using CPU')
        self.config = config

        # if config['search_space'] == 'tnb101':
        #     self.num_classes = 75
        # else:
        #     num_classes_dict = {'CIFAR-10': 10, 'CIFAR-100': 100, 'ImageNet16-120': 120}
        #     self.num_classes = None
        #     if self.config['dataset'] in num_classes_dict:
        #         self.num_classes = num_classes_dict[self.config['dataset']]
        #     else:
        #         raise KeyError(f'Not support {self.config["dataset"]} dataset. Just only supported "CIFAR-10"; "CIFAR-100" '
        #                        f'and ImageNet16-120 datasets')
        self.num_classes = None
        self.train_loader = None
        self.pre_process()

    def pre_process(self):
        self.train_loader, _, _, _, _ = get_train_val_loaders(self.config, mode='train')

    def query(self, network, metrics):
        network = network.to(self.device)
        metrics = metrics if isinstance(metrics, list) else [metrics]
        score = predictive.find_measures(
            network,
            self.train_loader,
            (self.dataload, self.num_imgs_or_batches, self.num_classes),
            self.device,
            measure_names=metrics,
        )
        for key in score:
            if math.isnan(score[key]) or math.isinf(score[key]):
                score[key] = -1e8

            if key == 'synflow':
                if score[key] == 0.:
                    return score

                score[key] = math.log(score[key]) if score[key] > 0 else -math.log(-score[key])

        torch.cuda.empty_cache()
        return score