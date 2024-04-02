from problems import Problem
from search_spaces import SS_DARTS
import numpy as np
import time
import tensorwatch as tw
import subprocess
import json

def get_model_stats(model, input_tensor_shape, clone_model=True) -> tw.ModelStats:
    model_stats = tw.ModelStats(model, input_tensor_shape, clone_model=clone_model)
    return model_stats

class DARTS(Problem):
    def __init__(self, max_eval, max_time, dataset, **kwargs):
        super().__init__(SS_DARTS(), max_eval, max_time)
        self.dataset = dataset
        self.save_path = kwargs['save_path']

    def evaluate(self, networks, **kwargs):
        end_iepoch = kwargs['iepoch']
        TOTAL_TIME, TOTAL_EPOCHS = 0.0, 0
        if not isinstance(networks, list) and not isinstance(networks, np.ndarray):
            networks = [networks]
        if bool(kwargs['using_zc_metric']):
            metric = kwargs['metric']
            for _network in networks:
                total_time = self.zc_evaluate(_network, metric=metric)
                TOTAL_TIME += total_time
        else:
            total_time, total_epoch = self._parallel_train(list_network=networks, end_iepoch=end_iepoch)
            TOTAL_TIME += total_time
            TOTAL_EPOCHS += total_epoch
        return TOTAL_TIME, TOTAL_EPOCHS, False

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

    def _parallel_train(self, list_network, **kwargs):
        start_iepoch = list_network[-1].info['cur_iepoch'][-1]
        end_iepoch = kwargs['end_iepoch']
        total_epoch = 0
        s = time.time()

        checklist = {}
        for network in list_network:
            network_id = ''.join(list(map(str, network.genotype)))
            checklist[network_id] = False

        fp = open(f'{self.save_path}/checklist_{start_iepoch}_{start_iepoch}_{end_iepoch}.json', 'w')
        json.dump(checklist, fp, indent=4)
        fp.close()

        for network in list_network:
            total_epoch += (end_iepoch - start_iepoch)
            genotype = network.genotype
            network_id = ''.join(list(map(str, network.genotype)))
            subprocess.run(["python", "./scripts/train_darts.py", "--save_path", f"{self.save_path}", "--genotype", f"{genotype}",
                            "--dataset", f"{self.dataset}", "--start_iepoch", f"{start_iepoch}" "--end_iepoch",
                            f"{end_iepoch}", f"&> {self.save_path}/log_{network_id}_{start_iepoch}_{end_iepoch}.txt &"])
        while True:
            fp = open(f'{self.save_path}/checklist_{start_iepoch}_{start_iepoch}_{end_iepoch}.json', 'r')
            checklist = json.load(fp)
            fp.close()
            done = True
            for network_id in checklist:
                if not checklist[network_id]:
                    done = False
                    break
            if done:
                break
            else:
                time.sleep(6000)
        total_time = time.time() - s
        return total_time, total_epoch

    def get_test_performance(self, network, **kwargs):
        return network.score
        # pass
        # network.model.to('cuda')
        # network_id = ''.join(list(map(str, network.genotype)))
        # checkpoint = torch.load(f'{self.save_path}/{network_id}/best_model.pth.tar')
        # network.model.load_state_dict(checkpoint['model_state_dict'])
        # network.model.drop_path_prob = 0.3
        # network.model._auxiliary = False
        # test_acc, test_objs = infer(self.test_queue, network.model, criterion)
        # return test_acc