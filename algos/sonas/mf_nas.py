from algos import Algorithm
from algos.sonas import IteratedLocalSearch, RandomSearch, SuccessiveHalving
import numpy as np
import pathlib
import pickle as p
from models import Network
root = pathlib.Path(__file__).parent.parent

class MF_NAS(Algorithm):
    def __init__(self):
        super().__init__()
        # Stage 1: Training-free Search (Local Search at current)
        self.using_zc_metric_stage1 = True
        self.metric_stage1 = None
        self.max_eval_stage1 = None
        self.optimizer_stage1 = None

        # Stage 2: Training-based Search (Successive Halving)
        self.using_zc_metric_stage2 = False
        self.metric_stage2 = None
        self.n_candidate = -1
        self.list_iepoch = None

    def _run(self, **kwargs):
        assert self.max_eval_stage1 is not None
        assert self.metric_stage1 is not None
        assert self.metric_stage2 is not None
        assert self.list_iepoch is not None

        best_network = self.search(**kwargs)
        return best_network, self.total_time, self.total_epoch

    def search(self, **kwargs):
        # Stage 1: Training-free Search
        print('- Stage 1: Training-free Local Search')
        if self.optimizer_stage1 == 'FLS':
            optimizer_stage1 = IteratedLocalSearch(first_improvement=True)
        elif self.optimizer_stage1 == 'BLS':
            optimizer_stage1 = IteratedLocalSearch(first_improvement=False)
        elif self.optimizer_stage1 == 'RS':
            optimizer_stage1 = RandomSearch()
        else:
            raise ValueError(f'Not support this optimizer in MF-NAS framework: {self.optimizer_stage1}')
        optimizer_stage1.adapt(self.problem)
        optimizer_stage1.using_zc_metric = self.using_zc_metric_stage1

        _ = optimizer_stage1.search(max_time=9999999999, max_eval=self.max_eval_stage1,
                                    metric=self.metric_stage1, **kwargs)

        self.total_time += optimizer_stage1.total_time
        self.total_epoch += optimizer_stage1.total_epoch

        network_history_stage1 = optimizer_stage1.network_history[:self.max_eval_stage1]
        score_history_stage1 = optimizer_stage1.score_history[:self.max_eval_stage1]

        ## Remove duplication
        if self.problem.search_space.name == 'NB-101':
            h_history_stage1 = []
            for network in network_history_stage1:
                h = self.problem.get_h(network.genotype)
                h_history_stage1.append(h)
            _, I = np.unique(h_history_stage1, return_index=True)

        else:
            genotype_history_stage1 = np.array([network.genotype for network in network_history_stage1])
            _, I = np.unique(genotype_history_stage1, axis=0, return_index=True)
        network_history_stage1 = np.array(network_history_stage1)[I]
        score_history_stage1 = np.array(score_history_stage1)[I]

        ## Sort
        I = np.flip(np.argsort(score_history_stage1))
        network_history_stage1 = network_history_stage1[I]

        # Stage 2: Training-based Search
        ## Get top-k best solutions in terms of training-free metric value. They are the input of SH.
        topK_found_solutions = network_history_stage1[:self.n_candidate]
        list_genotypes = [network.genotype for network in topK_found_solutions]
        list_scores = [network.genotype for network in topK_found_solutions]
        p.dump(list_genotypes, open(f'result_local_search_{self.metric_stage1}_genotype.p', 'wb'))
        p.dump(list_scores, open(f'result_local_search_{self.metric_stage1}_score.p', 'wb'))

        # topK_found_solutions = []
        # result_ls = p.load(open(f'{root}/databases/result_local_search.p', 'rb'))
        # for genotype in result_ls[0]:
        #     network = Network()
        #     network.genotype = genotype
        #     topK_found_solutions.append(network)
        print('- Stage 2: Successive Halving')
        return topK_found_solutions[0]
        ## Initialize Successive Halving
        # optimizer_stage2 = SuccessiveHalving()
        # optimizer_stage2.adapt(self.problem)
        # optimizer_stage2.using_zc_metric = self.using_zc_metric_stage2
        # optimizer_stage2.metric = self.metric_stage2
        # optimizer_stage2.list_iepoch = self.list_iepoch
        #
        # best_network = optimizer_stage2.search(topK_found_solutions, max_time=self.problem.max_time - optimizer_stage1.total_time)
        # self.total_time += optimizer_stage2.total_time
        # self.total_epoch += optimizer_stage2.total_epoch
        #
        # return best_network

