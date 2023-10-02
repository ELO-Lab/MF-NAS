from . import Algorithm, BestImprovementLS, FirstImprovementLS, SuccessiveHalving
import numpy as np

class MF_NAS(Algorithm):
    def __init__(self):
        super().__init__()
        self.list_iepoch = None
        self.using_zc_metric_stage1, self.using_zc_metric_stage2 = True, False
        self.metric_stage1, self.metric_stage2 = None, None
        self.max_eval_stage1 = None

    def _run(self, **kwargs):
        best_network, search_time, total_epoch = self.search(**kwargs)
        return best_network, search_time, total_epoch

    def search(self, **kwargs):
        assert self.max_eval_stage1 is not None
        assert self.metric_stage1 is not None
        assert self.metric_stage2 is not None
        assert self.list_iepoch is not None

        if self.optimizer_stage1 == 'FLS':
            optimizer_stage1 = FirstImprovementLS()
        elif self.optimizer_stage1 == 'BLS':
            optimizer_stage1 = BestImprovementLS()
        else:
            raise ValueError(f'Not support this optimizer in MF-NAS framework: {self.optimizer_stage1}')
        optimizer_stage1.adapt(self.problem)
        optimizer_stage1.using_zc_metric = self.using_zc_metric_stage1
        optimizer_stage1.metric = self.metric_stage1
        optimizer_stage1.max_eval = self.max_eval_stage1

        _, search_cost, _ = optimizer_stage1.search(**kwargs)

        network_history_stage1 = optimizer_stage1.network_history[:self.max_eval_stage1]
        genotype_history_stage1 = np.array([network.genotype for network in network_history_stage1])
        score_history_stage1 = optimizer_stage1.score_history[:self.max_eval_stage1]

        _, ids = np.unique(genotype_history_stage1, axis=0, return_index=True)
        network_history_stage1 = np.array(network_history_stage1)[ids]
        score_history_stage1 = np.array(score_history_stage1)[ids]

        ids = np.flip(np.argsort(score_history_stage1))
        network_history_stage1 = network_history_stage1[ids]

        topK_found_solutions = network_history_stage1[:self.k]

        optimizer_stage2 = SuccessiveHalving()
        optimizer_stage2.adapt(self.problem)
        optimizer_stage2.using_zc_metric = self.using_zc_metric_stage2
        optimizer_stage2.metric = self.metric_stage2
        optimizer_stage2.list_iepoch = self.list_iepoch
        best_network, search_time, total_epoch = optimizer_stage2.search(topK_found_solutions)
        return best_network, search_time, total_epoch

