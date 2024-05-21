from algos import Algorithm
import numpy as np
from algos.utils import ElitistArchive
from algos.monas import ParetoLocalSearch, MultiObjective_SuccessiveHalving, LOMONAS
from algos.monas.mo_sh import selection

class MOF_NAS(Algorithm):
    def __init__(self):
        super().__init__(nas_type='mo')
        # Stage 1: Training-free Search (Pareto Local Search at current)
        self.need_trained_stage1 = True
        self.list_metrics_stage1 = None
        self.max_eval_stage1 = None
        self.optimizer_stage1 = None

        # Stage 2: Training-based Search (Multi-objective Successive Halving)
        self.need_trained_stage2 = True
        self.list_metrics_stage2 = None
        self.n_remaining_candidates = None
        self.list_iepochs = None

        self.archive = ElitistArchive()

    def _reset(self):
        self.archive = ElitistArchive()

    def _run(self, **kwargs):
        approximation_set = self.search(**kwargs)
        return approximation_set, self.total_time, self.total_epoch

    def search(self, **kwargs):
        # Stage 1: Training-free Search
        print('- Stage 1: Training-free Pareto Local Search')
        if self.optimizer_stage1 == 'PLS':
            optimizer_stage1 = ParetoLocalSearch()
        elif self.optimizer_stage1 == 'LOMONAS':
            optimizer_stage1 = LOMONAS()
        else:
            raise ValueError(f'Not support this optimizer in MOF-NAS framework: {self.optimizer_stage1}')
        optimizer_stage1.adapt(self.problem)
        optimizer_stage1.list_metrics = self.list_metrics_stage1
        optimizer_stage1.need_trained = self.need_trained_stage1

        _ = optimizer_stage1.search(max_time=9999999999, max_eval=self.max_eval_stage1, **kwargs)

        self.total_time += optimizer_stage1.total_time
        self.total_epoch += optimizer_stage1.total_epoch

        network_history_stage1 = optimizer_stage1.network_history[:self.max_eval_stage1]

        h_history_stage1 = np.array([self.problem.get_hash(network) for network in network_history_stage1])
        _, I = np.unique(h_history_stage1, return_index=True)
        network_history_stage1 = np.array(network_history_stage1)[I]

        # Stage 2: Training-based Search
        ## Get top-k best solutions. They are the input of MO-SH.
        ids = selection(network_history_stage1, self.n_remaining_candidates[0])
        topK_found_solutions = np.array([network_history_stage1[i] for i in ids])

        print('- Stage 2: Multi-Objective Successive Halving')
        ## Initialize Successive Halving
        optimizer_stage2 = MultiObjective_SuccessiveHalving()
        optimizer_stage2.adapt(self.problem)
        optimizer_stage2.need_trained = self.need_trained_stage2
        optimizer_stage2.list_metrics = self.list_metrics_stage2
        optimizer_stage2.list_iepochs = self.list_iepochs
        optimizer_stage2.n_remaining_candidates = self.n_remaining_candidates

        approximation_front = optimizer_stage2.search(topK_found_solutions, max_time=self.problem.max_time - optimizer_stage1.total_time)
        self.total_time += optimizer_stage2.total_time
        self.total_epoch += optimizer_stage2.total_epoch

        return approximation_front