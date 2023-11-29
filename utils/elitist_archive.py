from .utils import fx_better_fy
import numpy as np

class ElitistArchive:
    def __init__(self):
        self.genotype, self.ID, self.fitness = [], [], []
        self.test_fitness = []
        self.search_fitness = []

    def update(self, sol, **kwargs):
        algo = kwargs['algorithm']

        genotype = sol.get('genotype')
        genotype_id = sol.get('ID')
        fx = sol.get('score')

        search_F = np.round((1 - fx[0]) * 100, 2)
        test_F = algo.problem.get_test_performance(sol)[0]

        if genotype_id not in self.ID:
            ranks = np.zeros(len(self.fitness), dtype=np.int8)
            status = True
            for j, fy in enumerate(self.fitness):
                better_sol = fx_better_fy(fx=fx, fy=fy)
                if better_sol == 0:
                    ranks[j] += 1
                elif better_sol == 1:
                    status = False
                    break
            if status:
                self.genotype.append(genotype)
                self.ID.append(genotype_id)
                self.fitness.append(fx)
                self.search_fitness.append(search_F)
                self.test_fitness.append(test_F)
                ranks = np.append(ranks, 0)

                self.genotype = np.array(self.genotype)[ranks == 0].tolist()
                self.ID = np.array(self.ID)[ranks == 0].tolist()
                self.fitness = np.array(self.fitness)[ranks == 0].tolist()

                self.search_fitness = np.array(self.search_fitness)[ranks == 0].tolist()
                self.test_fitness = np.array(self.test_fitness)[ranks == 0].tolist()

                if algo.n_eval == 1:
                    algo.best_search_arch = sol.genotype.copy()
                    algo.best_test_arch = sol.genotype.copy()
                    algo.fitness_search_arch = search_F
                    algo.fitness_test_arch = test_F
                    algo.fitness_test_arch1 = test_F
                else:
                    i = np.argmax(self.search_fitness)
                    algo.best_search_arch = self.genotype[i]
                    algo.fitness_search_arch = self.search_fitness[i]
                    algo.fitness_test_arch = self.test_fitness[i]

                    i = np.argmax(self.test_fitness)
                    algo.best_test_arch = self.genotype[i]
                    algo.fitness_test_arch1 = self.test_fitness[i]
            algo.search_log.append(
                [''.join(list(map(str, algo.best_search_arch))), algo.fitness_search_arch, algo.fitness_test_arch, ''.join(list(map(str, algo.best_test_arch))),
                 algo.fitness_test_arch1])
