from .utils import fx_better_fy
import numpy as np

class ElitistArchive:
    def __init__(self):
        self.genotype, self.ID, self.fitness = [], [], []

    def update(self, sol, **kwargs):
        genotype = sol.get('genotype')
        genotype_id = sol.get('ID')
        fx = sol.get('score')

        if genotype_id not in self.ID:
            ranks = np.zeros(len(self.fitness), dtype=np.int8)
            status = True
            for j, fy in enumerate(self.fitness):
                better_sol = x_better_y(x=fx, y=fy)
                if better_sol == 0:
                    ranks[j] += 1
                elif better_sol == 1:
                    status = False
                    break
            if status:
                self.genotype.append(genotype)
                self.ID.append(genotype_id)
                self.fitness.append(fx)
                ranks = np.append(ranks, 0)

                self.genotype = np.array(self.genotype)[ranks == 0].tolist()
                self.ID = np.array(self.ID)[ranks == 0].tolist()
                self.fitness = np.array(self.fitness)[ranks == 0].tolist()
