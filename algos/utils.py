from models import Network
import numpy as np
from numpy import ndarray

def sampling_solution(problem):
    while True:
        network = Network()
        network.genotype = problem.search_space.sample(genotype=True)
        if problem.search_space.is_valid(network.genotype):
            return network

def update_log(best_network, cur_network, **kwargs):
    algorithm = kwargs['algorithm']
    algorithm.trend_best_network.append(best_network)
    algorithm.trend_time.append(algorithm.total_time)

    algorithm.network_history.append(cur_network)
    algorithm.score_history.append(cur_network.score)

def is_equal(f1: ndarray, f2: ndarray) -> bool:
    """
    Takes in the objective function values of two solution (f1, f2.)
    Returns the better one using Pareto-dominance definition.

    :param f1: the objective function values of the first solution
    :param f2: the objective function values of the second solution
    :return: True or False
    """
    return np.all(f1 == f2)

def compare_f1_f2(f1: ndarray, f2: ndarray) -> int:
    """
    Takes in the objective function values of two solution (f1, f2). Returns the better one using Pareto-dominance definition.

    :param f1: the objective function values of the first solution
    :param f2: the objective function values of the second solution
    :return: -1 (no one is better); 0 (f1 is better); or 1 (f2 is better)
    """
    x_better = np.all(f1 <= f2)
    y_better = np.all(f2 <= f1)
    if x_better == y_better:
        return -1
    if y_better:  # False - True
        return 1
    return 0  # True - False

def not_existed(genotypeHash: str, **kwargs) -> bool:
    """
    Takes in the fingerprint of a solution and a set of checklists.
    Return True if the current solution have not existed on the set of checklists.

    :param genotypeHash: the fingerprint of the considering solution
    :return: True or False
    """
    return np.all([genotypeHash not in kwargs[L] for L in kwargs])

class ElitistArchive:
    """
        Note: No limit the size
    """
    def __init__(self):
        self.archive = []
        self.genotypeHash_archive = []

    def add(self, list_solution):
        for solution in list_solution:
            self.update(solution)
        return self

    def update_without_check(self, solution):
        self.archive.append(solution)

    def update(self, solution, **kwargs):
        length = len(self.archive)
        notDominated = np.ones(length).astype(bool)

        genotypeHash_solution = ''.join(map(str, solution.genotype))
        if genotypeHash_solution not in self.genotypeHash_archive:
            # Compare to every solutions in Elitist Archive
            for i, elitist in enumerate(self.archive):
                better_sol = compare_f1_f2(f1=solution.score, f2=elitist.score)
                if better_sol == 0:  # Filter out members that are dominated by new solution
                    notDominated[i] = False
                elif better_sol == 1:  # If new solution is dominated by any member, stop the checking process
                    return
            self.archive.append(solution)
            self.genotypeHash_archive.append(genotypeHash_solution)
            notDominated = np.append(notDominated, True)
            # Update Elitist Archive
            self.archive = np.array(self.archive)[notDominated].tolist()
            self.genotypeHash_archive = np.array(self.genotypeHash_archive)[notDominated].tolist()

    def isDominated(self, other) -> bool:
        """
        Check whether the current archive is dominated by the other or not.
        Returns the better one using Pareto-dominance definition.

        :param other: the comparing archive
        :return: True or False
        """
        fitness_self = np.array([s.F for s in self.archive])
        fitness_other = np.array([s_.F for s_ in other.archive])

        checklist = []
        for i, f_s1 in enumerate(fitness_self):
            res = 'non'
            for f_s2 in fitness_other:
                better_sol = compare_f1_f2(f1=f_s1, f2=f_s2)
                if better_sol == 1:
                    res = 'dom'
                    break
                elif better_sol == -1:
                    if is_equal(f1=f_s1, f2=f_s2):
                        res = 'eq'
            checklist.append(res)
        checklist = np.array(checklist)
        if np.all(checklist == 'dom'):
            return True
        if np.any(checklist == 'non'):
            return False
        return True

class Footprint:
    def __init__(self):
        self.data = {}