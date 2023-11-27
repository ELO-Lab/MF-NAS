import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_info(data):
    for key, value in data.items():
        if isinstance(value, list) or isinstance(value, np.ndarray):
            print(f'- {key}:')
            for v in value:
                print(f'\t{v}')
        else:
            print(f'- {key}: {value}')
    print()

def fx_better_fy(fx, fy):
    if isinstance(fx, list):
        fx = np.array(fx)
    if isinstance(fy, list):
        fy = np.array(fy)
    diff = fx - fy
    fx_better = np.all(diff <= 0)
    fy_better = np.all(diff >= 0)
    if fx_better == fy_better:  # True - True
        return -1
    if fy_better:  # False - True
        return 1
    return 0  # True - False

def check_valid(hash_key, **kwargs):
    """
    - Check if the current solution already exists on the set of checklists.
    """
    return np.all([hash_key not in kwargs[L] for L in kwargs])

def so_evaluation_phase(search_result, problem):
    best_network = search_result
    test_performance, evaluation_cost = problem.get_test_performance(best_network)

    return best_network, test_performance, evaluation_cost


def mo_evaluation_phase(search_result, problem):
    list_best_network = search_result
    list_test_performance = []
    list_evaluation_cost = []
    for network in list_best_network:
        test_performance, evaluation_cost = problem.get_test_performance(network)
        list_evaluation_cost.append(evaluation_cost)
        list_test_performance.append(test_performance)
    scores = np.array([network.get('score') for network in list_best_network])
    best_network_1 = list_best_network[np.argmin(scores[:, 0])]
    test_performance_1 = list_test_performance[np.argmin(scores[:, 0])]
    evaluation_cost_1 = list_evaluation_cost[np.argmin(scores[:, 0])]

    best_network_2 = list_best_network[np.argmax(list_test_performance)]
    test_performance_2 = list_test_performance[np.argmax(list_test_performance)]
    return best_network_1, test_performance_1, evaluation_cost_1, best_network_2, test_performance_2, sum(list_evaluation_cost), list_test_performance
