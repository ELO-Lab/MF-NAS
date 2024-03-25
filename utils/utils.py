import random
import numpy as np
import torch
import itertools
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
sorter = NonDominatedSorting()

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

def check_not_exist(hash_key, **kwargs):
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
    return best_network_1, test_performance_1, evaluation_cost_1, best_network_2, test_performance_2, list_evaluation_cost, list_test_performance


def compare(fx, fy):
    if fx > fy:
        return 1, 0
    elif fy > fx:
        return 0, 1
    winner = np.random.choice([0, 1])
    loser = 1 - winner
    return winner, loser

def proposed_selection(list_candidates, problem):
    init_iepoch = 12
    losers_at_each_iepoch = {}
    I = list(range(len(list_candidates)))
    np.random.shuffle(I)
    ids = np.random.choice(I, 2)
    first_candidate, second_candidate = list_candidates[ids[0]], list_candidates[ids[1]]
    I.remove(ids[0])
    I.remove(ids[1])

    total_time, total_epoch = 0.0, 0.0
    _list_candidates = [first_candidate, second_candidate]
    for network in _list_candidates:
        info, cost_time = problem.evaluate(network, using_zc_metric=False, metric='val_acc', iepoch=init_iepoch)
        network.score = info['val_acc']
        network.info['train_info'][init_iepoch]['score'] = network.score

        diff_epoch = network.info['cur_iepoch'][-1] - network.info['cur_iepoch'][-2]
        total_time += cost_time
        total_epoch += diff_epoch

    winner, loser = compare(_list_candidates[0].score, _list_candidates[1].score)
    losers_at_each_iepoch[init_iepoch] = loser

    while len(I) > 0:
        i = np.random.choice(I)
        new_candidate = list_candidates[i]
        I.remove(i)


def multi_fidelity_selection(list_candidates, problem, list_iepoch):
    # way #1: 8 -> 4 -> 2 -> 1
    # way #2: 16 -> 8 -> 4 -> 2
    # iepoch: 4 -> 12 -> 36 -> 108

    # way #1: 16 -> 8 -> 4 -> 2 -> 1
    # way #2: 32 -> 16 -> 8 -> 4 -> 2
    # iepoch: 12 -> 25 -> 50 -> 100 -> 200
    from algos import SuccessiveHalving
    opt = SuccessiveHalving()
    opt.adapt(problem)
    opt.using_zc_metric = False
    opt.metric = 'val_acc'
    opt.list_iepoch = list_iepoch

    best_network = opt.search(list_candidates, max_time=999999999)
    total_time = opt.total_time
    total_epoch = opt.total_epoch

    return best_network, total_time, total_epoch


def knee_extreme_selection(non_dominated_front, alpha):
    ids = range(non_dominated_front.shape[-1])
    I = list(range(len(non_dominated_front)))
    info_potential_sols_all = []
    for f_ids in itertools.combinations(ids, 2):
        f_ids = np.array(f_ids)
        obj_1, obj_2 = f'{f_ids[0]}', f'{f_ids[1]}'

        _non_dominated_front = non_dominated_front[:, f_ids].copy()

        ids_sol = np.array(list(range(len(non_dominated_front))))
        ids_fr0 = sorter.do(_non_dominated_front, only_non_dominated_front=True)

        ids_sol = ids_sol[ids_fr0]
        _non_dominated_front = _non_dominated_front[ids_fr0]

        sorted_idx = np.argsort(_non_dominated_front[:, 0])

        ids_sol = ids_sol[sorted_idx]
        _non_dominated_front = _non_dominated_front[sorted_idx]

        min_values, max_values = np.min(_non_dominated_front, axis=0), np.max(_non_dominated_front, axis=0)
        _non_dominated_front_norm = (_non_dominated_front - min_values) / (max_values - min_values)

        info_potential_sols = [
            [I[ids_sol[0]], f'best_f{obj_1}']  # (idx (in full set), property)
        ]

        l_non_front = len(_non_dominated_front)
        for i in range(l_non_front - 1):
            if np.sum(np.abs(_non_dominated_front[i] - _non_dominated_front[i + 1])) != 0:
                break
            else:
                info_potential_sols.append([I[ids_sol[i + 1]], f'best_f{obj_1}'])

        for i in range(l_non_front - 1, -1, -1):
            if np.sum(np.abs(_non_dominated_front[i] - _non_dominated_front[i - 1])) != 0:
                break
            else:
                info_potential_sols.append([I[ids_sol[i - 1]], f'best_f{obj_2}'])
        info_potential_sols.append([I[ids_sol[l_non_front - 1]], f'best_f{obj_2}'])

        ## find the knee solutions
        start_idx, end_idx = 0, l_non_front - 1

        for i in range(len(info_potential_sols)):
            if info_potential_sols[i + 1][-1] == f'best_f{obj_2}':
                break
            else:
                start_idx = info_potential_sols[i][0] + 1

        for i in range(len(info_potential_sols) - 1, -1, -1):
            if info_potential_sols[i - 1][-1] == f'best_f{obj_1}':
                break
            else:
                end_idx = info_potential_sols[i][0] - 1

        for i in range(start_idx, end_idx + 1):
            l = None
            h = None
            for m in range(i - 1, -1, -1):
                if np.sum(np.abs(_non_dominated_front[m] - _non_dominated_front[i])) != 0:
                    l = m
                    break
            for m in range(i + 1, l_non_front, 1):
                if np.sum(np.abs(_non_dominated_front[m] - _non_dominated_front[i])) != 0:
                    h = m
                    break

            if (h is not None) and (l is not None):
                position = above_or_below(considering_pt=_non_dominated_front[i],
                                          remaining_pt_1=_non_dominated_front[l],
                                          remaining_pt_2=_non_dominated_front[h])
                if position == -1:
                    angle_measure = calc_angle_measure(considering_pt=_non_dominated_front_norm[i],
                                                            neighbor_1=_non_dominated_front_norm[l],
                                                            neighbor_2=_non_dominated_front_norm[h])
                    if angle_measure > alpha:
                        info_potential_sols.append([I[ids_sol[i]], 'knee'])
        info_potential_sols_all += info_potential_sols
    return info_potential_sols_all


def above_or_below(considering_pt, remaining_pt_1, remaining_pt_2):
    """
    This function is used to check if the considering point is above or below
    the line connecting two remaining points.\n
    1: above\n
    -1: below
    """
    orthogonal_vector = remaining_pt_2 - remaining_pt_1
    line_connecting_pt1_and_pt2 = -orthogonal_vector[1] * (considering_pt[0] - remaining_pt_1[0]) \
                                  + orthogonal_vector[0] * (considering_pt[1] - remaining_pt_1[1])
    if line_connecting_pt1_and_pt2 > 0:
        return 1
    return -1


def calc_angle_measure(considering_pt, neighbor_1, neighbor_2):
    """
    This function is used to calculate the angle measure is created by the considering point
    and two its nearest neighbors
    """
    line_1 = neighbor_1 - considering_pt
    line_2 = neighbor_2 - considering_pt
    cosine_angle = (line_1[0] * line_2[0] + line_1[1] * line_2[1]) \
                   / (np.sqrt(np.sum(line_1 ** 2)) * np.sqrt(np.sum(line_2 ** 2)))
    if cosine_angle < -1:
        cosine_angle = -1
    if cosine_angle > 1:
        cosine_angle = 1
    angle = np.arccos(cosine_angle)
    return 360 - np.degrees(angle)
