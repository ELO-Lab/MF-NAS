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