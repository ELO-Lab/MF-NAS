import random
import numpy as np
import torch
import subprocess as sp
import os

def create_result_folder(nas_type, opt_name, search_space, opt, save_path):
    if nas_type == 'so':
        if 'MF-NAS' in opt_name:
            res_path = f'{save_path}/{opt_name}_{search_space}_{opt.optimizer_stage1}-{opt.metric_stage1}-{opt.max_eval_stage1}_SH-{opt.metric_stage2}-{opt.n_candidate}'
        else:
            if opt.using_zc_metric:
                res_path = f'{save_path}/{opt_name}_{search_space}_{opt.metric}'
            else:
                res_path = f'{save_path}/{opt_name}_{search_space}_{opt.metric}{opt.iepoch}'
    else:
        if opt_name == 'MOF-NAS':
            metrics_stage1 = '&'.join(opt.list_metrics_stage1)
            metrics_stage2 = '&'.join(opt.list_metrics_stage2)
            res_path = f'{save_path}/{opt_name}_{search_space}_{metrics_stage1}_{opt.max_eval_stage1}_{metrics_stage2}_{opt.n_remaining_candidates[0]}'
        else:
            metrics = '&'.join(opt.list_metrics)
            res_path = f'{save_path}/{opt_name}_{search_space}_{metrics}'
    os.makedirs(res_path, exist_ok=True)
    return res_path

def mean_std(X, verbose=True):
    mean = np.round(np.mean(X), 4)
    std = np.round(np.std(X), 4)
    if verbose:
        print(f'{mean:.4f} ({std:.4f})')
    return float(mean), float(std)

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

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = np.array([int(x.split()[0]) for i, x in enumerate(memory_free_info)])
    return memory_free_values
