import random
import numpy as np
import torch
import subprocess as sp
import os

def create_result_folder(nas_type, opt_name, search_space, opt, save_path):
    if nas_type == 'so':
        if opt_name == 'MF-NAS':
            res_path = f'{save_path}/{opt_name}_{search_space}_{opt.metric_stage1}_{opt.max_eval_stage1}_{opt.metric_stage2}_{opt.n_candidate}'
        else:
            if opt.using_zc_metric:
                res_path = f'{save_path}/{opt_name}_{search_space}_{opt.metric}'
            else:
                res_path = f'{save_path}/{opt_name}_{search_space}_{opt.metric}{opt.iepoch}'
    else:
        if opt_name == 'MOF-NAS':
            metrics_stage1 = '&'.join(opt.list_metrics_stage1)
            metrics_stage2 = '&'.join(opt.list_metrics_stage2)
            res_path = f'{save_path}/{opt_name}_{search_space}_{metrics_stage1}_{opt.max_eval_stage1}_{metrics_stage2}_{opt.n_candidate}'
        else:
            metrics = '&'.join(opt.list_metrics)
            res_path = f'{save_path}/{opt_name}_{search_space}_{metrics}'
    os.makedirs(res_path, exist_ok=True)
    return res_path

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