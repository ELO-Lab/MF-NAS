import random
import numpy as np
import torch
import subprocess as sp


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