import numpy as np
import torch
import torch.nn as nn
from . import measure

def count_parameters(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e3

def cal_regular_factor(model, mu, sigma):

    model_params = torch.as_tensor(count_parameters(model))
    regular_factor =  torch.exp(-(torch.pow((model_params-mu),2)/sigma))
   
    return regular_factor


class SampleWiseActivationPatterns(object):
    def __init__(self, device):
        self.swap = -1 
        self.activations = None
        self.device = device

    @torch.no_grad()
    def collect_activations(self, activations):
        n_sample = activations.size()[0]
        n_neuron = activations.size()[1]

        if self.activations is None:
            self.activations = torch.zeros(n_sample, n_neuron).to(self.device)  

        self.activations = torch.sign(activations)

    @torch.no_grad()
    def calSWAP(self, regular_factor):
        
        self.activations = self.activations.T # transpose the activation matrix: (samples, neurons) to (neurons, samples)
        self.swap = torch.unique(self.activations, dim=0).size(0)
        
        del self.activations
        self.activations = None
        torch.cuda.empty_cache()

        return self.swap * regular_factor


class SWAP:
    def __init__(self, model=None, inputs = None, device='cuda', regular=False, mu=None, sigma=None):
        self.model = model
        self.interFeature = []
        self.regular_factor = 1
        self.inputs = inputs
        self.device = device

        if regular and mu is not None and sigma is not None:
            self.regular_factor = cal_regular_factor(self.model, mu, sigma).item()

        self.reinit(self.model)

    def reinit(self, model=None, seed=None):
        if model is not None:
            self.model = model
            self.register_hook(self.model)
            self.swap = SampleWiseActivationPatterns(self.device)

        del self.interFeature
        self.interFeature = []
        torch.cuda.empty_cache()

    def clear(self):
        self.swap = SampleWiseActivationPatterns(self.device)
        del self.interFeature
        self.interFeature = []
        torch.cuda.empty_cache()

    def register_hook(self, model):
        for n, m in model.named_modules():
            if isinstance(m, nn.ReLU):
                m.register_forward_hook(hook=self.hook_in_forward)

    def hook_in_forward(self, module, input, output):
        if isinstance(input, tuple) and len(input[0].size()) == 4:
            self.interFeature.append(output.detach()) 

    def forward(self):
        self.interFeature = []
        with torch.no_grad():
            self.model.forward(self.inputs.to(self.device))
            if len(self.interFeature) == 0: return
            activtions = torch.cat([f.view(self.inputs.size(0), -1) for f in self.interFeature], 1)         
            self.swap.collect_activations(activtions)
            
            return self.swap.calSWAP(self.regular_factor)

def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net


@measure("swap")
def compute_swap_score(net, inputs, targets, loss_fn, split_data=1):
    device = inputs.get_device()
    if device == -1:
        device = 'cpu'
    else:
        device = f'cuda:{device}'
    swap = SWAP(model=net, inputs=inputs, device=device)
    swap_score = []
    net = net.apply(network_weight_gaussian_init)
    swap.reinit()
    swap_score.append(swap.forward())
    swap.clear()
    return swap_score[-1]

