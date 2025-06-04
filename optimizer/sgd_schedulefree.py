from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Union, Tuple, Optional, Callable
from torch.optim.optimizer import ParamsT

class SGDScheduleFree(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr = 0.01,
            r = 0.0,
            beta = 0.9
    ):
        
        defaults = dict(
            lr = lr,
            r = r,
            k = 0,
            train_mode = True,
            beta = beta
        )
        
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            if train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'x' in state:
                        p.copy_(state['x'])
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            if not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'y' in state:
                        # Set p to y
                        p.copy_(state['y'])
                group['train_mode'] = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        for group in self.param_groups:
            lr = group['lr']
            k = group['k']
            beta = group['beta']

            ckp1 = 1 / (k+1)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if 'z' not in state:
                    state['z'] = torch.clone(p, memory_format=torch.preserve_format)
                    state['x'] = torch.clone(p, memory_format=torch.preserve_format)
                    state['y'] = torch.clone(p, memory_format=torch.preserve_format)

                z = state['z']
                x = state['x']
                y = state['y']

                # 3 z_{t+1} = z_t - lr * g_t
                z.sub_(grad, alpha=lr)

                # 4 x_{t+1} = (1-ckp1) * x_t + ckp1 * z_{t+1}
                x.mul_(1-ckp1).add_(z, alpha=ckp1)

                # 5 y_t = (1-beta) * z_t + beta * x_t
                y.copy_(x.mul(beta).add_(z, alpha=1-beta))

                # 6 p = y_t
                p.copy_(y)

            group['k'] = k+1
        return loss