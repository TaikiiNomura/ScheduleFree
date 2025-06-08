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

class AdinaW(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr = 1e-3,
            beta1 = 0.9,
            beta2 = 0.999,
            eps = 1e-8,
            a = 0.1,
            b = 0.9,
            weight_decay = 0.0,
    ):
        
        defaults = dict(
            lr = lr,
            beta1 = beta1,
            beta2 = beta2,
            eps = eps,
            a = a,
            b = b,
            weight_decay = weight_decay
        )
        
        super(AdinaW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            a = group["a"]
            b = group["b"]
            weight_decay = group["weight_decay"]

            gamma = (1.0 - a * b) / a

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m = state["m"]
                v = state["v"]
                state["step"] += 1
                step = state["step"]

                alpha = lr * (1.0 - beta2 ** step) ** 0.5

                # m ← m + (γ * grad - m) * (1 - β1)
                m.mul_(beta1).add_(grad, alpha=(1 - beta1) * gamma)

                # v ← v + (grad^2) * (1 - β2)
                v.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

                # bias correction for m
                bias_correction1 = 1.0 - beta1 ** step

                # tmp ← grad * b
                tmp = grad.clone().mul_(b)
                tmp.add_(m, alpha=1.0 / bias_correction1)  # tmp ← grad * b + m / (1 - β1^t)

                denom = v.clone().sqrt_().add_(eps)

                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-weight_decay * lr)

                p.data.addcdiv_(tmp, denom, value=-alpha)

        return loss
