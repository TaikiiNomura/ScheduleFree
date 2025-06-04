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

class AdinaScheduleFree(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr = 1e-3,
        beta1 = 0.9,
        beta2 = 0.99,
        eps = 1e-8,
        a = 0.1,
        b = 0.9,
        num_schedulefree = 0.9
    ):
        
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            a=a,
            b=b,
            num_schedulefree=num_schedulefree,
            train_mode=False
        )
        
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self):
        first = True
        for group in self.param_groups:
            if group['train_mode']:
                if first:
                    print("Switching to eval mode")
                for p in group['params']:
                    state = self.state[p]
                    if 'x' in state:
                        p.copy_(state['x'])
                group['train_mode'] = False
                first = False

    @torch.no_grad()
    def train(self):
        first = True
        for group in self.param_groups:
            if not group['train_mode']:
                if first:
                    print("Switching to train mode")
                for p in group['params']:
                    state = self.state[p]
                    if 'y' in state:
                        p.copy_(state['y'])
                group['train_mode'] = True
                first = False

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if not self.param_groups[0]['train_mode']:
            raise Exception("train() が呼び出されていません")

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps = group['eps']
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            a = group['a']
            b = group['b']
            num_schedulefree = group['num_schedulefree']

            ganma = (1.0 - a * b) / a

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['z'] = p.clone(memory_format=torch.preserve_format)
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['x'] = p.clone(memory_format=torch.preserve_format)
                    state['y'] = p.clone(memory_format=torch.preserve_format)

                z = state['z']
                m = state['m']
                v = state['v']
                x = state['x']
                y = state['y']

                state['step'] += 1
                step = state['step']

                """
                    m_{t+1} = beta1*m_t + (1 - beta1)*c*g_t
                    v_{t+1} = beta2*v_t + (1 - beta2)*g_t*g_t
                    m_hat = m_t / (1 - beta1**(t+1) )
                    v_hat = v_t / (1 - beta2**(t+1) )
                    mole = m_hat + b*g_t
                    denom = v_hat + eps
                    z_{t+1} = z_t + lr*mole / denom
                    x_{t+1} = (1 - ckp1)*x_t + ckp1*z_{t+1} // schedulefree
                    y_{t+1} = (1 - ckp1)*z_t + ckp1*x_t // schedulefree
                """

                ckp1 = 1.0 / (step+1)
                alpha = lr * (1.0 - beta2 ** step) ** 0.5

                # モーメントの更新
                # m ← m + (γ * grad - m) * (1 - β1)
                m.mul_(beta1).add_(grad, alpha=(1 - beta1) * ganma)
                # v ← v + (grad^2) * (1 - β2)
                v.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))


                # バイアス補正
                bias_correction1 = 1.0 - beta1 ** step

                # 勾配補正と更新
                tmp = grad.clone().mul_(b)
                tmp.add_(m, alpha=1.0 / bias_correction1)  # tmp ← grad * b + m / (1 - β1^t)
                denom = v.clone().sqrt_().add_(eps)
                z.addcdiv_(tmp, denom, value=-alpha)

                # スケジュールフリーの更新
                x.mul_(1 - ckp1).add_(z, alpha=ckp1)
                y.copy_(x.mul(num_schedulefree).add_(z, alpha=1-num_schedulefree))

                # モデルパラメータの更新
                p.copy_(y)

        return loss