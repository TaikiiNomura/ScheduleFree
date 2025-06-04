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

class AdamWScheduleFree(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr = 0.0025,
            betas = (0.9, 0.999),
            eps = 1e-8,
            weight_decay = 0,
            num_schedulefree=0.9
        ):

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            k=0,
            train_mode = False,
            weight_decay=weight_decay,
            num_schedulefree=num_schedulefree
        )
        
        super().__init__(params, defaults)

    # --- モード切替用 ---
    @torch.no_grad()
    def eval(self):
        first = True

        for group in self.param_groups:
            train_mode = group['train_mode']

            if train_mode:
                if first:
                    print(f"Switching to eval mode")
                for p in group['params']:
                    state = self.state[p]

                    if 'x' in state:
                        # Switch p to x
                        p.copy_(state['x'])
                group['train_mode'] = False
                first = False

    @torch.no_grad()
    def train(self):
        first = True

        for group in self.param_groups:
            train_mode = group['train_mode']
            if not train_mode:
                if first:
                    print(f"Switching to train mode")
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

        for group in self.param_groups:
            eps = group['eps']
            lr = max(group['lr'], eps)
            decay = group['weight_decay']
            beta1, beta2 = group['betas']
            k = group['k']
            num_schedulefree = group['num_schedulefree']
            
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            # r = group['r']
            # warmup_steps = group['warmup_steps']
            # weight_lr_power = group['weight_lr_power']
            # decay_at_z = group['decay_at_z']

            #  --- 学習率スケジューリング ---
            # if k < warmup_steps:
            #   sched = (k+1) / warmup_steps
            # else:
            #   sched = 1.0
            # lr = group['lr']*sched
            # group['scheduled_lr'] = lr
            # lr_max = group['lr_max'] = max(lr, group['lr_max'])

            # --- 重み付け係数 ckp1 の計算 ---
            # weight = ((k+1)**r) * (lr_max**weight_lr_power)
            # weight_sum = group['weight_sum'] = group['weight_sum'] + weight
            # try:
            #     ckp1 = weight/weight_sum
            # except ZeroDivisionError:
            #     ckp1 = 0
            ckp1 = 1 / (k+1)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                # --- 必要な状態変数の初期化 ---
                if 'z' not in state:
                    state['z'] = torch.clone(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['x'] = torch.clone(p, memory_format=torch.preserve_format)
                    state['y'] = torch.clone(p, memory_format=torch.preserve_format)

                exp_avg_sq = state['exp_avg_sq']
                z = state['z']
                x = state['x']
                y = state['y']

                # --- 正則化 ---
                # if decay != 0:
                #     decay_point = z if decay_at_z else y
                #     z.sub_(decay_point, alpha=lr*decay)
                z.sub_(y, alpha=lr*decay)

                # --- 二次モーメント更新 ---
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                denom = exp_avg_sq.div( 1 - beta2 ** (k + 1) ).sqrt_().add_(eps)

                # --- 勾配方向への更新 ---
                z.addcdiv_(grad, denom, value=-lr)

                # --- 移動平均の更新 ---
                x.mul_(1-ckp1).add_(z, alpha=ckp1)

                y.copy_(x.mul(num_schedulefree).add_(z, alpha=1-num_schedulefree))

                # モデルのパラメータ更新
                p.copy_(y)

            group['k'] = k+1
        return loss