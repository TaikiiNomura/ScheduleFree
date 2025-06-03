from typing import Tuple, Union, Optional, Iterable, Dict, Callable, Any
from typing_extensions import TypeAlias
import torch
import torch.optim
try:
    from torch.optim.optimizer import ParamsT
except ImportError:
    ParamsT : TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]
import math

class AdamWScheduleFree(torch.optim.Optimizer):
    def __init__(
            self,
            params: ParamsT,
            lr: Union[float, torch.Tensor] = 0.0025,
            betas: Tuple[float, float] = (0.9, 0.999), # 第一引数=重み付け係数、第二引数=二次モーメント係数
            eps: float = 1e-8,
            weight_decay: float = 0,
            # warmup_steps: int = 0, # ウォームアップ期間のステップ数
            # r: float = 0,# 学習履歴の重み係数（多項式べき）
            # weight_lr_power: float = 2,
            # decay_at_z: bool = False,
            # foreach: Optional[bool] = False, # ツカワナイ
        ):

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            # r=r,
            k=0,
            # warmup_steps=warmup_steps,
            train_mode = False,
            # weight_sum=0.0,
            # lr_max=-1.0,
            # scheduled_lr=0.0,
            # weight_lr_power=weight_lr_power,
            # decay_at_z=decay_at_z,
            weight_decay=weight_decay
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

                y.copy_(x.mul(beta1).add_(z, alpha=1-beta1))

                # モデルのパラメータ更新
                p.copy_(y)

            group['k'] = k+1
        return loss