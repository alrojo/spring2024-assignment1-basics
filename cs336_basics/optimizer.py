import torch
import torch.nn as nn
import numpy as np
import math
from typing import Callable, Optional
from cs336_basics.utils.nn import softmax

def lr_schedule(t, lr_max, lr_min, T_w, T_c):
    if t <= T_w:
        return t/T_w * lr_max
    elif T_w <= t <= T_c:
        cos_term = math.cos((t-T_w)/(T_c-T_w)*math.pi)
        return lr_min + 1/2 * (1 + cos_term)*(lr_max-lr_min)
    else:
        return lr_min

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-5, eps=1e-8):
        if lr<0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr,
                    "betas": betas,
                    "weight_decay": weight_decay,
                    "eps": eps}
        # init state
        super().__init__(params, defaults)

    def set_lr(self, lr):
        for group in self.param_groups:
            group["lr"] = lr

    def step(self):#, closure: Optional[Callable]):
        #loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # get the learning rate
            beta1, beta2 = group["betas"] # get the learning rate
            weight_decay = group["weight_decay"] # get the learning rate
            eps = group["eps"] # get the learning rate
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p not in self.state:
                    self.state[p]={}
                    self.state[p]['t'] = 0
                    self.state[p]['m'] = torch.zeros_like(p.grad)
                    self.state[p]['v'] = torch.zeros_like(p.grad)
                g, t  = p.grad, self.state[p]['t']
                t = t+1
                m, v = self.state[p]['m'], self.state[p]['v']
                new_m = beta1*m + (1-beta1)*g
                new_v = beta2*v + (1-beta2)*torch.square(g)
                lr_t = lr * math.sqrt(1-beta2**t) / (1-beta1**t)
                p.data -= lr_t *new_m / (torch.torch.sqrt(new_v) + eps)
                p.data -= lr*weight_decay*p.data
                self.state[p]['t'] = t
                self.state[p]['m'] = new_m
                self.state[p]['v'] = new_v
