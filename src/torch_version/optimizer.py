import math
from typing import Tuple

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def update_lr(self, lr: float):
        for group in self.param_groups:
            group["lr"] = lr

    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                g = p.grad
                m = state["m"]
                v = state["v"]

                # update step
                state["step"] += 1

                # weight decay
                p.data.mul_(1 - lr * weight_decay)

                # momentum update, inplace operation
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # bias correction
                t = state["step"]
                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t
                bias_correction2_sqrt = math.sqrt(bias_correction2)
                lr_t = lr * bias_correction2_sqrt / bias_correction1

                grad_update = m / (v.sqrt() + eps)
                p.data.add_(grad_update, alpha=-lr_t)


def get_cosine_schedule_lr(step: int, lr_max: float, lr_min: float, warmup_steps: int, anneal_steps: int):
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    elif step < anneal_steps:
        return lr_min + 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (anneal_steps - warmup_steps))) * (
            lr_max - lr_min
        )
    else:
        return lr_min


def clip_grad_norm(params, max_norm: float, eps: float = 1e-6):
    # denominator is l2 norm of all gradients, not per parameter
    grads = [p.grad for p in params if p.grad is not None]  # Use python list to update gradients in place
    norms = torch.stack([g.norm(2) for g in grads])
    total_norm = norms.norm(2)
    coef = torch.clamp(max_norm / (total_norm + eps), max=1.0)
    for g in grads:
        g.mul_(coef)
    return total_norm
