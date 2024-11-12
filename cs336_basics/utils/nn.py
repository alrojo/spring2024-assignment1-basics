import torch
from typing import Iterable

def remove_max_features(in_features, dim):
    max_features, _ = torch.max(in_features, dim=dim, keepdims=True)
    normalized_features = in_features - max_features
    return normalized_features

def softmax(in_features: torch.FloatTensor, dim: int) -> torch.FloatTensor:
    normalized_features = remove_max_features(in_features, dim)
    exp_features = torch.exp(normalized_features)
    return exp_features / torch.sum(exp_features, dim=dim, keepdims=True)

def cross_entropy(inputs: torch.FloatTensor, targets: torch.LongTensor):
    if inputs.dim()>2:
        collapsed_dim = (-1, inputs.shape[-1])
        inputs = inputs.view(collapsed_dim)
        targets = targets.view(-1)
    # sum_i z_(i, targets_i) - sum_i log(sum_j exp z_j)
    normalized_features = remove_max_features(inputs, dim=1)
    N, d = inputs.shape
    z_i_targets = normalized_features[torch.arange(N), targets] 
    z_targets = torch.mean(z_i_targets)

    log_sum_exp_i = torch.log(torch.exp(normalized_features).sum(dim=-1))
    log_sum_exp = torch.mean(log_sum_exp_i)
    nll = - (z_targets - log_sum_exp)
    return nll

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    eps=1e-6
    grads = [p.grad for p in parameters if p.grad is not None]
    two_norm_grads = torch.sqrt(
            torch.sum(
                torch.stack([torch.sum(g**2) for g in grads])
                )
            )
    if two_norm_grads>max_l2_norm:
        scaling = max_l2_norm/(two_norm_grads + eps)
        for g in grads:
            g.data.mul_(scaling)
