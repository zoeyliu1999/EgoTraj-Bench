import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def modulate(x, shift, scale):
    if len(x.shape) == 3 and len(shift.shape) == 2:
        # [B, K, D] + [B, D]
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    elif len(x.shape) == len(shift.shape) == 3:
        # [B, K, D] + [B, K, D]
        return x * (1 + scale) + shift
    elif len(x.shape) == 4 and len(shift.shape) == 2:
        # [B, K, A, D] + [B, D]
        return x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(
            1
        )
    elif len(x.shape) == len(shift.shape) == 4:
        # [B, K, A, D] + [B, K, A, D]
        return x * (1 + scale) + shift
    else:
        raise ValueError("Invalid shapes to modulate")


def build_mlps(c_in, mlp_channels=None, ret_before_act=False, without_norm=False):
    layers = []
    num_layers = len(mlp_channels)

    for k in range(num_layers):
        if k + 1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[k], bias=True))
        else:
            if without_norm:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=True), nn.ReLU()])
            else:
                layers.extend(
                    [
                        nn.Linear(c_in, mlp_channels[k], bias=False),
                        nn.BatchNorm1d(mlp_channels[k]),
                        nn.ReLU(),
                    ]
                )
            c_in = mlp_channels[k]

    return nn.Sequential(*layers)
