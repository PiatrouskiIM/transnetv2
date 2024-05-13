import torch
import torch.nn.functional as functional
from torch.nn.functional import *
import numpy as np


def batched_bincount(inputs: torch.LongTensor, min_length: int = 512) -> torch.Tensor:
    reference = torch.eye(min_length, device=inputs.device)
    return reference[inputs.long()].sum(1)


@torch.jit.script
def get_bins(inputs):#: torch.LongTensor): # TODO: check if conversion is stil works
    # input shape B x 3 x N
    x = torch.divide(input=inputs, other=32, rounding_mode='trunc')
    return x[:, 0] * 64 + x[:, 1] * 8 + x[:, 2]


def get_color_histograms(inputs):
    # input shape B x C X H x W
    x = torch.flatten(inputs, start_dim=2).long()
    x = get_bins(x)
    x = batched_bincount(x)
    return x


def limited_pairwise_similarities(x, kernel_size):
    index = torch.unsqueeze(torch.arange(x.size(1), device=x.device), dim=1) + torch.unsqueeze(
        torch.arange(kernel_size, device=x.device), dim=0)
    a = functional.pad(x, pad=[0, 0, int((kernel_size - 1) // 2), int((kernel_size - 1) // 2)])[:, index]
    b = torch.unsqueeze(x, dim=-1)
    return torch.squeeze(torch.matmul(a, b), dim=-1)
