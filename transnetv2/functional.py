import torch
import torch.nn.functional as functional
from torch.nn.functional import *


@torch.jit.script
def batched_bincount(input: torch.LongTensor, minlength: int = 512) -> torch.Tensor:
    # version 0
    # return torch.vmap(partial(torch.bincount, minlength=minlength), in_dims=0, out_dims=0)(input) # unsupported by jit

    # version 1
    # batch_size = input.size(0)
    # out = torch.zeros([batch_size, minlength], device=input.device)
    # for i in range(batch_size):
    #     out[i] = torch.bincount(input[i], minlength=minlength)

    # version 2
    reference = torch.eye(minlength, device=input.device)
    return reference[input].sum(1)


@torch.jit.script
def get_bins(input):
    # input shape B x 3 x N

    # rgb = torch.moveaxis(input >> 5, source=1, destination=0)
    # return (rgb[0] << 6) + (rgb[1] << 3) + rgb[2]

    x = (input / 32).long()
    return x[:, 0] * 64 + x[:, 1] * 8 + x[:, 2]


@torch.jit.script
def get_color_histograms(input):
    # input shape B x C X H x W
    x = input.int()
    x = torch.flatten(x, start_dim=2)
    x = get_bins(x)
    x = batched_bincount(x)
    return x


@torch.jit.script
def limited_pairwise_similarities(x, kernel_size):
    index = torch.unsqueeze(torch.arange(x.size(1)), dim=1) + torch.unsqueeze(torch.arange(kernel_size), dim=0)
    a = functional.pad(x, pad=[0, 0, int((kernel_size - 1) // 2), int((kernel_size - 1) // 2)])[:, index]
    b = torch.unsqueeze(x, dim=-1)
    return torch.squeeze(torch.matmul(a, b), dim=-1)
