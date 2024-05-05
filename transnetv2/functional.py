import torch
import torch.nn.functional as functional
from torch.nn.functional import *
import numpy as np
from tqdm import tqdm


def batched_bincount(inputs: torch.LongTensor, min_length: int = 512) -> torch.Tensor:
    reference = torch.eye(min_length, device=inputs.device)
    return reference[inputs.long()].sum(1)


@torch.jit.script
def get_bins(inputs: torch.LongTensor):
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


def non_maximum_suppression(sequence, radius: int = 25):
    sequence = np.array(sequence)
    i = 0
    while i < len(sequence) - radius:
        index = np.argmax(sequence[i:i + radius]) + i
        value = sequence[index]
        sequence[i:i + radius] = 0.
        sequence[index] = value
        i = max(index, i + 1)

    index = np.argmax(sequence[-radius:]) + len(sequence) - radius
    value = sequence[index]
    sequence[i:i + radius] = 0.
    sequence[index] = value
    return sequence


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_sliding_window(inputs, kernel_size=100, stride=50, batch_size=1, padding=1):
    def calculate_pad(n, step_size=stride):
        return int(step_size / 2), int(step_size / 2 + np.ceil(n / step_size) * step_size - n)
    if padding != 0:
        pad_left, pad_right = calculate_pad(len(inputs))
        inputs = np.concatenate([inputs[:1]] * pad_left + [inputs] + [inputs[-1:]] * pad_right, axis=0)
    for i in tqdm(range(0, len(inputs) - stride * (batch_size + 1), stride * batch_size)):
        index = (np.arange(batch_size) * stride + i)[:, None] + np.arange(kernel_size)[None]
        yield inputs[index].transpose(0, 4, 1, 2, 3)
