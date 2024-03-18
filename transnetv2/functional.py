import torch
import torch.nn.functional as functional
from torch.nn.functional import *


@torch.jit.script
def batched_bincount(input: torch.IntTensor, minlength: int = 512) -> torch.Tensor:
    # return torch.vmap(partial(torch.bincount, minlength=minlength), in_dims=0, out_dims=0)(input) # unsupported by jit
    batch_size = input.size(0)
    out = torch.zeros([batch_size, minlength], device=input.device)
    for i in range(batch_size):
        out[i] = torch.bincount(input[i], minlength=minlength)
    return out


@torch.jit.script
def get_bins(input):
    # input shape B x 3 x N

    # rgb = torch.moveaxis(input >> 5, source=1, destination=0)
    # return (rgb[0] << 6) + (rgb[1] << 3) + rgb[2]

    x = (input / 32).int()
    return x[:, 0] * 64 + x[:, 1] * 8 + x[:, 2]


@torch.jit.script
def get_color_histograms(input):
    # input shape B x C X H x W
    # x = input.int()
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


# if __name__ == "__main__":
#     batch_size = 120
#     frame_count = 15
#     kernel_size = 5
#     output_dim = 18
#     channels = 3
#     print("BATCH SIZE", batch_size,
#           "FRAME COUNT", frame_count,
#           "KERNEL_SIZE", kernel_size,
#           "OUTPUT DIM", output_dim,
#           "CHANNELS", channels)
#
#     height = 150
#     width = 200
#     no_channels = channels
#     time_window = frame_count
#     # batch_size, time_window, height, width, no_channels = frames.shape
#     assert no_channels == 3
#     frames = (torch.rand(size=[batch_size, frame_count, height, width, channels]) * 255).int()
#
#     frames_flatten = frames.view(batch_size * time_window, height * width, 3)
#
#     binned_values = get_bin(frames_flatten)
#     # binned_values2 = get_bin2(frames_flatten)
#
#     # print(torch.sum(binned_values2 - binned_values),"GET BIN")
#
#     DVS = batched_bincount(binned_values)
#
#     frame_bin_prefix = (torch.arange(0, batch_size * time_window, device=frames.device) << 9).view(-1, 1)
#     AAA = binned_values.clone()
#     binned_values = (binned_values + frame_bin_prefix).view(-1)
#
#     vvv = torch.bincount(binned_values, minlength=batch_size * time_window * 512)
#
#     histograms = torch.zeros(batch_size * time_window * 512, dtype=torch.int32, device=frames.device)
#     histograms.scatter_add_(0, binned_values, torch.ones(len(binned_values), dtype=torch.int32, device=frames.device))
#     print((histograms - vvv).max())
#     print((histograms - DVS.view(-1)).max())
#
#     # eeee = run(AAA)
#
#     # print((eeee.view(-1) - histograms).max())
#     # torch.bincount(binned_values, minlength=512)
#
#     # torch.vmap(torch.bincount, in_dims=[0,1], out_dims=[0,1])()
#
#     histograms = histograms.view(batch_size, time_window, 512).float()
#     histograms_normalized = functional.normalize(histograms, p=2, dim=2)
#     # torch.bincount(
#     UUUU = get_color_histograms(
#         torch.moveaxis(frames.view(batch_size * frame_count, height * width, 3), source=-1, destination=1))
#     print((UUUU.view(batch_size, time_window, 512) - histograms_normalized).sum())
#
#     import numpy as np
#     import torch
#     import torch.nn as nn
#     import torch.nn.functional as functional
#
#     import random
#
#     #
#
#     batch_size = 120
#     frame_count = 15
#     kernel_size = 5
#     output_dim = 18
#     channels = 10
#
#     print("BATCH SIZE", batch_size, "FRAME COUNT", frame_count, "KERNEL_SIZE", kernel_size, "OUTPUT DIM", output_dim,
#           "CHANNELS", channels)
#
#     # [:, None, None]
#     fc = nn.Linear(kernel_size, output_dim)
#     # print(np.repeat(np.arange(0, batch_size), [1, time_window, lookup_window]))
#
#     # aa = torch.arange(0, batch_size * frame_count * channels).float().view(batch_size, frame_count, channels)
#     aa = torch.rand(size=[batch_size, frame_count, channels])
#     x = aa
#
#     # b = torch.unsqueeze(a, dim=-2).repeat(1, 1, kernel_size, 1)
#     # index = torch.unsqueeze(torch.arange(frame_count), dim=1) + torch.unsqueeze(torch.arange(kernel_size), dim=0)
#     # index = torch.flatten(index)
#     #
#     # a = functional.pad(a, pad=(0, 0, (kernel_size - 1) // 2, (kernel_size - 1) // 2))[:, index]
#     #
#     # print(torch.unsqueeze(torch.flatten(a, end_dim=-2), dim=-2).size())
#     # print(torch.unsqueeze(torch.flatten(b, end_dim=-2), dim=-1).size())
#     # X = torch.matmul(torch.unsqueeze(torch.flatten(a, end_dim=-2), dim=-2),
#     #                  torch.unsqueeze(torch.flatten(b, end_dim=-2), dim=-1)).reshape(batch_size, frame_count, kernel_size)
#
#     # def limited_similarities(x, kernel_size):
#     #     index = torch.unsqueeze(torch.arange(x.shape[1] - kernel_size), dim=1) + torch.unsqueeze(torch.arange(kernel_size), dim=0)
#     #     a = x[:, index]
#     #     print(a.size(), x.size())
#     #     b = torch.unsqueeze(x, dim=-1)
#     #     print(b.size(), "b")
#     #     A = torch.squeeze(torch.matmul(a, b[:, :a.size(1)]), dim=-1)
#     #     print(A.size(), "A")
#     #     B=  functional.pad(A, pad=(0, 0, (kernel_size - 1) // 2, (kernel_size - 1) // 2))
#     #     print(B.size())
#     #     return B
#
#     X = limited_pairwise_similarities(x, kernel_size)
#
#     print(X.size(), "SSSS")
#
#     x = aa
#     batch_size, frame_count = x.shape[0], x.shape[1]
#     similarities = torch.bmm(x, x.transpose(1, 2))  # [batch_size, time_window, time_window]
#     similarities_padded = functional.pad(similarities, [(kernel_size - 1) // 2, (kernel_size - 1) // 2])
#
#     batch_indices = torch.arange(0, batch_size, device=x.device).view([batch_size, 1, 1]).repeat(
#         [1, frame_count, kernel_size])
#     time_indices = torch.arange(0, frame_count, device=x.device).view([1, frame_count, 1]).repeat(
#         [batch_size, 1, kernel_size])
#     lookup_indices = torch.arange(0, kernel_size, device=x.device).view([1, 1, kernel_size]).repeat(
#         [batch_size, frame_count, 1]) + time_indices
#     similarities = similarities_padded[batch_indices, time_indices, lookup_indices]
#
#     # print(similarities)
#
#     print(torch.linalg.norm(X - similarities, axis=[-1, -2]).max())
#     # torch.jit.trace()
#     # similarities = torch.arange(0, time_window*time_window).view(1, time_window, time_window)
#     #
#     # # similarities = torch.arange()
#     #
#     #
#     # u = 0
#     # for i in range(time_window):
#     #     for j in range(i, time_window):
#     #         similarities[:, j, i] = u
#     #         similarities[:, i, j] = u
#     #
#     #         u+=1
#
#     # batch_size, time_window = x.shape[0], x.shape[1]
#     # similarities = torch.bmm(x, x.transpose(1, 2))  # [batch_size, time_window, time_window]
#     # similarities_padded = functional.pad(similarities, [(lookup_window - 1) // 2, (lookup_window - 1) // 2])
#     # # print(similarities_padded)
#     # batch_indices = torch.arange(0, batch_size).view([batch_size, 1, 1]).repeat([1, time_window, lookup_window])
#     # time_indices = torch.arange(0, time_window).view([1, time_window, 1]).repeat([batch_size, 1, lookup_window])
#     # lookup_indices = torch.arange(0, lookup_window).view([1, 1, lookup_window]).repeat([batch_size, time_window, 1]) + time_indices
#     #
#     # # print(similarities)
#     # similarities2 = similarities_padded[batch_indices, time_indices, lookup_indices]
#     # print(similarities2.size())
#     # print(similarities)
#     # print(similarities2)
#
#     # print(similarities2[..., 0])
#     #
#     # print(similarities2[..., 1])
#     #
#     # print(similarities2[..., 0,:])
#     #
#     # print(similarities2[..., 1,:])
#
#     # cv1 = nn.Conv1d(in_channels=1, out_channels=18, kernel_size=lookup_window, stride=lookup_window)
#     # print(cv1.weight.size())
#
#     # e = functional.conv1d(input=similarities_padded.float().reshape(batch_size, 1, -1),
#     #                   weight=fc.weight.view(18, 1, 11),
#     #                   bias=fc.bias,
#     #                   stride=time_window+lookup_window)
#     #
#     #
#     # ee = fc(similarities2.float())
#     # print(e.size())
#     # print(ee.size())
#     # # print(e.transpose(1,2))
#     # print(torch.linalg.norm(ee-e.transpose(1,2)))
