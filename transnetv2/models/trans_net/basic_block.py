import torch.nn as nn
import torch.nn.functional as functional
from transnetv2 import ops


class BasicBlock(nn.Module):
    def __init__(self, in_channels, multiplier, n_blocks=2):
        super(BasicBlock, self).__init__()
        self.intro = nn.Sequential(
            ops.MultiScaleConv2plus1D(in_channels=in_channels, multiplier=multiplier, n_scales=4, bias=False),
            nn.BatchNorm3d(multiplier * 4, eps=1e-3))

        self.layers = nn.ModuleList([nn.Sequential(
            ops.MultiScaleConv2plus1D(in_channels=multiplier * 4, multiplier=multiplier, n_scales=4, bias=False),
            nn.BatchNorm3d(multiplier * 4, eps=1e-3)) for _ in range(n_blocks - 1)])

    def forward(self, inputs):
        y = self.intro(inputs)
        y = functional.relu(y)
        x = y
        for module in self.layers:
            x = module(x)
            x = functional.relu(x)
        return x + y
