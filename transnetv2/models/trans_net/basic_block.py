from torch import nn
from transnetv2 import ops


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, multiplier: int, n_blocks: int = 2):
        super(BasicBlock, self).__init__()
        self.intro = ops.Sequential(
            ops.MultiScaleConv2plus1D(in_channels=in_channels, multiplier=multiplier, n_scales=4, bias=False),
            ops.BatchNorm3d(multiplier * 4, eps=1e-3),
            ops.ReLU(inplace=True))
        self.layers = ops.Sequential(*[ops.Sequential(
            ops.MultiScaleConv2plus1D(in_channels=multiplier * 4, multiplier=multiplier, n_scales=4, bias=False),
            ops.BatchNorm3d(multiplier * 4, eps=1e-3),
            ops.ReLU(inplace=True)) for _ in range(n_blocks - 1)])

    def forward(self, x):
        x = self.intro(x)
        return x + self.layers(x)

    def forward_ane(self, x):
        x = self.intro.forward_ane(x)
        return x + self.layers.forward_ane(x)
