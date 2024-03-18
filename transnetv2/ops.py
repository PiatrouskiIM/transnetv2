import torch
import torch.nn as nn


class Conv2plus1D(nn.Sequential):
    # (2+1)D convolution https://arxiv.org/pdf/1711.11248.pdf
    def __init__(self,
                 in_channels,
                 multiplier,
                 kernel_side=3,
                 dilation=1,
                 bias=True,
                 device=None,
                 dtype=None):
        super(Conv2plus1D, self).__init__(nn.Conv3d(in_channels=in_channels,
                                                    out_channels=2 * multiplier,
                                                    kernel_size=(1, kernel_side, kernel_side),
                                                    dilation=(1, 1, 1),
                                                    padding=(0, 1, 1),
                                                    bias=False,
                                                    device=device,
                                                    dtype=dtype),
                                          nn.Conv3d(in_channels=2 * multiplier,
                                                    out_channels=multiplier,
                                                    kernel_size=(kernel_side, 1, 1),
                                                    dilation=(dilation, 1, 1),
                                                    padding=(dilation, 0, 0),
                                                    bias=bias,
                                                    device=device,
                                                    dtype=dtype))


class MultiScaleConv2plus1D(nn.Module):
    def __init__(self, in_channels, multiplier, n_scales, kernel_side=3, bias=True, device=None, dtype=None):
        super(MultiScaleConv2plus1D, self).__init__()
        self.layers = nn.ModuleList([Conv2plus1D(in_channels=in_channels,
                                                 multiplier=multiplier,
                                                 kernel_side=kernel_side,
                                                 dilation=2 ** i,
                                                 bias=bias,
                                                 device=device,
                                                 dtype=dtype) for i in range(n_scales)])

    def forward(self, inputs):
        return torch.cat([layer(inputs) for layer in self.layers], dim=1)
