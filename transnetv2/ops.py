import torch
from torch import Tensor, nn


class Conv2plus1D(nn.Sequential):
    # (2+1)D convolution https://arxiv.org/pdf/1711.11248.pdf
    def __init__(self, in_channels, multiplier, kernel_side=3, dilation=1, bias=True, device=None, dtype=None):
        self.dilation = dilation
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

    def forward_ane(self, x):
        b, c, t, h, w = x.shape
        x = torch.transpose(x, 1, 2).flatten(start_dim=0, end_dim=1)  # -> b t c h w -> bt c h w
        x = nn.functional.conv2d(x, self[0].weight[:, :, 0], padding=1)  # -> bt c1 h w
        x = x.view(b, t, -1, h, w).permute(0, 3, 4, 2, 1).flatten(0, 2)  # -> bhw c1 x t
        x = nn.functional.conv1d(x, self[1].weight[:, :, :, 0, 0], padding=self.dilation, dilation=self.dilation)
        x = x.view(b, h, w, -1, t).permute(0, 3, 4, 1, 2)
        return x


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

    def forward_ane(self, inputs):
        return torch.cat([layer.forward_ane(inputs) for layer in self.layers], dim=1)


class BatchNorm3d(nn.BatchNorm3d):
    def forward_ane(self, x: Tensor) -> Tensor:
        b, c, t, h, w = x.shape
        x = nn.functional.batch_norm(input=x.view(b, c, t, h * w),
                                     running_mean=self.running_mean,
                                     running_var=self.running_var,
                                     weight=self.weight,
                                     bias=self.bias,
                                     eps=1e-3)
        return x.view(b, c, t, h, w)


class ReLU(nn.ReLU):
    def forward_ane(self, x: Tensor) -> Tensor:
        return self.forward(x)


class Sequential(nn.Sequential):
    def forward_ane(self, x):
        for module in self:
            x = module.forward_ane(x)
        return x
