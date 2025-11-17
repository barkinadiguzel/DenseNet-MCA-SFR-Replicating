import torch
import torch.nn as nn

class MultiScaleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=False)
        self.conv7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, bias=False)

        self.w1 = nn.Parameter(torch.ones(1))
        self.w3 = nn.Parameter(torch.ones(1))
        self.w5 = nn.Parameter(torch.ones(1))
        self.w7 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)

        fine = self.w1 * x1 + self.w3 * x3
        coarse = self.w5 * x5 + self.w7 * x7

        fine_stack = torch.stack([self.w1 * x1, self.w3 * x3], dim=0)
        fine_max, _ = torch.max(fine_stack, dim=0)

        coarse_stack = torch.stack([self.w5 * x5, self.w7 * x7], dim=0)
        coarse_max, _ = torch.max(coarse_stack, dim=0)

        out = torch.cat([fine_max, coarse_max], dim=1)
        return out
