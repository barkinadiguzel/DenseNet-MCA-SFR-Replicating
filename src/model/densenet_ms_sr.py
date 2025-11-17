import torch
import torch.nn as nn
from layers.conv1 import Conv1Layer
from layers.multi_scale_layer import MultiScaleLayer
from layers.stochastic_reuse import StochasticReuse
from layers.transition_layer import transition_layer
from blocks.dense_block import DenseBlock

class DenseNetMS_SR(nn.Module):
    def __init__(self, growth_rate=32, block_layers=[6,12,24,16], num_classes=1000):
        super().__init__()
        self.conv1 = Conv1Layer(3, growth_rate*2)
        self.mca = MultiScaleLayer(growth_rate*2, growth_rate*2)
        self.sfr = StochasticReuse(reuse_prob=0.8)

        num_channels = growth_rate*2
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_channels, growth_rate, num_layers)
            self.blocks.append(block)
            num_channels += num_layers * growth_rate
            if i != len(block_layers)-1:
                trans = transition_layer(num_channels, num_channels // 2)
                self.transitions.append(trans)
                num_channels = num_channels // 2

        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mca(x)
        x = self.sfr(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
