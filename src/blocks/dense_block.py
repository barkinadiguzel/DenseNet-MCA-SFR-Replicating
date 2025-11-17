import torch
import torch.nn as nn
from src.layers.dense_layer import DenseLayer
from src.layers.multi_scale_layer import MultiScaleLayer
from src.layers.stochastic_reuse import StochasticReuse

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, reuse_prob=0.8, use_mca=True):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.use_mca = use_mca
        self.reuse = StochasticReuse(reuse_prob)

        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            if use_mca and i == 0: 
                self.layers.append(MultiScaleLayer(layer_in_channels, growth_rate))
            else:
                self.layers.append(DenseLayer(layer_in_channels, growth_rate))

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            out = self.reuse(out)  
            x = out  
        return x
