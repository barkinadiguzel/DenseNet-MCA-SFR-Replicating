import torch.nn as nn

def avgpool_layer(kernel_size=2, stride=2, padding=0):
    return nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
