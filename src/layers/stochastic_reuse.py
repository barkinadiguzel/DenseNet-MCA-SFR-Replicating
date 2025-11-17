import torch
import torch.nn as nn

class StochasticReuse(nn.Module):
    def __init__(self, reuse_prob=0.8):
        super(StochasticReuse, self).__init__()
        self.reuse_prob = reuse_prob  

    def forward(self, x):
        if not self.training:  
            return x
        mask = torch.bernoulli(torch.full((x.size(1),), self.reuse_prob)).view(1, -1, 1, 1)
        out = x * mask  
        return out
