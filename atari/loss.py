import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedMSE(nn.Module):
    def __init__(self):
        super(WeightedMSE, self).__init__()
    
    def forward(self, inputs, targets, weight=1):
        return weight*(targets - inputs).pow(2)