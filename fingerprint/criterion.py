import torch.nn as nn
import torch 
import torch.nn.functional as F 

class Criterion(nn.Module):
    def __init__(slef, margin=2.0):
        super().__init__()
        slef.margin = margin

    def forward(self, out1, out2, label):
        euclidean_distance= F.pairwise_distance(out1, out2, keepdim=True)
        # positive = 0
        # negative = 1
        loss = torch.mean((1-label)*torch.pow(euclidean_distance, 2) + 
        label * torch.pow(torch.clamp(self.margin - euclidean_distance, min = 0.0), 2))

        return loss 