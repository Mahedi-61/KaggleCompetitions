import torch
import torch.nn  
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin = 1.0):
        super().__init__()
        self.margin = margin 
        self.eps = 1e-9

    def check_type_forward(self, inputs):
        assert len(inputs) == 3

        x0, x1, y = inputs 
        assert x0.size() == x1.size()
        assert x0.size()[0] == y.shape[0]
        assert x0.dim() == 2
        assert y.dim() == 1 

    def forward(self, x0, x1, y):
        self.check_type_forward((x0, x1, y))

        # euclidean distance
        dist_sq = torch.sum(torch.pow(x0 - x1, 2), dim=1)
        mid_dist = self.margin - torch.sqrt(dist_sq + self.eps)

        true_loss =  y * (dist_sq)
        wrong_loss = (1 - y) * torch.pow(F.relu(mid_dist), 2)

        return (true_loss + wrong_loss).sum() / 2.0