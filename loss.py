


import torch
import torch.nn as nn

class ProtoNetLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(ProtoNetLoss, self).__init__()
        self.reduction = reduction

    def squared_euclidian_distance(self, embedding_1, embedding_2):
        return torch.sum((embedding_1 - embedding_2) ** 2, dim=-1)
    def forward(self, outputs, target_centroid, non_target_centroids):
        target_distance = self.squared_euclidian_distance(outputs, target_centroid)
        non_target_distnaces = self.squared_euclidian_distance(outputs, non_target_centroids)

        loss = torch.mean(target_distance + torch.log(sum(torch.exp((-non_target_distnaces)))))

        if self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}")