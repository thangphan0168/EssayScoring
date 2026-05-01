import torch
import torch.nn as nn


class OrdinalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        num_thresholds = logits.shape[1]
        targets = torch.zeros_like(logits)
        for j in range(num_thresholds):
            targets[:, j] = (labels > j).float()
        return self.bce(logits, targets)
