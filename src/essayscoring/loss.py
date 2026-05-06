import torch
import torch.nn as nn


class OrdinalLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        num_thresholds = logits.shape[1]
        targets = torch.zeros_like(logits)
        for j in range(num_thresholds):
            targets[:, j] = (labels > j).float()
        targets = targets * (1 - self.smoothing) + (1 - targets) * self.smoothing
        return self.bce(logits, targets)
