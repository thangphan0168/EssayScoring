import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CORNLoss(nn.Module):
    def __init__(self, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        num_thresholds = logits.shape[1]
        loss = torch.zeros(1, device=logits.device, dtype=logits.dtype)

        for k in range(num_thresholds):
            # Only samples where label >= k are used for threshold k
            mask = (labels >= k)
            if mask.sum() == 0:
                continue

            logits_k = logits[mask, k]
            targets_k = (labels[mask] > k).float()
            targets_k = (targets_k * (1 - self.smoothing) + (1 - targets_k) * self.smoothing)
            loss = loss + F.binary_cross_entropy_with_logits(logits_k, targets_k)

        return loss / num_thresholds
