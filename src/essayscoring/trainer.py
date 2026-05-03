import numpy as np
import torch
from torch.optim import AdamW
from transformers import Trainer
from sklearn.metrics import cohen_kappa_score


def compute_metrics(eval_pred):
    """
    Converts raw logits → ordinal scores, then computes QWK.
    Passed directly to Trainer as compute_metrics=compute_metrics.
    """
    logits, labels = eval_pred
    preds = (logits > 0).sum(axis=-1).astype(int)
    return {"cohen_kappa": cohen_kappa_score(labels.astype(int), preds, weights="quadratic")}


class OrdinalTrainer(Trainer):
    """
    Subclasses HuggingFace Trainer so that:
      • compute_loss delegates to the model's own loss_fn
      • predict() converts cumulative logits → integer scores
    """
    def __init__(self, *args, backbone_lr: float, head_lr: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone_lr = backbone_lr
        self.head_lr = head_lr
 
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss
    
    def create_optimizer(self, model=None):
        model = self.model if model is None else model
        backbone_lr = self.backbone_lr
        head_lr = self.head_lr

        param_groups = [
            {
                "params": model.backbone.parameters(), # type:ignore
                "lr": backbone_lr,
            },
            {
                "params": model.scoring_head.parameters(), # type:ignore
                "lr": head_lr,
            },
        ]

        self.optimizer = AdamW(
            param_groups,
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer
 
    @staticmethod
    def logits_to_scores(logits: torch.Tensor) -> torch.Tensor:
        """
        Convert ordinal logits → integer scores.
        Each threshold k is "active" when sigmoid(logit_k) > 0.5, i.e. logit_k > 0.
        Score = number of active thresholds (0 … num_thresholds).
        """
        return (logits > 0).sum(dim=-1)
