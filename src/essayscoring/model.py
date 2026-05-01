import torch
import torch.nn as nn
from transformers import AutoModel

from essayscoring.loss import OrdinalLoss


class ScoringModel(nn.Module):
    def __init__(self, model_name, dropout=0.1, num_thresholds=5):
        super().__init__()        
        self.backbone = AutoModel.from_pretrained(model_name, dtype=torch.float)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.scoring_head = nn.Linear(hidden_size, num_thresholds)
        self.loss_fn = OrdinalLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.backbone(input_ids=input_ids, 
                               attention_mask=attention_mask)
        hidden_state = output.last_hidden_state[:, -1, :] # Last token
        hidden_state = self.dropout(hidden_state)
        logits = self.scoring_head(hidden_state)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
 
        return {"loss": loss, "logits": logits}


