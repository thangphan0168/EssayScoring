import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from essayscoring.loss import OrdinalLoss


class ScoringModel(nn.Module):
    def __init__(self, model_name, dropout=0.1, num_thresholds=5, from_pretrained=True):
        super().__init__()
        if from_pretrained:
            self.backbone = AutoModel.from_pretrained(model_name, dtype=torch.float)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_config(config, dtype=torch.float)
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

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str, model_name: str, dropout: float = 0.1):
        import os
        from safetensors.torch import load_file

        state_dict = load_file(os.path.join(checkpoint_dir, "model.safetensors"))
        num_thresholds = state_dict["scoring_head.weight"].shape[0]

        model = cls(model_name, dropout=dropout, num_thresholds=num_thresholds, from_pretrained=False)
        model.load_state_dict(state_dict, strict=False)
        return model
