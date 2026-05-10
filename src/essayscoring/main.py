from functools import partial

import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from essayscoring.dataloader import EssayDataset
from essayscoring.model import ScoringModel
from essayscoring.trainer import compute_metrics, OrdinalTrainer


def train(
    model_name: str = "bert-base-uncased",
    train_csv: str = "train.csv",
    eval_csv: str | None = None,
    output_dir: str = "./checkpoints",
    num_thresholds: int = 5,
    num_epochs: int = 3,
    batch_size: int = 8,
    backbone_lr: float = 1e-6,
    head_lr: float = 1e-5,
    dropout: float = 0.1,
    label_smoothing: float = 0.0,
    warmup_steps: int = 200,
    logging_steps: int = 50,
    eval_steps: int = 200,
    save_steps: int = 200,
    max_length: int = 1024,
    freeze_backbone: bool = False,
    checkpoint_dir: str | None = None,
    loss_type: str = "ordinal"
):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir or model_name)
    if checkpoint_dir:
        model = ScoringModel.from_checkpoint(
            checkpoint_dir, model_name, dropout=dropout, label_smoothing=label_smoothing, loss_type=loss_type
        )
    else:
        model = ScoringModel(model_name, dropout=dropout,
            num_thresholds=num_thresholds, label_smoothing=label_smoothing, loss_type=loss_type
        )
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
 
    train_dataset = EssayDataset(train_csv, tokenizer, max_length=max_length)
    eval_dataset = EssayDataset(eval_csv, tokenizer, max_length=max_length) if eval_csv else None
 
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=4,
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="cohen_kappa" if eval_dataset else None,
        greater_is_better=True,
        logging_steps=logging_steps,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to="none",
        warmup_steps=warmup_steps
    )
 
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
 
    trainer = OrdinalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=partial(compute_metrics, loss_type=loss_type),
        backbone_lr=backbone_lr,
        head_lr=head_lr
    )
 
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    return trainer
