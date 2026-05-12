# Essay Scoring

Automated essay scoring using transformer models with ordinal regression. Essays are scored on a 1–6 scale using either an Ordinal or CORN (Conditional Ordinal Regression for Neural networks) loss.

Data from: https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd essayscoring

# Install dependencies
uv pip install .
```

## Project Structure

```
essayscoring/
├── __init__.py        # Exports train()
├── dataloader.py      # EssayDataset – loads CSV, tokenizes text
├── loss.py            # OrdinalLoss and CORNLoss implementations
├── model.py           # ScoringModel (transformer backbone + linear head)
└── main.py            # train() entry point
```

## Data Format

Training and evaluation CSVs must have the following columns:

| Column | Type | Description |
|---|---|---|
| `essay_id` | string | Unique identifier |
| `full_text` | string | Essay body |
| `score` | int | Human score, 1–6 (inclusive) |

## Usage

```python
from essayscoring import train

# Basic training run
trainer = train(
    model_name="google/gemma-3-270m",
    train_csv="train.csv",
    eval_csv="eval.csv",
    output_dir="./checkpoints",
)
```

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `"google/gemma-3-270m"` | HuggingFace model identifier |
| `train_csv` | `"train.csv"` | Path to training data |
| `eval_csv` | `None` | Path to validation data (enables evaluation) |
| `output_dir` | `"./checkpoints"` | Where to save model checkpoints |
| `num_thresholds` | `5` | Number of ordinal thresholds (= num\_classes − 1) |
| `num_epochs` | `3` | Training epochs |
| `batch_size` | `8` | Per-device batch size |
| `backbone_lr` | `1e-6` | Learning rate for the transformer backbone |
| `head_lr` | `1e-5` | Learning rate for the scoring head |
| `dropout` | `0.1` | Dropout probability |
| `label_smoothing` | `0.0` | Label smoothing factor for the loss |
| `warmup_steps` | `200` | Linear LR warmup steps |
| `max_length` | `1024` | Max token length (essays are truncated) |
| `freeze_backbone` | `False` | Freeze backbone weights, train head only |
| `checkpoint_dir` | `None` | Resume/fine-tune from an existing checkpoint |
| `loss_type` | `"ordinal"` | Loss function: `"ordinal"` or `"corn"` |

### Fine-tuning from a Checkpoint

```python
trainer = train(
    model_name="google/gemma-3-270m",
    checkpoint_dir="./checkpoints-frozen-backbone",
    train_csv="train.csv",
    eval_csv="eval.csv",
    output_dir="./checkpoints-finetuned",
    backbone_lr=1e-6,
    head_lr=1e-5,
    warmup_steps=200,
)
```

### Using CORN Loss

```python
trainer = train(
    model_name="google/gemma-3-270m",
    train_csv="train.csv",
    eval_csv="eval.csv",
    output_dir="./checkpoints-corn",
    loss_type="corn",
)
```

## Evaluation Metric

The primary metric is **Quadratic Weighted Kappa (QWK)**, reported as `cohen_kappa` during training. The best checkpoint by QWK is saved automatically when `eval_csv` is provided.

---

## Experiment Results

All runs use `bert-base-uncased` unless otherwise noted. Best-checkpoint QWK is reported (higher is better).

| Run | Epochs | Backbone LR | Head LR | Warmup | Loss | Label Smoothing | Notes | **QWK** |
|---|---|---|---|---|---|---|---|---|
| checkpoints-1 | 1 | 1e-5 | 1e-5 | None | Ordinal | 0.0 | Baseline | 0.790 |
| checkpoints-2 | 5 | 1e-6 | 1e-6 | None | Ordinal | 0.0 | Longer training, lower LR | 0.797 |
| checkpoints-3 | 3 | 1e-6 | 1e-5 | None | Ordinal | 0.0 | Differential LR backbone/head | 0.805 |
| checkpoints-frozen-backbone | 3 | frozen | 1e-4 | None | Ordinal | 0.0 | Head-only training | 0.643 |
| checkpoints-4 | 3 | 2e-5 | 1e-4 | 200 | Ordinal | 0.0 | Fine-tune from frozen ckpt, fp16; loss diverges at end | 0.817 |
| checkpoints-5 | 3 | 1e-6 | 1e-5 | 200 | Ordinal | 0.0 | Same as #4, lower LR; stable | **0.831** |
| checkpoints-6 | 3 | 1e-5 | 5e-5 | 200 | Ordinal | 0.0 | Same as #4, mid LR; loss diverges at end | 0.828 |
| checkpoints-frozen-backbone-corn | 3 | frozen | 1e-4 | None | CORN | 0.0 | Head-only training with CORN loss | 0.597 |
| checkpoints-corn-1 | 3 | 1e-6 | 1e-5 | 200 | CORN | 0.0 | Fine-tune from frozen CORN ckpt, fp16 | 0.820 |
| checkpoints-corn-2 | 3 | 1e-6 | 1e-5 | 200 | CORN | 0.05 | Same as corn-1 with label smoothing | 0.810 |

### Observations

- **Best run: checkpoints-5** (QWK 0.831) — fine-tuning from a frozen-backbone checkpoint with differential learning rates (1e-6 backbone, 1e-5 head) and 200 warmup steps produces the best results. This two-stage approach (freeze → unfreeze) consistently outperforms training from scratch.
- **Loss divergence** occurs at higher learning rates (checkpoints-4 and -6)
- **CORN loss** performs worse than ordinal loss after fine-tuning (0.820 vs 0.831), need to investigate further.
