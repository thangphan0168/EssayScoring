import pandas as pd
from torch.utils.data import Dataset


class EssayDataset(Dataset):
    """Loads a CSV with columns: essay_id, full_text, score."""
 
    def __init__(self, csv_path: str, tokenizer, max_length=1024):
        df = pd.read_csv(csv_path)
        self.texts = df["full_text"].tolist()
        self.labels = df["score"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
 
    def __len__(self):
        return len(self.texts)
 
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": self.labels[idx] - 1, # Change labels from 1-6 to 0-5
        }
