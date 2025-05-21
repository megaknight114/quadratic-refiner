# filename: binary_over_under_classifier_no_norm.py
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

CSV_PATH   = Path("/home/xuzonghuan/quadratic-refiner/train_data/train_data_with_preds0.csv")
BATCH_SIZE = 64
EPOCHS     = 300
LR         = 1e-3
TRAIN_SPLIT = 0.7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# ------------------------ Data ------------------------ #
class QuadRefineDataset(Dataset):
    def __init__(self, csv_path: Path):
        df = pd.read_csv(csv_path)

        # 标签：prediction > target → 1 (过大) ，否则 0
        self.y = (df["prediction"].values > df["target"].values).astype(np.float32)

        # 原始特征 x1,x2,x3,prediction
        self.X = df[["x1", "x2", "x3", "prediction"]].values.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )

full_ds = QuadRefineDataset(CSV_PATH)
train_len = int(len(full_ds) * TRAIN_SPLIT)
val_len   = len(full_ds) - train_len
train_ds, val_ds = random_split(full_ds, [train_len, val_len])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# ------------------------ Model ------------------------ #
class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),   # logits
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # (B,) logits

model = MLP(in_dim=4).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ------------------------ Train / Validate ------------------------ #
def run_epoch(loader, training: bool):
    model.train() if training else model.eval()
    epoch_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            loss   = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item() * y.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total   += y.size(0)

    return epoch_loss / total, correct / total

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, training=True)
    val_loss,   val_acc   = run_epoch(val_loader,   training=False)
    print(f"Epoch {epoch:02d} | "
          f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "
          f"val loss {val_loss:.4f}, acc {val_acc:.4f}")


print("Training finished")
