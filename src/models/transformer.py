"""
Improved Transformer Model for IoT Device Identification
Optimized for GPU Training
"""

import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import *
from src.models.lstm import IoTSequenceDataset, Trainer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        x = x + self.pe[:, : x.size(1), :]

        return self.dropout(x)


class TransformerClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        num_classes,
        seq_length=SEQUENCE_LENGTH,
        config=None,
    ):
        super().__init__()

        config = config or TRANSFORMER_CONFIG

        d_model = config["d_model"]
        nhead = config["nhead"]
        num_encoder_layers = config["num_encoder_layers"]
        dim_feedforward = config["dim_feedforward"]
        dropout = config["dropout"]

        self.input_projection = nn.Linear(input_size, d_model)

        self.pos_encoder = PositionalEncoding(
            d_model,
            max_len=seq_length,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = self.pos_encoder(x)

        x = self.transformer_encoder(x)

        x = self.norm(x)

        x = x.mean(dim=1)

        output = self.classifier(x)

        return output


class LabelSmoothingLoss(nn.Module):

    def __init__(self, classes, smoothing=0.1):
        super().__init__()

        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes

    def forward(self, pred, target):

        log_probs = torch.nn.functional.log_softmax(pred, dim=-1)

        true_dist = torch.zeros_like(log_probs)

        true_dist.fill_(self.smoothing / (self.cls - 1))

        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


def train_transformer(data_path, save_path, epochs=30):

    data_path = Path(data_path)

    save_path = Path(save_path)

    save_path.mkdir(parents=True, exist_ok=True)

    X_train = np.load(data_path / "X_train.npy")
    X_val = np.load(data_path / "X_val.npy")
    X_test = np.load(data_path / "X_test.npy")

    y_train = np.load(data_path / "y_train.npy")
    y_val = np.load(data_path / "y_val.npy")
    y_test = np.load(data_path / "y_test.npy")

    input_size = X_train.shape[2]

    num_classes = len(np.unique(y_train))

    seq_length = X_train.shape[1]

    print(
        f"Input size: {input_size}, Sequence length: {seq_length}, Classes: {num_classes}"
    )

    train_dataset = IoTSequenceDataset(X_train, y_train)
    val_dataset = IoTSequenceDataset(X_val, y_val)
    test_dataset = IoTSequenceDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        num_workers=4,
        pin_memory=True,
    )

    model = TransformerClassifier(
        input_size,
        num_classes,
        seq_length,
    )

    trainer = Trainer(model)

    device = trainer.device

    print(f"\nTraining on: {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    criterion = LabelSmoothingLoss(num_classes, smoothing=0.1)

    scaler = torch.cuda.amp.GradScaler()

    history = {"train_acc": [], "val_acc": []}

    best_val = 0

    for epoch in range(epochs):

        model.train()

        total = 0
        correct = 0

        for x, y in tqdm(train_loader):

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():

                outputs = model(x)

                loss = criterion(outputs, y)

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)

            scaler.update()

            _, predicted = torch.max(outputs, 1)

            total += y.size(0)

            correct += (predicted == y).sum().item()

        scheduler.step()

        train_acc = correct / total

        val_loss, val_acc = trainer.evaluate(val_loader, nn.CrossEntropyLoss())

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val:

            best_val = val_acc

            torch.save(
                model.state_dict(),
                save_path / "best_transformer.pt",
            )

    test_loss, test_acc = trainer.evaluate(
        test_loader,
        nn.CrossEntropyLoss(),
    )

    print(f"\nTest Accuracy: {test_acc:.4f}")

    with open(save_path / "history.json", "w") as f:

        json.dump(history, f)

    results = {
        "model": "Transformer",
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "best_val_accuracy": best_val,
        "num_classes": num_classes,
        "input_size": input_size,
        "sequence_length": seq_length,
    }

    with open(save_path / "results.json", "w") as f:

        json.dump(results, f, indent=2)

    return model, history


if __name__ == "__main__":

    train_transformer(
        PROCESSED_DATA_DIR,
        RESULTS_DIR / "models" / "transformer",
    )
