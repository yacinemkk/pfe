"""
Transformer Model for IoT Device Identification
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
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
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
        self, input_size, num_classes, seq_length=SEQUENCE_LENGTH, config=None
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
            d_model, max_len=seq_length, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        output = self.classifier(x)
        return output


def train_transformer(data_path, save_path, epochs=NUM_EPOCHS):
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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = TransformerClassifier(input_size, num_classes, seq_length)
    trainer = Trainer(model)

    print(f"\nTraining on: {trainer.device}")
    history = trainer.fit(train_loader, val_loader, epochs=epochs, save_path=save_path)

    test_loss, test_acc = trainer.evaluate(test_loader, nn.CrossEntropyLoss())
    print(f"\nTest Accuracy: {test_acc:.4f}")

    with open(save_path / "history.json", "w") as f:
        json.dump(history, f)

    results = {
        "model": "Transformer",
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "best_val_accuracy": max(history["val_acc"]),
        "num_classes": num_classes,
        "input_size": input_size,
        "sequence_length": seq_length,
    }

    with open(save_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return model, history


if __name__ == "__main__":
    train_transformer(PROCESSED_DATA_DIR, RESULTS_DIR / "models" / "transformer")
