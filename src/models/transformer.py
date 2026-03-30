"""
Transformer Classifier for IoT Device Identification
======================================================
Architecture per docs/architectures §4 (MIND-IoT framework):

  Input → Linear Projection → Positional Encoding
    └── Transformer Encoder (6 layers)
         Each layer:
           ├── Multi-Head Self-Attention (4 heads, d_model=128)
           ├── Residual connection + LayerNorm
           ├── Feed-Forward Network (512 dims, GELU activation)
           └── Residual connection + LayerNorm
    └── Mean Pooling
    └── Classifier head → num_classes

Note: The paper (MIND-IoT) uses d_model=768, 12 heads, FFN=3072 dims.
      We use d_model=128, 4 heads, FFN=512 dims for practical GPU training
      while keeping the full 6-layer encoder depth as specified.
      These can be overridden via TRANSFORMER_CONFIG in config.py.
"""

import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import *
from src.models.lstm import IoTSequenceDataset, Trainer


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (as in the original Transformer paper)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    Encoder-only Transformer classifier.

    Per docs/architectures §4 (MIND-IoT):
    - 6 stacked encoder layers
    - Multi-Head Self-Attention (Query × Key → weights × Value)
    - FFN with GELU activation
    - Residual connections + LayerNorm (training stability)
    - Mean Pooling → FC head → classification
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        seq_length: int = SEQUENCE_LENGTH,
        config: dict = None,
    ):
        super().__init__()

        config = config or TRANSFORMER_CONFIG

        d_model          = config.get("d_model", 128)
        nhead            = config.get("nhead", 4)
        num_enc_layers   = config.get("num_encoder_layers", 6)   # 6 per doc
        dim_feedforward  = config.get("dim_feedforward", 512)
        dropout          = config.get("dropout", 0.2)

        # Project raw features to d_model
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length + 50, dropout=dropout)

        # Transformer encoder stack (6 layers per MIND-IoT)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",    # GELU per doc
            batch_first=True,
            norm_first=False,     # Post-norm (original Transformer)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_enc_layers
        )

        # Final layer normalisation
        self.norm = nn.LayerNorm(d_model)

        # Classifier head (Mean Pooling → FC)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = self.input_projection(x)      # → (B, T, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)   # 6 encoder layers
        x = self.norm(x)
        x = x.mean(dim=1)                 # Mean Pooling over time
        return self.classifier(x)


class NLPTransformerClassifier(nn.Module):
    """
    NLP-style Transformer that takes BPE token sequences.
    Per docs/architectures §4: La couche d'intégration utilise un 
    vocabulaire de 52 000 jetons.
    """
    def __init__(self, vocab_size, num_classes, seq_length=512, config=None):
        super().__init__()
        config = config or TRANSFORMER_CONFIG
        d_model = config.get("d_model", 128)
        nhead = config.get("nhead", 4)
        num_enc_layers = config.get("num_encoder_layers", 6)
        dim_feedforward = config.get("dim_feedforward", 512)
        dropout = config.get("dropout", 0.2)

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=2) # <pad> = 2
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length + 50, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_enc_layers)
        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch, seq_len) of token indices
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.classifier(x)

class LabelSmoothingLoss(nn.Module):
    """Label smoothing cross-entropy — helpful for the Transformer."""

    def __init__(self, classes: int, smoothing: float = 0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing  = smoothing
        self.cls        = classes

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
    X_val   = np.load(data_path / "X_val.npy")
    X_test  = np.load(data_path / "X_test.npy")
    y_train = np.load(data_path / "y_train.npy")
    y_val   = np.load(data_path / "y_val.npy")
    y_test  = np.load(data_path / "y_test.npy")

    input_size  = X_train.shape[2]
    num_classes = len(np.unique(y_train))
    seq_length  = X_train.shape[1]

    print(f"Input size: {input_size}, Seq length: {seq_length}, Classes: {num_classes}")
    print(f"Architecture: 6×TransformerEncoder(d_model=128, heads=4, ff=512, GELU)")

    train_dataset = IoTSequenceDataset(X_train, y_train)
    val_dataset   = IoTSequenceDataset(X_val,   y_val)
    test_dataset  = IoTSequenceDataset(X_test,  y_test)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=256, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=256, num_workers=4, pin_memory=True)

    model = TransformerClassifier(input_size, num_classes, seq_length)
    trainer = Trainer(model)
    device = trainer.device
    model = model.to(device)

    print(f"\nTraining on: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = LabelSmoothingLoss(num_classes, smoothing=0.1)
    history = {"train_acc": [], "val_acc": []}
    best_val = 0

    for epoch in range(epochs):
        model.train()
        total, correct = 0, 0
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()

        scheduler.step()
        train_acc = correct / total
        val_loss, val_acc = trainer.evaluate(val_loader, nn.CrossEntropyLoss())
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), save_path / "best_transformer.pt")

    test_loss, test_acc = trainer.evaluate(test_loader, nn.CrossEntropyLoss())
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
        "architecture": "6×TransformerEncoder(d_model=128, nhead=4, ff=512, GELU) → MeanPool → FC",
    }

    with open(save_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return model, history


if __name__ == "__main__":
    train_transformer(PROCESSED_DATA_DIR, RESULTS_DIR / "models" / "transformer")
