"""
Temporal Transformer + Flow Embedding
IoT Device Identification
GPU Optimized with High Accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import *
from src.models.lstm import IoTSequenceDataset, Trainer

# =========================================================
# Positional Encoding
# =========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# =========================================================
# Flow Embedding
# =========================================================
class FlowEmbedding(nn.Module):
    def __init__(self, input_size, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# =========================================================
# Temporal Transformer Classifier
# =========================================================
class TransformerClassifier(nn.Module):
    def __init__(self, input_size, num_classes, seq_length):
        super().__init__()
        d_model = 512
        nhead = 16
        num_layers = 4
        dim_feedforward = 1024
        dropout = 0.1

        # Flow embedding
        self.flow_embedding = FlowEmbedding(input_size, d_model, dropout)

        # Temporal projection
        self.temporal_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length + 1, dropout=dropout)

        # Transformer Encoder Layer (Pre-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.Tanh(),
            nn.Linear(d_model//2, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.flow_embedding(x)
        x = self.temporal_projection(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        attn_weights = torch.softmax(self.attention_pool(x), dim=1)
        x = torch.sum(attn_weights * x, dim=1)
        out = self.classifier(x)
        return out

# =========================================================
# Mixup Data Augmentation
# =========================================================
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# =========================================================
# Training Function
# =========================================================
def train_transformer(data_path, save_path, epochs=NUM_EPOCHS):
    data_path = Path(data_path)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train = np.load(data_path / "X_train.npy")
    X_val = np.load(data_path / "X_val.npy")
    X_test = np.load(data_path / "X_test.npy")
    y_train = np.load(data_path / "y_train.npy")
    y_val = np.load(data_path / "y_val.npy")
    y_test = np.load(data_path / "y_test.npy")

    input_size = X_train.shape[2]
    seq_length = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    print(f"Input size: {input_size}, Sequence length: {seq_length}, Classes: {num_classes}")

    # Datasets
    train_dataset = IoTSequenceDataset(X_train, y_train)
    val_dataset = IoTSequenceDataset(X_val, y_val)
    test_dataset = IoTSequenceDataset(X_test, y_test)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    # Model
    model = TransformerClassifier(input_size, num_classes, seq_length)
    trainer = Trainer(model)
    device = trainer.device
    print(f"\nTraining on device: {device}")

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min((step+1)/warmup_steps, 0.5*(1+np.cos(np.pi*(step-warmup_steps)/(total_steps-warmup_steps)))) if step >= warmup_steps else (step+1)/warmup_steps
    )

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Training loop
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x, y_a, y_b, lam = mixup_data(x, y)
            optimizer.zero_grad()
            outputs = model(x)
            loss = lam*criterion(outputs, y_a) + (1-lam)*criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * x.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (lam*preds.eq(y_a).sum().item() + (1-lam)*preds.eq(y_b).sum().item())
            total += x.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item() * x.size(0)
                _, preds = torch.max(outputs, 1)
                correct += preds.eq(y).sum().item()
                total += x.size(0)
        val_loss /= total
        val_acc = correct / total
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # Test evaluation
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            test_loss += loss.item() * x.size(0)
            _, preds = torch.max(outputs, 1)
            correct += preds.eq(y).sum().item()
            total += x.size(0)
    test_loss /= total
    test_acc = correct / total
    print(f"\nTest Accuracy: {test_acc:.4f}")

    # Save history
    with open(save_path / "history.json", "w") as f:
        json.dump(history, f)

    # Save results
    results = {
        "model": "Temporal Transformer",
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "best_val_accuracy": max(history["val_acc"]),
        "num_classes": num_classes,
        "input_size": input_size,
        "sequence_length": seq_length
    }
    with open(save_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return model, history

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    train_transformer(
        PROCESSED_DATA_DIR,
        RESULTS_DIR / "models" / "transformer_temporal_high_acc"
    )
