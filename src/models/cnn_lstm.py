"""
PHASE 4.3: Modèle CNN-LSTM
Per docs/important.md

Architecture:
- Couche Conv1D avec MaxPooling (extraction spatiale)
- Couche LSTM 64 neurones (return_sequences=True)
- MaxPool1D + Flatten
- Dense 100 neurones + sortie
"""

import torch
import torch.nn as nn
from typing import Optional
import yaml


class CNNLSTMClassifier(nn.Module):
    """
    Classificateur CNN-LSTM hybride.

    Architecture per docs/important.md §4.3:
    - Conv1D pour extraction spatiale
    - MaxPool1D pour réduction
    - LSTM pour modélisation temporelle
    - Dense(100) + sortie
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        cnn_channels: int = 64,
        cnn_kernel_size: int = 3,
        pool_size: int = 2,
        dense_units: int = 100,
        dropout: float = 0.3,
        config_path: str = "config/config.yaml",
    ):
        super().__init__()

        if config_path:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            model_config = config["models"]["cnn_lstm"]
            lstm_hidden = model_config.get("lstm_hidden", lstm_hidden)
            lstm_layers = model_config.get("lstm_layers", lstm_layers)
            cnn_channels = model_config.get("cnn_channels", cnn_channels)
            cnn_kernel_size = model_config.get("cnn_kernel_size", cnn_kernel_size)
            pool_size = model_config.get("pool_size", pool_size)
            dense_units = model_config.get("dense_units", dense_units)
            dropout = model_config.get("dropout", dropout)

        self.conv1d = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_channels,
            kernel_size=cnn_kernel_size,
            padding=cnn_kernel_size // 2,
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.batchnorm = nn.BatchNorm1d(cnn_channels)

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(dense_units, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_length, input_size)
        Returns:
            logits: (batch_size, num_classes)
        """
        x = x.permute(0, 2, 1)

        x = self.conv1d(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.batchnorm(x)

        x = x.permute(0, 2, 1)

        lstm_out, (hidden, cell) = self.lstm(x)

        x = lstm_out.permute(0, 2, 1)
        x = self.global_pool(x)

        return self.classifier(x)


class CNNClassifier(nn.Module):
    """
    Pure CNN classifier for IoT device identification.
    Uses 1D convolutions without LSTM for faster inference.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        cnn_channels: int = 64,
        cnn_kernel_size: int = 3,
        pool_size: int = 2,
        dense_units: int = 100,
        dropout: float = 0.3,
        config_path: str = "config/config.yaml",
    ):
        super().__init__()

        if config_path:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            model_config = config["models"]["cnn_lstm"]
            cnn_channels = model_config.get("cnn_channels", cnn_channels)
            cnn_kernel_size = model_config.get("cnn_kernel_size", cnn_kernel_size)
            pool_size = model_config.get("pool_size", pool_size)
            dense_units = model_config.get("dense_units", dense_units)
            dropout = model_config.get("dropout", dropout)

        self.conv1d = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_channels,
            kernel_size=cnn_kernel_size,
            padding=cnn_kernel_size // 2,
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.batchnorm = nn.BatchNorm1d(cnn_channels)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(cnn_channels, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(dense_units, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.batchnorm(x)
        return self.classifier(x)


class CNNLSTMConfig:
    """Configuration pour le modèle CNN-LSTM."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["models"]["cnn_lstm"]

        self.lstm_hidden = self.config.get("lstm_hidden", 64)
        self.lstm_layers = self.config.get("lstm_layers", 1)
        self.cnn_channels = self.config.get("cnn_channels", 64)
        self.cnn_kernel_size = self.config.get("cnn_kernel_size", 3)
        self.pool_size = self.config.get("pool_size", 2)
        self.dense_units = self.config.get("dense_units", 100)
        self.dropout = self.config.get("dropout", 0.3)


if __name__ == "__main__":
    batch_size = 32
    seq_length = 10
    input_size = 36
    num_classes = 18

    model = CNNLSTMClassifier(input_size, num_classes)
    x = torch.randn(batch_size, seq_length, input_size)

    output = model(x)
    print(f"Input: {x.shape} -> Output: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    assert output.shape == (batch_size, num_classes)
    print("✅ CNN-LSTM OK")
