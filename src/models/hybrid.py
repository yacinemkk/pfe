"""
PHASE 4.6: Architecture Hybride CNN-BiLSTM-Transformer
Per docs/important.md

Architecture:
- Branche CNN 1: Conv1D kernel=3, ReLU, MaxPool1D, BatchNorm, Flatten
- Branche CNN 2: Conv1D kernel=5, ReLU, MaxPool1D, BatchNorm, Flatten
- Couche Fusion: Concaténation des 2 branches
- Module BiLSTM: Modélisation temporelle bidirectionnelle
- Module Transformer: Encodeurs avec Multi-Head Attention
- Sortie: Mean Pooling + FC + Sigmoid/Softmax
"""

import math
import torch
import torch.nn as nn
from typing import Optional
import yaml


class PositionalEncoding(nn.Module):
    """Encodage positionnel sinusoïdal."""

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
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class CNNBranch(nn.Module):
    """Branche CNN pour extraction de features locales."""

    def __init__(
        self, input_size: int, channels: int, kernel_size: int, pool_size: int
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(
            kernel_size=pool_size, stride=1, padding=pool_size // 2
        )
        self.batchnorm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.batchnorm(x)
        return x


class HybridClassifier(nn.Module):
    """
    Classificateur hybride CNN-BiLSTM-Transformer.

    Architecture per docs/important.md §4.6:
    - Deux branches CNN parallèles (noyaux 3 et 5)
    - Fusion par concaténation
    - BiLSTM pour modélisation temporelle
    - Transformer pour attention globale
    - Mean Pooling + FC pour classification
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        seq_length: int = 10,
        cnn_channels: int = 64,
        cnn_kernel_small: int = 3,
        cnn_kernel_large: int = 5,
        cnn_pool_size: int = 2,
        bilstm_hidden: int = 128,
        bilstm_layers: int = 2,
        bilstm_dropout: float = 0.3,
        transformer_d_model: int = 256,
        transformer_nhead: int = 4,
        transformer_layers: int = 2,
        transformer_ff_dim: int = 512,
        transformer_dropout: float = 0.2,
        fc_dropout: float = 0.4,
        config_path: str = "config/config.yaml",
    ):
        super().__init__()

        if config_path:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            model_config = config["models"]["hybrid"]
            cnn_channels = model_config.get("cnn_channels", cnn_channels)
            cnn_kernel_small = model_config.get("cnn_kernel_small", cnn_kernel_small)
            cnn_kernel_large = model_config.get("cnn_kernel_large", cnn_kernel_large)
            cnn_pool_size = model_config.get("cnn_pool_size", cnn_pool_size)
            bilstm_hidden = model_config.get("bilstm_hidden", bilstm_hidden)
            bilstm_layers = model_config.get("bilstm_layers", bilstm_layers)
            bilstm_dropout = model_config.get("bilstm_dropout", bilstm_dropout)
            transformer_d_model = model_config.get(
                "transformer_d_model", transformer_d_model
            )
            transformer_nhead = model_config.get("transformer_nhead", transformer_nhead)
            transformer_layers = model_config.get(
                "transformer_layers", transformer_layers
            )
            transformer_ff_dim = model_config.get(
                "transformer_ff_dim", transformer_ff_dim
            )
            transformer_dropout = model_config.get(
                "transformer_dropout", transformer_dropout
            )
            fc_dropout = model_config.get("fc_dropout", fc_dropout)

        self.cnn_branch1 = CNNBranch(
            input_size, cnn_channels, cnn_kernel_small, cnn_pool_size
        )
        self.cnn_branch2 = CNNBranch(
            input_size, cnn_channels, cnn_kernel_large, cnn_pool_size
        )

        cnn_out_size = cnn_channels * 2

        self.bilstm = nn.LSTM(
            input_size=cnn_out_size,
            hidden_size=bilstm_hidden,
            num_layers=bilstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=bilstm_dropout if bilstm_layers > 1 else 0,
        )

        bilstm_out_size = bilstm_hidden * 2

        self.proj = (
            nn.Linear(bilstm_out_size, transformer_d_model)
            if bilstm_out_size != transformer_d_model
            else nn.Identity()
        )

        self.pos_encoder = PositionalEncoding(
            transformer_d_model, seq_length + 50, transformer_dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_ff_dim,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.layer_norm = nn.LayerNorm(transformer_d_model)

        self.classifier = nn.Sequential(
            nn.Dropout(fc_dropout),
            nn.Linear(transformer_d_model, transformer_d_model // 2),
            nn.ReLU(),
            nn.Dropout(fc_dropout / 2),
            nn.Linear(transformer_d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_length, input_size)
        Returns:
            logits: (batch_size, num_classes)
        """
        x_t = x.permute(0, 2, 1)

        b1 = self.cnn_branch1(x_t)
        b2 = self.cnn_branch2(x_t)

        fused = torch.cat([b1, b2], dim=1)
        fused = fused.permute(0, 2, 1)

        lstm_out, _ = self.bilstm(fused)

        out = self.proj(lstm_out)
        out = self.pos_encoder(out)

        out = self.transformer(out)
        out = self.layer_norm(out)

        out = out.mean(dim=1)

        return self.classifier(out)


class HybridConfig:
    """Configuration pour le modèle hybride."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["models"]["hybrid"]

        self.cnn_channels = self.config.get("cnn_channels", 64)
        self.cnn_kernel_small = self.config.get("cnn_kernel_small", 3)
        self.cnn_kernel_large = self.config.get("cnn_kernel_large", 5)
        self.cnn_pool_size = self.config.get("cnn_pool_size", 2)
        self.bilstm_hidden = self.config.get("bilstm_hidden", 128)
        self.bilstm_layers = self.config.get("bilstm_layers", 2)
        self.bilstm_dropout = self.config.get("bilstm_dropout", 0.3)
        self.transformer_d_model = self.config.get("transformer_d_model", 256)
        self.transformer_nhead = self.config.get("transformer_nhead", 4)
        self.transformer_layers = self.config.get("transformer_layers", 2)
        self.transformer_ff_dim = self.config.get("transformer_ff_dim", 512)
        self.transformer_dropout = self.config.get("transformer_dropout", 0.2)
        self.fc_dropout = self.config.get("fc_dropout", 0.4)


if __name__ == "__main__":
    batch_size = 32
    seq_length = 10
    input_size = 36
    num_classes = 18

    model = HybridClassifier(input_size, num_classes, seq_length)
    x = torch.randn(batch_size, seq_length, input_size)

    output = model(x)
    print(f"Input: {x.shape} -> Output: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    assert output.shape == (batch_size, num_classes)
    print("✅ Hybrid CNN-BiLSTM-Transformer OK")
