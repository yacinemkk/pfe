"""
PHASE 4.5: Modèle Transformer
Per docs/important.md

Architecture:
- 6 couches encodeur
- 768 dimensions, 12 têtes d'attention
- FFN 3072 dimensions, activation GELU
- Connexions résiduelles + LayerNorm
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


class TransformerClassifier(nn.Module):
    """
    Classificateur Transformer (encodeur uniquement).

    Architecture per docs/important.md §4.5:
    - 6 couches encodeur
    - Multi-Head Self-Attention (Query × Key → weights × Value)
    - FFN avec GELU
    - Connexions résiduelles + LayerNorm
    - Mean Pooling → FC → classification
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        d_model: int = 768,
        nhead: int = 12,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 3072,
        dropout: float = 0.2,
        max_seq_length: int = 512,
        config_path: str = "config/config.yaml",
    ):
        super().__init__()

        if config_path:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            model_config = config["models"]["transformer"]
            d_model = model_config.get("d_model", d_model)
            nhead = model_config.get("nhead", nhead)
            num_encoder_layers = model_config.get(
                "num_encoder_layers", num_encoder_layers
            )
            dim_feedforward = model_config.get("dim_feedforward", dim_feedforward)
            dropout = model_config.get("dropout", dropout)
            max_seq_length = model_config.get("max_seq_length", max_seq_length)

        self.d_model = d_model

        self.input_projection = nn.Linear(input_size, d_model)

        self.pos_encoder = PositionalEncoding(d_model, max_seq_length + 50, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_length, input_size)
        Returns:
            logits: (batch_size, num_classes)
        """
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.classifier(x)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Extrait les poids d'attention (pour visualisation)."""
        x = self.input_projection(x)
        x = self.pos_encoder(x)

        attention_weights = []
        for layer in self.transformer_encoder.layers:
            x = layer(x)

        return attention_weights


class NLPTransformerClassifier(nn.Module):
    """
    Transformer pour séquences tokenisées (NLP-style).

    Per docs/important.md §3: Vocabulaire de 52 000 tokens.
    """

    def __init__(
        self,
        vocab_size: int = 52000,
        num_classes: int = 18,
        d_model: int = 768,
        nhead: int = 12,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 3072,
        dropout: float = 0.2,
        max_seq_length: int = 512,
        config_path: str = "config/config.yaml",
    ):
        super().__init__()

        if config_path:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            model_config = config["models"]["transformer"]
            d_model = model_config.get("d_model", d_model)
            nhead = model_config.get("nhead", nhead)
            num_encoder_layers = model_config.get(
                "num_encoder_layers", num_encoder_layers
            )
            dim_feedforward = model_config.get("dim_feedforward", dim_feedforward)
            dropout = model_config.get("dropout", dropout)
            max_seq_length = model_config.get("max_seq_length", max_seq_length)

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=2)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length + 50, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_length) - token IDs
        Returns:
            logits: (batch_size, num_classes)
        """
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class TransformerConfig:
    """Configuration pour le Transformer."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["models"]["transformer"]

        self.d_model = self.config.get("d_model", 768)
        self.nhead = self.config.get("nhead", 12)
        self.num_encoder_layers = self.config.get("num_encoder_layers", 6)
        self.dim_feedforward = self.config.get("dim_feedforward", 3072)
        self.dropout = self.config.get("dropout", 0.2)
        self.activation = self.config.get("activation", "gelu")
        self.max_seq_length = self.config.get("max_seq_length", 512)


if __name__ == "__main__":
    batch_size = 32
    seq_length = 10
    input_size = 36
    num_classes = 18

    model = TransformerClassifier(input_size, num_classes)
    x = torch.randn(batch_size, seq_length, input_size)

    output = model(x)
    print(f"Input: {x.shape} -> Output: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    assert output.shape == (batch_size, num_classes)
    print("✅ Transformer OK")
