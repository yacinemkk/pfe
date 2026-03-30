"""
PHASE 4.1: Modèle LSTM
Per docs/important.md

Architecture:
- 2 couches LSTM, 64 unités/couche
- Activation ReLU
- Dropout pour régularisation
- Couche Dense de sortie (classification multi-classe)
- Vecteur embedding: 128 dimensions
"""

import torch
import torch.nn as nn
from typing import Optional
import yaml


class LSTMClassifier(nn.Module):
    """
    Classificateur LSTM pour l'identification d'appareils IoT.

    Architecture per docs/important.md §4.1:
    - 2 couches LSTM empilées
    - 64 unités par couche
    - Dropout pour régularisation
    - Classificateur dense en sortie
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        config_path: str = "config/config.yaml",
    ):
        super().__init__()

        if config_path:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            model_config = config["models"]["lstm"]
            hidden_size = model_config.get("hidden_size", hidden_size)
            num_layers = model_config.get("num_layers", num_layers)
            dropout = model_config.get("dropout", dropout)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_length, input_size)
        Returns:
            logits: (batch_size, num_classes)
        """
        lstm_out, (hidden, cell) = self.lstm(x)

        embedding = hidden[-1]

        return self.classifier(embedding)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extrait l'embedding de la séquence."""
        lstm_out, (hidden, cell) = self.lstm(x)
        return hidden[-1]


class LSTMConfig:
    """Configuration pour le modèle LSTM."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["models"]["lstm"]

        self.hidden_size = self.config.get("hidden_size", 64)
        self.num_layers = self.config.get("num_layers", 2)
        self.dropout = self.config.get("dropout", 0.3)
        self.embedding_dim = self.config.get("embedding_dim", 128)


if __name__ == "__main__":
    batch_size = 32
    seq_length = 10
    input_size = 36
    num_classes = 18

    model = LSTMClassifier(input_size, num_classes)
    x = torch.randn(batch_size, seq_length, input_size)

    output = model(x)
    print(f"Input: {x.shape} -> Output: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    assert output.shape == (batch_size, num_classes)
    print("✅ LSTM OK")
