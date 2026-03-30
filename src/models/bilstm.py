"""
PHASE 4.2: Modèle BiLSTM
Per docs/important.md

Architecture:
- Remplacer LSTM par BiLSTM bidirectionnel
- Même configuration: 2 couches, 64 unités
- Capture contexte avant/arrière
"""

import torch
import torch.nn as nn
from typing import Optional
import yaml


class BiLSTMClassifier(nn.Module):
    """
    Classificateur BiLSTM pour l'identification d'appareils IoT.

    Architecture per docs/important.md §4.2:
    - LSTM bidirectionnel
    - 2 couches, 64 unités
    - Capture le contexte avant et arrière
    - Embedding: 128 dimensions (64 × 2)
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
            model_config = config["models"]["bilstm"]
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
            bidirectional=True,
        )

        embedding_size = hidden_size * 2

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_size, hidden_size),
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

        embedding = torch.cat((hidden[-2], hidden[-1]), dim=1)

        return self.classifier(embedding)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extrait l'embedding bidirectionnel."""
        lstm_out, (hidden, cell) = self.lstm(x)
        return torch.cat((hidden[-2], hidden[-1]), dim=1)


class BiLSTMConfig:
    """Configuration pour le modèle BiLSTM."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["models"]["bilstm"]

        self.hidden_size = self.config.get("hidden_size", 64)
        self.num_layers = self.config.get("num_layers", 2)
        self.dropout = self.config.get("dropout", 0.3)


if __name__ == "__main__":
    batch_size = 32
    seq_length = 10
    input_size = 36
    num_classes = 18

    model = BiLSTMClassifier(input_size, num_classes)
    x = torch.randn(batch_size, seq_length, input_size)

    output = model(x)
    print(f"Input: {x.shape} -> Output: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    embedding = model.get_embedding(x)
    print(f"Embedding: {embedding.shape}")

    assert output.shape == (batch_size, num_classes)
    print("✅ BiLSTM OK")
