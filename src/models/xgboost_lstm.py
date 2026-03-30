"""
PHASE 4.4: Modèle XGBoost-LSTM
Per docs/important.md

Architecture:
- LSTM pré-entraîné comme extracteur de features
- Sortie: vecteur latent de taille fixe
- XGBoost avec hyperparamètres:
  - lr: [0.01-0.3]
  - max_depth: [3-15]
  - min_child_weight: [1-7]
  - gamma: [0-0.5]
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import yaml

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost not available. Install with: pip install xgboost")


class XGBoostLSTMClassifier(nn.Module):
    """
    Classificateur hybride XGBoost-LSTM.

    Architecture per docs/important.md §4.4:
    - LSTM comme extracteur de features
    - Embedding de taille fixe (128 dims pour BiLSTM)
    - XGBoost pour la classification finale
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        config_path: str = "config/config.yaml",
    ):
        super().__init__()

        if config_path:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            model_config = config["models"]["xgboost_lstm"]
            hidden_size = model_config.get("lstm_hidden", hidden_size)
            num_layers = model_config.get("lstm_layers", num_layers)
            bidirectional = model_config.get("bidirectional", bidirectional)
            dropout = model_config.get("dropout", dropout)

            self.xgb_params = {
                "n_estimators": model_config.get("n_estimators", 200),
                "learning_rate": model_config.get("learning_rate", 0.1),
                "max_depth": model_config.get("max_depth", 6),
                "min_child_weight": model_config.get("min_child_weight", 3),
                "gamma": model_config.get("gamma", 0.1),
                "subsample": model_config.get("subsample", 0.8),
                "colsample_bytree": model_config.get("colsample_bytree", 0.8),
                "eval_metric": "mlogloss",
                "use_label_encoder": False,
                "tree_method": "hist",
            }
        else:
            self.xgb_params = {
                "n_estimators": 200,
                "learning_rate": 0.1,
                "max_depth": 6,
                "min_child_weight": 3,
                "gamma": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "eval_metric": "mlogloss",
                "use_label_encoder": False,
                "tree_method": "hist",
            }

        self.num_classes = num_classes
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        embedding_size = hidden_size * 2 if bidirectional else hidden_size

        self.surrogate_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_size, embedding_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(embedding_size // 2, num_classes),
        )

        if XGBOOST_AVAILABLE:
            self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
        else:
            self.xgb_model = None
        self.xgb_fitted = False

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extrait l'embedding LSTM."""
        _, (hidden, _) = self.lstm(x)

        if self.bidirectional:
            embedding = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            embedding = hidden[-1]

        return embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_length, input_size)
        Returns:
            logits: (batch_size, num_classes)
        """
        embedding = self.extract_features(x)

        if self.xgb_fitted and not x.requires_grad and self.xgb_model is not None:
            features_np = embedding.detach().cpu().numpy()
            probs = self.xgb_model.predict_proba(features_np)
            return torch.tensor(probs, device=x.device, dtype=torch.float32)

        return self.surrogate_head(embedding)

    def fit_xgboost(self, X: np.ndarray, y: np.ndarray, device: torch.device = None):
        """
        Entraîne XGBoost sur les embeddings extraits.

        Args:
            X: Séquences d'entrée (n_samples, seq_length, input_size)
            y: Labels (n_samples,)
            device: Device pour l'extraction
        """
        if not XGBOOST_AVAILABLE or self.xgb_model is None:
            raise RuntimeError("XGBoost not available")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eval()
        self.to(device)

        X_tensor = torch.FloatTensor(X).to(device)

        with torch.no_grad():
            embeddings = self.extract_features(X_tensor)

        features_np = embeddings.cpu().numpy()

        print(f"  Entraînement XGBoost sur {len(features_np):,} échantillons...")
        print(
            f"    lr={self.xgb_params['learning_rate']}, "
            f"max_depth={self.xgb_params['max_depth']}, "
            f"min_child_weight={self.xgb_params['min_child_weight']}, "
            f"gamma={self.xgb_params['gamma']}"
        )

        self.xgb_model.fit(features_np, y)
        self.xgb_fitted = True

        print("  XGBoost entraîné avec succès ✓")


class XGBoostLSTMConfig:
    """Configuration pour le modèle XGBoost-LSTM."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["models"]["xgboost_lstm"]

        self.lstm_hidden = self.config.get("lstm_hidden", 64)
        self.lstm_layers = self.config.get("lstm_layers", 2)
        self.bidirectional = self.config.get("bidirectional", True)
        self.dropout = self.config.get("dropout", 0.3)
        self.n_estimators = self.config.get("n_estimators", 200)
        self.learning_rate = self.config.get("learning_rate", 0.1)
        self.max_depth = self.config.get("max_depth", 6)
        self.min_child_weight = self.config.get("min_child_weight", 3)
        self.gamma = self.config.get("gamma", 0.1)


if __name__ == "__main__":
    batch_size = 32
    seq_length = 10
    input_size = 36
    num_classes = 18

    model = XGBoostLSTMClassifier(input_size, num_classes)
    x = torch.randn(batch_size, seq_length, input_size)

    output = model(x)
    print(f"Input: {x.shape} -> Output: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    assert output.shape == (batch_size, num_classes)
    print("✅ XGBoost-LSTM OK")
