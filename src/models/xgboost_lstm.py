"""
XGBoost-LSTM Hybrid Classifier for IoT Device Identification
=============================================================
Architecture per docs/architectures §3:

  Input
    └── LSTM (temporal feature extractor)
         └── Embedding vector (fixed-size latent representation)
              └── XGBoost classifier (final classification)

  XGBoost hyper-parameter ranges (per doc):
    - learning_rate  : 0.01 – 0.3
    - max_depth      : 3    – 15
    - min_child_weight: 1   – 7
    - gamma          : 0    – 0.5

  For gradient-based adversarial attacks, a PyTorch surrogate linear
  head is attached to the LSTM so gradients can flow during attack generation.
"""

import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb
from torch.utils.data import DataLoader


class XGBoostLSTMClassifier(nn.Module):
    """
    XGBoost-LSTM hybrid model.

    The LSTM is pre-trained end-to-end via a surrogate PyTorch classifier.
    Once the LSTM is trained, XGBoost is fitted on the extracted embeddings
    for the final classification step.

    During adversarial attack generation (PGD), the surrogate head is used
    to compute gradients, making the attacks white-box compatible.
    """

    def __init__(self, input_size: int, num_classes: int, lstm_config: dict):
        super().__init__()
        self.num_classes = num_classes

        hidden_size   = lstm_config.get("hidden_size", 64)   # 64 per doc
        num_layers    = lstm_config.get("num_layers", 2)
        bidirectional = lstm_config.get("bidirectional", True)
        dropout       = lstm_config.get("dropout", 0.3)

        # ── LSTM feature extractor ──────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        # embedding_size: 128 for BiLSTM (64×2), 64 for unidirectional
        embedding_size = hidden_size * 2 if bidirectional else hidden_size

        # ── Surrogate PyTorch head (for gradient-based adversarial attacks) ─
        self.surrogate_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_size, embedding_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(embedding_size // 2, num_classes),
        )

        # ── XGBoost classifier ──────────────────────────────────────────────
        # Hyper-parameters per docs/architectures §3
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,        # in [0.01, 0.3] per doc
            max_depth=6,              # in [3, 15] per doc
            min_child_weight=3,       # in [1, 7] per doc
            gamma=0.1,                # in [0, 0.5] per doc
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            use_label_encoder=False,
            tree_method="hist",       # fast CPU/GPU training
        )
        self.xgb_fitted = False

    # ── Feature extraction ──────────────────────────────────────────────────

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run LSTM and return the final hidden-state embedding."""
        _, (hidden, _) = self.lstm(x)
        if self.lstm.bidirectional:
            # Concatenate last forward + last backward hidden → 128-dim
            embedding = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            embedding = hidden[-1]
        return embedding

    # ── Forward pass ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.extract_features(x)

        # Inference / evaluation: use the robust XGBoost predictions
        if self.xgb_fitted and not x.requires_grad:
            features_np = embedding.detach().cpu().numpy()
            probs = self.xgb_model.predict_proba(features_np)
            return torch.tensor(probs, device=x.device, dtype=torch.float32)

        # Training / attack generation: use surrogate classifier
        return self.surrogate_classifier(embedding)

    # ── XGBoost fitting ─────────────────────────────────────────────────────

    def fit_xgboost(self, dataloader: DataLoader, device: torch.device):
        """
        Extract LSTM embeddings from all batches and fit XGBoost.
        Called once after LSTM training (Phase 3).
        """
        self.eval()
        X_feats, Y_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(device)
                feats = self.extract_features(X_batch)
                X_feats.append(feats.cpu().numpy())
                Y_labels.append(y_batch.numpy())

        X_all = np.vstack(X_feats)
        Y_all = np.concatenate(Y_labels)

        print(f"Fitting XGBoost on {X_all.shape[0]:,} samples "
              f"with {X_all.shape[1]} LSTM embedding features...")
        print(f"  lr=0.1, max_depth=6, min_child_weight=3, gamma=0.1")

        self.xgb_model.fit(X_all, Y_all)
        self.xgb_fitted = True
        print("XGBoost trained successfully. ✅")
