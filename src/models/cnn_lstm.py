"""
CNN-LSTM Hybrid Model for IoT Device Identification
====================================================
Architecture per docs/architectures §2 (LSTM-1DCNN / LSTM-MLP):

  Input (seq_len, features)
    └── LSTM layer (64 neurons, return_sequences=True)
         └── Conv1D (1D convolution on the time axis)
              └── MaxPool1D
                   └── Flatten
                        └── Dense (100 neurons, ReLU)
                             └── Output (num_classes)

Key difference from a typical CNN-LSTM:
  The LSTM comes FIRST and returns the full sequence.
  The CNN then extracts local patterns from the LSTM output (temporal features).
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import LSTM_CONFIG


class CNNLSTMClassifier(nn.Module):
    """
    LSTM → CNN-LSTM Hybrid Architecture (docs/architectures §2):

    1. LSTM (64 units, return sequences) — captures temporal dependencies
    2. Conv1D (applies kernels on the LSTM sequence output)
    3. MaxPool1D — reduces temporal dimension
    4. Flatten — converts to 1D vector
    5. Dense (100 neurons, ReLU) — feature integration
    6. Output (num_classes)
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        lstm_hidden: int = 64,           # 64 neurons per doc
        lstm_layers: int = 1,            # single LSTM layer (return_sequences)
        cnn_channels: int = 64,          # Conv1D output channels
        cnn_kernel_size: int = 3,        # Conv1D kernel
        pool_size: int = 2,              # MaxPool1D
        dense_units: int = 100,          # Dense(100) per doc
        dropout: float = 0.3,
        bidirectional: bool = False,     # doc describes unidirectional for this variant
    ):
        super().__init__()

        self.bidirectional = bidirectional

        # ── Step 1: LSTM (return sequences) ────────────────────────────────
        # Outputs: (batch, seq_len, lstm_hidden [*2 if BiDir])
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0,          # single-layer → no inter-layer dropout
        )
        lstm_out_size = lstm_hidden * 2 if bidirectional else lstm_hidden

        # ── Step 2: Conv1D on LSTM sequence output ──────────────────────────
        # Input to Conv1d: (batch, lstm_out_size, seq_len) [after permute]
        self.conv1d = nn.Conv1d(
            in_channels=lstm_out_size,
            out_channels=cnn_channels,
            kernel_size=cnn_kernel_size,
            padding=cnn_kernel_size // 2,
        )
        self.relu_conv = nn.ReLU()

        # ── Step 3: MaxPool1D ───────────────────────────────────────────────
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)

        # ── Step 4+5: Flatten → Dense(100, ReLU) ───────────────────────────
        # We use AdaptiveAvgPool1d(1) to make Flatten size input-independent,
        # then Linear → 100 → ReLU
        self.global_pool = nn.AdaptiveAvgPool1d(1)   # → (batch, cnn_channels, 1)

        self.head = nn.Sequential(
            nn.Flatten(),                             # → (batch, cnn_channels)
            nn.Dropout(dropout),
            nn.Linear(cnn_channels, dense_units),     # Dense(100) per doc
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(dense_units, num_classes),      # Output
        )

    def forward(self, x):
        # x: (batch, seq_len, features)

        # 1. LSTM → return full sequence
        lstm_out, _ = self.lstm(x)                    # (B, T, lstm_hidden)

        # 2. Conv1D operates on (B, C, T) — permute
        x2 = lstm_out.permute(0, 2, 1)               # (B, lstm_out_size, T)
        x2 = self.relu_conv(self.conv1d(x2))          # (B, cnn_channels, T)

        # 3. MaxPool1D
        x3 = self.pool(x2)                            # (B, cnn_channels, T//2)

        # 4. Global pool → Flatten → Dense(100) → output
        x4 = self.global_pool(x3)                     # (B, cnn_channels, 1)
        return self.head(x4)                          # (B, num_classes)


class CNNClassifier(nn.Module):
    """
    1D CNN-only classifier (auxiliary / baseline comparison model).
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        in_channels = input_size

        for i in range(num_layers):
            out_channels = channels * (2 ** i)
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels // 2, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.features(x)
        x = self.global_pool(x)
        return self.classifier(x)


if __name__ == "__main__":
    batch_size = 32
    seq_len = 10
    input_size = 36
    num_classes = 17

    x = torch.randn(batch_size, seq_len, input_size)

    print("Testing CNN-LSTM (LSTM-first, per docs/architectures §2)...")
    model = CNNLSTMClassifier(input_size, num_classes)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == (batch_size, num_classes)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("PASS ✅")
