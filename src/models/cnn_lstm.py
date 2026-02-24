"""
CNN-LSTM Hybrid Model for IoT Device Identification
Combines CNN for local pattern extraction with LSTM for temporal modeling.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import LSTM_CONFIG


class CNNLSTMClassifier(nn.Module):
    """
    CNN-LSTM Hybrid Architecture:
    1. 1D CNN extracts local patterns from each feature sequence
    2. LSTM captures temporal dependencies across time steps
    3. Attention mechanism for feature importance
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        cnn_channels: int = 64,
        cnn_kernel_size: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()

        self.use_attention = use_attention

        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels, kernel_size=cnn_kernel_size, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                cnn_channels, cnn_channels * 2, kernel_size=cnn_kernel_size, padding=1
            ),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        cnn_output_size = cnn_channels * 2

        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        lstm_output_size = lstm_hidden * 2 if bidirectional else lstm_hidden

        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_size, lstm_output_size // 2),
                nn.Tanh(),
                nn.Linear(lstm_output_size // 2, 1),
            )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.cnn(x)

        x = x.permute(0, 2, 1)

        lstm_out, (hidden, cell) = self.lstm(x)

        if self.use_attention:
            attn_weights = self.attention(lstm_out)
            attn_weights = torch.softmax(attn_weights, dim=1)
            context = torch.sum(attn_weights * lstm_out, dim=1)
        else:
            if self.lstm.bidirectional:
                context = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                context = hidden[-1]

        output = self.classifier(context)
        return output


class CNNClassifier(nn.Module):
    """
    1D CNN-only classifier for comparison.
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
            out_channels = channels * (2**i)
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
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
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    batch_size = 32
    seq_len = 10
    input_size = 37
    num_classes = 18

    x = torch.randn(batch_size, seq_len, input_size)

    print("Testing CNN-LSTM...")
    model = CNNLSTMClassifier(input_size, num_classes)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTesting CNN...")
    model_cnn = CNNClassifier(input_size, num_classes)
    out_cnn = model_cnn(x)
    print(f"Input: {x.shape} -> Output: {out_cnn.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_cnn.parameters()):,}")
