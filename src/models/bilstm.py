import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from config.config import LSTM_CONFIG
except ImportError:
    LSTM_CONFIG = {}

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM Classifier
    Processes sequences in both forward and backward directions.
    """
    def __init__(self, input_size, num_classes, config=None):
        super().__init__()
        config = config or LSTM_CONFIG
        hidden_size = config.get("hidden_size", 64)
        num_layers = config.get("num_layers", 2)
        dropout = config.get("dropout", 0.3)
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)
        # Take the output of the last time step
        out = out[:, -1, :] 
        return self.fc(out)
