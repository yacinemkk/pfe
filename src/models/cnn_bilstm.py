import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class CNNBiLSTMClassifier(nn.Module):
    """
    CNN-BiLSTM Classifier
    Combines 1D CNN for local spatial motif extraction with BiLSTM for temporal modelling.
    """
    def __init__(self, input_size, num_classes, config=None):
        super().__init__()
        # Default config mimicking the hybrid but smaller
        cnn_filters = 64
        cnn_kernel = 3
        lstm_hidden = 64
        lstm_layers = 1
        dropout = 0.3
        
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, kernel_size=cnn_kernel, padding=cnn_kernel//2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(cnn_filters)
        )
        
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Linear(lstm_hidden, num_classes)
        )
        
    def forward(self, x):
        # x is (batch, seq_len, features)
        # Conv1d expects (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        
        # return to (batch, current_seq_len, features) for LSTM
        x = x.permute(0, 2, 1)
        
        out, _ = self.lstm(x)
        # Extract last output step from the sequence
        out = out[:, -1, :]
        return self.fc(out)
