import torch
import numpy as np
from src.models.cnn_bilstm_transformer import CNNBiLSTMTransformerClassifier
import os

BATCH_SIZE = 32
SEQ_LENGTH = 10

X_batch = torch.randn(BATCH_SIZE, SEQ_LENGTH, 16)
model = CNNBiLSTMTransformerClassifier(16, 18, seq_length=SEQ_LENGTH, config={
    'cnn_channels': 32, 'bilstm_hidden': 64, 'bilstm_layers': 2, 'bilstm_dropout': 0.3,
    'transformer_d_model': 128, 'transformer_nhead': 4, 'transformer_layers': 2, 'transformer_ff_dim': 256,
})

print("Input shape:", X_batch.shape)
out = model(X_batch)
print("Output shape:", out.shape)
print("Parameters:", sum(p.numel() for p in model.parameters()))
