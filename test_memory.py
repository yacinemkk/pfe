import torch
import torch.nn as nn
from src.models.cnn_bilstm_transformer import CNNBiLSTMTransformerClassifier

model = CNNBiLSTMTransformerClassifier(input_size=16, num_classes=18, seq_length=10).cuda()
x = torch.randn(32, 512, 16).cuda()
with torch.cuda.amp.autocast():
    out = model(x)
print(torch.cuda.memory_allocated() / 1e9, "GB")
