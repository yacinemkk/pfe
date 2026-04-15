import torch
import torch.nn as nn
from src.models.cnn_bilstm_transformer import CNNBiLSTMTransformerClassifier

model = CNNBiLSTMTransformerClassifier(input_size=16, num_classes=18, seq_length=10)
x = torch.randn(32, 10, 16)
x_t = x.permute(0, 2, 1)
b1 = model.cnn_branch1(x_t)
b2 = model.cnn_branch2(x_t)
fused = torch.cat([b1, b2], dim=1).permute(0, 2, 1).contiguous()
print("fused shape:", fused.shape)

out = model(x)
print("out shape:", out.shape)
