import torch
import torch.nn as nn

try:
    print(f"CUDA Available: {torch.cuda.is_available()}")
    model = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True).cuda()
    model.eval()

    # fused shape exactly as we calculated
    fused = torch.randn(32, 11, 64).cuda()
    with torch.cuda.amp.autocast():
        out, _ = model(fused)
    print("Test passed successfully:", out.shape)
except Exception as e:
    print("Error:", str(e))
