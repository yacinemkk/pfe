"""
CNN-BiLSTM-Transformer Hybrid Classifier
=========================================
Architecture hybride décrite dans docs/architectures §5:

1. Module CNN (Extraction spatiale locale) — deux branches parallèles:
   - Branche 1 : Conv1d(kernel=3) → ReLU → MaxPool1d → BatchNorm1d
   - Branche 2 : Conv1d(kernel=5) → ReLU → MaxPool1d → BatchNorm1d
   - Fusion : concaténation des deux branches

2. Module BiLSTM (Modélisation temporelle bidirectionnelle)

3. Module Transformer (Modélisation globale via Multi-Head Self-Attention)

4. Module de sortie : MeanPooling → FC → softmax (via CrossEntropyLoss)

Usage:
    from src.models.cnn_bilstm_transformer import CNNBiLSTMTransformerClassifier
    model = CNNBiLSTMTransformerClassifier(input_size=36, num_classes=17, seq_length=10)
"""

import math
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from config.config import CNN_BILSTM_TRANSFORMER_CONFIG
except ImportError:
    CNN_BILSTM_TRANSFORMER_CONFIG = {}


# ─── Default hyper-parameters ────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # CNN branches
    "cnn_channels": 64,          # channels per branch
    "cnn_kernel_small": 3,       # small kernel size
    "cnn_kernel_large": 5,       # large kernel size
    "cnn_pool_size": 2,          # MaxPool1d size
    # BiLSTM
    "bilstm_hidden": 128,        # hidden units per direction
    "bilstm_layers": 2,          # number of stacked BiLSTM layers
    "bilstm_dropout": 0.3,
    # Transformer encoder
    "transformer_d_model": 256,  # d_model (must equal bilstm output dim = hidden*2)
    "transformer_nhead": 4,
    "transformer_layers": 2,
    "transformer_ff_dim": 512,
    "transformer_dropout": 0.2,
    # FC head
    "fc_dropout": 0.4,
}


# ─── Positional Encoding ─────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ─── Main Hybrid Model ───────────────────────────────────────────────────────

class CNNBiLSTMTransformerClassifier(nn.Module):
    """
    CNN-BiLSTM-Transformer hybrid classifier for IoT device identification.

    Input shape:  (batch_size, seq_length, input_size)
    Output shape: (batch_size, num_classes)
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        seq_length: int = 10,
        vocab_size: int = None,
        padding_idx: int = 2,
        config: dict = None,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        if vocab_size is not None and vocab_size > 0:
            self.embedding = nn.Embedding(vocab_size, input_size, padding_idx=padding_idx)

        cfg = {**DEFAULT_CONFIG, **(CNN_BILSTM_TRANSFORMER_CONFIG or {}), **(config or {})}

        cnn_ch   = cfg["cnn_channels"]
        ks_small = cfg["cnn_kernel_small"]
        ks_large = cfg["cnn_kernel_large"]
        pool     = cfg["cnn_pool_size"]

        bilstm_h     = cfg["bilstm_hidden"]
        bilstm_lay   = cfg["bilstm_layers"]
        bilstm_drop  = cfg["bilstm_dropout"]

        t_d_model = cfg["transformer_d_model"]
        t_nhead   = cfg["transformer_nhead"]
        t_layers  = cfg["transformer_layers"]
        t_ff      = cfg["transformer_ff_dim"]
        t_drop    = cfg["transformer_dropout"]

        fc_drop   = cfg["fc_dropout"]

        # ── CNN Branch 1 (kernel_size = ks_small) ──────────────────────────
        # Input: (B, input_size, seq_len) — permuted inside forward()
        # After Conv1d: (B, cnn_ch, seq_len)
        # After pool:  (B, cnn_ch, seq_len // pool)
        self.cnn_branch1 = nn.Sequential(
            nn.Conv1d(input_size, cnn_ch, kernel_size=ks_small,
                      padding=ks_small // 2),
            nn.ReLU(),
            nn.MaxPool1d(pool, stride=1, padding=pool // 2),
            nn.BatchNorm1d(cnn_ch),
        )

        # ── CNN Branch 2 (kernel_size = ks_large) ──────────────────────────
        self.cnn_branch2 = nn.Sequential(
            nn.Conv1d(input_size, cnn_ch, kernel_size=ks_large,
                      padding=ks_large // 2),
            nn.ReLU(),
            nn.MaxPool1d(pool, stride=1, padding=pool // 2),
            nn.BatchNorm1d(cnn_ch),
        )

        # Fused CNN output size = cnn_ch * 2 (from two branches)
        cnn_out = cnn_ch * 2

        # ── BiLSTM ─────────────────────────────────────────────────────────
        # Transforms each time step from cnn_out → bilstm_h*2
        self.bilstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=bilstm_h,
            num_layers=bilstm_lay,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        bilstm_out = bilstm_h * 2  # bidirectional

        # ── Projection to Transformer d_model ──────────────────────────────
        self.proj = nn.Linear(bilstm_out, t_d_model) if bilstm_out != t_d_model else nn.Identity()

        # ── Positional Encoding ────────────────────────────────────────────
        self.pos_enc = PositionalEncoding(t_d_model, max_len=seq_length + 50, dropout=t_drop)

        # ── Transformer Encoder ─────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=t_d_model,
            nhead=t_nhead,
            dim_feedforward=t_ff,
            dropout=t_drop,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=t_layers)
        self.layer_norm  = nn.LayerNorm(t_d_model)

        # ── FC Head ────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(fc_drop),
            nn.Linear(t_d_model, t_d_model // 2),
            nn.ReLU(),
            nn.Dropout(fc_drop / 2),
            nn.Linear(t_d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size) OR (batch, seq_len) if vocab_size > 0
        Returns:
            logits: (batch, num_classes)
        """
        if self.vocab_size is not None and self.vocab_size > 0:
            if x.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                x = self.embedding(x)

        # ── CNN: permute to (B, input_size, seq_len) ──────────────────────
        x_t = x.permute(0, 2, 1)                 # (B, C_in, T)

        b1 = self.cnn_branch1(x_t)               # (B, cnn_ch, T)
        b2 = self.cnn_branch2(x_t)               # (B, cnn_ch, T)

        fused = torch.cat([b1, b2], dim=1)        # (B, cnn_ch*2, T)
        fused = fused.permute(0, 2, 1).contiguous()            # (B, T, cnn_ch*2) for LSTM

        # ── BiLSTM ─────────────────────────────────────────────────────────
        # Bypass PyTorch CuDNN LSTM 28GB Workspace Allocation bug
        with torch.backends.cudnn.flags(enabled=False):
            lstm_out, _ = self.bilstm(fused)          # (B, T, bilstm_h*2)

        # ── Projection + Positional Encoding ───────────────────────────────
        out = self.proj(lstm_out)                 # (B, T, d_model)
        out = self.pos_enc(out)

        # ── Transformer ────────────────────────────────────────────────────
        out = self.transformer(out)               # (B, T, d_model)
        out = self.layer_norm(out)

        # ── MeanPooling ────────────────────────────────────────────────────
        out = out.mean(dim=1)                     # (B, d_model)

        # ── Classification ─────────────────────────────────────────────────
        return self.classifier(out)               # (B, num_classes)


# ─── Quick self-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # CSV: 37 continuous features, 18 classes
    model_csv = CNNBiLSTMTransformerClassifier(input_size=37, num_classes=18, seq_length=10)
    x_csv = torch.randn(4, 10, 37)
    out_csv = model_csv(x_csv)
    print(f"CSV  → input {x_csv.shape}  output {out_csv.shape}")
    assert out_csv.shape == (4, 18), f"Expected (4,18), got {out_csv.shape}"

    # JSON: 36 features (28 cont + 8 binary), 17 classes
    model_json = CNNBiLSTMTransformerClassifier(input_size=36, num_classes=17, seq_length=10)
    x_json = torch.randn(4, 10, 36)
    out_json = model_json(x_json)
    print(f"JSON → input {x_json.shape}  output {out_json.shape}")
    assert out_json.shape == (4, 17), f"Expected (4,17), got {out_json.shape}"

    # NLP test: BPE tokens
    model_nlp = CNNBiLSTMTransformerClassifier(input_size=128, num_classes=17, seq_length=10, vocab_size=52000)
    x_nlp = torch.randint(0, 1000, (4, 10))
    out_nlp = model_nlp(x_nlp)
    print(f"NLP  → input {x_nlp.shape}  output {out_nlp.shape}")
    assert out_nlp.shape == (4, 17), f"Expected (4,17), got {out_nlp.shape}"

    total = sum(p.numel() for p in model_json.parameters())
    print(f"Parameters: {total:,}")
    print("✅ All assertions passed.")
