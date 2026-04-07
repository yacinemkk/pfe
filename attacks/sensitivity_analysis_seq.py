#!/usr/bin/env python3
"""
Sensitivity Analysis for Sequential Models (LSTM, BiLSTM, CNN-LSTM, Transformer)

Analyzes which features are most vulnerable to adversarial perturbations.
Tests 4 strategies per feature:
  - Zero: set feature to 0
  - Mimic_Mean: set to mean of benign samples
  - Mimic_95th: set to 95th percentile of benign samples
  - Padding_x10: multiply feature by 10

Usage:
    python sensitivity_analysis_seq.py \
        --model_path results/models/lstm_10_phases_xxx/best_model.pt \
        --model_type lstm \
        --test_data data/processed/test.csv \
        --output sensitivity_results.csv
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

import sys

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.adversarial.attacks import SensitivityAnalysis
from src.models.lstm import LSTMClassifier
from src.models.bilstm import BiLSTMClassifier
from src.models.cnn_lstm import CNNLSTMClassifier
from src.models.transformer import TransformerClassifier


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    model_path: Path,
    model_type: str,
    device: torch.device,
    num_classes: int,
    input_size: int,
    seq_length: int,
):
    """Load trained model from checkpoint."""
    model_classes = {
        "lstm": LSTMClassifier,
        "bilstm": BiLSTMClassifier,
        "cnn_lstm": CNNLSTMClassifier,
        "transformer": TransformerClassifier,
    }

    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model_classes[model_type](
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        seq_length=seq_length,
    )

    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def load_test_data(data_path: Path, seq_length: int):
    """Load and preprocess test data."""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    df = pd.read_csv(data_path)

    label_col = "label" if "label" in df.columns else "device_category"
    feature_cols = [
        c for c in df.columns if c != label_col and not c.startswith("Unnamed")
    ]

    X = df[feature_cols].values
    y = df[label_col].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    n_samples = len(X) // seq_length
    X = X[: n_samples * seq_length]
    y = y[: n_samples * seq_length]

    X = X.reshape(n_samples, seq_length, -1)
    y = y[::seq_length]

    return X, y, feature_cols, scaler, label_encoder


def analyze_sensitivity(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    features: List[str],
    num_classes: int,
    device: torch.device,
    batch_size: int = 64,
    n_continuous_features: Optional[int] = None,
    verbose: bool = True,
) -> List[Dict]:
    """Run sensitivity analysis on test data."""
    X_flat = X_test.reshape(-1, X_test.shape[-1])
    y_expanded = np.repeat(y_test, X_test.shape[1])

    sensitivity = SensitivityAnalysis(
        X_train=X_flat,
        y_train=y_expanded,
        feature_names=features,
        num_classes=num_classes,
        n_continuous_features=n_continuous_features,
    )

    results = sensitivity.analyze(
        model=model,
        X=X_test,
        y=y_test,
        device=device,
        batch_size=batch_size,
        verbose=verbose,
    )

    return results, sensitivity


def main():
    parser = argparse.ArgumentParser(
        description="Sensitivity Analysis for Sequential Models"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["lstm", "bilstm", "cnn_lstm", "transformer"],
    )
    parser.add_argument(
        "--test_data", type=str, required=True, help="Path to test data CSV"
    )
    parser.add_argument(
        "--output", type=str, default="sensitivity_results.csv", help="Output CSV path"
    )
    parser.add_argument("--seq_length", type=int, default=10, help="Sequence length")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Max samples to analyze"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    print(f"\nLoading test data from {args.test_data}...")
    X_test, y_test, features, scaler, label_encoder = load_test_data(
        Path(args.test_data), args.seq_length
    )

    if args.max_samples and len(X_test) > args.max_samples:
        indices = np.random.choice(len(X_test), args.max_samples, replace=False)
        X_test = X_test[indices]
        y_test = y_test[indices]

    num_classes = len(np.unique(y_test))
    input_size = X_test.shape[-1]

    print(f"  Samples: {len(X_test)}")
    print(f"  Sequence length: {args.seq_length}")
    print(f"  Features: {len(features)}")
    print(f"  Classes: {num_classes}")

    print(f"\nLoading model from {args.model_path}...")
    model = load_model(
        Path(args.model_path),
        args.model_type,
        device,
        num_classes,
        input_size,
        args.seq_length,
    )

    print(f"\nRunning sensitivity analysis...")
    results, sensitivity = analyze_sensitivity(
        model=model,
        X_test=X_test,
        y_test=y_test,
        features=features,
        num_classes=num_classes,
        device=device,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    print(f"\n{'=' * 60}")
    print("TOP 10 MOST VULNERABLE FEATURES")
    print(f"{'=' * 60}")

    seen = set()
    top_features = []
    for r in results:
        if r["feature"] not in seen:
            seen.add(r["feature"])
            top_features.append(r)
        if len(top_features) >= 10:
            break

    for i, r in enumerate(top_features, 1):
        print(
            f"  {i}. {r['feature']:<30s} | Strategy: {r['strategy']:<12s} | Drop: {r['drop'] * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
