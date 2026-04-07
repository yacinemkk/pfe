#!/usr/bin/env python3
"""
Evaluate Evasion Success for Sequential Models (LSTM, BiLSTM, CNN-LSTM, Transformer)

Tests adversarial samples against trained model and measures evasion success.

Success Thresholds:
  - Full Evasion: accuracy < 10% (model is blind)
  - Partial Evasion: accuracy < 50% (detection strongly degraded)
  - Evasion Failed: accuracy >= 50% (model still detects)

Usage:
    python evaluate_evasion_seq.py \
        --model_path results/models/lstm_10_phases_xxx/best_model.pt \
        --model_type lstm \
        --test_data data/processed/test.csv \
        --adversarial_samples adversarial_samples.npy \
        --output evasion_report.json
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import sys

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

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


def predict(
    model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 64
) -> np.ndarray:
    """Get model predictions."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_batch = torch.FloatTensor(X[i : i + batch_size]).to(device)
            outputs = model(X_batch)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
    return np.array(predictions)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict:
    """Compute comprehensive evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    per_class_metrics = {}
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision_i = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_i = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_i = (
            2 * precision_i * recall_i / (precision_i + recall_i)
            if (precision_i + recall_i) > 0
            else 0
        )

        per_class_metrics[f"class_{i}"] = {
            "precision": precision_i,
            "recall": recall_i,
            "f1": f1_i,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist(),
    }


def categorize_evasion(accuracy: float) -> str:
    """Categorize evasion success based on accuracy."""
    if accuracy < 0.1:
        return "FULL_EVASION"
    elif accuracy < 0.5:
        return "PARTIAL_EVASION"
    else:
        return "EVASION_FAILED"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Evasion Success for Sequential Models"
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
        "--adversarial_samples",
        type=str,
        required=True,
        help="Path to adversarial samples .npy",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evasion_report.json",
        help="Output JSON report path",
    )
    parser.add_argument("--seq_length", type=int, default=10, help="Sequence length")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Max samples to evaluate"
    )

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    print(f"\nLoading test data from {args.test_data}...")
    X_test, y_test, features, scaler, label_encoder = load_test_data(
        Path(args.test_data), args.seq_length
    )

    print(f"\nLoading adversarial samples from {args.adversarial_samples}...")
    X_adv = np.load(args.adversarial_samples)

    if args.max_samples:
        n = min(args.max_samples, len(X_test), len(X_adv))
        X_test = X_test[:n]
        y_test = y_test[:n]
        X_adv = X_adv[:n]

    if len(X_test) != len(X_adv):
        print(
            f"Warning: Adjusting sample count. Clean: {len(X_test)}, Adv: {len(X_adv)}"
        )
        n = min(len(X_test), len(X_adv))
        X_test = X_test[:n]
        y_test = y_test[:n]
        X_adv = X_adv[:n]

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

    print(f"\nEvaluating on CLEAN data...")
    y_pred_clean = predict(model, X_test, device, args.batch_size)
    clean_metrics = compute_metrics(y_test, y_pred_clean, num_classes)
    print(f"  Accuracy: {clean_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {clean_metrics['f1_macro']:.4f}")

    print(f"\nEvaluating on ADVERSARIAL data...")
    y_pred_adv = predict(model, X_adv, device, args.batch_size)
    adv_metrics = compute_metrics(y_test, y_pred_adv, num_classes)
    print(f"  Accuracy: {adv_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {adv_metrics['f1_macro']:.4f}")

    robustness_ratio = adv_metrics["accuracy"] / max(clean_metrics["accuracy"], 1e-8)
    accuracy_drop = clean_metrics["accuracy"] - adv_metrics["accuracy"]
    evasion_category = categorize_evasion(adv_metrics["accuracy"])

    report = {
        "model_path": args.model_path,
        "model_type": args.model_type,
        "num_samples": len(X_test),
        "seq_length": args.seq_length,
        "num_classes": num_classes,
        "clean_metrics": clean_metrics,
        "adversarial_metrics": adv_metrics,
        "robustness_ratio": robustness_ratio,
        "accuracy_drop": accuracy_drop,
        "evasion_category": evasion_category,
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {args.output}")

    print(f"\n{'=' * 60}")
    print("EVASION EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Clean Accuracy:         {clean_metrics['accuracy']:.4f}")
    print(f"  Adversarial Accuracy:   {adv_metrics['accuracy']:.4f}")
    print(f"  Accuracy Drop:          {accuracy_drop * 100:.1f}%")
    print(f"  Robustness Ratio:       {robustness_ratio:.4f}")
    print(f"  Evasion Category:       {evasion_category}")
    print(f"{'=' * 60}")

    if evasion_category == "FULL_EVASION":
        print("  *** FULL EVASION - Model is blind to this attack ***")
    elif evasion_category == "PARTIAL_EVASION":
        print("  *** PARTIAL EVASION - Detection strongly degraded ***")
    else:
        print("  *** EVASION FAILED - Model still detects attack ***")


if __name__ == "__main__":
    main()
