#!/usr/bin/env python3
"""
Adversarial Search for Sequential Models (LSTM, BiLSTM, CNN-LSTM, Transformer)

Greedy search that combines the most effective feature-strategy pairs
to minimize model accuracy (maximize evasion).

Takes sensitivity analysis results and greedily applies perturbations
one by one, keeping only those that reduce accuracy.

Usage:
    python adversarial_search_seq.py \
        --model_path results/models/lstm_10_phases_xxx/best_model.pt \
        --model_type lstm \
        --test_data data/processed/test.csv \
        --sensitivity_csv sensitivity_results.csv \
        --target_accuracy 0.5 \
        --output adversarial_samples.npy
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

import sys

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.adversarial.attacks import SensitivityAnalysis, AdversarialSearch
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


def load_test_data(data_path: Path, seq_length: int, preprocessor_path: Path = None):
    """Load and preprocess test data using the saved preprocessor (anti-leakage).

    If preprocessor_path is provided and exists, loads the fitted scaler and
    label_encoder from the pickle and uses .transform() (no fit on test data).
    Otherwise falls back to fit_transform with a deprecation warning.
    """
    import pandas as pd
    import warnings

    df = pd.read_csv(data_path)

    label_col = "label" if "label" in df.columns else "device_category"
    feature_cols = [
        c for c in df.columns if c != label_col and not c.startswith("Unnamed")
    ]

    X = df[feature_cols].values
    y = df[label_col].values

    if preprocessor_path is not None and preprocessor_path.exists():
        with open(preprocessor_path, "rb") as f:
            prep = pickle.load(f)
        scaler = prep.get("scaler") or prep.get("minmax_scaler")
        label_encoder = prep["label_encoder"]
        X = scaler.transform(X)
        y = np.array(
            [
                label_encoder.transform([lbl])[0]
                if lbl in label_encoder.classes_
                else -1
                for lbl in y
            ]
        )
        print(
            f"  Loaded preprocessor from {preprocessor_path} (scaler fitted on train)"
        )
    else:
        warnings.warn(
            "No preprocessor.pkl found — using fit_transform on test data. "
            "This causes data leakage. Pass --preprocessor_path to fix.",
            stacklevel=2,
        )
        from sklearn.preprocessing import StandardScaler, LabelEncoder

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


def load_sensitivity_results(csv_path: Path) -> List[Dict]:
    """Load sensitivity analysis results from CSV."""
    df = pd.read_csv(csv_path)
    results = df.to_dict("records")
    return results


def run_adversarial_search(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    features: List[str],
    num_classes: int,
    sensitivity_results: List[Dict],
    device: torch.device,
    target_accuracy: float = 0.5,
    batch_size: int = 64,
    n_continuous_features: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[Dict]]:
    """Run greedy adversarial search."""
    X_flat = X_test.reshape(-1, X_test.shape[-1])
    y_expanded = np.repeat(y_test, X_test.shape[1])

    sensitivity = SensitivityAnalysis(
        X_train=X_flat,
        y_train=y_expanded,
        feature_names=features,
        num_classes=num_classes,
        n_continuous_features=n_continuous_features,
    )

    search = AdversarialSearch(
        model=model,
        device=device,
        sensitivity_analysis=sensitivity,
        target_accuracy=target_accuracy,
        batch_size=batch_size,
    )

    X_adv, applied_strategies = search.search(
        sensitivity_results=sensitivity_results,
        X=X_test,
        y=y_test,
        verbose=verbose,
    )

    return X_adv, applied_strategies


def evaluate_evasion(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> float:
    """Evaluate model accuracy."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_batch = torch.FloatTensor(X[i : i + batch_size]).to(device)
            y_batch = torch.LongTensor(y[i : i + batch_size]).to(device)
            outputs = model(X_batch)
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Adversarial Search for Sequential Models"
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
        "--sensitivity_csv",
        type=str,
        required=True,
        help="Path to sensitivity results CSV",
    )
    parser.add_argument(
        "--target_accuracy",
        type=float,
        default=0.5,
        help="Target accuracy to achieve (default: 0.5)",
    )
    parser.add_argument(
        "--output_samples",
        type=str,
        default="adversarial_samples.npy",
        help="Output numpy file for adversarial samples",
    )
    parser.add_argument(
        "--output_strategies",
        type=str,
        default="applied_strategies.csv",
        help="Output CSV for applied strategies",
    )
    parser.add_argument("--seq_length", type=int, default=10, help="Sequence length")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Max samples to process"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--preprocessor_path",
        type=str,
        default=None,
        help="Path to preprocessor.pkl (auto-detected from model_path if not provided)",
    )

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    if args.preprocessor_path:
        preprocessor_path = Path(args.preprocessor_path)
    else:
        preprocessor_path = Path(args.model_path).parent / "preprocessor.pkl"

    print(f"\nLoading test data from {args.test_data}...")
    X_test, y_test, features, scaler, label_encoder = load_test_data(
        Path(args.test_data), args.seq_length, preprocessor_path
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

    print(f"\nLoading sensitivity results from {args.sensitivity_csv}...")
    sensitivity_results = load_sensitivity_results(Path(args.sensitivity_csv))
    print(f"  Loaded {len(sensitivity_results)} feature-strategy combinations")

    print(f"\nEvaluating baseline accuracy...")
    baseline_acc = evaluate_evasion(model, X_test, y_test, device, args.batch_size)
    print(f"  Baseline accuracy: {baseline_acc:.4f}")

    print(
        f"\nRunning greedy adversarial search (target accuracy: {args.target_accuracy})..."
    )
    X_adv, applied_strategies = run_adversarial_search(
        model=model,
        X_test=X_test,
        y_test=y_test,
        features=features,
        num_classes=num_classes,
        sensitivity_results=sensitivity_results,
        device=device,
        target_accuracy=args.target_accuracy,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    print(f"\nEvaluating adversarial accuracy...")
    adv_acc = evaluate_evasion(model, X_adv, y_test, device, args.batch_size)
    print(f"  Adversarial accuracy: {adv_acc:.4f}")
    print(f"  Accuracy drop: {(baseline_acc - adv_acc) * 100:.1f}%")

    np.save(args.output_samples, X_adv)
    print(f"\nAdversarial samples saved to {args.output_samples}")

    if applied_strategies:
        df_strategies = pd.DataFrame(applied_strategies)
        df_strategies.to_csv(args.output_strategies, index=False)
        print(f"Applied strategies saved to {args.output_strategies}")

    print(f"\n{'=' * 60}")
    print("ADVERSARIAL SEARCH SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Baseline accuracy:      {baseline_acc:.4f}")
    print(f"  Adversarial accuracy:   {adv_acc:.4f}")
    print(f"  Accuracy drop:          {(baseline_acc - adv_acc) * 100:.1f}%")
    print(f"  Strategies applied:     {len(applied_strategies)}")

    if adv_acc < 0.1:
        print(f"\n  *** FULL EVASION - Model is blind to attack ***")
    elif adv_acc < 0.5:
        print(f"\n  *** PARTIAL EVASION - Detection strongly degraded ***")
    else:
        print(f"\n  *** EVASION FAILED - Model still detects attack ***")


if __name__ == "__main__":
    main()
