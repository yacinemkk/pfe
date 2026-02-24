#!/usr/bin/env python3
"""
Hybrid Adversarial Training for IoT Device Identification

This script implements:
1. Multiple model architectures (LSTM, Transformer, CNN-LSTM)
2. Multiple sequence lengths for experimentation
3. Feature-level adversarial attacks (IoT-SDN style)
4. Sequence-level adversarial attacks (PGD via BPTT)
5. Hybrid adversarial training with combined attacks
6. Comprehensive robustness evaluation

Usage:
    python train_adversarial.py --model lstm --seq_length 10 --adv_method hybrid
    python train_adversarial.py --model cnn_lstm --seq_length 50 --adv_method all
    python train_adversarial.py --compare_all --seq_lengths 10,25,50
"""

import argparse
import gc
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    FEATURES_TO_KEEP,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    LSTM_CONFIG,
    TRANSFORMER_CONFIG,
)
from src.models.lstm import LSTMClassifier, IoTSequenceDataset
from src.models.transformer import TransformerClassifier
from src.models.cnn_lstm import CNNLSTMClassifier, CNNClassifier
from src.adversarial.attacks import (
    FeatureLevelAttack,
    SequenceLevelAttack,
    HybridAdversarialAttack,
    AdversarialEvaluator,
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_sequences_with_stride(
    X: np.ndarray,
    y: np.ndarray,
    seq_length: int,
    stride: int = 1,
    source_groups: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences from data with configurable stride."""
    X_seq, y_seq = [], []

    if source_groups is None:
        n_samples = len(X) - seq_length + 1
        for i in range(0, n_samples, stride):
            X_seq.append(X[i : i + seq_length])
            y_seq.append(y[i + seq_length - 1])
    else:
        for group_id in np.unique(source_groups):
            mask = source_groups == group_id
            X_group = X[mask]
            y_group = y[mask]

            n_samples = len(X_group) - seq_length + 1
            for i in range(0, max(1, n_samples), stride):
                if i + seq_length <= len(X_group):
                    X_seq.append(X_group[i : i + seq_length])
                    y_seq.append(y_group[i + seq_length - 1])

    return np.array(X_seq), np.array(y_seq)


def load_and_preprocess_data(
    seq_length: int, stride: int = 5, max_files: int = None
) -> Tuple[np.ndarray, ...]:
    """Load and preprocess data with specified sequence length."""
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    import pandas as pd

    print(f"Loading data with sequence length={seq_length}, stride={stride}")

    dfs = []
    csv_files = sorted(RAW_DATA_DIR.glob("home*_labeled.csv"))

    if max_files:
        csv_files = csv_files[:max_files]

    for f in csv_files:
        print(f"  Loading {f.name}...")
        df = pd.read_csv(f)
        df["source_file"] = f.stem
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    df = df.drop_duplicates()
    df = df.dropna(subset=["device"])

    class_counts = df["device"].value_counts()
    valid_classes = class_counts[class_counts >= 500].index
    df = df[df["device"].isin(valid_classes)]

    print(f"  Samples: {len(df):,} | Classes: {len(valid_classes)}")

    features = [c for c in FEATURES_TO_KEEP if c in df.columns]
    X = df[features].values
    y = df["device"].values
    source_groups = df["source_file"].values if "source_file" in df.columns else None

    del df, dfs
    gc.collect()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    del X, y
    gc.collect()

    print(f"  Creating sequences...")
    X_seq, y_seq = create_sequences_with_stride(
        X_scaled, y_encoded, seq_length, stride, source_groups
    )

    del X_scaled, y_encoded, source_groups
    gc.collect()

    print(f"  Total sequences: {len(X_seq):,}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )

    del X_seq, y_seq
    gc.collect()

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    del X_temp, y_temp
    gc.collect()

    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        features,
        scaler,
        label_encoder,
    )


class AdversarialTrainer:
    """Trainer with adversarial training support."""

    def __init__(
        self, model: nn.Module, device: torch.device, model_name: str = "model"
    ):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "adv_loss": [],
            "adv_acc": [],
        }

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        adv_generator: Optional[HybridAdversarialAttack] = None,
        adv_ratio: float = 0.0,
        adv_method: str = "pgd",
    ) -> Tuple[float, float]:
        """Train for one epoch with optional adversarial training."""
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        for X_batch, y_batch in tqdm(train_loader, desc="Training", leave=False):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            if adv_generator is not None and adv_ratio > 0:
                n_adv = int(len(X_batch) * adv_ratio)
                if n_adv > 0:
                    adv_indices = np.random.choice(len(X_batch), n_adv, replace=False)

                    X_adv = adv_generator.sequence_attack.generate_batch(
                        X_batch[adv_indices].cpu().numpy(),
                        y_batch[adv_indices].cpu().numpy(),
                        method=adv_method,
                    )

                    X_batch_adv = torch.FloatTensor(X_adv).to(self.device)

                    X_batch[adv_indices] = X_batch_adv

                    del X_adv, X_batch_adv
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

                    total += y_batch.size(0)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(y_batch).sum().item()
                    total_loss += loss.item()
                else:
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

                    total += y_batch.size(0)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(y_batch).sum().item()
                    total_loss += loss.item()
            else:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total += y_batch.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(y_batch).sum().item()
                total_loss += loss.item()

        return total_loss / len(train_loader), correct / total

    def evaluate(
        self, dataloader: DataLoader, criterion: nn.Module
    ) -> Tuple[float, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()

        return total_loss / len(dataloader), correct / total

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        adv_generator: Optional[HybridAdversarialAttack] = None,
        adv_ratio: float = 0.0,
        adv_method: str = "pgd",
        save_path: Optional[Path] = None,
        early_stopping_patience: int = 10,
    ) -> Dict:
        """Full training loop with adversarial training."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )

        best_val_acc = 0
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, criterion, adv_generator, adv_ratio, adv_method
            )
            val_loss, val_acc = self.evaluate(val_loader, criterion)

            scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0

                if save_path:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_acc": val_acc,
                            "history": self.history,
                        },
                        save_path / "best_model.pt",
                    )
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        return self.history

    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions."""
        self.model.eval()
        predictions, labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                _, predicted = outputs.max(1)

                predictions.extend(predicted.cpu().numpy())
                labels.extend(y_batch.numpy())

        return np.array(predictions), np.array(labels)


def create_model(
    model_type: str,
    input_size: int,
    num_classes: int,
    seq_length: int,
    device: torch.device,
) -> nn.Module:
    """Create model based on type."""
    if model_type == "lstm":
        model = LSTMClassifier(input_size, num_classes, LSTM_CONFIG)
    elif model_type == "transformer":
        model = TransformerClassifier(
            input_size, num_classes, seq_length, TRANSFORMER_CONFIG
        )
    elif model_type == "cnn_lstm":
        model = CNNLSTMClassifier(input_size, num_classes)
    elif model_type == "cnn":
        model = CNNClassifier(input_size, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


def run_experiment(
    model_type: str,
    seq_length: int,
    adv_method: str,
    adv_ratio: float,
    epochs: int,
    batch_size: int,
    max_files: Optional[int] = None,
    save_results: bool = True,
) -> Dict:
    """Run a single experiment."""
    device = get_device()
    print(f"\n{'=' * 60}")
    print(f"Experiment: {model_type.upper()} | Seq={seq_length} | Adv={adv_method}")
    print(f"Device: {device}")
    print(f"{'=' * 60}\n")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        features,
        scaler,
        label_encoder,
    ) = load_and_preprocess_data(
        seq_length, stride=max(1, seq_length // 2), max_files=max_files
    )

    train_dataset = IoTSequenceDataset(X_train, y_train)
    val_dataset = IoTSequenceDataset(X_val, y_val)
    test_dataset = IoTSequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    input_size = X_train.shape[2]
    num_classes = len(np.unique(y_train))

    print(f"\nInput size: {input_size}, Classes: {num_classes}")

    model = create_model(model_type, input_size, num_classes, seq_length, device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = (
        RESULTS_DIR / "models" / f"{model_type}_{seq_length}_{adv_method}_{timestamp}"
    )
    if save_results:
        save_path.mkdir(parents=True, exist_ok=True)

    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    y_train_expanded = np.repeat(y_train, X_train.shape[1])
    feature_attack = FeatureLevelAttack(
        X_train_flat, y_train_expanded, features, num_classes
    )

    del X_train_flat, y_train_expanded
    gc.collect()

    sequence_attack = SequenceLevelAttack(
        model, device, epsilon=0.1, alpha=0.01, num_steps=10
    )

    adv_generator = HybridAdversarialAttack(
        feature_attack, sequence_attack, features, combine_ratio=0.5
    )

    trainer = AdversarialTrainer(model, device, model_type)

    print(f"\nTraining with adversarial ratio: {adv_ratio}")
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        adv_generator=adv_generator if adv_method != "none" else None,
        adv_ratio=adv_ratio if adv_method != "none" else 0.0,
        adv_method=adv_method if adv_method in ["pgd", "fgsm"] else "pgd",
        save_path=save_path if save_results else None,
    )

    del train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = trainer.evaluate(test_loader, criterion)
    print(f"\nClean Test - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    results = {
        "model_type": model_type,
        "sequence_length": seq_length,
        "adversarial_method": adv_method,
        "adversarial_ratio": adv_ratio,
        "test_accuracy_clean": test_acc,
        "test_loss_clean": test_loss,
        "best_val_accuracy": max(history["val_acc"]),
        "num_classes": num_classes,
        "input_size": input_size,
        "parameters": sum(p.numel() for p in model.parameters()),
    }

    adversarial_results = {}

    if adv_method != "none":
        print("\nGenerating adversarial examples for evaluation...")
        n_eval = min(1000, len(X_test))
        eval_indices = np.random.choice(len(X_test), n_eval, replace=False)

        X_eval = X_test[eval_indices].copy()
        y_eval = y_test[eval_indices].copy()

        print("  Feature-level attack...")
        X_adv_feature = feature_attack.generate_batch(
            X_eval.reshape(-1, X_eval.shape[-1]), y_eval, verbose=True
        ).reshape(X_eval.shape)

        eval_dataset = IoTSequenceDataset(X_adv_feature, y_eval)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
        loss_f, acc_f = trainer.evaluate(eval_loader, criterion)
        adversarial_results["feature_level"] = {"loss": loss_f, "accuracy": acc_f}
        print(f"  Feature-level attack - Loss: {loss_f:.4f}, Acc: {acc_f:.4f}")

        del X_adv_feature, eval_dataset, eval_loader
        gc.collect()

        print("  Sequence-level PGD attack...")
        X_adv_pgd = sequence_attack.generate_batch(
            X_eval, y_eval, method="pgd", verbose=True
        )

        eval_dataset = IoTSequenceDataset(X_adv_pgd, y_eval)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
        loss_p, acc_p = trainer.evaluate(eval_loader, criterion)
        adversarial_results["sequence_pgd"] = {"loss": loss_p, "accuracy": acc_p}
        print(f"  Sequence PGD attack - Loss: {loss_p:.4f}, Acc: {acc_p:.4f}")

        del X_adv_pgd, eval_dataset, eval_loader
        gc.collect()

        print("  Sequence-level FGSM attack...")
        X_adv_fgsm = sequence_attack.generate_batch(
            X_eval, y_eval, method="fgsm", verbose=True
        )

        eval_dataset = IoTSequenceDataset(X_adv_fgsm, y_eval)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
        loss_fgsm, acc_fgsm = trainer.evaluate(eval_loader, criterion)
        adversarial_results["sequence_fgsm"] = {"loss": loss_fgsm, "accuracy": acc_fgsm}
        print(f"  Sequence FGSM attack - Loss: {loss_fgsm:.4f}, Acc: {acc_fgsm:.4f}")

        del X_adv_fgsm, eval_dataset, eval_loader
        gc.collect()

        print("  Hybrid attack...")
        X_adv_hybrid = adv_generator.generate_batch(
            X_eval, y_eval, method="hybrid", verbose=True
        )

        eval_dataset = IoTSequenceDataset(X_adv_hybrid, y_eval)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
        loss_h, acc_h = trainer.evaluate(eval_loader, criterion)
        adversarial_results["hybrid"] = {"loss": loss_h, "accuracy": acc_h}
        print(f"  Hybrid attack - Loss: {loss_h:.4f}, Acc: {acc_h:.4f}")

        del X_adv_hybrid, eval_dataset, eval_loader, X_eval, y_eval
        gc.collect()

        results["adversarial_results"] = adversarial_results

        results["robustness_ratios"] = {
            k: v["accuracy"] / test_acc for k, v in adversarial_results.items()
        }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if save_results:
        with open(save_path / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        with open(save_path / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        with open(save_path / "preprocessor.pkl", "wb") as f:
            pickle.dump(
                {
                    "scaler": scaler,
                    "label_encoder": label_encoder,
                    "features": features,
                    "seq_length": seq_length,
                },
                f,
            )

        print(f"\nResults saved to: {save_path}")

    del X_train, X_val, X_test, y_train, y_val, y_test
    del test_loader, test_dataset, train_dataset, val_dataset
    del model, trainer, feature_attack, sequence_attack, adv_generator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def compare_models(
    seq_lengths: List[int],
    models: List[str],
    adv_methods: List[str],
    epochs: int,
    max_files: Optional[int] = None,
) -> Dict:
    """Compare multiple model architectures and adversarial methods."""
    all_results = {}

    for seq_len in seq_lengths:
        for model_type in models:
            for adv_method in adv_methods:
                key = f"{model_type}_seq{seq_len}_{adv_method}"
                print(f"\n{'=' * 60}")
                print(f"Running: {key}")
                print(f"{'=' * 60}")

                results = run_experiment(
                    model_type=model_type,
                    seq_length=seq_len,
                    adv_method=adv_method,
                    adv_ratio=0.2 if adv_method != "none" else 0.0,
                    epochs=epochs,
                    batch_size=BATCH_SIZE,
                    max_files=max_files,
                    save_results=True,
                )
                all_results[key] = results

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    print("\n| Model | Seq | Adv Method | Clean Acc | Feature | PGD | FGSM | Hybrid |")
    print("|-------|-----|------------|-----------|---------|-----|------|--------|")

    for key, res in all_results.items():
        clean_acc = res["test_accuracy_clean"]

        if "adversarial_results" in res:
            feat = (
                res["adversarial_results"].get("feature_level", {}).get("accuracy", 0)
            )
            pgd = res["adversarial_results"].get("sequence_pgd", {}).get("accuracy", 0)
            fgsm = (
                res["adversarial_results"].get("sequence_fgsm", {}).get("accuracy", 0)
            )
            hybrid = res["adversarial_results"].get("hybrid", {}).get("accuracy", 0)
        else:
            feat = pgd = fgsm = hybrid = "-"

        print(
            f"| {key.split('_')[0]} | {res['sequence_length']} | {res['adversarial_method']:10} | {clean_acc:.4f} | {feat:.4f} | {pgd:.4f} | {fgsm:.4f} | {hybrid:.4f} |"
        )

    comparison_path = RESULTS_DIR / "comparison_results.json"
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nComparison saved to: {comparison_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Adversarial Training for IoT Device Identification"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="lstm",
        choices=["lstm", "transformer", "cnn_lstm", "cnn"],
        help="Model architecture",
    )
    parser.add_argument("--seq_length", type=int, default=10, help="Sequence length")
    parser.add_argument(
        "--adv_method",
        type=str,
        default="hybrid",
        choices=["none", "feature", "pgd", "fgsm", "hybrid"],
        help="Adversarial attack method",
    )
    parser.add_argument(
        "--adv_ratio",
        type=float,
        default=0.2,
        help="Ratio of adversarial samples in training",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of data files to load",
    )

    parser.add_argument(
        "--compare_all",
        action="store_true",
        help="Compare all models and sequence lengths",
    )
    parser.add_argument(
        "--seq_lengths",
        type=str,
        default="10,25,50",
        help="Comma-separated sequence lengths for comparison",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="lstm,transformer,cnn_lstm",
        help="Comma-separated model types for comparison",
    )
    parser.add_argument(
        "--adv_methods",
        type=str,
        default="none,feature,pgd,hybrid",
        help="Comma-separated adversarial methods for comparison",
    )

    args = parser.parse_args()

    if args.compare_all:
        seq_lengths = [int(x) for x in args.seq_lengths.split(",")]
        models = args.models.split(",")
        adv_methods = args.adv_methods.split(",")

        compare_models(
            seq_lengths=seq_lengths,
            models=models,
            adv_methods=adv_methods,
            epochs=args.epochs,
            max_files=args.max_files,
        )
    else:
        run_experiment(
            model_type=args.model,
            seq_length=args.seq_length,
            adv_method=args.adv_method,
            adv_ratio=args.adv_ratio,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_files=args.max_files,
            save_results=True,
        )


if __name__ == "__main__":
    main()
