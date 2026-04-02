#!/usr/bin/env python3
"""
Hybrid Adversarial Training for IoT Device Identification

This script implements:
1. Multiple model architectures (LSTM, Transformer, CNN-LSTM)
2. Multiple sequence lengths for experimentation
3. Feature-level adversarial attacks (IoT-SDN style)
4. Sequence-level adversarial attacks (adversarial search via mimicry)
5. Hybrid adversarial training with 60% clean + 20% feature-level + 20% sequence-level
6. Comprehensive robustness evaluation

Usage:
    # Default hybrid training: 60% clean + 20% feature + 20% sequence
    python train_adversarial.py --model lstm --seq_length 10 --adv_method hybrid
    
    # Custom hybrid split
    python train_adversarial.py --model cnn_lstm --seq_length 50 --adv_method hybrid \
        --hybrid_clean 0.5 --hybrid_feature 0.3 --hybrid_sequence 0.2
    
    # Compare all configurations
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
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server/Colab
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import sys

BASE_DIR = Path(__file__).parent
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
    PIPELINE_MODE,
    JSON_DATA_DIR,
    JSON_PROCESSED_DATA_DIR,
    JSON_N_CONTINUOUS,
    JSON_N_BINARY,
    JSON_INPUT_SIZE,
)
from src.models.lstm import LSTMClassifier
import sys, importlib

_trainer_mod = importlib.import_module("src.training.trainer")
IoTSequenceDataset = _trainer_mod.IoTSequenceDataset
from src.models.transformer import TransformerClassifier, NLPTransformerClassifier
from src.data.tokenizer import IoTTokenizer
from src.models.cnn_lstm import CNNLSTMClassifier, CNNClassifier
from src.models.xgboost_lstm import XGBoostLSTMClassifier
from src.models.cnn_bilstm_transformer import CNNBiLSTMTransformerClassifier
from src.adversarial.attacks import (
    FeatureLevelAttack,
    SequenceLevelAttack,
    HybridAdversarialAttack,
    AdversarialEvaluator,
)

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
        for i in range(0, max(1, n_samples), stride):
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
    seq_length: int,
    stride: int = 5,
    max_files: Optional[int] = None,
    data_dir: Optional[Path] = None,
    pipeline_mode: Optional[str] = None,
    max_records: Optional[int] = None,
    dataset: Optional[str] = None,
    low_memory: bool = True,
) -> Tuple[np.ndarray, ...]:
    """Load and preprocess data with specified sequence length.

    Supports two pipeline modes:
    - 'csv': CSV pipeline (IPFIX ML Instances, 18 classes) — anti-leakage temporal split
    - 'json': JSON-native pipeline (IPFIX Records, 17-26 classes)

    Args:
        dataset: Explicit override for pipeline selection ('csv' or 'json').
                 Takes precedence over pipeline_mode / PIPELINE_MODE config.
        low_memory: If True, uses aggressive memory cleanup during loading.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test,
                  features, scaler, label_encoder, n_continuous_features)
    """
    if low_memory:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if dataset is not None:
        pipeline_mode = dataset
    if pipeline_mode is None:
        pipeline_mode = PIPELINE_MODE

    print(f"Pipeline mode: {pipeline_mode.upper()}")

    if pipeline_mode == "json":
        result = _load_json_pipeline(seq_length, stride, data_dir, max_records)
    else:
        result = _load_csv_pipeline(seq_length, stride, max_files, data_dir)

    if low_memory:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


def _load_json_pipeline(
    seq_length: int,
    stride: int = 5,
    data_dir: Optional[Path] = None,
    max_records: Optional[int] = None,
) -> Tuple[np.ndarray, ...]:
    """Load JSON IPFIX Records using the native JSON preprocessor.

    Addresses:
    - Issue 1: Uses JSON dataset natively
    - Issue 2: Labels via MAC-to-device mapping
    - Issue 3: Bidirectional flow labeling
    - Issue 4: Drops identifying columns
    - Issue 5: Decodes hex packet directions
    """
    from src.data.json_preprocessor import JsonIoTDataProcessor

    if data_dir is None:
        data_dir = JSON_DATA_DIR

    print(f"  JSON Data directory: {data_dir}")

    processor = JsonIoTDataProcessor()
    result = processor.process_all(
        data_dir=data_dir,
        seq_length=seq_length,
        stride=stride,
        max_records=max_records,
    )

    X_train, X_val, X_test, y_train, y_val, y_test, features, scaler, label_encoder = (
        result
    )
    n_continuous = JSON_N_CONTINUOUS

    print(f"\n  JSON Pipeline Summary:")
    print(
        f"    Features: {len(features)} ({n_continuous} continuous + {JSON_N_BINARY} binary)"
    )
    print(f"    Classes: {len(label_encoder.classes_)}")
    print(f"    Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

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
        n_continuous,
    )


def _load_csv_pipeline(
    seq_length: int,
    stride: int = 5,
    max_files: Optional[int] = None,
    data_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, ...]:
    """CSV pipeline — delegates to IoTDataProcessor for strict anti-leakage split.

    Uses IoTDataProcessor.process_all() which implements:
      - Per-device temporal 80/20 split (docs/general §Étape 3)
      - Sequences built independently on train and test (docs/general §Étape 4)
      - StandardScaler fitted on training data ONLY
    """
    from src.data.preprocessor import IoTDataProcessor

    print(f"Loading CSV data with sequence length={seq_length}, stride={stride}")

    if data_dir is None:
        data_dir = RAW_DATA_DIR

    print(f"  Data directory: {data_dir}")

    processor = IoTDataProcessor()
    result = processor.process_all(
        max_files=max_files,
        data_dir=data_dir,
        seq_length=seq_length,
        stride=stride,
    )

    X_train, X_val, X_test, y_train, y_val, y_test, features, scaler, label_encoder = (
        result
    )

    # CSV pipeline: all features are continuous (no binary packet direction bits)
    n_continuous = len(features)

    print(f"\n  CSV Pipeline Summary:")
    print(f"    Features: {n_continuous} (all continuous)")
    print(f"    Classes: {len(label_encoder.classes_)}")
    print(f"    Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

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
        n_continuous,
    )


class AdversarialTrainer:
    """Trainer with adversarial training support."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        model_name: str = "model",
        tokenizer=None,
        features: Optional[List[str]] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.features = features
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
        adv_method: str = "hybrid",
        hybrid_split: Optional[Dict[str, float]] = None,
        X_raw_batch: Optional[np.ndarray] = None,
        sensitivity_results: Optional[List[Dict]] = None,
    ) -> Tuple[float, float]:
        """Train for one epoch with optional adversarial training.

        Args:
            hybrid_split: Dict with 'clean', 'feature', 'sequence' ratios (must sum to 1.0)
                         Default: {'clean': 0.6, 'feature': 0.2, 'sequence': 0.2}
            X_raw_batch: Raw feature data for re-tokenization (NLP transformer mode)
            sensitivity_results: Output from sensitivity analysis for adversarial search
        """
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        # Default hybrid split: 60% clean, 20% feature-level, 20% sequence-level
        if hybrid_split is None:
            hybrid_split = {"clean": 0.6, "feature": 0.2, "sequence": 0.2}

        is_nlp = self.tokenizer is not None and self.features is not None

        for batch_idx, (X_batch, y_batch) in enumerate(
            tqdm(train_loader, desc="Training", leave=False)
        ):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            if is_nlp and X_raw_batch is not None:
                raw_slice = X_raw_batch[
                    batch_idx * len(X_batch) : (batch_idx + 1) * len(X_batch)
                ]
            else:
                raw_slice = None

            if adv_generator is not None and adv_ratio > 0:
                n_adv = int(len(X_batch) * adv_ratio)
                if n_adv > 0:
                    # Calculate split sizes based on hybrid ratios
                    n_feature = int(
                        n_adv
                        * hybrid_split.get("feature", 0.2)
                        / (
                            hybrid_split.get("feature", 0.2)
                            + hybrid_split.get("sequence", 0.2)
                        )
                    )
                    n_sequence = n_adv - n_feature

                    adv_indices = np.random.choice(len(X_batch), n_adv, replace=False)

                    # Split indices for feature-level and sequence-level attacks
                    idx_feature = adv_indices[:n_feature]
                    idx_sequence = adv_indices[n_feature : n_feature + n_sequence]

                    if is_nlp and raw_slice is not None:
                        raw_adv = raw_slice.copy()
                        raw_adv_feature = raw_adv[idx_feature]
                        raw_adv_sequence = raw_adv[idx_sequence]
                    else:
                        raw_adv_feature = X_batch[idx_feature].cpu().numpy()
                        raw_adv_sequence = X_batch[idx_sequence].cpu().numpy()

                    # Apply feature-level attack
                    if len(idx_feature) > 0:
                        X_adv_feature = adv_generator.feature_attack.generate_batch(
                            raw_adv_feature,
                            y_batch[idx_feature].cpu().numpy(),
                        )
                        if is_nlp:
                            tok_adv_feature = tokenize_adversarial_batch(
                                X_adv_feature, self.features, self.tokenizer
                            )
                            X_batch[idx_feature] = torch.LongTensor(tok_adv_feature).to(
                                self.device
                            )
                        else:
                            X_batch[idx_feature] = torch.FloatTensor(X_adv_feature).to(
                                self.device
                            )
                        del X_adv_feature

                    # Apply sequence-level attack (adversarial search)
                    if len(idx_sequence) > 0:
                        if sensitivity_results is None:
                            X_adv_sequence = (
                                adv_generator.feature_attack.generate_batch(
                                    raw_adv_sequence,
                                    y_batch[idx_sequence].cpu().numpy(),
                                )
                            )
                        else:
                            X_adv_sequence = (
                                adv_generator.sequence_attack.generate_batch(
                                    raw_adv_sequence,
                                    y_batch[idx_sequence].cpu().numpy(),
                                    sensitivity_results=sensitivity_results,
                                )
                            )
                        if is_nlp:
                            tok_adv_sequence = tokenize_adversarial_batch(
                                X_adv_sequence, self.features, self.tokenizer
                            )
                            X_batch[idx_sequence] = torch.LongTensor(
                                tok_adv_sequence
                            ).to(self.device)
                        else:
                            X_batch[idx_sequence] = torch.FloatTensor(
                                X_adv_sequence
                            ).to(self.device)
                        del X_adv_sequence

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

    def test_model(
        self, test_loader: DataLoader, criterion: nn.Module, phase_name: str = "Test"
    ) -> Tuple[float, float]:
        """Test the model and print results."""
        test_loss, test_acc = self.evaluate(test_loader, criterion)
        print(f"\n{'=' * 50}")
        print(f"  {phase_name} - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        print(f"{'=' * 50}\n")
        return test_loss, test_acc

    def compute_per_class_metrics(
        self,
        dataloader: DataLoader,
        label_encoder,
        num_classes: int,
    ) -> Dict:
        """Compute per-class Precision, Recall, F1 and macro averages."""
        from sklearn.metrics import (
            precision_recall_fscore_support,
            classification_report,
        )

        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = float(np.mean(all_preds == all_labels))

        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )

        class_names = (
            list(label_encoder.classes_)
            if label_encoder is not None
            else [str(i) for i in range(num_classes)]
        )

        per_class = {}
        for i in range(min(len(class_names), len(precision))):
            per_class[class_names[i]] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }

        return {
            "accuracy": accuracy,
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "per_class": per_class,
            "all_preds": all_preds,
            "all_labels": all_labels,
        }

    def _compute_detailed_metrics(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        num_classes: int,
    ) -> Dict:
        """Compute detailed metrics: Loss, Accuracy, Precision, Recall, F1 (macro)."""
        from sklearn.metrics import precision_recall_fscore_support

        self.model.eval()
        total_loss, total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y_batch.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = np.mean(all_preds == all_labels)
        avg_loss = total_loss / max(len(dataloader), 1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )

        return {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }

    def _crash_test(
        self,
        phase_num: int,
        test_loader: DataLoader,
        criterion: nn.Module,
        feature_attack: Optional[FeatureLevelAttack],
        sequence_attack: Optional[SequenceLevelAttack],
        X_test: Optional[np.ndarray],
        y_test: Optional[np.ndarray],
        batch_size: int,
        num_classes: int,
        label_encoder=None,
        save_path: Optional[Path] = None,
    ) -> Dict:
        """Crash Test exhaustif après chaque phase.

        Évalue le modèle sur 4 loaders:
        1. Test_Loader_Normal (données pures)
        2. Test_Loader_Adv_Feature (attaque feature-level mimicry)
        3. Test_Loader_Adv_Seq_Search (recherche adversariale gloutonne)

        Métriques par loader: Loss, Accuracy, Precision, Recall, F1, Robustness Ratio.
        Generates per-phase plots for the jury presentation.
        """
        print(f"\n{'─' * 60}")
        print(f"  CRASH TEST — Phase {phase_num}")
        print(f"{'─' * 60}")

        crash_results = {"phase": phase_num}

        # 1. Clean Test (with per-class metrics)
        clean_metrics = self._compute_detailed_metrics(
            test_loader, criterion, num_classes
        )
        clean_per_class_data = self.compute_per_class_metrics(
            test_loader, label_encoder, num_classes
        )
        crash_results["clean"] = clean_metrics
        crash_results["clean"]["macro_f1"] = clean_per_class_data["macro_f1"]
        crash_results["clean"]["per_class"] = clean_per_class_data["per_class"]
        clean_acc = clean_metrics["accuracy"]
        print(
            f"  [Clean]       Loss={clean_metrics['loss']:.4f}  Acc={clean_acc:.4f}  "
            f"P={clean_metrics['precision']:.4f}  R={clean_metrics['recall']:.4f}  "
            f"F1={clean_metrics['f1_score']:.4f}  Macro-F1={clean_per_class_data['macro_f1']:.4f}"
        )

        adv_feature_per_class = {}
        adv_search_per_class = {}

        # 2. Feature-level attack
        if feature_attack is not None and X_test is not None and y_test is not None:
            n_eval = min(1000, len(X_test))
            eval_indices = np.random.choice(len(X_test), n_eval, replace=False)
            X_eval = X_test[eval_indices].copy()
            y_eval = y_test[eval_indices].copy()

            y_eval_expanded = np.repeat(y_eval, X_eval.shape[1])
            X_adv_feature = feature_attack.generate_batch(
                X_eval.reshape(-1, X_eval.shape[-1]), y_eval_expanded, verbose=False
            ).reshape(X_eval.shape)

            eval_dataset = IoTSequenceDataset(X_adv_feature, y_eval)
            eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
            feat_metrics = self._compute_detailed_metrics(
                eval_loader, criterion, num_classes
            )
            feat_per_class = self.compute_per_class_metrics(
                eval_loader, label_encoder, num_classes
            )
            feat_metrics["robustness_ratio"] = feat_metrics["accuracy"] / max(
                clean_acc, 1e-8
            )
            feat_metrics["macro_f1"] = feat_per_class["macro_f1"]
            feat_metrics["per_class"] = feat_per_class["per_class"]
            adv_feature_per_class = feat_per_class["per_class"]
            crash_results["feature_attack"] = feat_metrics
            print(
                f"  [FeatureAdv]  Loss={feat_metrics['loss']:.4f}  Acc={feat_metrics['accuracy']:.4f}  "
                f"P={feat_metrics['precision']:.4f}  R={feat_metrics['recall']:.4f}  "
                f"F1={feat_metrics['f1_score']:.4f}  Macro-F1={feat_per_class['macro_f1']:.4f}  "
                f"RR={feat_metrics['robustness_ratio']:.4f}"
            )

            del X_adv_feature, eval_dataset, eval_loader
            gc.collect()

        # 3. Sequence-level adversarial search
        if sequence_attack is not None and X_test is not None and y_test is not None:
            n_eval = min(1000, len(X_test))
            eval_indices = np.random.choice(len(X_test), n_eval, replace=False)
            X_eval = X_test[eval_indices].copy()
            y_eval = y_test[eval_indices].copy()

            sensitivity_results = feature_attack.analyze_sensitivity(
                self.model,
                X_eval,
                y_eval,
                self.device,
                batch_size=batch_size,
                verbose=False,
            )

            X_adv_search = sequence_attack.generate_batch(
                X_eval, y_eval, sensitivity_results=sensitivity_results, verbose=False
            )
            eval_dataset = IoTSequenceDataset(X_adv_search, y_eval)
            eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
            search_metrics = self._compute_detailed_metrics(
                eval_loader, criterion, num_classes
            )
            search_per_class = self.compute_per_class_metrics(
                eval_loader, label_encoder, num_classes
            )
            search_metrics["robustness_ratio"] = search_metrics["accuracy"] / max(
                clean_acc, 1e-8
            )
            search_metrics["macro_f1"] = search_per_class["macro_f1"]
            search_metrics["per_class"] = search_per_class["per_class"]
            adv_search_per_class = search_per_class["per_class"]
            crash_results["sequence_search"] = search_metrics
            print(
                f"  [SeqSearch]   Loss={search_metrics['loss']:.4f}  Acc={search_metrics['accuracy']:.4f}  "
                f"P={search_metrics['precision']:.4f}  R={search_metrics['recall']:.4f}  "
                f"F1={search_metrics['f1_score']:.4f}  Macro-F1={search_per_class['macro_f1']:.4f}  "
                f"RR={search_metrics['robustness_ratio']:.4f}"
            )

            del X_adv_search, eval_dataset, eval_loader, X_eval, y_eval
            gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Generate per-phase plots ────────────────────────────
        if save_path and clean_per_class_data.get("per_class"):
            try:
                phase_label = f"phase{phase_num}"
                print(f"\n  Generating Phase {phase_num} plots...")

                # Plot 1: Per-device P/R/F1
                generate_device_identification_plot(
                    clean_per_class_data["per_class"],
                    save_path,
                    f"{self.model_name} (Phase {phase_num})",
                )
                # Rename to include phase
                src = save_path / "fig_device_identification_scores.png"
                dst = save_path / f"fig_device_identification_scores_{phase_label}.png"
                if src.exists():
                    src.rename(dst)

                # Plot 2: Adversarial effect
                if adv_feature_per_class or adv_search_per_class:
                    generate_adversarial_effect_plot(
                        clean_per_class_data["per_class"],
                        adv_feature_per_class,
                        adv_search_per_class,
                        save_path,
                        f"{self.model_name} (Phase {phase_num})",
                    )
                    src = save_path / "fig_adversarial_effect.png"
                    dst = save_path / f"fig_adversarial_effect_{phase_label}.png"
                    if src.exists():
                        src.rename(dst)

                # Plot 3: Robustness summary
                summary_data = {
                    "clean_metrics": {
                        "accuracy": clean_acc,
                        "macro_f1": clean_per_class_data["macro_f1"],
                    }
                }
                if "feature_attack" in crash_results:
                    summary_data["adv_feature_metrics"] = {
                        "accuracy": crash_results["feature_attack"]["accuracy"],
                        "macro_f1": crash_results["feature_attack"].get("macro_f1", 0),
                    }
                if "sequence_search" in crash_results:
                    summary_data["adv_search_metrics"] = {
                        "accuracy": crash_results["sequence_search"]["accuracy"],
                        "macro_f1": crash_results["sequence_search"].get("macro_f1", 0),
                    }
                generate_robustness_summary_plot(
                    summary_data,
                    save_path,
                    f"{self.model_name} (Phase {phase_num})",
                )
                src = save_path / "fig_robustness_summary.png"
                dst = save_path / f"fig_robustness_summary_{phase_label}.png"
                if src.exists():
                    src.rename(dst)

            except Exception as e:
                print(f"  ⚠️ Phase {phase_num} plot error (non-fatal): {e}")

        return crash_results

    def fit_with_phase_checkpoints(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 30,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        adv_generator: Optional[HybridAdversarialAttack] = None,
        adv_ratio: float = 0.0,
        adv_method: str = "search",
        hybrid_split: Optional[Dict[str, float]] = None,
        save_path: Optional[Path] = None,
        early_stopping_patience: int = 5,
        feature_attack: Optional[FeatureLevelAttack] = None,
        sequence_attack: Optional[SequenceLevelAttack] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        batch_size: int = 64,
        is_xgboost: bool = False,
        label_encoder=None,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        X_train_raw: Optional[np.ndarray] = None,
    ) -> Dict:
        """Entraînement en 2 phases per docs/train.

        PHASE 1 : Entraînement Standard et Constat de Vulnérabilité
        1. Entraînement initial (Données Bénignes Uniquement)
        2. Crash Test 1:
           - Test 1: Données Bénignes
           - Test 2: Données Adversaires Uniquement
           - Test 3: Mélange Bénignes + Adversaires

        PHASE 2 : Entraînement Antagoniste (Adversarial Training) et Robustesse
        1. Création du Corpus Conjoint (bénignes + adversaires)
        2. Ré-entraînement sur corpus conjoint
        3. Crash Test 2 (mêmes 3 tests)

        Early Stopping par phase avec patience réinitialisée.
        Pour XGBoost-LSTM: fit XGBoost avant l'évaluation finale.
        """
        import copy

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )

        sample_y = next(iter(test_loader))[1]
        num_classes = int(sample_y.max().item()) + 1

        phase1_epochs = 20
        phase2_epochs = 10

        print(f"\n{'=' * 70}")
        print("  ENTRAÎNEMENT EN 2 PHASES (per docs/train)")
        print(f"{'=' * 70}")
        print(f"  Phase 1 (Entraînement Standard):       max {phase1_epochs} epochs")
        if adv_generator and adv_ratio > 0:
            print(
                f"  Phase 2 (Entraînement Antagoniste):    max {phase2_epochs} epochs"
            )
        print(f"{'=' * 70}\n")

        all_crash_results = {}

        # ─────────────────────────────────────────────────────────
        # PHASE 1 : Entraînement Standard (Données Bénignes Uniquement)
        # ─────────────────────────────────────────────────────────
        print(f"\n{'=' * 60}")
        print("PHASE 1: ENTRAÎNEMENT STANDARD (Données Bénignes Uniquement)")
        print(f"{'=' * 60}\n")

        best_val_acc_p1 = 0.0
        best_state_p1 = copy.deepcopy(self.model.state_dict())
        patience_counter = 0

        for epoch in range(phase1_epochs):
            train_loss, train_acc = self.train_epoch(
                train_loader,
                optimizer,
                criterion,
                None,
                0.0,
                adv_method,
                None,
                X_raw_batch=X_train_raw,
            )
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(
                f"  P1 Epoch {epoch + 1}/{phase1_epochs}  "
                f"Train[loss={train_loss:.4f} acc={train_acc:.4f}]  "
                f"Val[loss={val_loss:.4f} acc={val_acc:.4f}]"
            )

            if val_acc > best_val_acc_p1:
                best_val_acc_p1 = val_acc
                best_state_p1 = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"  ⚡ Early Stopping Phase 1 à l'epoch {epoch + 1}")
                    break

        self.model.load_state_dict(best_state_p1)
        print(f"  ✓ Meilleurs poids Phase 1 rechargés (val_acc={best_val_acc_p1:.4f})")

        if save_path:
            checkpoint_path = save_path / "checkpoint_phase1.pt"
            torch.save(
                {
                    "model_state_dict": best_state_p1,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc_p1,
                    "phase": 1,
                },
                checkpoint_path,
            )
            print(f"  💾 Checkpoint P1: {checkpoint_path}")

        # Crash Test 1: 3 tests (benign, adversarial, mixed)
        crash_p1 = self._crash_test_2phase(
            1,
            test_loader,
            criterion,
            feature_attack,
            sequence_attack,
            X_test,
            y_test,
            batch_size,
            num_classes,
            label_encoder=label_encoder,
            save_path=save_path,
        )
        all_crash_results["phase_1"] = crash_p1

        # ─────────────────────────────────────────────────────────
        # GÉNÉRATION DES ATTAQUES (Post-Phase 1)
        # ─────────────────────────────────────────────────────────
        if (
            adv_generator is not None
            and adv_ratio > 0
            and sequence_attack is not None
            and X_train is not None
            and y_train is not None
        ):
            print(f"\n{'=' * 70}")
            print("  GÉNÉRATION DES ÉCHANTILLONS ADVERSAIRES (Post-Phase 1)")
            print(f"{'=' * 70}\n")

            n_adv = int(len(X_train) * adv_ratio)
            adv_indices = np.random.choice(len(X_train), n_adv, replace=False)

            # 1) Sensitivity analysis
            print("  [1/3] Analyse de sensibilité du modèle Phase 1...")
            sensitivity_results = feature_attack.analyze_sensitivity(
                self.model,
                X_train[adv_indices],
                y_train[adv_indices],
                self.device,
                batch_size=batch_size,
                verbose=False,
            )

            # Display top vulnerable features
            sorted_sens = sorted(
                sensitivity_results, key=lambda r: r["drop"], reverse=True
            )
            seen_features = set()
            top_vulnerable = []
            for entry in sorted_sens:
                if entry["feature"] not in seen_features:
                    seen_features.add(entry["feature"])
                    top_vulnerable.append(entry)
                if len(top_vulnerable) >= 5:
                    break

            print(f"  Top 5 features les plus vulnérables :")
            for rank, entry in enumerate(top_vulnerable, 1):
                print(
                    f"    {rank}. {entry['feature']:<25s} → Stratégie: {entry['strategy']:<15s} "
                    f"→ Drop: {entry['drop'] * 100:.1f}%"
                )

            # 2) Generate adversarial batch
            print(
                f"\n  [2/3] Génération du batch adversaire (n={n_adv:,} échantillons)..."
            )
            print(
                f"  Méthode : {adv_method} (60% clean + 20% feature-level + 20% sequence-level)"
            )

            X_adv = sequence_attack.generate_batch(
                X_train[adv_indices],
                y_train[adv_indices],
                sensitivity_results=sensitivity_results,
                verbose=False,
            )

            # 3) Evaluate adversarial effectiveness
            print(f"\n  [3/3] Évaluation de l'efficacité des attaques...")

            # Clean accuracy on the same subset
            clean_acc = self._evaluate_accuracy(
                X_train[adv_indices], y_train[adv_indices], batch_size=batch_size
            )
            adv_acc = self._evaluate_accuracy(
                X_adv, y_train[adv_indices], batch_size=batch_size
            )
            degradation = clean_acc - adv_acc

            print(f"  Résultats d'attaque :")
            print(
                f"    Accuracy sur données bénignes (Phase 1) : {clean_acc * 100:.1f}%"
            )
            print(
                f"    Accuracy sur données adversaires         : {adv_acc * 100:.1f}%"
            )
            print(
                f"    ↓ Dégradation                            : {degradation * 100:.1f}%"
            )

            if degradation > 0.5:
                print(
                    f"\n  ⚠️  Le modèle Phase 1 est VULNÉRABLE aux attaques adversaires."
                )
            else:
                print(f"\n  ✓ Le modèle Phase 1 montre une bonne résistance.")

            # Store for Phase 2
            X_adv_precomputed = X_adv
            y_adv_precomputed = y_train[adv_indices].copy()
            if X_train_raw is not None:
                X_adv_raw_precomputed = X_train_raw[adv_indices]
            else:
                X_adv_raw_precomputed = None
            sensitivity_results_precomputed = sensitivity_results
        else:
            X_adv_precomputed = None
            y_adv_precomputed = None
            X_adv_raw_precomputed = None
            sensitivity_results_precomputed = None

        # Early exit si pas d'entraînement adversarial
        if adv_generator is None or adv_ratio <= 0:
            self.history["phase_checkpoints_evaluation"] = all_crash_results
            if save_path:
                self._save_phase_report(all_crash_results, save_path)
            return self.history

        # ─────────────────────────────────────────────────────────
        # PHASE 2 : Entraînement Antagoniste (Corpus Conjoint)
        # ─────────────────────────────────────────────────────────
        print(f"\n{'=' * 60}")
        print("PHASE 2: ENTRAÎNEMENT ANTAGONISTE (Corpus Conjoint)")
        print(f"{'=' * 60}\n")

        # Créer le corpus conjoint avec les attaques pré-générées
        print("  Création du corpus conjoint (bénignes + adversaires)...")

        adv_train_loader = train_loader
        X_joint_raw = X_train_raw
        if (
            X_train is not None
            and y_train is not None
            and X_adv_precomputed is not None
        ):
            n_adv = len(X_adv_precomputed)
            if n_adv > 0:
                X_joint = np.concatenate([X_train, X_adv_precomputed], axis=0)
                y_joint = np.concatenate([y_train, y_adv_precomputed], axis=0)
                if X_train_raw is not None and X_adv_raw_precomputed is not None:
                    X_joint_raw = np.concatenate(
                        [X_train_raw, X_adv_raw_precomputed], axis=0
                    )
                joint_dataset = IoTSequenceDataset(X_joint, y_joint)
                adv_train_loader = DataLoader(
                    joint_dataset, batch_size=batch_size, shuffle=True
                )
                print(
                    f"    Corpus conjoint: {len(X_train):,} bénignes + {n_adv:,} adversaires = {len(X_joint):,} total"
                )

        best_val_acc_p2 = 0.0
        best_state_p2 = copy.deepcopy(self.model.state_dict())
        patience_counter = 0

        for epoch in range(phase2_epochs):
            train_loss, train_acc = self.train_epoch(
                adv_train_loader,
                optimizer,
                criterion,
                None,
                0.0,
                adv_method,
                None,
                X_raw_batch=X_joint_raw,
            )
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(
                f"  P2 Epoch {epoch + 1}/{phase2_epochs}  "
                f"Train[loss={train_loss:.4f} acc={train_acc:.4f}]  "
                f"Val[loss={val_loss:.4f} acc={val_acc:.4f}]"
            )

            if val_acc > best_val_acc_p2:
                best_val_acc_p2 = val_acc
                best_state_p2 = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"  ⚡ Early Stopping Phase 2 à l'epoch {epoch + 1}")
                    break

        self.model.load_state_dict(best_state_p2)
        print(f"  ✓ Meilleurs poids Phase 2 rechargés (val_acc={best_val_acc_p2:.4f})")

        # XGBoost: fit l'arbre AVANT le Crash Test Phase 2
        if is_xgboost:
            print("\n  🌲 Fitting XGBoost sur features LSTM extraites...")
            self.model.fit_xgboost(adv_train_loader, self.device)

        if save_path:
            checkpoint_path = save_path / "checkpoint_phase2.pt"
            torch.save(
                {
                    "model_state_dict": best_state_p2,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc_p2,
                    "phase": 2,
                },
                checkpoint_path,
            )
            print(f"  💾 Checkpoint P2: {checkpoint_path}")

        # Crash Test 2: 3 tests (benign, adversarial, mixed)
        crash_p2 = self._crash_test_2phase(
            2,
            test_loader,
            criterion,
            feature_attack,
            sequence_attack,
            X_test,
            y_test,
            batch_size,
            num_classes,
            label_encoder=label_encoder,
            save_path=save_path,
        )
        all_crash_results["phase_2"] = crash_p2

        if save_path:
            final_path = save_path / "best_model.pt"
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc_p2,
                    "history": self.history,
                    "phases": {
                        "phase1_epochs": phase1_epochs,
                        "phase2_epochs": phase2_epochs,
                    },
                },
                final_path,
            )

        self._print_comparative_summary_2phase(all_crash_results)

        self.history["phase_checkpoints_evaluation"] = all_crash_results

        if save_path:
            self._save_phase_report(all_crash_results, save_path)

        return self.history

    def _evaluate_accuracy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 64,
    ) -> float:
        """Evaluate accuracy on raw numpy arrays (no DataLoader needed)."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                X_batch = torch.FloatTensor(X[i : i + batch_size]).to(self.device)
                y_batch = torch.LongTensor(y[i : i + batch_size]).to(self.device)
                outputs = self.model(X_batch)
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()
        return correct / total if total > 0 else 0.0

    def _crash_test_2phase(
        self,
        phase_num: int,
        test_loader: DataLoader,
        criterion: nn.Module,
        feature_attack: Optional[FeatureLevelAttack],
        sequence_attack: Optional[SequenceLevelAttack],
        X_test: Optional[np.ndarray],
        y_test: Optional[np.ndarray],
        batch_size: int,
        num_classes: int,
        label_encoder=None,
        save_path: Optional[Path] = None,
    ) -> Dict:
        """Crash Test per docs/train: 3 tests (benign, adversarial, mixed)."""
        print(f"\n{'─' * 60}")
        print(f"  CRASH TEST — Phase {phase_num}")
        print(f"{'─' * 60}")

        crash_results = {"phase": phase_num}

        clean_metrics = self._compute_detailed_metrics(
            test_loader, criterion, num_classes
        )
        clean_per_class = self.compute_per_class_metrics(
            test_loader, label_encoder, num_classes
        )
        crash_results["test1_benign"] = clean_metrics
        crash_results["test1_benign"]["macro_f1"] = clean_per_class["macro_f1"]
        crash_results["test1_benign"]["per_class"] = clean_per_class["per_class"]
        print(
            f"  [Test 1 - Bénignes]     Loss={clean_metrics['loss']:.4f}  Acc={clean_metrics['accuracy']:.4f}  "
            f"F1={clean_metrics['f1_score']:.4f}  Macro-F1={clean_per_class['macro_f1']:.4f}"
        )

        if (
            sequence_attack is not None
            and feature_attack is not None
            and X_test is not None
            and y_test is not None
        ):
            n_eval = min(1000, len(X_test))
            eval_indices = np.random.choice(len(X_test), n_eval, replace=False)

            X_eval = X_test[eval_indices].copy()
            y_eval = y_test[eval_indices].copy()

            sensitivity_results = feature_attack.analyze_sensitivity(
                self.model,
                X_eval,
                y_eval,
                self.device,
                batch_size=batch_size,
                verbose=False,
            )
            X_adv = sequence_attack.generate_batch(
                X_eval, y_eval, sensitivity_results=sensitivity_results, verbose=False
            )

            del sensitivity_results
            gc.collect()

            adv_dataset = IoTSequenceDataset(X_adv, y_eval)
            adv_loader = DataLoader(adv_dataset, batch_size=batch_size)
            adv_metrics = self._compute_detailed_metrics(
                adv_loader, criterion, num_classes
            )
            adv_per_class = self.compute_per_class_metrics(
                adv_loader, label_encoder, num_classes
            )
            adv_metrics["robustness_ratio"] = adv_metrics["accuracy"] / max(
                clean_metrics["accuracy"], 1e-8
            )
            adv_metrics["macro_f1"] = adv_per_class["macro_f1"]
            crash_results["test2_adversarial"] = adv_metrics
            print(
                f"  [Test 2 - Adversaires]  Loss={adv_metrics['loss']:.4f}  Acc={adv_metrics['accuracy']:.4f}  "
                f"F1={adv_metrics['f1_score']:.4f}  Macro-F1={adv_per_class['macro_f1']:.4f}  "
                f"RR={adv_metrics['robustness_ratio']:.4f}"
            )

            n_half = n_eval // 2
            X_mix = np.concatenate([X_eval[:n_half], X_adv[n_half:]], axis=0)
            y_mix = np.concatenate([y_eval[:n_half], y_eval[n_half:]], axis=0)
            mix_dataset = IoTSequenceDataset(X_mix, y_mix)
            mix_loader = DataLoader(mix_dataset, batch_size=batch_size)
            mix_metrics = self._compute_detailed_metrics(
                mix_loader, criterion, num_classes
            )
            mix_per_class = self.compute_per_class_metrics(
                mix_loader, label_encoder, num_classes
            )
            mix_metrics["macro_f1"] = mix_per_class["macro_f1"]
            crash_results["test3_mixed"] = mix_metrics
            print(
                f"  [Test 3 - Mélangé]      Loss={mix_metrics['loss']:.4f}  Acc={mix_metrics['accuracy']:.4f}  "
                f"F1={mix_metrics['f1_score']:.4f}  Macro-F1={mix_per_class['macro_f1']:.4f}"
            )

            del X_adv, X_eval, y_eval, adv_dataset, adv_loader, mix_dataset, mix_loader
            del X_mix, y_mix, n_eval, eval_indices
            gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return crash_results

    def _print_comparative_summary_2phase(self, all_crash_results: Dict):
        """Affiche le résumé comparatif pour 2 phases."""
        print(f"\n{'=' * 80}")
        print("  RAPPORT COMPARATIF — Entraînement 2 Phases")
        print(f"{'=' * 80}")
        header = f"{'Phase':<8} {'Test':<20} {'Loss':<8} {'Acc':<8} {'F1':<8} {'RR':<8}"
        print(header)
        print(f"{'-' * 80}")

        for phase_key in sorted(all_crash_results.keys()):
            results = all_crash_results[phase_key]
            phase_num = results["phase"]
            for test_name in ["test1_benign", "test2_adversarial", "test3_mixed"]:
                if test_name in results:
                    m = results[test_name]
                    rr = (
                        m.get("robustness_ratio", 1.0)
                        if "adversarial" in test_name
                        else 1.0
                    )
                    print(
                        f"  P{phase_num:<5} {test_name:<20} "
                        f"{m['loss']:<8.4f} {m['accuracy']:<8.4f} "
                        f"{m.get('f1_score', 0):<8.4f} {rr:<8.4f}"
                    )
            print(f"{'-' * 80}")

        print(f"{'=' * 80}\n")

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


# ═══════════════════════════════════════════════════════════════════════════
# PLOT GENERATION — Publication-Quality Figures for Jury Presentation
# ═══════════════════════════════════════════════════════════════════════════


def generate_device_identification_plot(
    per_class_metrics: Dict,
    save_path: Path,
    model_type: str = "LSTM",
):
    """
    Figure: IoT Device Identification Scores in IPFIX Records.

    Per-device Precision / Recall / F1 grouped bar chart (like Figure 6 in paper).
    """
    devices = sorted(per_class_metrics.keys())
    precisions = [per_class_metrics[d]["precision"] * 100 for d in devices]
    recalls = [per_class_metrics[d]["recall"] * 100 for d in devices]
    f1s = [per_class_metrics[d]["f1"] * 100 for d in devices]

    x = np.arange(len(devices))
    width = 0.25

    fig, ax = plt.subplots(figsize=(18, 8))
    bars_p = ax.bar(
        x - width,
        precisions,
        width,
        label="Precision",
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.5,
    )
    bars_r = ax.bar(
        x,
        recalls,
        width,
        label="Recall",
        color="#ff7f0e",
        edgecolor="white",
        linewidth=0.5,
    )
    bars_f = ax.bar(
        x + width,
        f1s,
        width,
        label="F1-Score",
        color="#2ca02c",
        edgecolor="white",
        linewidth=0.5,
    )

    # Add value labels
    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    fontweight="bold",
                )

    ax.set_ylabel("Score (%)", fontsize=14, fontweight="bold")
    ax.set_title(
        f"IoT Device Identification Scores in IPFIX Records ({model_type.upper()})",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(devices, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=12, loc="lower right")
    ax.set_ylim(
        bottom=max(0, min(min(precisions), min(recalls), min(f1s)) - 15), top=105
    )
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    plot_path = save_path / "fig_device_identification_scores.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Plot saved: {plot_path}")
    return plot_path


def generate_adversarial_effect_plot(
    clean_per_class: Dict,
    adv_feature_per_class: Dict,
    adv_search_per_class: Dict,
    save_path: Path,
    model_type: str = "LSTM",
):
    """
    Figure: Adversarial Effect on IPFIX Records Identification Devices.

    Shows per-device accuracy under clean vs each adversarial attack.
    """
    devices = sorted(clean_per_class.keys())
    clean_f1 = [clean_per_class[d]["f1"] * 100 for d in devices]

    attack_data = {}
    if adv_feature_per_class:
        attack_data["Feature-Level"] = [
            adv_feature_per_class.get(d, {}).get("f1", 0) * 100 for d in devices
        ]
    if adv_search_per_class:
        attack_data["Adversarial Search"] = [
            adv_search_per_class.get(d, {}).get("f1", 0) * 100 for d in devices
        ]

    n_groups = 1 + len(attack_data)
    width = 0.8 / n_groups
    x = np.arange(len(devices))

    fig, ax = plt.subplots(figsize=(18, 9))
    colors = ["#2ca02c", "#e74c3c", "#9467bd", "#ff7f0e"]

    ax.bar(
        x - width * (n_groups - 1) / 2,
        clean_f1,
        width,
        label="Clean (No Attack)",
        color=colors[0],
        edgecolor="white",
        linewidth=0.5,
    )
    for i, (attack_name, values) in enumerate(attack_data.items()):
        offset = x - width * (n_groups - 1) / 2 + width * (i + 1)
        ax.bar(
            offset,
            values,
            width,
            label=f"{attack_name} Attack",
            color=colors[i + 1],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_ylabel("F1-Score (%)", fontsize=14, fontweight="bold")
    ax.set_title(
        f"Adversarial Effect on IPFIX Records Device Identification ({model_type.upper()})",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(devices, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_ylim(bottom=0, top=105)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    plot_path = save_path / "fig_adversarial_effect.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Plot saved: {plot_path}")
    return plot_path


def generate_robustness_summary_plot(
    results: Dict,
    save_path: Path,
    model_type: str = "LSTM",
):
    """
    Figure: Performance Evaluation of Device Identification in IPFIX Records
    with Robustness Measures.

    Horizontal bar chart showing Accuracy, Macro-F1 for Clean and each attack type.
    """
    categories = []
    accuracies = []
    macro_f1s = []

    # Clean
    if "clean_metrics" in results:
        categories.append("Clean\n(No Attack)")
        accuracies.append(results["clean_metrics"].get("accuracy", 0) * 100)
        macro_f1s.append(results["clean_metrics"].get("macro_f1", 0) * 100)

    # Adversarial attacks
    adv_mapping = [
        ("adv_feature_metrics", "Feature-Level\nAttack"),
        ("adv_search_metrics", "Sequence Search\nAttack"),
        ("adv_hybrid_metrics", "Hybrid\nAttack"),
    ]
    for key, label in adv_mapping:
        if key in results:
            categories.append(label)
            accuracies.append(results[key].get("accuracy", 0) * 100)
            macro_f1s.append(results[key].get("macro_f1", 0) * 100)

    if not categories:
        return None

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    bars_acc = ax.bar(
        x - width / 2,
        accuracies,
        width,
        label="Accuracy",
        color="#3498db",
        edgecolor="white",
        linewidth=0.5,
    )
    bars_f1 = ax.bar(
        x + width / 2,
        macro_f1s,
        width,
        label="Macro F1-Score",
        color="#e67e22",
        edgecolor="white",
        linewidth=0.5,
    )

    for bars in [bars_acc, bars_f1]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_ylabel("Score (%)", fontsize=14, fontweight="bold")
    ax.set_title(
        f"Performance Evaluation of Device Identification\nin IPFIX Records with Robustness Measures ({model_type.upper()})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=12, loc="upper right")
    ax.set_ylim(bottom=0, top=110)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    plot_path = save_path / "fig_robustness_summary.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Plot saved: {plot_path}")
    return plot_path


def generate_training_history_plot(
    history: Dict, save_path: Path, model_type: str = "LSTM"
):
    """Plot training and validation loss/accuracy curves across phases."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    epochs_range = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs_range, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs_range, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title(
        f"Training & Validation Loss ({model_type.upper()})",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Accuracy
    ax2.plot(epochs_range, history["train_acc"], "b-", label="Train Acc", linewidth=2)
    ax2.plot(epochs_range, history["val_acc"], "r-", label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title(
        f"Training & Validation Accuracy ({model_type.upper()})",
        fontsize=14,
        fontweight="bold",
    )
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plot_path = save_path / "fig_training_history.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Plot saved: {plot_path}")
    return plot_path


def tokenize_data(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    features: List[str],
    config_path: str = "config/config.yaml",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, IoTTokenizer]:
    """Tokenize raw features for NLP Transformer.

    Args:
        X_train: (n_train, seq_len, n_features)
        X_val: (n_val, seq_len, n_features)
        X_test: (n_test, seq_len, n_features)
        features: list of feature names

    Returns:
        token_ids_train, token_ids_val, token_ids_test, pad_token_id, tokenizer
    """
    print("\n[TOKENIZER] Fitting BPE tokenizer on training data...")
    tokenizer = IoTTokenizer(config_path)
    tokenizer.fit(X_train, features, verbose=True)

    pad_token_id = tokenizer.tokenizer.token_to_id("<pad>")
    print(f"  pad_token_id: {pad_token_id}")

    print("  Transforming train data...")
    token_ids_train = tokenizer.transform(X_train, features)
    print("  Transforming val data...")
    token_ids_val = tokenizer.transform(X_val, features)
    print("  Transforming test data...")
    token_ids_test = tokenizer.transform(X_test, features)

    print(
        f"  Token IDs shape: train={token_ids_train.shape}, val={token_ids_val.shape}, test={token_ids_test.shape}"
    )

    return token_ids_train, token_ids_val, token_ids_test, pad_token_id, tokenizer


def tokenize_adversarial_batch(
    X_adv: np.ndarray,
    features: List[str],
    tokenizer: IoTTokenizer,
) -> np.ndarray:
    """Tokenize adversarial examples (re-tokenize after perturbation)."""
    return tokenizer.transform(X_adv, features)


def create_model(
    model_type: str,
    input_size: int,
    num_classes: int,
    seq_length: int,
    device: torch.device,
    pad_token_id: int = 2,
) -> nn.Module:
    """Create model based on type."""
    if model_type == "lstm":
        model = LSTMClassifier(input_size, num_classes, LSTM_CONFIG)
    elif model_type == "transformer":
        model = TransformerClassifier(
            input_size, num_classes, seq_length, TRANSFORMER_CONFIG
        )
    elif model_type == "bilstm":
        from config.config import LSTM_CONFIG

        _bilstm_cfg = dict(LSTM_CONFIG)
        _bilstm_cfg["bidirectional"] = True  # always bidirectional
        model = LSTMClassifier(input_size, num_classes, _bilstm_cfg)
    elif model_type == "cnn_lstm":
        model = CNNLSTMClassifier(input_size, num_classes)
    elif model_type == "xgboost_lstm":
        model = XGBoostLSTMClassifier(input_size, num_classes, LSTM_CONFIG)
    elif model_type == "cnn":
        model = CNNClassifier(input_size, num_classes)
    elif model_type == "cnn_bilstm_transformer":
        from config.config import CNN_BILSTM_TRANSFORMER_CONFIG

        model = CNNBiLSTMTransformerClassifier(
            input_size, num_classes, seq_length, CNN_BILSTM_TRANSFORMER_CONFIG
        )
    elif model_type == "cnn_bilstm":
        from src.models.cnn_bilstm import CNNBiLSTMClassifier

        model = CNNBiLSTMClassifier(input_size, num_classes)
    elif model_type == "nlp_transformer":
        model = NLPTransformerClassifier(
            vocab_size=52000,
            num_classes=num_classes,
            max_seq_length=576,
            pad_token_id=pad_token_id,
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            "Choices: lstm, bilstm, transformer, nlp_transformer, cnn_lstm, xgboost_lstm, cnn, cnn_bilstm, cnn_bilstm_transformer"
        )

    return model.to(device)


def run_experiment_with_phase_checkpoints(
    model_type: str,
    seq_length: int,
    adv_method: str,
    adv_ratio: float,
    epochs: int,
    batch_size: int,
    max_files: Optional[int] = None,
    save_results: bool = True,
    hybrid_split: Optional[Dict[str, float]] = None,
    data_dir: Optional[Path] = None,
    dataset: Optional[str] = None,
    max_records: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
) -> Dict:
    """Run experiment avec sauvegarde et evaluation des checkpoints de phase.

    Args:
        eval_batch_size: Batch size for evaluation (defaults to batch_size//2 for RAM savings)
    """
    device = get_device()
    print(f"\n{'=' * 60}")
    print(
        f"Experiment (Phase Checkpoints): {model_type.upper()} | Seq={seq_length} | Adv={adv_method}"
    )
    print(f"Device: {device}")
    print(f"{'=' * 60}\n")

    if eval_batch_size is None:
        eval_batch_size = max(16, batch_size // 2)

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
        n_continuous_features,
    ) = load_and_preprocess_data(
        seq_length,
        stride=max(1, seq_length // 2),
        max_files=max_files,
        data_dir=data_dir,
        dataset=dataset,
        max_records=max_records,
    )

    input_size = X_train.shape[2]
    num_classes = len(label_encoder.classes_)

    use_nlp = model_type == "nlp_transformer"
    pad_token_id = 2
    tokenizer = None

    if use_nlp:
        from src.training.trainer import NLPTransformerDataset

        X_tok_train, X_tok_val, X_tok_test, pad_token_id, tokenizer = tokenize_data(
            X_train, X_val, X_test, features
        )
        train_dataset = NLPTransformerDataset(X_tok_train, y_train)
        val_dataset = NLPTransformerDataset(X_tok_val, y_val)
        test_dataset = NLPTransformerDataset(X_tok_test, y_test)
        print(f"\n  NLP Transformer mode: token IDs input, pad_token_id={pad_token_id}")
    else:
        train_dataset = IoTSequenceDataset(X_train, y_train)
        val_dataset = IoTSequenceDataset(X_val, y_val)
        test_dataset = IoTSequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size)

    print(f"\nInput size: {input_size}, Classes: {num_classes}")
    print(
        f"Continuous features: {n_continuous_features}, Binary features: {input_size - n_continuous_features}"
    )

    model = create_model(
        model_type, input_size, num_classes, seq_length, device, pad_token_id
    )
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = (
        RESULTS_DIR
        / "models"
        / f"{model_type}_{seq_length}_{adv_method}_phases_{timestamp}"
    )
    if save_results:
        save_path.mkdir(parents=True, exist_ok=True)

    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    y_train_expanded = np.repeat(y_train, X_train.shape[1])
    feature_attack = FeatureLevelAttack(
        X_train_flat,
        y_train_expanded,
        features,
        num_classes,
        n_continuous_features=n_continuous_features,
    )

    del X_train_flat, y_train_expanded
    gc.collect()

    sequence_attack = SequenceLevelAttack(
        model,
        device,
        feature_attack,
        batch_size=batch_size,
    )

    adv_generator = HybridAdversarialAttack(
        feature_attack, sequence_attack, features, combine_ratio=0.5
    )

    trainer = AdversarialTrainer(
        model, device, model_type, tokenizer=tokenizer, features=features
    )

    if hybrid_split is None:
        hybrid_split = {"clean": 0.6, "feature": 0.2, "sequence": 0.2}

    print(f"\nTraining avec checkpoints de phase - adversarial ratio: {adv_ratio}")

    X_train_raw = X_tok_train if use_nlp else X_train

    history = trainer.fit_with_phase_checkpoints(
        train_loader,
        val_loader,
        test_loader,
        epochs=epochs,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        adv_generator=adv_generator if adv_method != "none" else None,
        adv_ratio=adv_ratio if adv_method != "none" else 0.0,
        adv_method=adv_method
        if adv_method in ["search", "feature", "hybrid"]
        else "search",
        hybrid_split=hybrid_split,
        save_path=save_path if save_results else None,
        feature_attack=feature_attack,
        sequence_attack=sequence_attack,
        X_test=X_test,
        y_test=y_test,
        batch_size=eval_batch_size,
        is_xgboost=(model_type == "xgboost_lstm"),
        label_encoder=label_encoder,
        X_train_raw=X_train_raw,
    )

    if save_results and save_path:
        try:
            generate_training_history_plot(history, save_path, model_type)
        except Exception as e:
            print(f"  ⚠️ Training history plot error: {e}")

    phase_results = history.get("phase_checkpoints_evaluation", {})

    results = {
        "model_type": model_type,
        "sequence_length": seq_length,
        "adversarial_method": adv_method,
        "adversarial_ratio": adv_ratio,
        "hybrid_split": hybrid_split,
        "num_classes": num_classes,
        "input_size": input_size,
        "parameters": sum(p.numel() for p in model.parameters()),
        "phase_checkpoints_evaluation": phase_results,
    }

    if phase_results:
        last_phase = max(phase_results.keys(), key=lambda k: int(k.split("_")[1]))
        final_metrics = phase_results[last_phase]
        results["test_accuracy_clean"] = final_metrics.get("clean", {}).get(
            "accuracy", 0
        )
        results["test_loss_clean"] = final_metrics.get("clean", {}).get("loss", 0)
        results["final_phase"] = int(last_phase.split("_")[1])

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
    del train_loader, val_loader, test_loader
    del train_dataset, val_dataset, test_dataset
    del model, trainer, feature_attack, sequence_attack, adv_generator
    if use_nlp:
        del X_tok_train, X_tok_val, X_tok_test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def run_experiment(
    model_type: str,
    seq_length: int,
    adv_method: str,
    adv_ratio: float,
    epochs: int,
    batch_size: int,
    max_files: Optional[int] = None,
    save_results: bool = True,
    hybrid_split: Optional[Dict[str, float]] = None,
    data_dir: Optional[Path] = None,
    dataset: Optional[str] = None,
    max_records: Optional[int] = None,
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
        n_continuous_features,
    ) = load_and_preprocess_data(
        seq_length,
        stride=max(1, seq_length // 2),
        max_files=max_files,
        data_dir=data_dir,
        dataset=dataset,
        max_records=max_records,
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
    print(
        f"Continuous features: {n_continuous_features}, Binary features: {input_size - n_continuous_features}"
    )

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
        X_train_flat,
        y_train_expanded,
        features,
        num_classes,
        n_continuous_features=n_continuous_features,
    )

    del X_train_flat, y_train_expanded
    gc.collect()

    sequence_attack = SequenceLevelAttack(
        model,
        device,
        feature_attack,
        batch_size=batch_size,
    )

    adv_generator = HybridAdversarialAttack(
        feature_attack, sequence_attack, features, combine_ratio=0.5
    )

    trainer = AdversarialTrainer(model, device, model_type)

    # Default hybrid split: 60% clean, 20% feature-level, 20% sequence-level
    if hybrid_split is None:
        hybrid_split = {"clean": 0.6, "feature": 0.2, "sequence": 0.2}

    print(f"\nTraining with adversarial ratio: {adv_ratio}")
    if adv_method == "hybrid":
        print(
            f"  Phased training: 60% clean -> 20% feature-level -> 20% sequence-level"
        )
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        adv_generator=adv_generator if adv_method != "none" else None,
        adv_ratio=adv_ratio if adv_method != "none" else 0.0,
        adv_method=adv_method
        if adv_method in ["search", "feature", "hybrid"]
        else "search",
        hybrid_split=hybrid_split,
        save_path=save_path if save_results else None,
        test_loader=test_loader,
    )

    if model_type == "xgboost_lstm":
        print("\n" + "=" * 60)
        print("FITTING XGBOOST ON EXTRACTED LSTM FEATURES")
        print("=" * 60)
        model.fit_xgboost(train_loader, device)

    del train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("EVALUATION (with per-class metrics & macro F1)")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = trainer.evaluate(test_loader, criterion)
    print(f"\nClean Test - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    # ── Per-class metrics on Clean data ──────────────────────────
    print("\n── Per-Class Metrics (Clean) ──")
    clean_detailed = trainer.compute_per_class_metrics(
        test_loader, label_encoder, num_classes
    )
    print(f"  Accuracy:        {clean_detailed['accuracy']:.4f}")
    print(f"  Macro Precision: {clean_detailed['macro_precision']:.4f}")
    print(f"  Macro Recall:    {clean_detailed['macro_recall']:.4f}")
    print(f"  Macro F1-Score:  {clean_detailed['macro_f1']:.4f}")
    print(f"\n  {'Device':<30} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>10}")
    print(f"  {'-' * 66}")
    for dev, m in sorted(clean_detailed["per_class"].items()):
        print(
            f"  {dev:<30} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['support']:>10,}"
        )

    results = {
        "model_type": model_type,
        "sequence_length": seq_length,
        "adversarial_method": adv_method,
        "adversarial_ratio": adv_ratio,
        "hybrid_split": hybrid_split,
        "test_accuracy_clean": test_acc,
        "test_loss_clean": test_loss,
        "best_val_accuracy": max(history["val_acc"]),
        "num_classes": num_classes,
        "input_size": input_size,
        "parameters": sum(p.numel() for p in model.parameters()),
        "clean_metrics": {
            "accuracy": clean_detailed["accuracy"],
            "macro_precision": clean_detailed["macro_precision"],
            "macro_recall": clean_detailed["macro_recall"],
            "macro_f1": clean_detailed["macro_f1"],
            "per_class": clean_detailed["per_class"],
        },
    }

    adversarial_results = {}
    adv_feature_per_class = {}
    adv_search_per_class = {}

    if adv_method != "none":
        print("\nGenerating adversarial examples for evaluation...")
        n_eval = min(1000, len(X_test))
        eval_indices = np.random.choice(len(X_test), n_eval, replace=False)

        X_eval = X_test[eval_indices].copy()
        y_eval = y_test[eval_indices].copy()

        # ── Feature-level attack ────────────────────────────────
        print("  Feature-level attack...")
        seq_len = X_eval.shape[1]
        y_eval_expanded = np.repeat(y_eval, seq_len)
        X_adv_feature = feature_attack.generate_batch(
            X_eval.reshape(-1, X_eval.shape[-1]), y_eval_expanded, verbose=True
        ).reshape(X_eval.shape)

        eval_dataset = IoTSequenceDataset(X_adv_feature, y_eval)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
        loss_f, acc_f = trainer.evaluate(eval_loader, criterion)

        feat_detailed = trainer.compute_per_class_metrics(
            eval_loader, label_encoder, num_classes
        )
        adv_feature_per_class = feat_detailed["per_class"]
        adversarial_results["feature_level"] = {
            "loss": loss_f,
            "accuracy": acc_f,
            "macro_f1": feat_detailed["macro_f1"],
            "macro_precision": feat_detailed["macro_precision"],
            "macro_recall": feat_detailed["macro_recall"],
        }
        results["adv_feature_metrics"] = adversarial_results["feature_level"].copy()
        results["adv_feature_metrics"]["per_class"] = adv_feature_per_class
        print(
            f"  Feature-level - Loss: {loss_f:.4f}, Acc: {acc_f:.4f}, Macro-F1: {feat_detailed['macro_f1']:.4f}"
        )

        del X_adv_feature, eval_dataset, eval_loader
        gc.collect()

        # ── Sequence-level adversarial search ─────────────────────
        print("  Sequence-level adversarial search...")
        sensitivity_results = feature_attack.analyze_sensitivity(
            model, X_eval, y_eval, device, batch_size=batch_size, verbose=True
        )
        X_adv_search = sequence_attack.generate_batch(
            X_eval, y_eval, sensitivity_results=sensitivity_results, verbose=True
        )

        eval_dataset = IoTSequenceDataset(X_adv_search, y_eval)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
        loss_s, acc_s = trainer.evaluate(eval_loader, criterion)

        search_detailed = trainer.compute_per_class_metrics(
            eval_loader, label_encoder, num_classes
        )
        adv_search_per_class = search_detailed["per_class"]
        adversarial_results["sequence_search"] = {
            "loss": loss_s,
            "accuracy": acc_s,
            "macro_f1": search_detailed["macro_f1"],
            "macro_precision": search_detailed["macro_precision"],
            "macro_recall": search_detailed["macro_recall"],
        }
        results["adv_search_metrics"] = adversarial_results["sequence_search"].copy()
        results["adv_search_metrics"]["per_class"] = adv_search_per_class
        print(
            f"  Sequence Search - Loss: {loss_s:.4f}, Acc: {acc_s:.4f}, Macro-F1: {search_detailed['macro_f1']:.4f}"
        )

        del X_adv_search, eval_dataset, eval_loader
        gc.collect()

        # ── Hybrid attack ───────────────────────────────────────
        print("  Hybrid attack...")
        X_adv_hybrid = adv_generator.generate_batch(
            X_eval, y_eval, method="hybrid", verbose=True
        )

        eval_dataset = IoTSequenceDataset(X_adv_hybrid, y_eval)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
        loss_h, acc_h = trainer.evaluate(eval_loader, criterion)

        hybrid_detailed = trainer.compute_per_class_metrics(
            eval_loader, label_encoder, num_classes
        )
        adversarial_results["hybrid"] = {
            "loss": loss_h,
            "accuracy": acc_h,
            "macro_f1": hybrid_detailed["macro_f1"],
            "macro_precision": hybrid_detailed["macro_precision"],
            "macro_recall": hybrid_detailed["macro_recall"],
        }
        results["adv_hybrid_metrics"] = adversarial_results["hybrid"].copy()
        results["adv_hybrid_metrics"]["per_class"] = hybrid_detailed["per_class"]
        print(
            f"  Hybrid        - Loss: {loss_h:.4f}, Acc: {acc_h:.4f}, Macro-F1: {hybrid_detailed['macro_f1']:.4f}"
        )

        del X_adv_hybrid, eval_dataset, eval_loader, X_eval, y_eval
        gc.collect()

        results["adversarial_results"] = adversarial_results

        results["robustness_ratios"] = {
            k: v["accuracy"] / max(test_acc, 1e-8)
            for k, v in adversarial_results.items()
        }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Generate Plots ──────────────────────────────────────────
    if save_results and save_path:
        print("\n" + "=" * 60)
        print("GENERATING PLOTS FOR PRESENTATION")
        print("=" * 60)

        try:
            # Plot 1: Per-device P/R/F1 (Figure 6 style)
            print("\n  [1/4] Per-device identification scores...")
            generate_device_identification_plot(
                clean_detailed["per_class"], save_path, model_type
            )

            # Plot 2: Adversarial effect on per-device F1
            if adv_method != "none":
                print("  [2/4] Adversarial effect per device...")
                generate_adversarial_effect_plot(
                    clean_detailed["per_class"],
                    adv_feature_per_class,
                    adv_search_per_class,
                    save_path,
                    model_type,
                )

                # Plot 3: Robustness summary (Accuracy + Macro-F1 clean vs attacks)
                print("  [3/4] Robustness summary...")
                generate_robustness_summary_plot(results, save_path, model_type)

            # Plot 4: Training history curves
            print("  [4/4] Training history curves...")
            generate_training_history_plot(history, save_path, model_type)

        except Exception as e:
            print(f"  ⚠️ Plot generation error (non-fatal): {e}")

    # ── Save Results ────────────────────────────────────────────
    if save_results:
        # Remove numpy arrays from results before JSON serialization
        results_json = json.loads(json.dumps(results, default=str))

        with open(save_path / "results.json", "w") as f:
            json.dump(results_json, f, indent=2)

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

        print(f"\n✅ All results and plots saved to: {save_path}")

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
    data_dir: Optional[Path] = None,
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
                    data_dir=data_dir,
                )
                all_results[key] = results

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    print("\n| Model | Seq | Adv Method | Clean Acc | Feature | Search | Hybrid |")
    print("|-------|-----|------------|-----------|---------|--------|--------|")

    for key, res in all_results.items():
        clean_acc = res["test_accuracy_clean"]

        if "adversarial_results" in res:
            feat = (
                res["adversarial_results"].get("feature_level", {}).get("accuracy", 0)
            )
            search = (
                res["adversarial_results"].get("sequence_search", {}).get("accuracy", 0)
            )
            hybrid = res["adversarial_results"].get("hybrid", {}).get("accuracy", 0)
        else:
            feat = search = hybrid = "-"

        print(
            f"| {key.split('_')[0]} | {res['sequence_length']} | {res['adversarial_method']:10} | {clean_acc:.4f} | {feat:.4f} | {search:.4f} | {hybrid:.4f} |"
        )

    comparison_path = RESULTS_DIR / "comparison_results.json"
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nComparison saved to: {comparison_path}")

    return all_results


def run_dual_dataset_study(
    model_type: str,
    seq_length: int,
    adv_method: str,
    adv_ratio: float,
    epochs: int,
    batch_size: int,
    max_files: Optional[int] = None,
    max_records: Optional[int] = None,
    hybrid_split: Optional[Dict[str, float]] = None,
    csv_data_dir: Optional[Path] = None,
    json_data_dir: Optional[Path] = None,
) -> Dict:
    """
    Runs the full adversarial-training study on BOTH datasets:
      1. IPFIX ML Instances (CSV, 18 classes) ← per-device temporal split
      2. IPFIX Records (JSON, 17 classes) ← per-device temporal split

    Generates a comparison report: results/dual_study_<model>_<timestamp>.json
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison = {}

    datasets = [
        ("csv", csv_data_dir),
        ("json", json_data_dir),
    ]

    for ds_name, ds_dir in datasets:
        print(f"\n{'#' * 70}")
        print(f"  DUAL-DATASET STUDY — Dataset: {ds_name.upper()}")
        print(f"{'#' * 70}")
        try:
            result = run_experiment_with_phase_checkpoints(
                model_type=model_type,
                seq_length=seq_length,
                adv_method=adv_method,
                adv_ratio=adv_ratio,
                epochs=epochs,
                batch_size=batch_size,
                max_files=max_files if ds_name == "csv" else None,
                save_results=True,
                hybrid_split=hybrid_split,
                data_dir=ds_dir,
                dataset=ds_name,
                max_records=max_records if ds_name == "json" else None,
            )
            comparison[ds_name] = result
        except Exception as e:
            print(f"  ⚠️  {ds_name.upper()} study failed: {e}")
            comparison[ds_name] = {"error": str(e)}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save comparison report
    report_path = RESULTS_DIR / f"dual_study_{model_type}_{timestamp}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\n📊 Dual-dataset comparison report: {report_path}")

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Adversarial Training for IoT Device Identification"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="lstm",
        choices=[
            "lstm",
            "bilstm",
            "transformer",
            "nlp_transformer",
            "cnn_lstm",
            "cnn",
            "xgboost_lstm",
            "cnn_bilstm",
            "cnn_bilstm_transformer",
        ],
        help="Model architecture (bilstm = bidirectional LSTM, nlp_transformer = tokenized BPE)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["csv", "json"],
        help=(
            "Dataset to use: 'csv' (IPFIX ML Instances, 18 classes) or "
            "'json' (IPFIX Records, 17 classes). "
            "Defaults to the PIPELINE_MODE value in config.py."
        ),
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Maximum JSON records to load (for quick smoke tests; JSON pipeline only)",
    )
    parser.add_argument("--seq_length", type=int, default=10, help="Sequence length")
    parser.add_argument(
        "--adv_method",
        type=str,
        default="hybrid",
        choices=["none", "feature", "search", "hybrid"],
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
        "--data_dir",
        type=str,
        default=None,
        help="Path to JSON data directory (default: use config)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory to save models and results (default: results/ in project root). "
        "Set to a Google Drive path to save directly to Drive.",
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
        default="lstm,transformer,cnn_lstm,xgboost_lstm",
        help="Comma-separated model types for comparison",
    )
    parser.add_argument(
        "--adv_methods",
        type=str,
        default="none,feature,search,hybrid",
        help="Comma-separated adversarial methods for comparison",
    )
    parser.add_argument(
        "--hybrid_clean",
        type=float,
        default=0.6,
        help="Ratio of clean samples in hybrid training (default: 0.6)",
    )
    parser.add_argument(
        "--hybrid_feature",
        type=float,
        default=0.2,
        help="Ratio of feature-level adversarial samples in hybrid training (default: 0.2)",
    )
    parser.add_argument(
        "--hybrid_sequence",
        type=float,
        default=0.2,
        help="Ratio of sequence-level adversarial samples in hybrid training (default: 0.2)",
    )
    parser.add_argument(
        "--phase_checkpoints",
        action="store_true",
        help="Utiliser la méthode avec sauvegarde et évaluation de checkpoints après chaque phase",
    )
    parser.add_argument(
        "--dual_dataset",
        action="store_true",
        help=(
            "Run the full study on BOTH datasets (CSV + JSON) sequentially "
            "and generate a comparison report (requires --phase_checkpoints logic)."
        ),
    )
    parser.add_argument(
        "--csv_data_dir",
        type=str,
        default=None,
        help="Path to CSV data directory for dual-dataset study (default: config RAW_DATA_DIR)",
    )
    parser.add_argument(
        "--json_data_dir",
        type=str,
        default=None,
        help="Path to JSON data directory for dual-dataset study (default: config JSON_DATA_DIR)",
    )

    args = parser.parse_args()

    # Enforce sequence length of 25 for transformer if it's currently default
    if args.model in ("transformer", "nlp_transformer") and args.seq_length == 10:
        print("Transformer model selected, defaulting sequence length to 25.")
        args.seq_length = 25

    # Validate hybrid split ratios
    hybrid_split = {
        "clean": args.hybrid_clean,
        "feature": args.hybrid_feature,
        "sequence": args.hybrid_sequence,
    }
    total = sum(hybrid_split.values())
    if abs(total - 1.0) > 1e-6:
        parser.error(f"Hybrid split ratios must sum to 1.0, got {total}")

    # Parse data_dir / results_dir if provided
    data_dir = Path(args.data_dir) if args.data_dir else None

    # Override RESULTS_DIR globally if --results_dir is supplied
    if args.results_dir:
        _results_dir = Path(args.results_dir)
        _results_dir.mkdir(parents=True, exist_ok=True)
        # Patch the module-level RESULTS_DIR so all save paths point there
        globals()["RESULTS_DIR"] = _results_dir
        print(f"Results will be saved to: {_results_dir}")

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
            data_dir=data_dir,
        )
    elif args.phase_checkpoints:
        # Utiliser la nouvelle méthode avec checkpoints de phase
        run_experiment_with_phase_checkpoints(
            model_type=args.model,
            seq_length=args.seq_length,
            adv_method=args.adv_method,
            adv_ratio=args.adv_ratio,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_files=args.max_files,
            save_results=True,
            hybrid_split=hybrid_split,
            data_dir=data_dir,
            dataset=args.dataset,
            max_records=getattr(args, "max_records", None),
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
            hybrid_split=hybrid_split,
            data_dir=data_dir,
            dataset=args.dataset,
            max_records=getattr(args, "max_records", None),
        )


if __name__ == "__main__":
    main()
