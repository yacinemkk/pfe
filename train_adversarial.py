#!/usr/bin/env python3
"""
Hybrid Adversarial Training for IoT Device Identification

This script implements:
1. Multiple model architectures (LSTM, Transformer, CNN-LSTM)
2. Multiple sequence lengths for experimentation
3. Feature-level adversarial attacks (IoT-SDN style)
4. Sequence-level adversarial attacks (PGD via BPTT)
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
from src.models.lstm import LSTMClassifier, IoTSequenceDataset
from src.models.transformer import TransformerClassifier
from src.models.cnn_lstm import CNNLSTMClassifier, CNNClassifier
from src.models.xgboost_lstm import XGBoostLSTMClassifier
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
    seq_length: int, stride: int = 5, max_files: Optional[int] = None,
    data_dir: Optional[Path] = None, pipeline_mode: Optional[str] = None,
    max_records: Optional[int] = None,
) -> Tuple[np.ndarray, ...]:
    """Load and preprocess data with specified sequence length.
    
    Supports two pipeline modes:
    - 'csv': Original CSV pipeline (IoT IPFIX Home dataset, 18 classes)
    - 'json': New JSON-native pipeline (IPFIX Records dataset, 17-26 classes)
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test,
                  features, scaler, label_encoder, n_continuous_features)
    """
    if pipeline_mode is None:
        pipeline_mode = PIPELINE_MODE
    
    print(f"Pipeline mode: {pipeline_mode.upper()}")
    
    if pipeline_mode == "json":
        return _load_json_pipeline(seq_length, stride, data_dir, max_records)
    else:
        return _load_csv_pipeline(seq_length, stride, max_files, data_dir)


def _load_json_pipeline(
    seq_length: int, stride: int = 5,
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
    
    X_train, X_val, X_test, y_train, y_val, y_test, features, scaler, label_encoder = result
    n_continuous = JSON_N_CONTINUOUS
    
    print(f"\n  JSON Pipeline Summary:")
    print(f"    Features: {len(features)} ({n_continuous} continuous + {JSON_N_BINARY} binary)")
    print(f"    Classes: {len(label_encoder.classes_)}")
    print(f"    Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        features, scaler, label_encoder,
        n_continuous,
    )


def _load_csv_pipeline(
    seq_length: int, stride: int = 5,
    max_files: Optional[int] = None,
    data_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, ...]:
    """Original CSV pipeline (backward compatible)."""
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    import pandas as pd

    print(f"Loading CSV data with sequence length={seq_length}, stride={stride}")
    
    if data_dir is None:
        data_dir = RAW_DATA_DIR
    
    print(f"  Data directory: {data_dir}")

    dfs = []
    csv_files = sorted(data_dir.glob("home*_labeled.csv"))

    if len(csv_files) == 0:
        print(f"\n⚠️  ERROR: No CSV files found in {data_dir}")
        print(f"    Expected files matching pattern: home*_labeled.csv")
        print(f"\n    To fix this:")
        print(f"    1. If running locally: Ensure data is at {data_dir}")
        print(f"    2. If running on Colab: Use --data_dir /content/drive/MyDrive/PFE/IPFIX_ML_Instances")
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    if max_files:
        csv_files = csv_files[:max_files]

    for f in csv_files:
        print(f"  Loading {f.name}...")
        df = pd.read_csv(f)
        df["source_file"] = f.stem
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    df = df.drop_duplicates()
    df = df.dropna(subset=["name"])

    from config.config import IOT_DEVICE_CLASSES

    df = df[df["name"].isin(IOT_DEVICE_CLASSES)]

    class_counts = df["name"].value_counts()
    valid_classes = class_counts.index
    df = df[df["name"].isin(valid_classes)]

    print(f"  Samples: {len(df):,} | Classes: {len(valid_classes)} (filtered to 18 IoT classes)")

    features = [c for c in FEATURES_TO_KEEP if c in df.columns]
    X = df[features].values
    y = df["name"].values
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

    # CSV pipeline: all features are continuous (no binary packet direction bits)
    n_continuous = len(features)

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        features, scaler, label_encoder,
        n_continuous,
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
        hybrid_split: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float]:
        """Train for one epoch with optional adversarial training.
        
        Args:
            hybrid_split: Dict with 'clean', 'feature', 'sequence' ratios (must sum to 1.0)
                         Default: {'clean': 0.6, 'feature': 0.2, 'sequence': 0.2}
        """
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        # Default hybrid split: 60% clean, 20% feature-level, 20% sequence-level
        if hybrid_split is None:
            hybrid_split = {"clean": 0.6, "feature": 0.2, "sequence": 0.2}

        for X_batch, y_batch in tqdm(train_loader, desc="Training", leave=False):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            if adv_generator is not None and adv_ratio > 0:
                n_adv = int(len(X_batch) * adv_ratio)
                if n_adv > 0:
                    # Calculate split sizes based on hybrid ratios
                    n_feature = int(n_adv * hybrid_split.get("feature", 0.2) / (hybrid_split.get("feature", 0.2) + hybrid_split.get("sequence", 0.2)))
                    n_sequence = n_adv - n_feature

                    adv_indices = np.random.choice(len(X_batch), n_adv, replace=False)
                    
                    # Split indices for feature-level and sequence-level attacks
                    idx_feature = adv_indices[:n_feature]
                    idx_sequence = adv_indices[n_feature:n_feature + n_sequence]

                    # Apply feature-level attack
                    if len(idx_feature) > 0:
                        X_adv_feature = adv_generator.feature_attack.generate_batch(
                            X_batch[idx_feature].cpu().numpy(),
                            y_batch[idx_feature].cpu().numpy(),
                        )
                        X_batch[idx_feature] = torch.FloatTensor(X_adv_feature).to(self.device)
                        del X_adv_feature

                    # Apply sequence-level attack
                    if len(idx_sequence) > 0:
                        seq_method = "pgd" if adv_method == "hybrid" else adv_method
                        X_adv_sequence = adv_generator.sequence_attack.generate_batch(
                            X_batch[idx_sequence].cpu().numpy(),
                            y_batch[idx_sequence].cpu().numpy(),
                            method=seq_method,
                        )
                        X_batch[idx_sequence] = torch.FloatTensor(X_adv_sequence).to(self.device)
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
        self,
        test_loader: DataLoader,
        criterion: nn.Module,
        phase_name: str = "Test"
    ) -> Tuple[float, float]:
        """Test the model and print results."""
        test_loss, test_acc = self.evaluate(test_loader, criterion)
        print(f"\n{'='*50}")
        print(f"  {phase_name} - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        print(f"{'='*50}\n")
        return test_loss, test_acc

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
    ) -> Dict:
        """Crash Test exhaustif après chaque phase.

        Évalue le modèle sur 4 loaders:
        1. Test_Loader_Normal (données pures)
        2. Test_Loader_Adv_Feature (attaque feature-level)
        3. Test_Loader_Adv_Seq_PGD (attaque PGD)
        4. Test_Loader_Adv_Seq_FGSM (attaque FGSM)

        Métriques par loader: Loss, Accuracy, Precision, Recall, F1, Robustness Ratio.
        """
        print(f"\n{'─'*60}")
        print(f"  CRASH TEST — Phase {phase_num}")
        print(f"{'─'*60}")

        crash_results = {"phase": phase_num}

        # 1. Clean Test
        clean_metrics = self._compute_detailed_metrics(test_loader, criterion, num_classes)
        crash_results["clean"] = clean_metrics
        clean_acc = clean_metrics["accuracy"]
        print(f"  [Clean]       Loss={clean_metrics['loss']:.4f}  Acc={clean_acc:.4f}  "
              f"P={clean_metrics['precision']:.4f}  R={clean_metrics['recall']:.4f}  "
              f"F1={clean_metrics['f1_score']:.4f}")

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
            feat_metrics = self._compute_detailed_metrics(eval_loader, criterion, num_classes)
            feat_metrics["robustness_ratio"] = feat_metrics["accuracy"] / max(clean_acc, 1e-8)
            crash_results["feature_attack"] = feat_metrics
            print(f"  [FeatureAdv]  Loss={feat_metrics['loss']:.4f}  Acc={feat_metrics['accuracy']:.4f}  "
                  f"P={feat_metrics['precision']:.4f}  R={feat_metrics['recall']:.4f}  "
                  f"F1={feat_metrics['f1_score']:.4f}  RR={feat_metrics['robustness_ratio']:.4f}")

            del X_adv_feature, eval_dataset, eval_loader
            gc.collect()

        # 3. Sequence-level PGD
        if sequence_attack is not None and X_test is not None and y_test is not None:
            n_eval = min(1000, len(X_test))
            eval_indices = np.random.choice(len(X_test), n_eval, replace=False)
            X_eval = X_test[eval_indices].copy()
            y_eval = y_test[eval_indices].copy()

            X_adv_pgd = sequence_attack.generate_batch(X_eval, y_eval, method="pgd", verbose=False)
            eval_dataset = IoTSequenceDataset(X_adv_pgd, y_eval)
            eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
            pgd_metrics = self._compute_detailed_metrics(eval_loader, criterion, num_classes)
            pgd_metrics["robustness_ratio"] = pgd_metrics["accuracy"] / max(clean_acc, 1e-8)
            crash_results["sequence_pgd"] = pgd_metrics
            print(f"  [SeqPGD]      Loss={pgd_metrics['loss']:.4f}  Acc={pgd_metrics['accuracy']:.4f}  "
                  f"P={pgd_metrics['precision']:.4f}  R={pgd_metrics['recall']:.4f}  "
                  f"F1={pgd_metrics['f1_score']:.4f}  RR={pgd_metrics['robustness_ratio']:.4f}")

            del X_adv_pgd, eval_dataset, eval_loader
            gc.collect()

            # 4. Sequence-level FGSM
            X_adv_fgsm = sequence_attack.generate_batch(X_eval, y_eval, method="fgsm", verbose=False)
            eval_dataset = IoTSequenceDataset(X_adv_fgsm, y_eval)
            eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
            fgsm_metrics = self._compute_detailed_metrics(eval_loader, criterion, num_classes)
            fgsm_metrics["robustness_ratio"] = fgsm_metrics["accuracy"] / max(clean_acc, 1e-8)
            crash_results["sequence_fgsm"] = fgsm_metrics
            print(f"  [SeqFGSM]     Loss={fgsm_metrics['loss']:.4f}  Acc={fgsm_metrics['accuracy']:.4f}  "
                  f"P={fgsm_metrics['precision']:.4f}  R={fgsm_metrics['recall']:.4f}  "
                  f"F1={fgsm_metrics['f1_score']:.4f}  RR={fgsm_metrics['robustness_ratio']:.4f}")

            del X_adv_fgsm, eval_dataset, eval_loader, X_eval, y_eval
            gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        adv_method: str = "pgd",
        hybrid_split: Optional[Dict[str, float]] = None,
        save_path: Optional[Path] = None,
        early_stopping_patience: int = 10,
        feature_attack: Optional[FeatureLevelAttack] = None,
        sequence_attack: Optional[SequenceLevelAttack] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        batch_size: int = 64,
        is_xgboost: bool = False,
    ) -> Dict:
        """Curriculum Learning avec Early Stopping par Phase & Crash Test.

        Implémente le plan de dernier_correction.md:
        1. Phase 1 (Normal)     – Entraînement sur données propres.
        2. Phase 2 (FeatureAdv) – Perturbations feature-level.
        3. Phase 3 (SeqAdv)     – Attaques PGD/FGSM séquentielles.

        Chaque phase dispose de son propre Early Stopping:
          - Patience réinitialisée au début de chaque phase.
          - Meilleurs poids sauvegardés et rechargés en fin de phase.

        Crash Test exhaustif après chaque phase (4 loaders × métriques complètes).
        Pour XGBoost-LSTM: fit XGBoost avant l'évaluation Phase 3.
        Génère un rapport JSON récapitulatif comparant P1, P2 et P3.
        """
        import copy

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )

        # Determine num_classes from test_loader
        sample_y = next(iter(test_loader))[1]
        num_classes = int(sample_y.max().item()) + 1

        # Calculate phase epochs
        phase1_epochs = max(1, int(epochs * 0.6))
        phase2_epochs = max(1, int(epochs * 0.2))
        phase3_epochs = max(1, epochs - phase1_epochs - phase2_epochs)

        print(f"\n{'='*70}")
        print("  CURRICULUM LEARNING — Early Stopping par Phase")
        print(f"{'='*70}")
        print(f"  Phase 1 (Clean Training):        max {phase1_epochs} epochs  |  patience={early_stopping_patience}")
        if adv_generator and adv_ratio > 0:
            print(f"  Phase 2 (Feature-Level Attack):   max {phase2_epochs} epochs  |  patience={early_stopping_patience}")
            print(f"  Phase 3 (Sequence-Level Attack):  max {phase3_epochs} epochs  |  patience={early_stopping_patience}")
        print(f"{'='*70}\n")

        all_crash_results = {}

        # ─────────────────────────────────────────────────────────
        # PHASE 1 : Clean Training
        # ─────────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("PHASE 1: ENTRAÎNEMENT NORMAL (Benign First)")
        print(f"{'='*60}\n")

        best_val_acc_p1 = 0.0
        best_state_p1 = copy.deepcopy(self.model.state_dict())
        patience_counter = 0

        for epoch in range(phase1_epochs):
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, criterion, None, 0.0, adv_method, None
            )
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"  P1 Epoch {epoch+1}/{phase1_epochs}  "
                  f"Train[loss={train_loss:.4f} acc={train_acc:.4f}]  "
                  f"Val[loss={val_loss:.4f} acc={val_acc:.4f}]")

            if val_acc > best_val_acc_p1:
                best_val_acc_p1 = val_acc
                best_state_p1 = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"  ⚡ Early Stopping Phase 1 à l'epoch {epoch+1}")
                    break

        # Recharger les meilleurs poids de Phase 1
        self.model.load_state_dict(best_state_p1)
        print(f"  ✓ Meilleurs poids Phase 1 rechargés (val_acc={best_val_acc_p1:.4f})")

        # Sauvegarder checkpoint Phase 1
        if save_path:
            checkpoint_path = save_path / "checkpoint_phase1.pt"
            torch.save({
                "model_state_dict": best_state_p1,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc_p1,
                "phase": 1,
            }, checkpoint_path)
            print(f"  💾 Checkpoint P1: {checkpoint_path}")

        # Crash Test Phase 1
        crash_p1 = self._crash_test(
            1, test_loader, criterion,
            feature_attack, sequence_attack, X_test, y_test, batch_size, num_classes
        )
        all_crash_results["phase_1"] = crash_p1

        # Early exit si pas d'entraînement adversarial
        if adv_generator is None or adv_ratio <= 0:
            self.history["phase_checkpoints_evaluation"] = all_crash_results
            if save_path:
                self._save_phase_report(all_crash_results, save_path)
            return self.history

        # ─────────────────────────────────────────────────────────
        # PHASE 2 : Feature-Level Adversarial Training
        # ─────────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("PHASE 2: ATTAQUES FEATURE-LEVEL (Perturbations discrètes)")
        print(f"{'='*60}\n")

        best_val_acc_p2 = 0.0
        best_state_p2 = copy.deepcopy(self.model.state_dict())
        patience_counter = 0  # RESET patience

        for epoch in range(phase2_epochs):
            global_epoch = phase1_epochs + epoch + 1
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, criterion, adv_generator, adv_ratio,
                "feature", {"clean": 0.0, "feature": 1.0, "sequence": 0.0}
            )
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"  P2 Epoch {epoch+1}/{phase2_epochs} (Global: {global_epoch}/{epochs})  "
                  f"Train[loss={train_loss:.4f} acc={train_acc:.4f}]  "
                  f"Val[loss={val_loss:.4f} acc={val_acc:.4f}]")

            if val_acc > best_val_acc_p2:
                best_val_acc_p2 = val_acc
                best_state_p2 = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"  ⚡ Early Stopping Phase 2 à l'epoch {epoch+1}")
                    break

        # Recharger les meilleurs poids de Phase 2
        self.model.load_state_dict(best_state_p2)
        print(f"  ✓ Meilleurs poids Phase 2 rechargés (val_acc={best_val_acc_p2:.4f})")

        # Sauvegarder checkpoint Phase 2
        if save_path:
            checkpoint_path = save_path / "checkpoint_phase2.pt"
            torch.save({
                "model_state_dict": best_state_p2,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc_p2,
                "phase": 2,
            }, checkpoint_path)
            print(f"  💾 Checkpoint P2: {checkpoint_path}")

        # Crash Test Phase 2
        crash_p2 = self._crash_test(
            2, test_loader, criterion,
            feature_attack, sequence_attack, X_test, y_test, batch_size, num_classes
        )
        all_crash_results["phase_2"] = crash_p2

        # ─────────────────────────────────────────────────────────
        # PHASE 3 : Sequence-Level Adversarial Training (PGD/FGSM)
        # ─────────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("PHASE 3: ATTAQUES SÉQUENTIELLES (PGD / FGSM — boîte blanche)")
        print(f"{'='*60}\n")

        best_val_acc_p3 = 0.0
        best_state_p3 = copy.deepcopy(self.model.state_dict())
        patience_counter = 0  # RESET patience

        for epoch in range(phase3_epochs):
            global_epoch = phase1_epochs + phase2_epochs + epoch + 1
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, criterion, adv_generator, adv_ratio,
                "pgd", {"clean": 0.0, "feature": 0.0, "sequence": 1.0}
            )
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"  P3 Epoch {epoch+1}/{phase3_epochs} (Global: {global_epoch}/{epochs})  "
                  f"Train[loss={train_loss:.4f} acc={train_acc:.4f}]  "
                  f"Val[loss={val_loss:.4f} acc={val_acc:.4f}]")

            if val_acc > best_val_acc_p3:
                best_val_acc_p3 = val_acc
                best_state_p3 = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"  ⚡ Early Stopping Phase 3 à l'epoch {epoch+1}")
                    break

        # Recharger les meilleurs poids de Phase 3
        self.model.load_state_dict(best_state_p3)
        print(f"  ✓ Meilleurs poids Phase 3 rechargés (val_acc={best_val_acc_p3:.4f})")

        # XGBoost: fit l'arbre AVANT le Crash Test Phase 3
        if is_xgboost:
            print("\n  🌲 Fitting XGBoost sur features LSTM extraites (avant Crash Test P3)...")
            self.model.fit_xgboost(train_loader, self.device)

        # Sauvegarder checkpoint Phase 3
        if save_path:
            checkpoint_path = save_path / "checkpoint_phase3.pt"
            torch.save({
                "model_state_dict": best_state_p3,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc_p3,
                "phase": 3,
            }, checkpoint_path)
            print(f"  💾 Checkpoint P3: {checkpoint_path}")

        # Crash Test Phase 3 (Final)
        crash_p3 = self._crash_test(
            3, test_loader, criterion,
            feature_attack, sequence_attack, X_test, y_test, batch_size, num_classes
        )
        all_crash_results["phase_3"] = crash_p3

        # Sauvegarder modèle final
        if save_path:
            final_path = save_path / "best_model.pt"
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc_p3,
                "history": self.history,
                "phases": {
                    "phase1_epochs": phase1_epochs,
                    "phase2_epochs": phase2_epochs,
                    "phase3_epochs": phase3_epochs,
                },
            }, final_path)

        # ─────────────────────────────────────────────────
        # RAPPORT COMPARATIF P1 vs P2 vs P3
        # ─────────────────────────────────────────────────
        self._print_comparative_summary(all_crash_results)

        self.history["phase_checkpoints_evaluation"] = all_crash_results

        if save_path:
            self._save_phase_report(all_crash_results, save_path)

        return self.history

    def _print_comparative_summary(self, all_crash_results: Dict):
        """Affiche le résumé comparatif avec toutes les métriques."""
        print(f"\n{'='*90}")
        print("  RAPPORT COMPARATIF — Curriculum Learning")
        print(f"{'='*90}")
        header = (f"{'Phase':<8} {'Loader':<15} {'Loss':<8} {'Acc':<8} "
                  f"{'Prec':<8} {'Recall':<8} {'F1':<8} {'RR':<8}")
        print(header)
        print(f"{'-'*90}")

        for phase_key in sorted(all_crash_results.keys()):
            results = all_crash_results[phase_key]
            phase_num = results["phase"]
            for loader_name in ["clean", "feature_attack", "sequence_pgd", "sequence_fgsm"]:
                if loader_name in results:
                    m = results[loader_name]
                    rr = m.get("robustness_ratio", 1.0) if loader_name != "clean" else 1.0
                    print(f"  P{phase_num:<5} {loader_name:<15} "
                          f"{m['loss']:<8.4f} {m['accuracy']:<8.4f} "
                          f"{m.get('precision', 0):<8.4f} {m.get('recall', 0):<8.4f} "
                          f"{m.get('f1_score', 0):<8.4f} {rr:<8.4f}")
            print(f"{'-'*90}")

        print(f"{'='*90}\n")

    def _save_phase_report(self, all_crash_results: Dict, save_path: Path):
        """Sauvegarde le rapport JSON récapitulatif comparant P1, P2 et P3."""
        report = {
            "model": self.model_name,
            "description": "Curriculum Learning — Rapport comparatif par phase",
            "phases": all_crash_results,
        }
        report_path = save_path / "curriculum_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  📄 Rapport JSON sauvegardé: {report_path}")

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
        hybrid_split: Optional[Dict[str, float]] = None,
        save_path: Optional[Path] = None,
        early_stopping_patience: int = 10,
        test_loader: Optional[DataLoader] = None,
    ) -> Dict:
        """Curriculum Learning avec Early Stopping par Phase (version légère).
        
        Phases:
        1. Phase 1 (60% epochs): Clean training only -> TEST
        2. Phase 2 (20% epochs): Feature-level adversarial training -> TEST
        3. Phase 3 (20% epochs): Sequence-level adversarial training -> TEST
        
        Chaque phase dispose de son propre Early Stopping avec patience réinitialisée
        et meilleurs poids rechargés en fin de phase.
        """
        import copy

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )

        # Calculate phase epochs (60% - 20% - 20%)
        phase1_epochs = max(1, int(epochs * 0.6))
        phase2_epochs = max(1, int(epochs * 0.2))
        phase3_epochs = max(1, epochs - phase1_epochs - phase2_epochs)

        print(f"\n{'='*60}")
        print("CURRICULUM LEARNING — Early Stopping par Phase")
        print(f"{'='*60}")
        print(f"Phase 1 (Clean Training):        max {phase1_epochs} epochs  |  patience={early_stopping_patience}")
        if adv_generator and adv_ratio > 0:
            print(f"Phase 2 (Feature-Level Attack):  max {phase2_epochs} epochs  |  patience={early_stopping_patience}")
            print(f"Phase 3 (Sequence-Level Attack): max {phase3_epochs} epochs  |  patience={early_stopping_patience}")
        print(f"{'='*60}\n")

        # ─── Phase 1: Clean Training ─────────────────────────────
        print(f"\n{'='*60}")
        print("PHASE 1: ENTRAÎNEMENT NORMAL (Benign First)")
        print(f"{'='*60}\n")

        best_val_acc_p1 = 0.0
        best_state_p1 = copy.deepcopy(self.model.state_dict())
        patience_counter = 0
        
        for epoch in range(phase1_epochs):
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, criterion, None, 0.0, adv_method, None
            )
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"  P1 Epoch {epoch+1}/{phase1_epochs}  "
                  f"Train[loss={train_loss:.4f} acc={train_acc:.4f}]  "
                  f"Val[loss={val_loss:.4f} acc={val_acc:.4f}]")

            if val_acc > best_val_acc_p1:
                best_val_acc_p1 = val_acc
                best_state_p1 = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                if save_path:
                    torch.save({
                        "model_state_dict": self.model.state_dict(),
                        "val_acc": val_acc,
                        "phase": 1,
                    }, save_path / "best_model_phase1.pt")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"  ⚡ Early Stopping Phase 1 à l'epoch {epoch+1}")
                    break

        # Recharger meilleurs poids Phase 1
        self.model.load_state_dict(best_state_p1)
        print(f"  ✓ Meilleurs poids Phase 1 rechargés (val_acc={best_val_acc_p1:.4f})")

        # Test after Phase 1
        if test_loader:
            self.test_model(test_loader, criterion, "TEST APRES PHASE 1 (Clean)")

        # Early exit if no adversarial training
        if adv_generator is None or adv_ratio <= 0:
            return self.history

        # ─── Phase 2: Feature-Level Adversarial Training ─────────
        print(f"\n{'='*60}")
        print("PHASE 2: ATTAQUES FEATURE-LEVEL (Perturbations discrètes)")
        print(f"{'='*60}\n")

        best_val_acc_p2 = 0.0
        best_state_p2 = copy.deepcopy(self.model.state_dict())
        patience_counter = 0  # RESET
        
        for epoch in range(phase2_epochs):
            global_epoch = phase1_epochs + epoch + 1
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, criterion, adv_generator, adv_ratio, 
                "feature", {"clean": 0.0, "feature": 1.0, "sequence": 0.0}
            )
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"  P2 Epoch {epoch+1}/{phase2_epochs} (Global: {global_epoch}/{epochs})  "
                  f"Train[loss={train_loss:.4f} acc={train_acc:.4f}]  "
                  f"Val[loss={val_loss:.4f} acc={val_acc:.4f}]")

            if val_acc > best_val_acc_p2:
                best_val_acc_p2 = val_acc
                best_state_p2 = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                if save_path:
                    torch.save({
                        "model_state_dict": self.model.state_dict(),
                        "val_acc": val_acc,
                        "phase": 2,
                    }, save_path / "best_model_phase2.pt")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"  ⚡ Early Stopping Phase 2 à l'epoch {epoch+1}")
                    break

        # Recharger meilleurs poids Phase 2
        self.model.load_state_dict(best_state_p2)
        print(f"  ✓ Meilleurs poids Phase 2 rechargés (val_acc={best_val_acc_p2:.4f})")

        # Test after Phase 2
        if test_loader:
            self.test_model(test_loader, criterion, "TEST APRES PHASE 2 (Feature-Level)")

        # ─── Phase 3: Sequence-Level Adversarial Training ────────
        print(f"\n{'='*60}")
        print("PHASE 3: ATTAQUES SÉQUENTIELLES (PGD / FGSM — boîte blanche)")
        print(f"{'='*60}\n")

        best_val_acc_p3 = 0.0
        best_state_p3 = copy.deepcopy(self.model.state_dict())
        patience_counter = 0  # RESET
        
        for epoch in range(phase3_epochs):
            global_epoch = phase1_epochs + phase2_epochs + epoch + 1
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, criterion, adv_generator, adv_ratio,
                "pgd", {"clean": 0.0, "feature": 0.0, "sequence": 1.0}
            )
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"  P3 Epoch {epoch+1}/{phase3_epochs} (Global: {global_epoch}/{epochs})  "
                  f"Train[loss={train_loss:.4f} acc={train_acc:.4f}]  "
                  f"Val[loss={val_loss:.4f} acc={val_acc:.4f}]")

            if val_acc > best_val_acc_p3:
                best_val_acc_p3 = val_acc
                best_state_p3 = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                if save_path:
                    torch.save({
                        "model_state_dict": self.model.state_dict(),
                        "val_acc": val_acc,
                        "phase": 3,
                    }, save_path / "best_model_phase3.pt")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"  ⚡ Early Stopping Phase 3 à l'epoch {epoch+1}")
                    break

        # Recharger meilleurs poids Phase 3
        self.model.load_state_dict(best_state_p3)
        print(f"  ✓ Meilleurs poids Phase 3 rechargés (val_acc={best_val_acc_p3:.4f})")

        # Test after Phase 3 (Final)
        if test_loader:
            self.test_model(test_loader, criterion, "TEST FINAL APRES PHASE 3 (Sequence-Level)")

        # Save final best model
        if save_path:
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc_p3,
                "history": self.history,
                "phases": {
                    "phase1_epochs": phase1_epochs,
                    "phase2_epochs": phase2_epochs,
                    "phase3_epochs": phase3_epochs,
                },
            }, save_path / "best_model.pt")

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
    elif model_type == "xgboost_lstm":
        model = XGBoostLSTMClassifier(input_size, num_classes, LSTM_CONFIG)
    elif model_type == "cnn":
        model = CNNClassifier(input_size, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

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
) -> Dict:
    """Run experiment avec sauvegarde et evaluation des checkpoints de phase."""
    device = get_device()
    print(f"\n{'=' * 60}")
    print(f"Experiment (Phase Checkpoints): {model_type.upper()} | Seq={seq_length} | Adv={adv_method}")
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
        seq_length, stride=max(1, seq_length // 2), max_files=max_files, data_dir=data_dir
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
    print(f"Continuous features: {n_continuous_features}, Binary features: {input_size - n_continuous_features}")

    model = create_model(model_type, input_size, num_classes, seq_length, device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = (
        RESULTS_DIR / "models" / f"{model_type}_{seq_length}_{adv_method}_phases_{timestamp}"
    )
    if save_results:
        save_path.mkdir(parents=True, exist_ok=True)

    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    y_train_expanded = np.repeat(y_train, X_train.shape[1])
    feature_attack = FeatureLevelAttack(
        X_train_flat, y_train_expanded, features, num_classes,
        n_continuous_features=n_continuous_features,
    )

    del X_train_flat, y_train_expanded
    gc.collect()

    sequence_attack = SequenceLevelAttack(
        model, device, epsilon=0.1, alpha=0.01, num_steps=10,
        n_continuous_features=n_continuous_features,
    )

    adv_generator = HybridAdversarialAttack(
        feature_attack, sequence_attack, features, combine_ratio=0.5
    )

    trainer = AdversarialTrainer(model, device, model_type)

    if hybrid_split is None:
        hybrid_split = {"clean": 0.6, "feature": 0.2, "sequence": 0.2}

    print(f"\nTraining avec checkpoints de phase - adversarial ratio: {adv_ratio}")
    
    history = trainer.fit_with_phase_checkpoints(
        train_loader,
        val_loader,
        test_loader,
        epochs=epochs,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        adv_generator=adv_generator if adv_method != "none" else None,
        adv_ratio=adv_ratio if adv_method != "none" else 0.0,
        adv_method=adv_method if adv_method in ["pgd", "fgsm"] else "pgd",
        hybrid_split=hybrid_split,
        save_path=save_path if save_results else None,
        feature_attack=feature_attack,
        sequence_attack=sequence_attack,
        X_test=X_test,
        y_test=y_test,
        batch_size=batch_size,
        is_xgboost=(model_type == "xgboost_lstm"),
    )

    # Récupérer les résultats d'évaluation des phases
    phase_results = history.get("phase_checkpoints_evaluation", {})

    # Construire le résultat final
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

    # Ajouter les métriques finales
    if phase_results:
        last_phase = max(phase_results.keys(), key=lambda k: int(k.split("_")[1]))
        final_metrics = phase_results[last_phase]
        results["test_accuracy_clean"] = final_metrics.get("clean", {}).get("accuracy", 0)
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
        seq_length, stride=max(1, seq_length // 2), max_files=max_files, data_dir=data_dir
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
    print(f"Continuous features: {n_continuous_features}, Binary features: {input_size - n_continuous_features}")

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
        X_train_flat, y_train_expanded, features, num_classes,
        n_continuous_features=n_continuous_features,
    )

    del X_train_flat, y_train_expanded
    gc.collect()

    sequence_attack = SequenceLevelAttack(
        model, device, epsilon=0.1, alpha=0.01, num_steps=10,
        n_continuous_features=n_continuous_features,
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
        print(f"  Phased training: 60% clean -> 20% feature-level -> 20% sequence-level")
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        adv_generator=adv_generator if adv_method != "none" else None,
        adv_ratio=adv_ratio if adv_method != "none" else 0.0,
        adv_method=adv_method if adv_method in ["pgd", "fgsm"] else "pgd",
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
        "hybrid_split": hybrid_split,
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
        seq_len = X_eval.shape[1]
        y_eval_expanded = np.repeat(y_eval, seq_len)
        X_adv_feature = feature_attack.generate_batch(
            X_eval.reshape(-1, X_eval.shape[-1]), y_eval_expanded, verbose=True
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
        choices=["lstm", "transformer", "cnn_lstm", "cnn", "xgboost_lstm"],
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
        default="none,feature,pgd,hybrid",
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

    args = parser.parse_args()

    # Enforce sequence length of 25 for transformer if it's currently default
    if args.model == "transformer" and args.seq_length == 10:
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
        )


if __name__ == "__main__":
    main()
