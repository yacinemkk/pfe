#!/usr/bin/env python3
"""
Main Entry Point for IoT Device Identification Pipeline

Per docs/important.md - Complete Implementation:
- PHASE 1: Data Preprocessing
- PHASE 2: Anti-Data Leakage Pipeline
- PHASE 3: Tokenization (optional, for Transformer)
- PHASE 4: Model Architectures
- PHASE 5: Standard Training
- PHASE 6: Adversarial Training
- PHASE 7: Comparison & Reporting
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import DataLoader as IoTDataLoader
from src.data.preprocessing import Preprocessor
from src.data.split import TemporalSplitter
from src.data.sequence import SequenceGenerator
from src.models.lstm import LSTMClassifier
from src.models.bilstm import BiLSTMClassifier
from src.models.cnn_lstm import CNNLSTMClassifier
from src.models.transformer import TransformerClassifier
from src.models.cnn_bilstm_transformer import CNNBiLSTMTransformerClassifier
from src.models.xgboost_lstm import XGBoostLSTMClassifier
from src.training.trainer import Trainer, IoTSequenceDataset, create_dataloaders
from src.training.evaluator import Evaluator, ModelComparator, CrashTestEvaluator
from src.adversarial.attacks import (
    SensitivityAnalysis,
    AdversarialSearch,
)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_preprocessing(
    config: dict, max_files: Optional[int] = None, verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    PHASE 1: Data Preprocessing.

    Returns:
        X, y, feature_names
    """
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 1: Data Preprocessing")
        print("=" * 60)

    loader = IoTDataLoader()
    df = loader.load_iot_ipfix_home(max_files=max_files, verbose=verbose)

    if verbose:
        stats = loader.analyze_distribution(df)
        print(f"\n  Classes: {stats['num_classes']}")
        print(f"  Imbalance ratio: {stats['imbalance_ratio']:.2f}")

    preprocessor = Preprocessor()
    X, y = preprocessor.process(df, verbose=verbose)

    return X, y, preprocessor.feature_names


def run_anti_leakage_split(
    df, config: dict, verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    PHASE 2: Anti-Data Leakage Pipeline.

    Returns:
        X_train, y_train, X_test, y_test
    """
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 2: Anti-Data Leakage Split")
        print("=" * 60)

    splitter = TemporalSplitter()
    df_train, df_test = splitter.temporal_split(df, verbose=verbose)
    splitter.validate_integrity(df_train, df_test)

    preprocessor = Preprocessor()
    X_train, y_train = preprocessor.extract_features(df_train, verbose=verbose)
    X_test, y_test = preprocessor.extract_features(df_test, verbose=verbose)

    return X_train, y_train, X_test, y_test


def create_sequences(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate sequences for sequential models.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Generating Sequences")
        print("=" * 60)

    generator = SequenceGenerator()
    X_train_seq, y_train_seq, X_test_seq, y_test_seq = (
        generator.create_train_test_sequences(X_train, y_train, X_test, y_test)
    )

    generator.validate_sequence_integrity(X_train_seq, y_train_seq)
    generator.validate_sequence_integrity(X_test_seq, y_test_seq)

    return X_train_seq, y_train_seq, X_test_seq, y_test_seq


def get_model(model_name: str, input_size: int, num_classes: int, seq_length: int):
    """Get model instance by name."""
    models = {
        "lstm": LSTMClassifier,
        "bilstm": BiLSTMClassifier,
        "cnn_lstm": CNNLSTMClassifier,
        "xgboost_lstm": XGBoostLSTMClassifier,
        "transformer": TransformerClassifier,
        "hybrid": CNNBiLSTMTransformerClassifier,
    }

    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(models.keys())}"
        )

    model_class = models[model_name]

    if model_name in ["transformer", "hybrid"]:
        return model_class(input_size, num_classes, seq_length)
    else:
        return model_class(input_size, num_classes)


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    save_path: str,
    verbose: bool = True,
) -> Tuple[Trainer, Dict]:
    """
    PHASE 5: Standard Training.
    """
    trainer = Trainer(model)
    history = trainer.fit(
        train_loader, val_loader, save_path=save_path, verbose=verbose
    )
    return trainer, history


def generate_adversarial_samples(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model: torch.nn.Module,
    feature_names: List[str],
    config: dict,
    device: torch.device,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate adversarial samples using sensitivity analysis + adversarial search.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Generating Adversarial Samples")
        print("=" * 60)

    num_classes = len(np.unique(y_train))

    sensitivity_analysis = SensitivityAnalysis(
        X_train=X_train.reshape(-1, X_train.shape[-1]),
        y_train=np.repeat(y_train, X_train.shape[1]),
        feature_names=feature_names,
        num_classes=num_classes,
    )

    sensitivity_results = sensitivity_analysis.analyze(
        model,
        X_train,
        y_train,
        device,
        batch_size=config["training"]["batch_size"],
        verbose=verbose,
    )

    adversarial_search = AdversarialSearch(
        model=model,
        device=device,
        sensitivity_analysis=sensitivity_analysis,
        target_accuracy=config["adversarial"]["target_accuracy"],
        batch_size=config["training"]["batch_size"],
    )

    X_adv = adversarial_search.generate_adversarial(
        X_train,
        y_train,
        sensitivity_results=sensitivity_results,
        verbose=verbose,
    )

    return X_adv, y_train.copy()


def run_adversarial_training(
    model: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_adv: np.ndarray,
    y_adv: np.ndarray,
    config: dict,
    save_path: str,
    verbose: bool = True,
) -> Trainer:
    """
    PHASE 6: Adversarial Training.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 6: Adversarial Training")
        print("=" * 60)

    clean_ratio = 0.8
    adv_ratio = 0.2

    n_clean = int(len(X_train) * clean_ratio)
    n_adv = int(len(X_adv) * adv_ratio)

    X_combined = np.vstack([X_train[:n_clean], X_adv[:n_adv]])
    y_combined = np.concatenate([y_train[:n_clean], y_adv[:n_adv]])

    indices = np.random.permutation(len(X_combined))
    X_combined = X_combined[indices]
    y_combined = y_combined[indices]

    if verbose:
        print(f"  Combined dataset: {len(X_combined):,} samples")
        print(f"    Clean: {n_clean:,} | Adversarial: {n_adv:,}")

    train_dataset = IoTSequenceDataset(X_combined, y_combined)
    val_dataset = IoTSequenceDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"])

    trainer = Trainer(model)
    trainer.fit(train_loader, val_loader, save_path=save_path, verbose=verbose)

    return trainer


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    save_dir: str,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate model and save results.
    """
    evaluator = Evaluator(model, device, class_names=class_names)
    metrics = evaluator.evaluate(test_loader, "test", verbose=verbose)

    evaluator.save_results(save_dir, "test")

    return metrics


def split_test_for_crash(
    X_test: np.ndarray,
    y_test: np.ndarray,
    clean_ratio: float = 0.5,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split test set into clean_test (reserved for evaluation) and adv_test (for adversarial generation).

    Temporal split by label group: first portion → clean, last portion → adv.
    This preserves the chronological ordering within each device group.
    """
    unique_labels = np.unique(y_test)
    label_to_group = {label: i for i, label in enumerate(unique_labels)}
    groups = np.array([label_to_group[label] for label in y_test])

    clean_indices = []
    adv_indices = []

    for group_id in np.unique(groups):
        group_mask = groups == group_id
        group_indices = np.where(group_mask)[0]
        n_group = len(group_indices)

        if n_group < 10:
            clean_indices.extend(group_indices)
        else:
            split_point = int(n_group * clean_ratio)
            clean_indices.extend(group_indices[:split_point])
            adv_indices.extend(group_indices[split_point:])

    X_clean = X_test[clean_indices]
    y_clean = y_test[clean_indices]
    X_adv_src = X_test[adv_indices]
    y_adv_src = y_test[adv_indices]

    if verbose:
        print(
            f"\n  Test split → Clean: {len(X_clean):,} | Adv source: {len(X_adv_src):,}"
        )

    return X_clean, y_clean, X_adv_src, y_adv_src


def run_crash_test(
    model: torch.nn.Module,
    X_clean: np.ndarray,
    y_clean: np.ndarray,
    X_adv_src: np.ndarray,
    y_adv_src: np.ndarray,
    feature_names: List[str],
    config: dict,
    device: torch.device,
    verbose: bool = True,
) -> Dict:
    """
    Run crash test (adversarial robustness evaluation) using sensitivity + search.

    X_clean/y_clean: Reserved clean test data (never used for attack generation)
    X_adv_src/y_adv_src: Data used to generate adversarial samples
    """
    num_classes = len(np.unique(y_adv_src))

    sensitivity_analysis = SensitivityAnalysis(
        X_adv_src.reshape(-1, X_adv_src.shape[-1]),
        np.repeat(y_adv_src, X_adv_src.shape[1]),
        feature_names,
        num_classes,
    )

    sensitivity_results = sensitivity_analysis.analyze(
        model,
        X_adv_src,
        y_adv_src,
        device,
        batch_size=config["training"]["batch_size"],
        verbose=verbose,
    )

    adversarial_search = AdversarialSearch(
        model=model,
        device=device,
        sensitivity_analysis=sensitivity_analysis,
        target_accuracy=config["adversarial"]["target_accuracy"],
        batch_size=config["training"]["batch_size"],
    )

    X_adv = adversarial_search.generate_adversarial(
        X_adv_src,
        y_adv_src,
        sensitivity_results=sensitivity_results,
        verbose=verbose,
    )

    crash_evaluator = CrashTestEvaluator(model, device)
    results = crash_evaluator.run_crash_test(
        X_clean, y_clean, X_adv, y_adv_src, verbose=verbose
    )

    return results


def run_full_pipeline(
    config: dict,
    models_to_train: List[str] = None,
    max_files: Optional[int] = None,
    skip_adversarial: bool = False,
    verbose: bool = True,
):
    """
    Run the complete IoT device identification pipeline.
    """
    if models_to_train is None:
        models_to_train = ["lstm", "bilstm", "cnn_lstm", "transformer", "hybrid"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"\nDevice: {device}")

    X, y, feature_names = run_preprocessing(config, max_files, verbose)

    preprocessor = Preprocessor()
    preprocessor.feature_names = feature_names

    # ─── Anti-Leakage Temporal Split ─────────────────────────────────────────
    # WARNING: This split is by label group only, NOT chronological.
    # The 'start' timestamp column has already been removed by SDN filtering.
    # For a proper chronological split, use:
    #   python -m src.data.preprocessor  (CSV pipeline)
    #   python -m src.data.json_preprocessor  (JSON pipeline)
    # These handle: group by device → sort by flow_start → 80/20 temporal split
    #
    # The current split groups by label and takes first 80% as train.
    # This is NOT a true chronological split and may introduce leakage.
    # Consider this a legacy fallback; prefer the dedicated preprocessors.

    splitter = TemporalSplitter()

    unique_labels = np.unique(y)
    label_to_group = {label: i for i, label in enumerate(unique_labels)}
    groups = np.array([label_to_group[label] for label in y])

    # Chronological split: sort within each group, then take first 80% as train
    indices = np.arange(len(X))
    train_indices = []
    test_indices = []

    for group_id in np.unique(groups):
        group_mask = groups == group_id
        group_indices = indices[group_mask]
        n_group = len(group_indices)

        if n_group < 20:
            train_indices.extend(group_indices)
        else:
            split_point = int(n_group * 0.8)
            train_indices.extend(group_indices[:split_point])
            test_indices.extend(group_indices[split_point:])

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    if verbose:
        print(f"\n  Train: {len(X_train):,} | Test: {len(X_test):,}")

    X_train, X_test = preprocessor.normalize(X_train, X_test, verbose=verbose)
    y_train, y_test = preprocessor.encode_labels(y_train, y_test, verbose=verbose)

    num_classes = preprocessor.num_classes
    class_names = list(preprocessor.label_encoder.classes_)

    generator = SequenceGenerator()
    X_train_seq, y_train_seq, X_test_seq, y_test_seq = (
        generator.create_train_test_sequences(X_train, y_train, X_test, y_test)
    )

    n_val = int(len(X_train_seq) * 0.1)
    X_val_seq, y_val_seq = X_train_seq[:n_val], y_train_seq[:n_val]
    X_train_seq, y_train_seq = X_train_seq[n_val:], y_train_seq[n_val:]

    # Split test set: clean_test (reserved for evaluation) + adv_test (for adversarial generation)
    # Temporal split by device: first 50% clean, last 50% for adversarial
    X_clean_test, y_clean_test, X_adv_test, y_adv_test = split_test_for_crash(
        X_test_seq, y_test_seq, clean_ratio=0.5, verbose=verbose
    )

    if verbose:
        print(f"\n  Train seq: {len(X_train_seq):,}")
        print(f"  Val seq: {len(X_val_seq):,}")
        print(f"  Test clean: {len(X_clean_test):,}")
        print(f"  Test adv source: {len(X_adv_test):,}")

    seq_length = X_train_seq.shape[1]
    input_size = X_train_seq.shape[2]

    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    comparator = ModelComparator()

    for model_name in models_to_train:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Training: {model_name.upper()}")
            print("=" * 60)

        model = get_model(model_name, input_size, num_classes, seq_length)
        model = model.to(device)

        train_loader, val_loader, test_loader = create_dataloaders(
            X_train_seq,
            y_train_seq,
            X_val_seq,
            y_val_seq,
            X_clean_test,
            y_clean_test,
            config["training"]["batch_size"],
        )

        model_dir = results_dir / "models" / model_name
        trainer, history = train_model(
            model, train_loader, val_loader, config, str(model_dir), verbose
        )

        metrics = evaluate_model(
            model, test_loader, class_names, device, str(model_dir), verbose
        )

        comparator.add_model_results(model_name, metrics, phase="standard")

        if not skip_adversarial:
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Adversarial Training: {model_name.upper()}")
                print("=" * 60)

            X_adv, y_adv = generate_adversarial_samples(
                X_train_seq, y_train_seq, model, feature_names, config, device, verbose
            )

            adv_model = get_model(model_name, input_size, num_classes, seq_length)
            adv_model = adv_model.to(device)

            adv_trainer = run_adversarial_training(
                adv_model,
                X_train_seq,
                y_train_seq,
                X_val_seq,
                y_val_seq,
                X_adv,
                y_adv,
                config,
                str(model_dir / "adversarial"),
                verbose,
            )

            adv_metrics = evaluate_model(
                adv_model,
                test_loader,
                class_names,
                device,
                str(model_dir / "adversarial"),
                verbose,
            )

            comparator.add_model_results(
                f"{model_name}_adv", adv_metrics, phase="adversarial"
            )

            crash_results = run_crash_test(
                adv_model,
                X_clean_test,
                y_clean_test,
                X_adv_test,
                y_adv_test,
                feature_names,
                config,
                device,
                verbose,
            )

    comparator.save_report(str(results_dir / "comparison"))

    if verbose:
        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        print(f"\nResults saved to: {results_dir}")
        print(f"\nBest model (standard): {comparator.get_best_model('standard')}")
        if not skip_adversarial:
            print(
                f"Best model (adversarial): {comparator.get_best_model('adversarial')}"
            )

    return comparator


def main():
    parser = argparse.ArgumentParser(description="IoT Device Identification Pipeline")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["lstm", "bilstm", "cnn_lstm", "xgboost_lstm", "transformer", "hybrid"],
        help="Models to train",
    )
    parser.add_argument(
        "--max-files", type=int, default=None, help="Max data files to load"
    )
    parser.add_argument(
        "--skip-adversarial", action="store_true", help="Skip adversarial training"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output")

    args = parser.parse_args()

    config = load_config(args.config)

    run_full_pipeline(
        config=config,
        models_to_train=args.models,
        max_files=args.max_files,
        skip_adversarial=args.skip_adversarial,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
