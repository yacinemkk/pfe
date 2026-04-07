"""
Adversarial Attacks for IoT Device Identification (Sequential Models)

Unified adversarial attack pipeline:
1. SensitivityAnalysis: Identify vulnerable features using mimicry strategies
2. AdversarialSearch: Greedy search to construct minimal evasion
3. AdversarialEvaluator: Evaluate model robustness against attacks
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import gc


class SensitivityAnalysis:
    """
    Sensitivity analysis: identifies vulnerable features by testing perturbation strategies.

    Strategies per feature:
      - Zero: set feature to 0
      - Mimic_Mean: set to mean of benign samples
      - Mimic_95th: set to 95th percentile of benign samples
      - Padding_x10: multiply feature by 10

    Returns ranked list of {feature, strategy, accuracy, drop} sorted by impact.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        num_classes: int,
        non_modifiable: List[str] = None,
        dependent_pairs: Dict[str, str] = None,
        n_continuous_features: Optional[int] = None,
    ):
        self.feature_names = feature_names
        self.num_classes = num_classes
        self.n_continuous_features = n_continuous_features

        if non_modifiable is not None:
            self.non_modifiable = non_modifiable
        else:
            has_pkt_dir = any(f.startswith("pkt_dir_") for f in feature_names)
            if has_pkt_dir:
                self.non_modifiable = [
                    "protocolIdentifier",
                ] + [f"pkt_dir_{i}" for i in range(8)]
            else:
                self.non_modifiable = ["ipProto"]

        if dependent_pairs is not None:
            self.dependent_pairs = dependent_pairs
        else:
            has_pkt_dir = any(f.startswith("pkt_dir_") for f in feature_names)
            if has_pkt_dir:
                self.dependent_pairs = {
                    "reversePacketTotalCount": "packetTotalCount",
                    "reverseOctetTotalCount": "octetTotalCount",
                    "reverseAverageInterarrivalTime": "averageInterarrivalTime",
                }
            else:
                self.dependent_pairs = {
                    "inPacketCount": "outPacketCount",
                    "inByteCount": "outByteCount",
                    "inAvgIAT": "outAvgIAT",
                    "inAvgPacketSize": "outAvgPacketSize",
                }

        self.modifiable_indices = self._get_modifiable_indices()
        self.dependent_indices = self._get_dependent_indices()
        self.benign_stats = self._compute_benign_stats(X_train, y_train)

    def _get_modifiable_indices(self) -> np.ndarray:
        modifiable = []
        for i, name in enumerate(self.feature_names):
            if name not in self.non_modifiable:
                modifiable.append(i)
        return np.array(modifiable)

    def _get_dependent_indices(self) -> List[Tuple[int, int]]:
        pairs = []
        for dep, indep in self.dependent_pairs.items():
            if dep in self.feature_names and indep in self.feature_names:
                pairs.append(
                    (self.feature_names.index(indep), self.feature_names.index(dep))
                )
        return pairs

    def _compute_benign_stats(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """Compute mean and 95th percentile for each class (used as benign reference)."""
        stats = {}
        for cls in range(self.num_classes):
            mask = y_train == cls
            if np.sum(mask) > 0:
                data = X_train[mask]
                stats[cls] = {
                    "mean": np.mean(data, axis=0),
                    "p95": np.percentile(data, 95, axis=0),
                }
        return stats

    def _apply_strategy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_idx: int,
        strategy: str,
    ) -> np.ndarray:
        """Apply a perturbation strategy to a specific feature on attack samples."""
        X_perturbed = X.copy().astype(float)
        n_features = X.shape[-1]

        if X.ndim == 3:
            for cls in range(self.num_classes):
                cls_mask = y == cls
                if not np.any(cls_mask):
                    continue
                if cls not in self.benign_stats:
                    continue
                benign = self.benign_stats[cls]

                if strategy == "Zero":
                    X_perturbed[cls_mask, :, feature_idx] = 0.0
                elif strategy == "Mimic_Mean":
                    X_perturbed[cls_mask, :, feature_idx] = benign["mean"][feature_idx]
                elif strategy == "Mimic_95th":
                    X_perturbed[cls_mask, :, feature_idx] = benign["p95"][feature_idx]
                elif strategy == "Padding_x10":
                    X_perturbed[cls_mask, :, feature_idx] *= 10.0
        else:
            for cls in range(self.num_classes):
                cls_mask = y == cls
                if not np.any(cls_mask):
                    continue
                if cls not in self.benign_stats:
                    continue
                benign = self.benign_stats[cls]

                if strategy == "Zero":
                    X_perturbed[cls_mask, feature_idx] = 0.0
                elif strategy == "Mimic_Mean":
                    X_perturbed[cls_mask, feature_idx] = benign["mean"][feature_idx]
                elif strategy == "Mimic_95th":
                    X_perturbed[cls_mask, feature_idx] = benign["p95"][feature_idx]
                elif strategy == "Padding_x10":
                    X_perturbed[cls_mask, feature_idx] *= 10.0

        return X_perturbed

    def projection(self, X: np.ndarray) -> np.ndarray:
        """Clip perturbed values to valid ranges and enforce dependent constraints."""
        X_proj = X.copy()
        n_cont = self.n_continuous_features

        if n_cont is not None:
            if X_proj.ndim == 3:
                X_proj[:, :, :n_cont] = np.clip(X_proj[:, :, :n_cont], -3.0, 3.0)
                X_proj[:, :, n_cont:] = np.clip(np.round(X_proj[:, :, n_cont:]), 0, 1)
            elif X_proj.ndim == 2:
                X_proj[:, :n_cont] = np.clip(X_proj[:, :n_cont], -3.0, 3.0)
                X_proj[:, n_cont:] = np.clip(np.round(X_proj[:, n_cont:]), 0, 1)
            elif X_proj.ndim == 1:
                X_proj[:n_cont] = np.clip(X_proj[:n_cont], -3.0, 3.0)
                X_proj[n_cont:] = np.clip(np.round(X_proj[n_cont:]), 0, 1)
        else:
            X_proj = np.clip(X_proj, -3.0, 3.0)

        for indep_idx, dep_idx in self.dependent_indices:
            if n_cont is not None and (dep_idx >= n_cont or indep_idx >= n_cont):
                continue
            if X_proj.ndim == 3:
                ratio = np.abs(X_proj[:, :, dep_idx]) / (
                    np.abs(X_proj[:, :, indep_idx]) + 1e-8
                )
                X_proj[:, :, dep_idx] = X_proj[:, :, indep_idx] * np.clip(
                    ratio, 0.5, 2.0
                )
            elif X_proj.ndim == 2:
                ratio = np.abs(X_proj[:, dep_idx]) / (
                    np.abs(X_proj[:, indep_idx]) + 1e-8
                )
                X_proj[:, dep_idx] = X_proj[:, indep_idx] * np.clip(ratio, 0.5, 2.0)
            elif X_proj.ndim == 1:
                ratio = np.abs(X_proj[dep_idx]) / (np.abs(X_proj[indep_idx]) + 1e-8)
                X_proj[dep_idx] = X_proj[indep_idx] * np.clip(ratio, 0.5, 2.0)

        return X_proj

    def analyze(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        device: torch.device,
        batch_size: int = 64,
        verbose: bool = False,
    ) -> List[Dict]:
        """
        Analyze sensitivity: for each feature, test all 4 strategies
        and measure accuracy drop.

        Returns list of dicts sorted by drop (descending):
        {feature, feature_idx, strategy, accuracy, drop}
        """
        model.eval()
        results = []

        base_acc = self._evaluate_model(model, X, y, device, batch_size)

        for i, feat in enumerate(self.feature_names):
            if i not in self.modifiable_indices:
                continue
            if verbose:
                print(f"  Analyzing feature {i + 1}/{len(self.feature_names)}: {feat}")

            for strategy in ["Zero", "Mimic_Mean", "Mimic_95th", "Padding_x10"]:
                X_pert = self._apply_strategy(X, y, i, strategy)
                X_pert = self.projection(X_pert)
                pert_acc = self._evaluate_model(model, X_pert, y, device, batch_size)
                drop = base_acc - pert_acc

                results.append(
                    {
                        "feature": feat,
                        "feature_idx": i,
                        "strategy": strategy,
                        "accuracy": pert_acc,
                        "drop": drop,
                    }
                )

                if drop > 0.05 and verbose:
                    print(
                        f"    -> {strategy}: Acc dropped to {pert_acc:.4f} (Drop: {drop:.4f})"
                    )

        results.sort(key=lambda r: r["drop"], reverse=True)
        return results

    def generate_adversarial(
        self,
        X: np.ndarray,
        y: np.ndarray,
        strategies: Optional[List[Dict]] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Generate adversarial batch using specified or default strategies.

        Args:
            X: input data (N, seq_len, n_features) or (N, n_features)
            y: labels (N,)
            strategies: list of {feature_idx, strategy} to apply.
                       If None, applies Mimic_Mean on all modifiable features.
        """
        X_adv = np.zeros_like(X, dtype=float)

        iterator = range(len(X))
        if verbose:
            iterator = tqdm(iterator, desc="Generating adversarial samples")

        for i in iterator:
            X_adv[i] = self._generate_single(X[i], int(y[i]), strategies)

        return X_adv

    def _generate_single(
        self,
        x0: np.ndarray,
        true_class: int,
        strategies: Optional[List[Dict]] = None,
    ) -> np.ndarray:
        """Generate adversarial sample using specified strategies."""
        x_adv = x0.copy().astype(float)

        if strategies is None:
            for idx in self.modifiable_indices:
                if true_class in self.benign_stats:
                    x_adv[..., idx] = self.benign_stats[true_class]["mean"][idx]
        else:
            for s in strategies:
                fidx = s["feature_idx"]
                strat = s["strategy"]
                if fidx >= x_adv.shape[-1]:
                    continue

                if strat == "Zero":
                    x_adv[..., fidx] = 0.0
                elif strat == "Mimic_Mean":
                    if true_class in self.benign_stats:
                        x_adv[..., fidx] = self.benign_stats[true_class]["mean"][fidx]
                elif strat == "Mimic_95th":
                    if true_class in self.benign_stats:
                        x_adv[..., fidx] = self.benign_stats[true_class]["p95"][fidx]
                elif strat == "Padding_x10":
                    x_adv[..., fidx] *= 10.0

        x_adv = self.projection(x_adv)
        return x_adv

    def _evaluate_model(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        device: torch.device,
        batch_size: int,
    ) -> float:
        """Evaluate model accuracy on data."""
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


class AdversarialSearch:
    """
    Greedy adversarial search: combines the most effective feature-strategy pairs
    to minimize model accuracy.

    Takes sensitivity analysis results and greedily applies perturbations
    one by one, keeping only those that reduce accuracy.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        sensitivity_analysis: SensitivityAnalysis,
        target_accuracy: float = 0.5,
        batch_size: int = 64,
    ):
        self.model = model
        self.device = device
        self.sensitivity_analysis = sensitivity_analysis
        self.target_accuracy = target_accuracy
        self.batch_size = batch_size

    def _evaluate_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Evaluate model accuracy."""
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                X_batch = torch.FloatTensor(X[i : i + self.batch_size]).to(self.device)
                y_batch = torch.LongTensor(y[i : i + self.batch_size]).to(self.device)
                outputs = self.model(X_batch)
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()
        return correct / total if total > 0 else 0.0

    def search(
        self,
        sensitivity_results: List[Dict],
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Greedy adversarial search: combine best strategies to minimize accuracy.

        Args:
            sensitivity_results: output from SensitivityAnalysis.analyze()
            X: input data
            y: labels

        Returns:
            (X_adversarial, applied_strategies)
        """
        sorted_results = sorted(
            sensitivity_results, key=lambda r: r["drop"], reverse=True
        )

        current_X = X.copy()
        base_acc = self._evaluate_model(current_X, y)
        if verbose:
            print(f"  Baseline Accuracy: {base_acc:.4f}")

        applied_strategies = []
        applied_features = set()

        for entry in sorted_results:
            feat_idx = entry["feature_idx"]
            strat = entry["strategy"]

            if feat_idx in applied_features:
                continue

            if verbose:
                print(
                    f"  Testing {entry['feature']} with {strat} (Current Acc: {base_acc:.4f})"
                )

            temp_X = current_X.copy()
            for i in range(len(X)):
                temp_X[i] = self.sensitivity_analysis._generate_single(
                    X[i],
                    int(y[i]),
                    strategies=[{"feature_idx": feat_idx, "strategy": strat}],
                )

            new_acc = self._evaluate_model(temp_X, y)

            if new_acc < base_acc:
                base_acc = new_acc
                current_X = temp_X
                applied_strategies.append(
                    {
                        "feature": entry["feature"],
                        "feature_idx": feat_idx,
                        "strategy": strat,
                        "accuracy": new_acc,
                    }
                )
                applied_features.add(feat_idx)
                if verbose:
                    print(f"    -> ADDED: Acc dropped to {new_acc:.4f}")
            else:
                if verbose:
                    print(f"    -> SKIPPED: No improvement (Acc: {new_acc:.4f})")

            if base_acc <= self.target_accuracy:
                if verbose:
                    print(
                        f"  Target reached! Accuracy ({base_acc:.4f}) <= Target ({self.target_accuracy:.4f})"
                    )
                break

        return current_X, applied_strategies

    def generate_adversarial(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sensitivity_results: List[Dict],
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Generate adversarial batch using greedy search.

        Args:
            X: input data
            y: labels
            sensitivity_results: output from SensitivityAnalysis.analyze()
        """
        X_adv, applied = self.search(sensitivity_results, X, y, verbose=verbose)
        return X_adv


class AdversarialEvaluator:
    """Evaluates model robustness against adversarial attacks."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attacks: Dict[str, np.ndarray],
        batch_size: int = 64,
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """
        Evaluate model on clean and adversarial data.

        Returns:
            results: dict with accuracy for each attack variant
            robustness: dict with robustness ratio per attack
        """
        results = {}
        results["clean"] = self._evaluate_clean(X, y, batch_size)

        for attack_name, X_adv in attacks.items():
            results[attack_name] = self._evaluate_adversarial(X_adv, y, batch_size)

        robustness = {}
        for attack_name in attacks.keys():
            robustness[attack_name] = {
                "robustness_ratio": results[attack_name]["accuracy"]
                / results["clean"]["accuracy"]
            }

        return results, robustness

    def _evaluate_clean(
        self, X: np.ndarray, y: np.ndarray, batch_size: int
    ) -> Dict[str, float]:
        """Evaluate model on clean data."""
        self.model.eval()
        correct, total = 0, 0
        n_batches = (len(X) + batch_size - 1) // batch_size
        with torch.no_grad():
            for i in tqdm(
                range(0, len(X), batch_size),
                total=n_batches,
                desc="Evaluating clean",
                leave=False,
            ):
                X_batch = torch.FloatTensor(X[i : i + batch_size]).to(self.device)
                y_batch = torch.LongTensor(y[i : i + batch_size]).to(self.device)
                outputs = self.model(X_batch)
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()
        return {"accuracy": correct / total if total > 0 else 0}

    def _evaluate_adversarial(
        self, X_adv: np.ndarray, y: np.ndarray, batch_size: int
    ) -> Dict[str, float]:
        """Evaluate model on adversarial data."""
        return self._evaluate_clean(X_adv, y, batch_size)
