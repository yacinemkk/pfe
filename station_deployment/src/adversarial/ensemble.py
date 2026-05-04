"""
Heterogeneous Ensemble for IoT Device Identification

Combines multiple model architectures to provide diverse robustness.
An attacker must fool ALL models simultaneously, which is significantly
harder than fooling a single model.

Supported architectures:
    - LSTM (sequential patterns)
    - BiLSTM (bidirectional context)
    - CNN-LSTM (local + temporal patterns)
    - Transformer (global attention)

Voting strategies:
    - Hard voting: majority class vote
    - Soft voting: average probabilities, argmax
    - Weighted voting: weighted average by validation accuracy

Reference:
    - "Ensemble Adversarial Training: Attacks and Defenses"
    - "Improving Robustness Using Ensemble Learning"

Usage:
    ensemble = HeterogeneousEnsemble(device)
    ensemble.add_model('lstm', model_lstm, weight=1.0)
    ensemble.add_model('cnn_lstm', model_cnn_lstm, weight=1.0)
    preds = ensemble.predict(X)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score


class HeterogeneousEnsemble:
    """
    Heterogeneous ensemble of diverse model architectures.

    Combines models with different inductive biases to improve
    robustness against adversarial attacks.

    Args:
        device: torch device
        voting: Voting strategy ('hard', 'soft', 'weighted')
    """

    def __init__(
        self,
        device: torch.device,
        voting: str = "soft",
    ):
        self.device = device
        self.voting = voting
        self.models: Dict[str, nn.Module] = {}
        self.weights: Dict[str, float] = {}
        self.val_accuracies: Dict[str, float] = {}

    def add_model(
        self,
        name: str,
        model: nn.Module,
        weight: float = 1.0,
        val_accuracy: Optional[float] = None,
    ):
        """
        Add a model to the ensemble.

        Args:
            name: Model identifier (e.g., 'lstm', 'cnn_lstm')
            model: Trained model
            weight: Voting weight (used for weighted voting)
            val_accuracy: Validation accuracy (used to auto-compute weights)
        """
        self.models[name] = model.to(self.device)
        self.weights[name] = weight
        if val_accuracy is not None:
            self.val_accuracies[name] = val_accuracy

        if self.voting == "weighted" and self.val_accuracies:
            total_acc = sum(self.val_accuracies.values())
            for n in self.models:
                if n in self.val_accuracies:
                    self.weights[n] = self.val_accuracies[n] / total_acc

    def remove_model(self, name: str):
        if name in self.models:
            del self.models[name]
            del self.weights[name]
            if name in self.val_accuracies:
                del self.val_accuracies[name]

    @torch.no_grad()
    def predict_proba(
        self,
        X: torch.Tensor,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """
        Get ensemble probability predictions.

        Args:
            X: Input tensor (N, seq_len, features)
            batch_size: Batch size for inference

        Returns:
            Probabilities (N, num_classes)
        """
        all_probs = []
        weight_sum = 0.0

        for name, model in self.models.items():
            model.eval()
            weight = self.weights.get(name, 1.0)
            weight_sum += weight

            model_probs = []
            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size].to(self.device)
                logits = model(batch)
                probs = torch.softmax(logits, dim=1)
                model_probs.append(probs.cpu())

            model_probs = torch.cat(model_probs, dim=0)
            all_probs.append((model_probs, weight))

        if self.voting == "hard":
            num_classes = all_probs[0][0].size(1)
            N = X.size(0)
            vote_counts = torch.zeros(N, num_classes)

            for model_probs, weight in all_probs:
                preds = model_probs.argmax(dim=1)
                for i in range(N):
                    vote_counts[i, preds[i]] += weight

            vote_counts = vote_counts / vote_counts.sum(dim=1, keepdim=True)
            return vote_counts

        else:
            ensemble_probs = torch.zeros_like(all_probs[0][0])
            for model_probs, weight in all_probs:
                ensemble_probs += weight * model_probs
            ensemble_probs /= weight_sum
            return ensemble_probs

    def predict(
        self,
        X: torch.Tensor,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """
        Get ensemble class predictions.

        Returns:
            Predictions (N,)
        """
        probs = self.predict_proba(X, batch_size)
        return probs.argmax(dim=1)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 64,
    ) -> Dict:
        """
        Evaluate ensemble on data.

        Args:
            X: Input data (N, seq_len, features) numpy array
            y: True labels (N,) numpy array
            batch_size: Batch size

        Returns:
            Dict with accuracy, macro_f1, per_model_accuracies
        """
        X_t = torch.FloatTensor(X)
        y_t = torch.LongTensor(y)

        preds = self.predict(X_t, batch_size).numpy()
        y_np = y_t.numpy()

        acc = accuracy_score(y_np, preds)
        f1 = f1_score(y_np, preds, average="macro", zero_division=0)

        per_model = {}
        for name, model in self.models.items():
            model.eval()
            model_preds = []
            with torch.no_grad():
                for i in range(0, len(X_t), batch_size):
                    batch = X_t[i:i + batch_size].to(self.device)
                    logits = model(batch)
                    p = logits.argmax(dim=1).cpu().numpy()
                    model_preds.append(p)
            model_preds = np.concatenate(model_preds)
            model_acc = accuracy_score(y_np, model_preds)
            model_f1 = f1_score(y_np, model_preds, average="macro", zero_division=0)
            per_model[name] = {"accuracy": model_acc, "macro_f1": model_f1}

        diversity = self._compute_diversity(X_t, y_t, batch_size)

        return {
            "ensemble_accuracy": acc,
            "ensemble_macro_f1": f1,
            "per_model": per_model,
            "diversity": diversity,
            "n_models": len(self.models),
            "voting": self.voting,
        }

    def _compute_diversity(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = 64,
    ) -> Dict:
        """
        Compute ensemble diversity metrics.

        Diversity is crucial: if all models make the same mistakes,
        the ensemble provides no robustness benefit.
        """
        if len(self.models) < 2:
            return {"pairwise_disagreement": 0.0}

        all_preds = {}
        for name, model in self.models.items():
            model.eval()
            preds = []
            with torch.no_grad():
                for i in range(0, len(X), batch_size):
                    batch = X[i:i + batch_size].to(self.device)
                    logits = model(batch)
                    p = logits.argmax(dim=1).cpu().numpy()
                    preds.append(p)
            all_preds[name] = np.concatenate(preds)

        names = list(all_preds.keys())
        disagreements = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                dis = np.mean(all_preds[names[i]] != all_preds[names[j]])
                disagreements.append(dis)

        return {
            "pairwise_disagreement": np.mean(disagreements) if disagreements else 0.0,
            "min_disagreement": np.min(disagreements) if disagreements else 0.0,
            "max_disagreement": np.max(disagreements) if disagreements else 0.0,
        }

    def crash_test(
        self,
        X_clean: np.ndarray,
        y_clean: np.ndarray,
        X_adv: np.ndarray,
        y_adv: np.ndarray,
        batch_size: int = 64,
    ) -> Dict:
        """
        Run crash test on the ensemble.

        Same 3-test protocol as CrashTestEvaluator:
        1. Clean benign data
        2. Adversarial data only
        3. Mixed (benign + adversarial)
        """
        results = {}

        X_clean_t = torch.FloatTensor(X_clean)
        X_adv_t = torch.FloatTensor(X_adv)

        clean_preds = self.predict(X_clean_t, batch_size).numpy()
        test1_acc = accuracy_score(y_clean, clean_preds)
        test1_f1 = f1_score(y_clean, clean_preds, average="macro", zero_division=0)
        results["test1_clean"] = {"accuracy": test1_acc, "macro_f1": test1_f1}

        adv_preds = self.predict(X_adv_t, batch_size).numpy()
        test2_acc = accuracy_score(y_adv, adv_preds)
        test2_f1 = f1_score(y_adv, adv_preds, average="macro", zero_division=0)
        results["test2_adversarial"] = {"accuracy": test2_acc, "macro_f1": test2_f1}

        X_mix = np.concatenate([X_clean, X_adv])
        y_mix = np.concatenate([y_clean, y_adv])
        X_mix_t = torch.FloatTensor(X_mix)
        mix_preds = self.predict(X_mix_t, batch_size).numpy()
        test3_acc = accuracy_score(y_mix, mix_preds)
        test3_f1 = f1_score(y_mix, mix_preds, average="macro", zero_division=0)
        results["test3_mixed"] = {"accuracy": test3_acc, "macro_f1": test3_f1}

        f1_drop = test1_f1 - test2_f1
        rr = test2_f1 / max(test1_f1, 1e-8)
        results["f1_drop"] = f1_drop
        results["robustness_ratio"] = rr

        return results

    def state_dict(self) -> Dict:
        return {name: model.state_dict() for name, model in self.models.items()}

    def load_state_dict(self, state_dicts: Dict[str, dict]):
        for name, sd in state_dicts.items():
            if name in self.models:
                self.models[name].load_state_dict(sd)


def create_heterogeneous_ensemble(
    input_size: int,
    num_classes: int,
    device: torch.device,
    architectures: Optional[List[str]] = None,
    voting: str = "soft",
) -> HeterogeneousEnsemble:
    """
    Factory function to create a heterogeneous ensemble.

    Args:
        input_size: Number of input features
        num_classes: Number of output classes
        device: torch device
        architectures: List of architecture names (default: ['lstm', 'bilstm', 'cnn_lstm'])
        voting: Voting strategy

    Returns:
        Untrained HeterogeneousEnsemble with models added
    """
    from src.models.lstm import LSTMClassifier
    from src.models.bilstm import BiLSTMClassifier
    from src.models.cnn_lstm import CNNLSTMClassifier

    if architectures is None:
        architectures = ["lstm", "bilstm", "cnn_lstm"]

    ensemble = HeterogeneousEnsemble(device=device, voting=voting)

    for arch in architectures:
        if arch == "lstm":
            model = LSTMClassifier(
                input_size=input_size,
                num_classes=num_classes,
                config_path="",
            )
            ensemble.add_model("lstm", model, weight=1.0)
        elif arch == "bilstm":
            model = BiLSTMClassifier(
                input_size=input_size,
                num_classes=num_classes,
                config_path="",
            )
            ensemble.add_model("bilstm", model, weight=1.0)
        elif arch == "cnn_lstm":
            model = CNNLSTMClassifier(
                input_size=input_size,
                num_classes=num_classes,
                config_path="",
            )
            ensemble.add_model("cnn_lstm", model, weight=1.0)

    return ensemble
