"""
Adversarial Attacks for IoT Device Identification

Implements:
1. Feature-level attack (IoT-SDN style)
2. Sequence-level gradient-based attack (PGD/BPTT)
3. Combined hybrid attack
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import gc


class FeatureLevelAttack:
    """
    Feature-level adversarial attack (IoT-SDN style).
    Targets statistical features while respecting semantic constraints.

    x_adv = Projection[x0 + c·t·mask·sign(μ_target - x0)·|Δ|]
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        num_classes: int,
        non_modifiable: List[str] = None,
        dependent_pairs: Dict[str, str] = None,
    ):
        self.feature_names = feature_names
        self.num_classes = num_classes

        self.non_modifiable = non_modifiable or [
            "ipProto",
            "http",
            "https",
            "dns",
            "ntp",
            "tcp",
            "udp",
            "ssdp",
        ]

        self.dependent_pairs = dependent_pairs or {
            "inPacketCount": "outPacketCount",
            "inByteCount": "outByteCount",
            "inAvgIAT": "outAvgIAT",
            "inAvgPacketSize": "outAvgPacketSize",
        }

        self.modifiable_indices = self._get_modifiable_indices()
        self.dependent_indices = self._get_dependent_indices()
        self.class_centroids = self._compute_centroids(X_train, y_train)
        self.nearest_classes = self._find_nearest_classes(k=3)
        self.masks = self._generate_masks(X_train, n_masks=15)

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

    def _compute_centroids(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Dict[int, np.ndarray]:
        centroids = {}
        for cls in range(self.num_classes):
            mask = y_train == cls
            if np.sum(mask) > 0:
                centroids[cls] = np.mean(X_train[mask], axis=0)
        return centroids

    def _find_nearest_classes(self, k: int = 3) -> Dict[int, List[int]]:
        nearest = {}
        class_ids = list(self.class_centroids.keys())

        if len(class_ids) < 2:
            return {c: [] for c in class_ids}

        centroids_matrix = np.array([self.class_centroids[c] for c in class_ids])

        for i, cls in enumerate(class_ids):
            centroid = self.class_centroids[cls]
            distances = np.sqrt(np.sum((centroids_matrix - centroid) ** 2, axis=1))
            distances[i] = np.inf
            nearest_indices = np.argsort(distances)[:k]
            nearest[cls] = [class_ids[j] for j in nearest_indices]

        return nearest

    def _generate_masks(self, X_train: np.ndarray, n_masks: int = 15) -> np.ndarray:
        n_features = len(self.feature_names)
        masks = []

        masks.append(np.ones(n_features))

        n_modifiable = len(self.modifiable_indices)
        for pct in [0.25, 0.5, 0.75]:
            n_active = max(1, int(n_modifiable * pct))
            for _ in range(2):
                mask = np.zeros(n_features)
                active = np.random.choice(
                    self.modifiable_indices, n_active, replace=False
                )
                mask[active] = 1
                masks.append(mask)

        if len(X_train) > 0:
            variances = np.var(X_train, axis=0)
            for top_k in [5, 10, 15]:
                mask = np.zeros(n_features)
                top_indices = np.argsort(variances)[-top_k:]
                mask[top_indices] = 1
                masks.append(mask)

        return np.array(masks[:n_masks])

    def projection(self, X: np.ndarray) -> np.ndarray:
        X_proj = X.copy()
        X_proj = np.clip(X_proj, -3.0, 3.0)

        for indep_idx, dep_idx in self.dependent_indices:
            if X_proj.ndim == 1:
                ratio = np.abs(X_proj[dep_idx]) / (np.abs(X_proj[indep_idx]) + 1e-8)
                X_proj[dep_idx] = X_proj[indep_idx] * np.clip(ratio, 0.5, 2.0)
            else:
                ratio = np.abs(X_proj[:, dep_idx]) / (
                    np.abs(X_proj[:, indep_idx]) + 1e-8
                )
                X_proj[:, dep_idx] = X_proj[:, indep_idx] * np.clip(ratio, 0.5, 2.0)

        return X_proj

    def generate_single(
        self,
        x0: np.ndarray,
        true_class: int,
        target_class: Optional[int] = None,
        max_iter: int = 20,
        c: float = 0.1,
    ) -> np.ndarray:
        if target_class is None:
            if (
                true_class in self.nearest_classes
                and len(self.nearest_classes[true_class]) > 0
            ):
                target_class = np.random.choice(self.nearest_classes[true_class])
            else:
                other = [c for c in range(self.num_classes) if c != true_class]
                if not other:
                    return x0
                target_class = np.random.choice(other)

        if target_class not in self.class_centroids:
            return x0

        mu_target = self.class_centroids[target_class]
        mask = self.masks[np.random.randint(len(self.masks))]

        x_adv = x0.copy()
        for t in range(1, max_iter + 1):
            diff = np.abs(mu_target - x0)
            direction = np.sign(mu_target - x0)
            perturbation = c * t * mask * direction * diff
            x_adv = self.projection(x0 + perturbation)

        return x_adv

    def generate_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_iter: int = 20,
        c: float = 0.1,
        verbose: bool = False,
    ) -> np.ndarray:
        X_adv = np.zeros_like(X)

        iterator = range(len(X))
        if verbose:
            iterator = tqdm(iterator, desc="Feature-level attack")

        for i in iterator:
            X_adv[i] = self.generate_single(X[i], y[i], max_iter=max_iter, c=c)

        return X_adv


class SequenceLevelAttack:
    """
    Sequence-level adversarial attack using gradient-based methods.
    Uses Backpropagation Through Time (BPTT) to compute gradients w.r.t. input sequences.

    Implements PGD (Projected Gradient Descent) adapted for sequences.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        epsilon: float = 0.1,
        alpha: float = 0.01,
        num_steps: int = 10,
        clip_min: float = -3.0,
        clip_max: float = 3.0,
        feature_mask: Optional[np.ndarray] = None,
        preserve_positions: bool = True,
    ):
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.feature_mask = feature_mask
        self.preserve_positions = preserve_positions
        self._original_training_state = None

    def _enable_grad_mode(self):
        self._original_training_state = self.model.training
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()

    def _restore_mode(self):
        if self._original_training_state is not None:
            if self._original_training_state:
                self.model.train()
            else:
                self.model.eval()
            for module in self.model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    module.eval()

    def _apply_feature_mask(self, grad: torch.Tensor) -> torch.Tensor:
        if self.feature_mask is not None:
            mask = torch.FloatTensor(self.feature_mask).to(self.device)
            mask = mask.view(1, 1, -1)
            grad = grad * mask
        return grad

    def _preserve_temporal_structure(
        self, x_adv: torch.Tensor, x_orig: torch.Tensor
    ) -> torch.Tensor:
        if not self.preserve_positions:
            return x_adv

        temporal_std = torch.std(x_orig, dim=2, keepdim=True)
        perturbation = x_adv - x_orig
        max_perturbation = 0.5 * temporal_std
        perturbation = torch.clamp(perturbation, -max_perturbation, max_perturbation)

        return x_orig + perturbation

    def pgd_attack(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        target_class: Optional[int] = None,
    ) -> torch.Tensor:
        self._enable_grad_mode()

        try:
            X = X.to(self.device)
            y = y.to(self.device)

            X_orig = X.clone().detach()

            target = None
            if targeted and target_class is not None:
                target = torch.full_like(y, target_class)

            X_adv = X.clone().detach()
            X_adv.requires_grad = True

            for step in range(self.num_steps):
                outputs = self.model(X_adv)

                if targeted and target_class is not None and target is not None:
                    loss = nn.CrossEntropyLoss()(outputs, target)
                    loss = -loss
                else:
                    loss = nn.CrossEntropyLoss()(outputs, y)

                loss.backward()

                grad = X_adv.grad.data
                grad = self._apply_feature_mask(grad)

                X_adv = X_adv + self.alpha * grad.sign()

                eta = torch.clamp(X_adv - X_orig, -self.epsilon, self.epsilon)
                X_adv = X_orig + eta
                X_adv = torch.clamp(X_adv, self.clip_min, self.clip_max)

                X_adv = self._preserve_temporal_structure(X_adv, X_orig)

                X_adv = X_adv.detach()
                X_adv.requires_grad = True

            return X_adv.detach()
        finally:
            self._restore_mode()

    def fgsm_attack(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        target_class: Optional[int] = None,
    ) -> torch.Tensor:
        self._enable_grad_mode()

        try:
            X = X.to(self.device)
            y = y.to(self.device)

            X_adv = X.clone().detach()
            X_adv.requires_grad = True

            outputs = self.model(X_adv)

            if targeted and target_class is not None:
                target = torch.full_like(y, target_class)
                loss = -nn.CrossEntropyLoss()(outputs, target)
            else:
                loss = nn.CrossEntropyLoss()(outputs, y)

            loss.backward()

            grad = X_adv.grad.data
            grad = self._apply_feature_mask(grad)

            X_adv = X_adv + self.epsilon * grad.sign()
            X_adv = torch.clamp(X_adv, self.clip_min, self.clip_max)

            return X_adv.detach()
        finally:
            self._restore_mode()

    def generate_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 64,
        method: str = "pgd",
        verbose: bool = False,
    ) -> np.ndarray:
        X_adv = []

        n_batches = (len(X) + batch_size - 1) // batch_size
        iterator = range(n_batches)
        if verbose:
            iterator = tqdm(iterator, desc="Sequence-level attack")

        attack_fn = self.pgd_attack if method == "pgd" else self.fgsm_attack

        for i in iterator:
            start = i * batch_size
            end = min(start + batch_size, len(X))

            X_batch = torch.FloatTensor(X[start:end])
            y_batch = torch.LongTensor(y[start:end])

            X_batch_adv = attack_fn(X_batch, y_batch)
            X_adv.append(X_batch_adv.cpu().numpy())

        return np.vstack(X_adv)


class HybridAdversarialAttack:
    """
    Combines feature-level and sequence-level attacks.

    Phase 1: Feature-level attack (targets statistical features)
    Phase 2: Sequence-level attack (targets temporal ordering)
    """

    def __init__(
        self,
        feature_attack: FeatureLevelAttack,
        sequence_attack: SequenceLevelAttack,
        feature_names: List[str],
        combine_ratio: float = 0.5,
    ):
        self.feature_attack = feature_attack
        self.sequence_attack = sequence_attack
        self.feature_names = feature_names
        self.combine_ratio = combine_ratio

    def generate_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = "hybrid",
        batch_size: int = 64,
        verbose: bool = False,
    ) -> np.ndarray:
        if method == "feature":
            return self.feature_attack.generate_batch(X, y, verbose=verbose)

        elif method == "sequence":
            return self.sequence_attack.generate_batch(
                X, y, batch_size=batch_size, verbose=verbose
            )

        elif method == "hybrid":
            n = len(X)
            n_feature = int(n * self.combine_ratio)

            idx = np.random.permutation(n)
            idx_feature = idx[:n_feature]
            idx_sequence = idx[n_feature:]

            X_adv = np.zeros_like(X)

            if len(idx_feature) > 0:
                X_adv[idx_feature] = self.feature_attack.generate_batch(
                    X[idx_feature], y[idx_feature], verbose=verbose
                )

            if len(idx_sequence) > 0:
                X_adv[idx_sequence] = self.sequence_attack.generate_batch(
                    X[idx_sequence],
                    y[idx_sequence],
                    batch_size=batch_size,
                    verbose=verbose,
                )

            return X_adv

        else:
            raise ValueError(f"Unknown method: {method}")


class AdversarialEvaluator:
    """Evaluates model robustness against various attacks."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attacks: Dict[str, np.ndarray],
        batch_size: int = 64,
    ) -> Dict[str, Dict[str, float]]:
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
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
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
        return self._evaluate_clean(X_adv, y, batch_size)
