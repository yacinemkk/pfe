"""
Adversarial Feature Dropout (AFD) for IoT Device Identification

During training, randomly zeros out entire features across ALL timesteps,
simulating the "Zero" adversarial strategy. This forces the model to learn
representations that do NOT depend on any single feature.

Without AFD: model relies heavily on "duration" → Zero attack kills accuracy
With AFD: model learns distributed representations → losing one feature is survivable

Reference:
    - Inspired by "Feature Denoising" and "Adversarial Logit Pairing"
    - Analogous to Dropout but at the FEATURE dimension (not neuron dimension)

Usage:
    afd = AdversarialFeatureDropout(p_single=0.3, p_double=0.15)
    X_dropped = afd(X, training=True)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List


class AdversarialFeatureDropout(nn.Module):
    """
    Adversarial Feature Dropout: zeros out entire features across all timesteps.

    Three modes:
    1. Single dropout (p_single): zero out 1 feature across the whole sequence
    2. Double dropout (p_double): zero out 2 features simultaneously
    3. Mimic replacement (p_mimic): replace feature with class mean (if stats provided)

    This directly simulates the adversarial attack strategies (Zero, Mimic_Mean)
    during training, making the model robust to them at test time.

    Args:
        n_features: Total number of input features
        p_single: Probability of dropping a single feature per sample
        p_double: Probability of dropping two features per sample
        p_mimic: Probability of replacing a feature with class mean
        non_modifiable_indices: Feature indices that should never be dropped
        benign_stats: Dict mapping class_idx -> {"mean": np.ndarray, "p95": np.ndarray}
        n_continuous_features: Number of continuous features (only drop these)
    """

    def __init__(
        self,
        n_features: int,
        p_single: float = 0.3,
        p_double: float = 0.15,
        p_mimic: float = 0.1,
        non_modifiable_indices: Optional[List[int]] = None,
        benign_stats: Optional[dict] = None,
        n_continuous_features: Optional[int] = None,
    ):
        super().__init__()
        self.n_features = n_features
        self.p_single = p_single
        self.p_double = p_double
        self.p_mimic = p_mimic
        self.non_modifiable_indices = set(non_modifiable_indices or [])
        self.n_continuous_features = n_continuous_features

        self._droppable_indices = self._compute_droppable_indices()

        self._benign_means_torch = None
        self._benign_stats_np = benign_stats
        if benign_stats is not None:
            self._prepare_torch_stats(benign_stats)

    def _compute_droppable_indices(self) -> List[int]:
        indices = []
        for i in range(self.n_features):
            if i in self.non_modifiable_indices:
                continue
            if (
                self.n_continuous_features is not None
                and i >= self.n_continuous_features
            ):
                continue
            indices.append(i)
        return indices

    def _prepare_torch_stats(self, benign_stats: dict):
        num_classes = max(benign_stats.keys()) + 1
        means = np.zeros((num_classes, self.n_features), dtype=np.float32)
        for cls, stats in benign_stats.items():
            means[cls] = stats["mean"].astype(np.float32)
        self._benign_means_np = means
        self._benign_means_torch = None
        self._device = None

    def _ensure_torch_stats(self, device: torch.device):
        if self._device == device and self._benign_means_torch is not None:
            return
        if self._benign_means_np is not None:
            self._benign_means_torch = torch.from_numpy(self._benign_means_np).to(
                device
            )
            self._device = device

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply adversarial feature dropout.

        Args:
            x: Input tensor (batch, seq_len, features) or (batch, features)
            y: Optional labels for mimic replacement (batch,)

        Returns:
            Tensor with features dropped/replaced
        """
        if not self.training:
            return x

        if len(self._droppable_indices) == 0:
            return x

        batch_size = x.size(0)
        device = x.device
        n_drop = len(self._droppable_indices)

        drop_idx = torch.tensor(self._droppable_indices, device=device)

        mask = torch.ones(batch_size, n_drop, device=device)

        for b in range(batch_size):
            r = torch.rand(1).item()

            if r < self.p_single + self.p_double + self.p_mimic:
                if r < self.p_double:
                    n_features_to_drop = 2
                elif r < self.p_single + self.p_double:
                    n_features_to_drop = 1
                else:
                    n_features_to_drop = 0

                if n_features_to_drop > 0:
                    perm = torch.randperm(n_drop)
                    for k in range(min(n_features_to_drop, n_drop)):
                        mask[b, perm[k]] = 0.0

                if (
                    r >= self.p_single + self.p_double
                    and self._benign_stats_np is not None
                    and y is not None
                ):
                    self._ensure_torch_stats(device)
                    cls = int(y[b].item())
                    if cls < self._benign_means_torch.size(0):
                        feat_to_mimic = torch.randint(0, n_drop, (1,)).item()
                        mask[b, feat_to_mimic] = 1.0
                        mimic_val = self._benign_means_torch[
                            cls, drop_idx[feat_to_mimic]
                        ]
                        if x.dim() == 3:
                            x[b, :, drop_idx[feat_to_mimic]] = mimic_val
                        else:
                            x[b, drop_idx[feat_to_mimic]] = mimic_val

        if x.dim() == 3:
            mask = mask.unsqueeze(1)
        x[..., drop_idx] = x[..., drop_idx] * mask

        return x


class GradientFeatureMasking(nn.Module):
    """
    Gradient Feature Masking: stops gradient flow through randomly selected features.

    Unlike AFD which zeros features, this stops gradient propagation, encouraging
    the model to not rely on any single feature for its gradient signal.

    This is a softer regularizer than AFD - the feature value is preserved
    but the model cannot learn to depend on it via gradient.
    """

    def __init__(
        self,
        n_features: int,
        p_mask: float = 0.2,
        non_modifiable_indices: Optional[List[int]] = None,
        n_continuous_features: Optional[int] = None,
    ):
        super().__init__()
        self.n_features = n_features
        self.p_mask = p_mask
        self.non_modifiable_indices = set(non_modifiable_indices or [])
        self.n_continuous_features = n_continuous_features
        self._maskable_indices = self._compute_maskable_indices()

    def _compute_maskable_indices(self) -> List[int]:
        indices = []
        for i in range(self.n_features):
            if i in self.non_modifiable_indices:
                continue
            if (
                self.n_continuous_features is not None
                and i >= self.n_continuous_features
            ):
                continue
            indices.append(i)
        return indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or len(self._maskable_indices) == 0:
            return x

        x_out = x.clone()
        batch_size = x.size(0)
        device = x.device
        n_mask = len(self._maskable_indices)

        mask_idx = torch.tensor(self._maskable_indices, device=device)

        for b in range(batch_size):
            if torch.rand(1).item() < self.p_mask:
                feat_to_mask = torch.randint(0, n_mask, (1,)).item()
                idx = mask_idx[feat_to_mask]
                if x.dim() == 3:
                    x_out[b, :, idx] = x_out[b, :, idx].detach()
                else:
                    x_out[b, idx] = x_out[b, idx].detach()

        return x_out


class FeatureImportanceRegularizer:
    """
    Regularizer that penalizes uneven feature importance.

    Computes per-feature gradient magnitude and adds a penalty
    proportional to the variance of feature importances.

    This prevents the model from depending too heavily on any
    single feature (like "duration" in the crash test).
    """

    def __init__(
        self,
        n_features: int,
        lambda_fim: float = 0.1,
        non_modifiable_indices: Optional[List[int]] = None,
        n_continuous_features: Optional[int] = None,
    ):
        self.n_features = n_features
        self.lambda_fim = lambda_fim
        self.non_modifiable_indices = set(non_modifiable_indices or [])
        self.n_continuous_features = n_continuous_features
        self._reg_indices = self._compute_reg_indices()

    def _compute_reg_indices(self) -> List[int]:
        indices = []
        for i in range(self.n_features):
            if i in self.non_modifiable_indices:
                continue
            if (
                self.n_continuous_features is not None
                and i >= self.n_continuous_features
            ):
                continue
            indices.append(i)
        return indices

    def compute_penalty(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        x_reg = x.clone().detach().requires_grad_(True)
        logits = model(x_reg)
        loss = logits.gather(1, y.unsqueeze(1)).sum()
        grad = torch.autograd.grad(loss, x_reg, create_graph=False)[0]

        if x.dim() == 3:
            feature_grads = grad[:, :, self._reg_indices].abs().mean(dim=(0, 1))
        else:
            feature_grads = grad[:, self._reg_indices].abs().mean(dim=0)

        mean_grad = feature_grads.mean()
        variance = ((feature_grads - mean_grad) ** 2).mean()

        return self.lambda_fim * variance


def create_adversarial_feature_dropout(
    n_features: int,
    p_single: float = 0.3,
    p_double: float = 0.15,
    p_mimic: float = 0.1,
    non_modifiable_indices: Optional[List[int]] = None,
    benign_stats: Optional[dict] = None,
    n_continuous_features: Optional[int] = None,
) -> AdversarialFeatureDropout:
    return AdversarialFeatureDropout(
        n_features=n_features,
        p_single=p_single,
        p_double=p_double,
        p_mimic=p_mimic,
        non_modifiable_indices=non_modifiable_indices,
        benign_stats=benign_stats,
        n_continuous_features=n_continuous_features,
    )
