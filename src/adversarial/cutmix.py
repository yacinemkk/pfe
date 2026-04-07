"""
CutMix Adversarial for IoT Device Identification

Implements CutMix augmentation adapted for adversarial training:
1. Standard CutMix: Cut and paste regions between samples
2. Adversarial CutMix: Mix clean and adversarial examples
3. Sequence CutMix: Cut time steps from one sample to another

Reference:
    - "CutMix: Regularization Strategy to Boost Strong Supervised Learning"
    - "Interpolation Consistency Training for Semi-Supervised Learning"

Usage:
    cutmix = AdversarialCutMix(alpha=1.0, prob=0.5)
    X_mixed, y_mixed = cutmix(X_clean, X_adv, y)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class CutMix:
    """
    Standard CutMix augmentation.

    Randomly cuts a rectangular region from one sample and pastes it
    onto another sample.

    For sequence data (batch, seq_len, features), we cut along the
    sequence dimension.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        prob: float = 0.5,
    ):
        self.alpha = alpha
        self.prob = prob

    def get_cutmix_params(
        self,
        seq_len: int,
    ) -> Tuple[int, int, float]:
        """
        Generate CutMix parameters for sequence data.

        Returns:
            (cut_start, cut_end, lambda)
        """
        lam = np.random.beta(self.alpha, self.alpha)

        cut_len = int(seq_len * (1 - lam))
        cut_len = max(1, cut_len)

        cut_start = np.random.randint(0, seq_len - cut_len + 1)
        cut_end = cut_start + cut_len

        actual_lam = 1 - (cut_end - cut_start) / seq_len

        return cut_start, cut_end, actual_lam

    def __call__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply CutMix augmentation.

        Args:
            X: Input tensor (batch, seq_len, features)
            y: Labels (batch,)

        Returns:
            (X_mixed, y_mixed) - mixed samples and soft labels
        """
        if torch.rand(1).item() > self.prob:
            return X, y

        batch_size, seq_len, n_features = X.shape

        cut_start, cut_end, lam = self.get_cutmix_params(seq_len)

        index = torch.randperm(batch_size, device=X.device)

        X_mixed = X.clone()
        X_mixed[:, cut_start:cut_end, :] = X[index, cut_start:cut_end, :]

        num_classes = y.max().item() + 1
        y_one_hot = torch.zeros(batch_size, num_classes, device=X.device)
        y_one_hot.scatter_(1, y.unsqueeze(1), 1)

        y_mixed = lam * y_one_hot + (1 - lam) * y_one_hot[index]

        return X_mixed, y_mixed


class AdversarialCutMix:
    """
    CutMix between clean and adversarial examples.

    Instead of mixing between different samples, we mix between
    clean and adversarial versions of the SAME sample:

        X_mixed[:, cut_start:cut_end, :] = X_adv[:, cut_start:cut_end, :]

    This forces the model to be robust even when part of the input
    is adversarial.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        prob: float = 0.5,
        adv_only_prob: float = 0.3,
    ):
        self.alpha = alpha
        self.prob = prob
        self.adv_only_prob = adv_only_prob

    def __call__(
        self,
        X_clean: torch.Tensor,
        X_adv: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply adversarial CutMix.

        Args:
            X_clean: Clean input tensor (batch, seq_len, features)
            X_adv: Adversarial input tensor (batch, seq_len, features)
            y: True labels (batch,)

        Returns:
            (X_mixed, y) - mixed samples and original labels
        """
        if torch.rand(1).item() > self.prob:
            return X_clean, y

        batch_size, seq_len, n_features = X_clean.shape

        lam = np.random.beta(self.alpha, self.alpha)

        if torch.rand(1).item() < self.adv_only_prob:
            X_mixed = X_adv.clone()
        else:
            cut_len = int(seq_len * (1 - lam))
            cut_len = max(1, cut_len)

            cut_start = np.random.randint(0, seq_len - cut_len + 1)
            cut_end = cut_start + cut_len

            X_mixed = X_clean.clone()
            X_mixed[:, cut_start:cut_end, :] = X_adv[:, cut_start:cut_end, :]

        return X_mixed, y


class FeatureCutMix:
    """
    CutMix along the feature dimension instead of sequence.

    Useful when different features have different importance levels.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        prob: float = 0.5,
        n_continuous_features: Optional[int] = None,
    ):
        self.alpha = alpha
        self.prob = prob
        self.n_continuous_features = n_continuous_features

    def __call__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply feature-level CutMix.

        Args:
            X: Input tensor (batch, seq_len, features)
            y: Labels (batch,)

        Returns:
            (X_mixed, y_mixed)
        """
        if torch.rand(1).item() > self.prob:
            return X, y

        batch_size = X.size(0)
        n_features = X.size(-1)

        if self.n_continuous_features is not None:
            n_features_to_cut = self.n_continuous_features
        else:
            n_features_to_cut = n_features

        lam = np.random.beta(self.alpha, self.alpha)

        n_cut = int(n_features_to_cut * (1 - lam))
        n_cut = max(1, n_cut)

        cut_start = np.random.randint(0, n_features_to_cut - n_cut + 1)
        cut_end = cut_start + n_cut

        index = torch.randperm(batch_size, device=X.device)

        X_mixed = X.clone()
        X_mixed[:, :, cut_start:cut_end] = X[index, :, cut_start:cut_end]

        num_classes = y.max().item() + 1
        y_one_hot = torch.zeros(batch_size, num_classes, device=X.device)
        y_one_hot.scatter_(1, y.unsqueeze(1), 1)

        y_mixed = lam * y_one_hot + (1 - lam) * y_one_hot[index]

        return X_mixed, y_mixed


class HybridCutMix:
    """
    Combines sequence and feature CutMix for maximum robustness.

    Randomly applies either:
    1. Sequence CutMix (cut time steps)
    2. Feature CutMix (cut features)
    3. Both
    """

    def __init__(
        self,
        alpha_seq: float = 1.0,
        alpha_feat: float = 1.0,
        prob: float = 0.5,
        n_continuous_features: Optional[int] = None,
    ):
        self.seq_cutmix = CutMix(alpha=alpha_seq, prob=1.0)
        self.feat_cutmix = FeatureCutMix(
            alpha=alpha_feat,
            prob=1.0,
            n_continuous_features=n_continuous_features,
        )
        self.prob = prob

    def __call__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply hybrid CutMix.

        Args:
            X: Input tensor (batch, seq_len, features)
            y: Labels (batch,)

        Returns:
            (X_mixed, y_mixed)
        """
        if torch.rand(1).item() > self.prob:
            return X, y

        choice = np.random.randint(0, 3)

        if choice == 0:
            return self.seq_cutmix(X, y)
        elif choice == 1:
            return self.feat_cutmix(X, y)
        else:
            X_mixed, y_mixed = self.seq_cutmix(X, y)
            return self.feat_cutmix(
                X_mixed, y.argmax(dim=-1) if y_mixed.dim() > 1 else y_mixed
            )


def create_adversarial_cutmix(
    alpha: float = 1.0,
    prob: float = 0.5,
) -> AdversarialCutMix:
    """
    Factory function to create AdversarialCutMix with standard settings.
    """
    return AdversarialCutMix(
        alpha=alpha,
        prob=prob,
        adv_only_prob=0.3,
    )
