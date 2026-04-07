"""
Input Transformation for Adversarial Robustness

Applies transformations to input data during training to improve robustness:
1. Gaussian Noise: Add random noise to continuous features
2. Feature Dropout: Randomly mask features
3. Quantization: Reduce precision of continuous features
4. Mixup: Linear interpolation between samples

Reference:
    - "Benchmarking Neural Network Robustness to Common Perturbations"
    - "Adversarial Examples Are a Natural Consequence of Test Error in Noise"

Usage:
    transform = InputTransform(noise_std=0.02, dropout_prob=0.1)
    X_transformed = transform(X, training=True)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class InputTransform:
    """
    Input transformation for adversarial robustness training.

    Applies multiple transformations to improve model robustness:
    - Gaussian noise injection
    - Random feature dropout
    - Quantization noise
    """

    def __init__(
        self,
        noise_std: float = 0.02,
        dropout_prob: float = 0.1,
        quantization_bits: int = 8,
        n_continuous_features: Optional[int] = None,
        apply_noise: bool = True,
        apply_dropout: bool = True,
        apply_quantization: bool = False,
    ):
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.quantization_bits = quantization_bits
        self.n_continuous_features = n_continuous_features
        self.apply_noise = apply_noise
        self.apply_dropout = apply_dropout
        self.apply_quantization = apply_quantization

    def add_gaussian_noise(
        self,
        X: torch.Tensor,
        std: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Add Gaussian noise to continuous features.

        Args:
            X: Input tensor (batch, seq_len, features) or (batch, features)
            std: Noise standard deviation (uses self.noise_std if None)

        Returns:
            Noisy tensor
        """
        if std is None:
            std = self.noise_std

        noise = torch.randn_like(X) * std

        if self.n_continuous_features is not None:
            if X.dim() == 3:
                noise[:, :, self.n_continuous_features :] = 0
            else:
                noise[:, self.n_continuous_features :] = 0

        return X + noise

    def feature_dropout(
        self,
        X: torch.Tensor,
        prob: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Randomly drop (zero out) features.

        Args:
            X: Input tensor
            prob: Dropout probability (uses self.dropout_prob if None)

        Returns:
            Tensor with randomly dropped features
        """
        if prob is None:
            prob = self.dropout_prob

        if self.n_continuous_features is not None:
            if X.dim() == 3:
                mask = torch.rand(
                    X.size(0), 1, self.n_continuous_features, device=X.device
                )
            else:
                mask = torch.rand(
                    X.size(0), self.n_continuous_features, device=X.device
                )
            mask = (mask > prob).float()

            X_dropped = X.clone()
            if X.dim() == 3:
                X_dropped[:, :, : self.n_continuous_features] *= mask
            else:
                X_dropped[:, : self.n_continuous_features] *= mask
            return X_dropped
        else:
            mask = (torch.rand_like(X) > prob).float()
            return X * mask

    def quantize(
        self,
        X: torch.Tensor,
        bits: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Quantize continuous features to reduce precision.

        Args:
            X: Input tensor
            bits: Number of bits for quantization (uses self.quantization_bits if None)

        Returns:
            Quantized tensor
        """
        if bits is None:
            bits = self.quantization_bits

        levels = 2**bits - 1

        X_min = X.min()
        X_max = X.max()

        X_scaled = (X - X_min) / (X_max - X_min + 1e-8)
        X_quantized = torch.round(X_scaled * levels) / levels
        X_restored = X_quantized * (X_max - X_min) + X_min

        return X_restored

    def __call__(
        self,
        X: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """
        Apply all enabled transformations.

        Args:
            X: Input tensor
            training: If True, apply transformations; if False, return unchanged

        Returns:
            Transformed tensor
        """
        if not training:
            return X

        X_transformed = X.clone()

        if self.apply_noise:
            X_transformed = self.add_gaussian_noise(X_transformed)

        if self.apply_dropout:
            X_transformed = self.feature_dropout(X_transformed)

        if self.apply_quantization:
            X_transformed = self.quantize(X_transformed)

        return X_transformed


class MixupTransform:
    """
    Mixup augmentation for adversarial robustness.

    Creates virtual training examples through linear interpolation:
        X_mixed = λ * X_i + (1 - λ) * X_j
        y_mixed = λ * y_i + (1 - λ) * y_j

    Reference:
        "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)
    """

    def __init__(
        self,
        alpha: float = 0.4,
        prob: float = 0.5,
    ):
        self.alpha = alpha
        self.prob = prob

    def __call__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mixup augmentation.

        Args:
            X: Input tensor (batch, ...)
            y: Labels (batch,)

        Returns:
            (X_mixed, y_mixed) - mixed samples and soft labels
        """
        if torch.rand(1).item() > self.prob:
            return X, y

        batch_size = X.size(0)

        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1 - lam)

        index = torch.randperm(batch_size, device=X.device)

        X_mixed = lam * X + (1 - lam) * X[index]

        y_one_hot = torch.zeros(batch_size, y.max().item() + 1, device=X.device)
        y_one_hot.scatter_(1, y.unsqueeze(1), 1)

        y_mixed = lam * y_one_hot + (1 - lam) * y_one_hot[index]

        return X_mixed, y_mixed


class AdversarialMixup:
    """
    Mixup between clean and adversarial examples.

    Creates:
        X_mixed = λ * X_clean + (1 - λ) * X_adv
        y_mixed = y (true label)

    This encourages the model to learn robust features that are
    consistent across the clean-to-adversarial spectrum.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        prob: float = 0.5,
    ):
        self.alpha = alpha
        self.prob = prob

    def __call__(
        self,
        X_clean: torch.Tensor,
        X_adv: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply adversarial mixup.

        Args:
            X_clean: Clean input tensor
            X_adv: Adversarial input tensor
            y: True labels

        Returns:
            (X_mixed, y) - mixed samples and labels
        """
        if torch.rand(1).item() > self.prob:
            return X_clean, y

        lam = np.random.beta(self.alpha, self.alpha)

        X_mixed = lam * X_clean + (1 - lam) * X_adv

        return X_mixed, y


def create_input_transform(
    noise_std: float = 0.02,
    dropout_prob: float = 0.1,
    n_continuous_features: Optional[int] = None,
) -> InputTransform:
    """
    Factory function to create InputTransform with standard settings.
    """
    return InputTransform(
        noise_std=noise_std,
        dropout_prob=dropout_prob,
        n_continuous_features=n_continuous_features,
        apply_noise=True,
        apply_dropout=True,
        apply_quantization=False,
    )
