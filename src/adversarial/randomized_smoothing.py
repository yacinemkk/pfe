"""
Randomized Smoothing for IoT Device Identification

At inference time, adds Gaussian noise to inputs and averages predictions
over multiple noisy copies. This provides certified robustness guarantees.

Algorithm:
    1. For input x, create N copies: {x + ε_1, x + ε_2, ..., x + ε_N}
       where ε_i ~ N(0, σ²I)
    2. Get predictions: {f(x + ε_1), f(x + ε_2), ..., f(x + ε_N)}
    3. Final prediction = argmax (vote majoritaire)

Certified radius:
    If the top class gets p_A votes and second class gets p_B votes,
    the prediction is certified within radius:
        r = σ/2 * (Φ⁻¹(p_A) - Φ⁻¹(p_B))
    where Φ⁻¹ is the inverse standard normal CDF.

Reference:
    - "Certified Robustness to Adversarial Examples via Randomized Smoothing"
      (Cohen et al., 2019)
    - https://arxiv.org/abs/1902.02918

Usage:
    smoother = RandomizedSmoothing(model, sigma=0.25, n_samples=50)
    prediction = smoother.predict(x)  # single sample
    predictions = smoother.predict_batch(X)  # batch
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict
from scipy.stats import norm


class RandomizedSmoothing:
    """
    Randomized Smoothing wrapper for robust inference.

    Wraps a trained classifier and provides:
    1. Robust predictions via majority voting over noisy samples
    2. Certified robustness radius per prediction
    3. Adjustable noise level and sample count

    Args:
        model: Trained classifier (nn.Module)
        device: torch device
        sigma: Noise standard deviation (higher = more robust but less accurate)
        n_samples: Number of noisy samples for prediction
        n_samples_certify: Number of samples for certification (usually larger)
        batch_size: Batch size for processing noisy samples
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        sigma: float = 0.25,
        n_samples: int = 50,
        n_samples_certify: int = 500,
        batch_size: int = 256,
    ):
        self.model = model
        self.device = device
        self.sigma = sigma
        self.n_samples = n_samples
        self.n_samples_certify = n_samples_certify
        self.batch_size = batch_size

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
    ) -> int:
        """
        Robust prediction for a single input.

        Args:
            x: Single input (seq_len, features) or (features,)

        Returns:
            Predicted class index
        """
        self.model.eval()

        if x.dim() == 2:
            x = x.unsqueeze(0)

        n_samples = self.n_samples
        x_repeated = x.repeat(n_samples, 1, 1)

        noise = torch.randn_like(x_repeated) * self.sigma
        x_noisy = x_repeated + noise

        all_preds = []
        for i in range(0, n_samples, self.batch_size):
            batch = x_noisy[i:i + self.batch_size].to(self.device)
            logits = self.model(batch)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())

        all_preds = torch.cat(all_preds)
        counts = torch.bincount(all_preds, minlength=self._get_num_classes())
        return counts.argmax().item()

    @torch.no_grad()
    def predict_batch(
        self,
        X: torch.Tensor,
        return_certified: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Robust predictions for a batch of inputs.

        Args:
            X: Input batch (batch, seq_len, features)
            return_certified: If True, also return certified radii

        Returns:
            predictions: (batch,) predicted class indices
            certified_radii: (batch,) certified radii (if return_certified=True)
        """
        self.model.eval()
        batch_size = X.size(0)
        n_samples = self.n_samples_certify if return_certified else self.n_samples

        all_predictions = torch.zeros(batch_size, dtype=torch.long)
        all_radii = torch.zeros(batch_size) if return_certified else None

        for idx in range(batch_size):
            x = X[idx:idx + 1]
            x_repeated = x.repeat(n_samples, 1, 1)
            noise = torch.randn_like(x_repeated) * self.sigma
            x_noisy = x_repeated + noise

            all_preds = []
            for i in range(0, n_samples, self.batch_size):
                batch = x_noisy[i:i + self.batch_size].to(self.device)
                logits = self.model(batch)
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())

            all_preds = torch.cat(all_preds)
            num_classes = max(self._get_num_classes(), all_preds.max().item() + 1)
            counts = torch.bincount(all_preds, minlength=num_classes)

            top2 = counts.topk(2)
            top_class = top2.indices[0].item()
            all_predictions[idx] = top_class

            if return_certified:
                p_A = top2.values[0].float() / n_samples
                p_B = top2.values[1].float() / n_samples

                if p_A > 0.5 and p_B < 0.5:
                    radius = (
                        self.sigma / 2 * (norm.ppf(p_A.item()) - norm.ppf(p_B.item()))
                    )
                    all_radii[idx] = max(0, radius)
                else:
                    all_radii[idx] = 0.0

        return all_predictions, all_radii

    def _get_num_classes(self) -> int:
        try:
            last_layer = None
            for name, param in self.model.named_parameters():
                if "weight" in name:
                    last_layer = param
            if last_layer is not None and last_layer.dim() >= 2:
                return last_layer.size(0)
        except Exception:
            pass
        return 20

    def evaluate_robust(
        self,
        X_clean: np.ndarray,
        y_clean: np.ndarray,
        X_adv: Optional[np.ndarray] = None,
        detailed: bool = False,
    ) -> Dict:
        """
        Evaluate robust accuracy with randomized smoothing.

        Args:
            X_clean: Clean test data (N, seq_len, features)
            y_clean: True labels (N,)
            X_adv: Optional adversarial data (same shape as X_clean)
            detailed: If True, return per-sample certified radii

        Returns:
            Dict with accuracy metrics and certified robustness statistics
        """
        self.model.eval()

        X_t = torch.FloatTensor(X_clean)
        y_t = torch.LongTensor(y_clean)

        predictions, radii = self.predict_batch(X_t, return_certified=True)

        clean_acc = (predictions == y_t).float().mean().item()
        mean_radius = radii.mean().item()
        median_radius = radii.median().item()
        pct_certified = (radii > 0).float().mean().item()

        results = {
            "clean_accuracy": clean_acc,
            "mean_certified_radius": mean_radius,
            "median_certified_radius": median_radius,
            "pct_certified": pct_certified,
            "sigma": self.sigma,
            "n_samples": self.n_samples,
            "n_samples_certify": self.n_samples_certify,
        }

        if X_adv is not None:
            X_adv_t = torch.FloatTensor(X_adv)
            adv_predictions, adv_radii = self.predict_batch(
                X_adv_t, return_certified=True
            )
            adv_acc = (adv_predictions == y_t).float().mean().item()
            results["adversarial_accuracy"] = adv_acc
            results["adversarial_mean_radius"] = adv_radii.mean().item()
            results["robustness_ratio"] = adv_acc / max(clean_acc, 1e-8)

        return results


class AdaptiveRandomizedSmoothing(RandomizedSmoothing):
    """
    Adaptive Randomized Smoothing: adjusts sigma per sample.

    Uses the model's prediction confidence to set sigma:
    - High confidence → lower sigma (less smoothing needed)
    - Low confidence → higher sigma (more smoothing for safety)

    This balances accuracy and robustness adaptively.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        sigma_min: float = 0.1,
        sigma_max: float = 0.5,
        n_samples: int = 50,
        batch_size: int = 256,
    ):
        super().__init__(
            model, device, sigma=sigma_min, n_samples=n_samples, batch_size=batch_size
        )
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @torch.no_grad()
    def _compute_adaptive_sigma(self, x: torch.Tensor) -> float:
        self.model.eval()
        logits = self.model(x.to(self.device))
        probs = torch.softmax(logits, dim=1)
        max_prob = probs.max().item()

        confidence = max_prob
        sigma = self.sigma_max - confidence * (self.sigma_max - self.sigma_min)
        return sigma

    @torch.no_grad()
    def predict_batch(
        self,
        X: torch.Tensor,
        return_certified: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self.model.eval()
        batch_size = X.size(0)
        n_samples = self.n_samples_certify if return_certified else self.n_samples

        all_predictions = torch.zeros(batch_size, dtype=torch.long)
        all_radii = torch.zeros(batch_size) if return_certified else None

        for idx in range(batch_size):
            x = X[idx:idx + 1]
            sigma = self._compute_adaptive_sigma(x)

            x_repeated = x.repeat(n_samples, 1, 1)
            noise = torch.randn_like(x_repeated) * sigma
            x_noisy = x_repeated + noise

            all_preds = []
            for i in range(0, n_samples, self.batch_size):
                batch = x_noisy[i:i + self.batch_size].to(self.device)
                logits = self.model(batch)
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())

            all_preds = torch.cat(all_preds)
            num_classes = max(self._get_num_classes(), all_preds.max().item() + 1)
            counts = torch.bincount(all_preds, minlength=num_classes)

            top2 = counts.topk(2)
            top_class = top2.indices[0].item()
            all_predictions[idx] = top_class

            if return_certified:
                p_A = top2.values[0].float() / n_samples
                p_B = top2.values[1].float() / n_samples

                if p_A > 0.5 and p_B < 0.5:
                    radius = sigma / 2 * (norm.ppf(p_A.item()) - norm.ppf(p_B.item()))
                    all_radii[idx] = max(0, radius)
                else:
                    all_radii[idx] = 0.0

        return all_predictions, all_radii


def create_randomized_smoothing(
    model: nn.Module,
    device: torch.device,
    sigma: float = 0.25,
    n_samples: int = 50,
    adaptive: bool = False,
) -> RandomizedSmoothing:
    """
    Factory function to create RandomizedSmoothing wrapper.
    """
    if adaptive:
        return AdaptiveRandomizedSmoothing(
            model=model,
            device=device,
            sigma_min=sigma * 0.4,
            sigma_max=sigma * 2.0,
            n_samples=n_samples,
        )
    return RandomizedSmoothing(
        model=model,
        device=device,
        sigma=sigma,
        n_samples=n_samples,
    )
