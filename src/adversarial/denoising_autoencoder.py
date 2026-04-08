"""
Feature Denoising Autoencoder for IoT Device Identification

Lightweight autoencoder that reconstructs clean features from perturbed inputs.
Trained jointly with the classifier: the classifier operates on reconstructed
(denoised) features instead of raw (potentially adversarial) inputs.

Architecture:
    Encoder: input_size -> 32 -> latent_dim (16)
    Decoder: latent_dim -> 32 -> input_size

Training:
    1. Perturb input features (Zero, Mimic, Padding, Noise)
    2. Autoencoder reconstructs clean features from perturbed
    3. Classifier predicts on reconstructed features
    4. Loss = MSE(reconstructed, clean) + CE(classifier(reconstructed), y)

This is a defense preprocessing layer: at inference time, adversarial
perturbations are attenuated before reaching the classifier.

Reference:
    - "Defending Against Adversarial Attacks Using Random Forests"
    - "Feature Squeezing: Detecting Adversarial Examples in DNNs"

Usage:
    dae = DenoisingAutoencoder(input_size=16, latent_dim=16)
    # Training: X_clean = dae.train_step(X_perturbed, X_clean, classifier, y, optimizer)
    # Inference: X_denoised = dae(X_raw)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple, List
from torch.utils.data import DataLoader


class PerturbationGenerator:
    """
    Generates perturbed versions of input for autoencoder training.

    Applied to training data to create (perturbed, clean) pairs
    that the autoencoder learns to reconstruct.
    """

    def __init__(
        self,
        n_features: int,
        p_zero: float = 0.3,
        p_noise: float = 0.3,
        p_scale: float = 0.2,
        p_mimic: float = 0.2,
        noise_std: float = 0.3,
        benign_stats: Optional[dict] = None,
        non_modifiable_indices: Optional[List[int]] = None,
        n_continuous_features: Optional[int] = None,
    ):
        self.n_features = n_features
        self.p_zero = p_zero
        self.p_noise = p_noise
        self.p_scale = p_scale
        self.p_mimic = p_mimic
        self.noise_std = noise_std
        self.benign_stats = benign_stats
        self.n_continuous_features = n_continuous_features
        self.non_modifiable_indices = set(non_modifiable_indices or [])

        self._perturbable = [
            i
            for i in range(n_features)
            if i not in self.non_modifiable_indices
            and (n_continuous_features is None or i < n_continuous_features)
        ]

        self._benign_means_torch = None
        self._device = None
        if benign_stats is not None:
            self._prepare_torch_stats(benign_stats)

    def _prepare_torch_stats(self, benign_stats: dict):
        num_classes = max(benign_stats.keys()) + 1
        means = np.zeros((num_classes, self.n_features), dtype=np.float32)
        for cls, stats in benign_stats.items():
            means[cls] = stats["mean"].astype(np.float32)
        self._benign_means_np = means
        self._benign_means_torch = None

    def _ensure_torch_stats(self, device: torch.device):
        if self._benign_means_torch is not None and self._device == device:
            return
        if hasattr(self, "_benign_means_np") and self._benign_means_np is not None:
            self._benign_means_torch = torch.from_numpy(self._benign_means_np).to(
                device
            )
            self._device = device

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate a perturbed version of x.

        Args:
            x: Clean input (batch, seq_len, features) or (batch, features)
            y: Optional labels for mimic perturbation

        Returns:
            Perturbed version of x
        """
        x_pert = x.clone()
        batch_size = x.size(0)
        device = x.device
        n_pert = len(self._perturbable)

        if n_pert == 0:
            return x_pert

        pert_idx = torch.tensor(self._perturbable, device=device)

        for b in range(batch_size):
            r = torch.rand(1).item()

            if r < self.p_zero:
                n_drop = torch.randint(1, min(4, n_pert + 1), (1,)).item()
                perm = torch.randperm(n_pert)
                for k in range(n_drop):
                    feat = pert_idx[perm[k]]
                    if x_pert.dim() == 3:
                        x_pert[b, :, feat] = 0.0
                    else:
                        x_pert[b, feat] = 0.0

            elif r < self.p_zero + self.p_noise:
                if x_pert.dim() == 3:
                    noise = (
                        torch.randn(x_pert.size(1), self.n_features, device=device)
                        * self.noise_std
                    )
                    n_cont = self.n_continuous_features or self.n_features
                    noise[:, n_cont:] = 0
                    x_pert[b] = x_pert[b] + noise
                else:
                    noise = torch.randn(self.n_features, device=device) * self.noise_std
                    n_cont = self.n_continuous_features or self.n_features
                    noise[n_cont:] = 0
                    x_pert[b] = x_pert[b] + noise

            elif r < self.p_zero + self.p_noise + self.p_scale:
                n_scale = torch.randint(1, min(3, n_pert + 1), (1,)).item()
                perm = torch.randperm(n_pert)
                for k in range(n_scale):
                    feat = pert_idx[perm[k]]
                    scale = torch.FloatTensor(1).uniform_(0.1, 10.0).item()
                    if x_pert.dim() == 3:
                        x_pert[b, :, feat] = x_pert[b, :, feat] * scale
                    else:
                        x_pert[b, feat] = x_pert[b, feat] * scale

            elif r < self.p_zero + self.p_noise + self.p_scale + self.p_mimic:
                if y is not None and self.benign_stats is not None:
                    self._ensure_torch_stats(device)
                    cls = int(y[b].item())
                    if cls < self._benign_means_torch.size(0):
                        n_replace = torch.randint(1, min(4, n_pert + 1), (1,)).item()
                        perm = torch.randperm(n_pert)
                        for k in range(n_replace):
                            feat = pert_idx[perm[k]]
                            mimic_val = self._benign_means_torch[cls, feat]
                            if x_pert.dim() == 3:
                                x_pert[b, :, feat] = mimic_val
                            else:
                                x_pert[b, feat] = mimic_val

        return x_pert


class DenoisingAutoencoder(nn.Module):
    """
    Lightweight autoencoder that reconstructs clean features from perturbed inputs.

    Architecture:
        Encoder: Linear(input_size, 32) -> ReLU -> Linear(32, latent_dim)
        Decoder: Linear(latent_dim, 32) -> ReLU -> Linear(32, input_size)

    The autoencoder is trained to denoise adversarial perturbations,
    producing features that are closer to the original clean distribution.

    At inference time, it acts as a preprocessing defense layer:
        x_denoised = autoencoder(x_raw)
        logits = classifier(x_denoised)
    """

    def __init__(
        self,
        input_size: int,
        latent_dim: int = 16,
        hidden_dim: int = 32,
        n_continuous_features: Optional[int] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.n_continuous_features = n_continuous_features

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_size),
        )

        if n_continuous_features is not None:
            self._clamp_min = 0.0
            self._clamp_max = 1.0
        else:
            self._clamp_min = -3.0
            self._clamp_max = 3.0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return self.encoder(x.reshape(-1, x.size(-1))).reshape(
                x.size(0), x.size(1), self.latent_dim
            )
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 3:
            decoded = self.decoder(z.reshape(-1, z.size(-1))).reshape(
                z.size(0), z.size(1), self.input_size
            )
        else:
            decoded = self.decoder(z)

        if self.n_continuous_features is not None:
            if decoded.dim() == 3:
                decoded[:, :, : self.n_continuous_features] = torch.clamp(
                    decoded[:, :, : self.n_continuous_features], 0.0, 1.0
                )
            else:
                decoded[:, : self.n_continuous_features] = torch.clamp(
                    decoded[:, : self.n_continuous_features], 0.0, 1.0
                )
        else:
            decoded = torch.clamp(decoded, -3.0, 3.0)

        return decoded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denoise input - alias for forward(), for clarity.
        Used at inference time to clean adversarial inputs.
        """
        return self.forward(x)


class DenoisingAETrainer:
    """
    Trainer for the Denoising Autoencoder + Classifier joint training.

    Two-stage training:
    1. Pre-train autoencoder on perturbation reconstruction
    2. Fine-tune autoencoder + classifier jointly

    Loss:
        L = alpha * MSE(x_reconstructed, x_clean) + beta * CE(classifier(x_reconstructed), y)
    """

    def __init__(
        self,
        autoencoder: DenoisingAutoencoder,
        classifier: nn.Module,
        device: torch.device,
        perturbation_generator: PerturbationGenerator,
        alpha: float = 1.0,
        beta: float = 0.5,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.autoencoder = autoencoder.to(device)
        self.classifier = classifier.to(device)
        self.device = device
        self.pert_gen = perturbation_generator
        self.alpha = alpha
        self.beta = beta

        self.ae_optimizer = torch.optim.AdamW(
            autoencoder.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.clf_optimizer = torch.optim.AdamW(
            classifier.parameters(), lr=lr * 0.1, weight_decay=weight_decay
        )
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def pretrain(
        self,
        train_loader: DataLoader,
        epochs: int = 10,
        verbose: bool = True,
    ) -> Dict:
        """
        Pre-train autoencoder on perturbation reconstruction only.
        """
        history = {"loss": []}

        for epoch in range(1, epochs + 1):
            self.autoencoder.train()
            total_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                X_pert = self.pert_gen.generate(X_batch, y_batch)

                X_recon = self.autoencoder(X_pert)
                loss = self.mse_loss(X_recon, X_batch)

                self.ae_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), 1.0)
                self.ae_optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            history["loss"].append(avg_loss)
            if verbose:
                print(
                    f"  DAE Pretrain Epoch {epoch}/{epochs} | MSE Loss: {avg_loss:.4f}"
                )

        return history

    def joint_train_step(
        self,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
    ) -> Tuple[float, float, float]:
        """
        One joint training step: autoencoder + classifier.

        Returns:
            (total_loss, mse_loss, ce_loss)
        """
        self.autoencoder.train()
        self.classifier.train()

        X_pert = self.pert_gen.generate(X_batch, y_batch)

        X_recon = self.autoencoder(X_pert)
        mse = self.mse_loss(X_recon, X_batch)

        logits = self.classifier(X_recon)
        ce = self.ce_loss(logits, y_batch)

        loss = self.alpha * mse + self.beta * ce

        self.ae_optimizer.zero_grad()
        self.clf_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 1.0)
        self.ae_optimizer.step()
        self.clf_optimizer.step()

        return loss.item(), mse.item(), ce.item()


def create_denoising_autoencoder(
    input_size: int,
    latent_dim: int = 16,
    hidden_dim: int = 32,
    n_continuous_features: Optional[int] = None,
    benign_stats: Optional[dict] = None,
    non_modifiable_indices: Optional[List[int]] = None,
) -> Tuple[DenoisingAutoencoder, PerturbationGenerator]:
    """
    Factory function to create autoencoder + perturbation generator.

    Returns:
        (autoencoder, perturbation_generator)
    """
    ae = DenoisingAutoencoder(
        input_size=input_size,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_continuous_features=n_continuous_features,
    )

    pert_gen = PerturbationGenerator(
        n_features=input_size,
        p_zero=0.3,
        p_noise=0.3,
        p_scale=0.2,
        p_mimic=0.2,
        noise_std=0.3,
        benign_stats=benign_stats,
        non_modifiable_indices=non_modifiable_indices,
        n_continuous_features=n_continuous_features,
    )

    return ae, pert_gen
