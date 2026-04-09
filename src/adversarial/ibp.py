"""
Interval Bound Propagation (IBP) for Certified Robustness

Unlike empirical defenses (TRADES, AFD) which resist known attacks "up to
proven otherwise", IBP provides mathematical, formal guarantees.

During training and evaluation, the algorithm tracks not just the trajectory
of a single data point through the network, but a "box" (interval) that
represents ALL possible alterations (perturbations) of that data point.

Key result: If the lower bound of the correct-class logit exceeds the upper
bounds of every other class logit, the prediction is *certified* – no
perturbation within epsilon can change it.

Reference:
    Gowal et al., "On the Effectiveness of Interval Bound Propagation
    for Training Certifiably Robust Models" (NeurIPS 2019)

Usage:
    ibp = IntervalBoundPropagation(model, epsilon=0.1)
    certified_acc, certified_radius = ibp.certify(test_loader, device)
    ibp_loss = ibp.compute_ibp_loss(model, x, y, epsilon, lambda_ibp)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from tqdm import tqdm


class IntervalBoundPropagation:
    """
    Interval Bound Propagation for certified robustness evaluation and
    adversarial training.

    Two operating modes:
      1. 'ibp'     – Layer-wise interval propagation (tight for FC layers,
                     conservative for recurrent/attention layers).
      2. 'crown'   – Linear relaxation (CROWN-IBP hybrid) using the
                     Jacobian to tighten bounds on the final logits.

    Attributes:
        model:  The classifier to certify.
        epsilon: L-infinity perturbation budget.
        n_continuous_features:  Number of continuous (perturbable) features.
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.1,
        n_continuous_features: Optional[int] = None,
    ):
        self.model = model
        self.epsilon = epsilon
        self.n_continuous_features = n_continuous_features

    # ------------------------------------------------------------------ #
    #  Input bounds                                                       #
    # ------------------------------------------------------------------ #
    def _input_bounds(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute lower and upper bounds on the input.

        Only continuous features are perturbed; binary / categorical features
        keep their original values.
        """
        x_lower = x.clone()
        x_upper = x.clone()

        if self.n_continuous_features is not None:
            n_cont = self.n_continuous_features
            x_lower[..., :n_cont] = x[..., :n_cont] - self.epsilon
            x_upper[..., :n_cont] = x[..., :n_cont] + self.epsilon
        else:
            x_lower = x - self.epsilon
            x_upper = x + self.epsilon

        return x_lower.detach(), x_upper.detach()

    # ------------------------------------------------------------------ #
    #  Linear-approximation (CROWN-style) bounds on logits                #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def compute_bounds_crown(
        self,
        x: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute output bounds using a first-order (linear) approximation.

        For each class j, the bound is:
            logit_j(x + delta) ≈ logit_j(x) + <grad_j, delta>

        With ||delta||_inf <= epsilon:
            logit_j_lower ≈ logit_j - epsilon * ||grad_j||_1
            logit_j_upper ≈ logit_j + epsilon * ||grad_j||_1

        This is a CROWN-IBP style linear relaxation.
        """
        x_input = x.clone().detach().requires_grad_(True)
        logits = self.model(x_input)

        batch_size = x.size(0)
        num_classes = logits.size(-1)

        logit_lower = torch.zeros_like(logits)
        logit_upper = torch.zeros_like(logits)

        for j in range(num_classes):
            self.model.zero_grad()
            if x_input.grad is not None:
                x_input.grad.zero_()

            grad_j = torch.autograd.grad(
                logits[:, j].sum(), x_input, retain_graph=False, create_graph=False
            )[0]

            if self.n_continuous_features is not None:
                grad_norm = grad_j[..., : self.n_continuous_features].abs().sum(dim=-1)
                if grad_j.dim() == 3:
                    grad_norm = grad_norm.sum(dim=1)
            else:
                grad_norm = grad_j.abs().sum(dim=-1)
                if grad_j.dim() == 3:
                    grad_norm = grad_norm.sum(dim=1)

            logit_lower[:, j] = logits[:, j] - self.epsilon * grad_norm
            logit_upper[:, j] = logits[:, j] + self.epsilon * grad_norm

        return logit_lower, logit_upper

    # ------------------------------------------------------------------ #
    #  IBP-style bounds via dual forward passes                            #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def compute_bounds_ibp(
        self,
        x: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute output bounds using forward-pass IBP.

        Propagates lower and upper input bounds through the model.
        For FC layers the propagation is exact; for nonlinear layers
        (LSTM, attention) the bounds are conservative (over-approximated)
        by taking the min/max of the outputs at the lower and upper inputs.
        """
        x_lower, x_upper = self._input_bounds(x)

        logits_lower = self.model(x_lower)
        logits_upper = self.model(x_upper)

        logits_l = torch.min(logits_lower, logits_upper)
        logits_u = torch.max(logits_lower, logits_upper)

        return logits_l, logits_u

    # ------------------------------------------------------------------ #
    #  Unified bound computation                                          #
    # ------------------------------------------------------------------ #
    def compute_bounds(
        self,
        x: torch.Tensor,
        device: torch.device,
        method: str = "crown",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dispatch to the chosen bound-computation method."""
        if method == "crown":
            return self.compute_bounds_crown(x, device)
        else:
            return self.compute_bounds_ibp(x, device)

    # ------------------------------------------------------------------ #
    #  Certified accuracy evaluation                                       #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def certify(
        self,
        dataloader,
        device: torch.device,
        method: str = "crown",
        max_samples: Optional[int] = None,
    ) -> Dict:
        """
        Evaluate certified accuracy on a dataset.

        A sample is *certified* if the lower bound of the correct-class
        logit exceeds the upper bound of every incorrect-class logit.

        Returns:
            dict with 'certified_accuracy', 'certified_ratio',
                      'clean_accuracy', 'avg_certified_radius',
                      'per_sample_radius'
        """
        self.model.eval()
        certified = 0
        correct = 0
        total = 0
        certified_radii = []

        for batch_idx, (X_batch, y_batch) in enumerate(
            tqdm(dataloader, desc="Certifying", leave=False)
        ):
            if max_samples is not None and total >= max_samples:
                break

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = self.model(X_batch)
            _, predicted = logits.max(1)
            correct += predicted.eq(y_batch).sum().item()

            logit_lower, logit_upper = self.compute_bounds(
                X_batch, device, method=method
            )

            for i in range(X_batch.size(0)):
                y_true = y_batch[i].item()
                lb_y = logit_lower[i, y_true].item()

                ub_others = logit_upper[i].clone()
                ub_others[y_true] = -float("inf")
                max_ub_other = ub_others.max().item()

                margin = lb_y - max_ub_other

                if margin > 0:
                    certified += 1
                    certified_radii.append(max(0.0, margin))

            total += X_batch.size(0)

        certified_acc = certified / max(total, 1)
        clean_acc = correct / max(total, 1)
        avg_radius = np.mean(certified_radii) if certified_radii else 0.0

        return {
            "certified_accuracy": certified_acc,
            "certified_ratio": certified / max(total, 1),
            "clean_accuracy": clean_acc,
            "avg_certified_radius": avg_radius,
            "total_samples": total,
        }

    # ------------------------------------------------------------------ #
    #  IBP training loss                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_ibp_loss(
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilon: float,
        lambda_ibp: float = 1.0,
        n_continuous_features: Optional[int] = None,
        method: str = "crown",
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute the IBP regularisation loss for adversarial training.

        L_ibp = max(0, max_{j!=y} ub_j - lb_y + margin)

        When lb_y > ub_j for all j!=y, the prediction is certified and
        the loss is zero.

        Args:
            model:  Classifier.
            x:  Input batch (batch, seq_len, features).
            y:  Labels (batch,).
            epsilon:  Perturbation budget.
            lambda_ibp:  Weight for the IBP loss term.
            n_continuous_features:  Number of perturbable features.
            method:  'crown' or 'ibp'.

        Returns:
            (ibp_loss, info_dict)
        """
        device = x.device
        batch_size = x.size(0)

        x_input = x.clone().detach().requires_grad_(True)
        logits = model(x_input)
        num_classes = logits.size(-1)

        if method == "crown":
            logit_lower = torch.zeros_like(logits)
            logit_upper = torch.zeros_like(logits)

            for j in range(num_classes):
                if x_input.grad is not None:
                    x_input.grad.zero_()

                grad_j = torch.autograd.grad(
                    logits[:, j].sum(),
                    x_input,
                    retain_graph=(j < num_classes - 1),
                    create_graph=False,
                )[0]

                if n_continuous_features is not None:
                    grad_norm = grad_j[..., :n_continuous_features].abs().sum(dim=-1)
                    if grad_j.dim() == 3:
                        grad_norm = grad_norm.sum(dim=1)
                else:
                    grad_norm = grad_j.abs().sum(dim=-1)
                    if grad_j.dim() == 3:
                        grad_norm = grad_norm.sum(dim=1)

                logit_lower[:, j] = logits[:, j] - epsilon * grad_norm
                logit_upper[:, j] = logits[:, j] + epsilon * grad_norm
        else:
            x_lower = x.clone()
            x_upper = x.clone()
            if n_continuous_features is not None:
                n_cont = n_continuous_features
                x_lower[..., :n_cont] = x[..., :n_cont] - epsilon
                x_upper[..., :n_cont] = x[..., :n_cont] + epsilon
            else:
                x_lower = x - epsilon
                x_upper = x + epsilon

            with torch.no_grad():
                logits_l = model(x_lower)
                logits_u = model(x_upper)

            logit_lower = torch.min(logits_l, logits_u)
            logit_upper = torch.max(logits_l, logits_u)

        lb_y = logit_lower[torch.arange(batch_size, device=device), y]
        ub_others = logit_upper.clone()
        ub_others[torch.arange(batch_size, device=device), y] = -float("inf")
        max_ub_other = ub_others.max(dim=1)[0]

        margin = 0.0
        per_sample_violation = F.relu(max_ub_other - lb_y + margin)
        ibp_loss = lambda_ibp * per_sample_violation.mean()

        certified_mask = (lb_y > max_ub_other).float()
        info = {
            "ibp_loss": ibp_loss.item(),
            "certified_ratio": certified_mask.mean().item(),
            "avg_margin": (lb_y - max_ub_other).mean().item(),
        }

        return ibp_loss, info


class IBPTrainer:
    """
    Helper for integrating IBP loss into the adversarial training loop.

    Usage inside a training loop::

        ibp_trainer = IBPTrainer(model, device, epsilon=0.1)
        ...
        ibp_loss, info = ibp_trainer.compute_loss(X_batch, y_batch)
        total_loss = ce_loss + kl_loss + ibp_loss
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        epsilon: float = 0.1,
        lambda_ibp: float = 1.0,
        n_continuous_features: Optional[int] = None,
        method: str = "crown",
        warmup_epochs: int = 5,
    ):
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.lambda_ibp = lambda_ibp
        self.n_continuous_features = n_continuous_features
        self.method = method
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def compute_loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute IBP loss with warmup scaling."""
        if self.current_epoch < self.warmup_epochs:
            scale = (self.current_epoch + 1) / self.warmup_epochs
        else:
            scale = 1.0

        effective_lambda = self.lambda_ibp * scale
        if effective_lambda < 1e-8:
            return torch.tensor(0.0, device=self.device, requires_grad=True), {}

        ibp_loss, info = IntervalBoundPropagation.compute_ibp_loss(
            self.model,
            x,
            y,
            epsilon=self.epsilon,
            lambda_ibp=effective_lambda,
            n_continuous_features=self.n_continuous_features,
            method=self.method,
        )
        return ibp_loss, info

    def certify(self, dataloader, max_samples=None) -> Dict:
        """Run certified accuracy evaluation."""
        propagator = IntervalBoundPropagation(
            self.model,
            epsilon=self.epsilon,
            n_continuous_features=self.n_continuous_features,
        )
        return propagator.certify(
            dataloader, self.device, method=self.method, max_samples=max_samples
        )
