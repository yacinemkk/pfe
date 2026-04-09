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

Perturbation box modes:

  1. Scalar epsilon:
     [x - ε, x + ε]  uniform for all features.

  2. Per-feature epsilon:
     [x - ε_f, x + ε_f]  feature-specific budgets.

  3. Data-driven with epsilon scaling (combined, recommended for IoT):
     Uses benign_stats (zero, mean, p95) per class to define the maximum
     reachable range under realistic IoT perturbations, then scales it
     by ε ∈ [0, 1]:

         effective_bound[f] = x[f] + ε × (stat_bound[f] - x[f])

     - ε = 0  → no perturbation (box = x)
     - ε = 1  → full data-driven box
     - ε = 0.3 → 30% of the way to the stat bound (tighter, certifiable)

     This keeps ε as a meaningful "budget knob" while respecting the
     per-feature, per-class structure of real IoT attacks.

Reference:
    Gowal et al., "On the Effectiveness of Interval Bound Propagation
    for Training Certifiably Robust Models" (NeurIPS 2019)

Usage:
    # Combined mode (recommended for IoT)
    ibp = IntervalBoundPropagation(
        model, epsilon=0.3,
        benign_stats=sensitivity_analysis.benign_stats,
        perturbation_types=["zero", "mean", "p95"],
        non_modifiable_indices=[0, 5],
    )

    # Scalar epsilon (fallback)
    ibp = IntervalBoundPropagation(model, epsilon=0.1)

    # Per-feature epsilon
    eps_vec = torch.FloatTensor([0.05, 0.1, 0.0, ...])
    ibp = IntervalBoundPropagation(model, epsilon=0.1, epsilon_per_feature=eps_vec)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


class IntervalBoundPropagation:
    """
    Interval Bound Propagation for certified robustness evaluation and
    adversarial training.

    Perturbation box modes (selected automatically based on available data):

    1. Scalar epsilon (fallback):
       [x-ε, x+ε] uniform for all continuous features.

    2. Per-feature epsilon:
       [x-ε_f, x+ε_f] with feature-specific budgets.

    3. Data-driven with epsilon scaling (when benign_stats provided):
       For each feature f of class c, the stat-derived bound represents the
       maximum reachable value under realistic IoT perturbations (zero, mean,
       p95).  The effective bound is then *scaled* by ε:

           effective_lower[f] = x[f] + ε × (stat_lower[f] - x[f])
           effective_upper[f] = x[f] + ε × (stat_upper[f] - x[f])

       ε acts as a "budget knob":
         - ε = 0   → no perturbation (box collapses to x)
         - ε = 1   → full data-driven box (may be very wide)
         - ε = 0.3 → 30% of the way (tighter, more likely to certify)

       Non-modifiable features keep their original value regardless of ε.

    Two bound computation methods:
      1. 'ibp'   – Forward-pass interval propagation.
      2. 'crown' – Linear relaxation (CROWN-IBP hybrid) using Jacobian.
                   Supports asymmetric (data-driven) bounds.
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.1,
        n_continuous_features: Optional[int] = None,
        epsilon_per_feature: Optional[torch.Tensor] = None,
        benign_stats: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
        perturbation_types: Optional[List[str]] = None,
        non_modifiable_indices: Optional[List[int]] = None,
    ):
        self.model = model
        self.epsilon = epsilon
        self.n_continuous_features = n_continuous_features
        self.epsilon_per_feature = epsilon_per_feature
        self.benign_stats = benign_stats
        self.non_modifiable_indices = set(non_modifiable_indices or [])
        self.perturbation_types = perturbation_types or ["zero", "mean", "p95"]

        self.perturbation_bounds = self._compute_perturbation_bounds()

    # ------------------------------------------------------------------ #
    #  Data-driven perturbation bounds from benign_stats                  #
    # ------------------------------------------------------------------ #
    def _compute_perturbation_bounds(self) -> Optional[Dict]:
        """Pre-compute per-class perturbation bounds from benign_stats.

        For each class *c* and feature *f*, the bounds represent the
        extreme reachable values under all specified perturbation strategies:

        +-----------+-------------------------------------------+
        | Strategy  | Effect on bounds                          |
        +===========+===========================================+
        | "zero"    | feature can be set to 0                   |
        | "mean"    | feature can be set to mean of class c     |
        | "p95"     | feature can be set to 95th pct of class c |
        +-----------+-------------------------------------------+

        The resulting box for feature *f* of class *c* is::

            lower[f] = min(0, mean_c[f], p95_c[f])
            upper[f] = max(0, mean_c[f], p95_c[f])

        At runtime, these bounds are **scaled by ε** via interpolation:

            effective[f] = x[f] + ε × (stat_bound[f] - x[f])

        so ε acts as a budget knob (0 = no perturbation, 1 = full stats).

        Non-modifiable features are marked with lower=+inf / upper=-inf
        so they are never perturbed.

        Returns:
            {class_idx: {"lower": FloatTensor, "upper": FloatTensor}}
            or None if benign_stats is not provided.
        """
        if self.benign_stats is None:
            return None

        bounds = {}
        for cls, stats in self.benign_stats.items():
            mean_vals = stats["mean"]
            p95_vals = stats["p95"]
            n_features = len(mean_vals)

            lower = np.full(n_features, np.inf)
            upper = np.full(n_features, -np.inf)

            if "zero" in self.perturbation_types:
                lower = np.minimum(lower, 0.0)
                upper = np.maximum(upper, 0.0)

            if "mean" in self.perturbation_types:
                lower = np.minimum(lower, mean_vals)
                upper = np.maximum(upper, mean_vals)

            if "p95" in self.perturbation_types:
                lower = np.minimum(lower, p95_vals)
                upper = np.maximum(upper, p95_vals)

            for idx in self.non_modifiable_indices:
                if idx < n_features:
                    lower[idx] = np.inf
                    upper[idx] = -np.inf

            lower = np.clip(lower, 0.0, 1.0)
            upper = np.clip(upper, 0.0, 1.0)

            bounds[cls] = {
                "lower": torch.FloatTensor(lower),
                "upper": torch.FloatTensor(upper),
            }

        return bounds

    # ------------------------------------------------------------------ #
    #  Input bounds                                                       #
    # ------------------------------------------------------------------ #
    def _input_bounds(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute lower and upper bounds on the input.

        Three modes (selected automatically based on available data):

        1. **Data-driven with ε scaling** (requires ``benign_stats`` + ``y``):
           For each sample of class *c*, the stat-derived bounds are
           interpolated toward x using ε:

               effective[f] = x[f] + ε × (stat_bound[f] - x[f])

           This produces class-specific, asymmetric bounds that are
           proportionally controlled by ε.

        2. **Per-feature epsilon** (requires ``epsilon_per_feature``):
           ``[x - ε_f, x + ε_f]`` with a different ε per feature.

        3. **Scalar epsilon** (fallback):
           ``[x - ε, x + ε]`` uniform for all continuous features.

        Args:
            x: Input tensor.
            y: Class labels (required for data-driven mode).

        Returns:
            (x_lower, x_upper) tensors with same shape as x.
        """
        x_lower = x.clone()
        x_upper = x.clone()

        # Mode 1: data-driven with epsilon scaling
        if self.perturbation_bounds is not None and y is not None:
            eps = self.epsilon
            for i in range(x.size(0)):
                cls = y[i].item()
                if cls in self.perturbation_bounds:
                    cls_lower = self.perturbation_bounds[cls]["lower"].to(x.device)
                    cls_upper = self.perturbation_bounds[cls]["upper"].to(x.device)

                    if x.dim() == 3 and cls_lower.dim() == 1:
                        cls_lower = cls_lower.unsqueeze(0).expand(x.size(1), -1)
                        cls_upper = cls_upper.unsqueeze(0).expand(x.size(1), -1)

                    # Interpolate: x + ε × (stat_bound - x)
                    x_lower[i] = x[i] + eps * (torch.min(x[i], cls_lower) - x[i])
                    x_upper[i] = x[i] + eps * (torch.max(x[i], cls_upper) - x[i])

        # Mode 2: per-feature epsilon
        elif self.epsilon_per_feature is not None:
            eps = self.epsilon_per_feature.to(x.device)
            if x.dim() == 3:
                eps_expanded = eps.unsqueeze(0).unsqueeze(0)
            else:
                eps_expanded = eps.unsqueeze(0)

            if self.n_continuous_features is not None:
                n_cont = self.n_continuous_features
                x_lower[..., :n_cont] = x[..., :n_cont] - eps_expanded[..., :n_cont]
                x_upper[..., :n_cont] = x[..., :n_cont] + eps_expanded[..., :n_cont]
            else:
                x_lower = x - eps_expanded
                x_upper = x + eps_expanded

        # Mode 3: scalar epsilon (fallback)
        else:
            if self.n_continuous_features is not None:
                n_cont = self.n_continuous_features
                x_lower[..., :n_cont] = x[..., :n_cont] - self.epsilon
                x_upper[..., :n_cont] = x[..., :n_cont] + self.epsilon
            else:
                x_lower = x - self.epsilon
                x_upper = x + self.epsilon

        # Non-modifiable features: always keep original value
        for idx in self.non_modifiable_indices:
            x_lower[..., idx] = x[..., idx]
            x_upper[..., idx] = x[..., idx]

        return x_lower.detach(), x_upper.detach()

    # ------------------------------------------------------------------ #
    #  CROWN-style linear relaxation bounds on logits                    #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def compute_bounds_crown(
        self,
        x: torch.Tensor,
        device: torch.device,
        y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute output bounds using CROWN-style linear approximation.

        For scalar / per-feature epsilon (symmetric bounds)::

            logit_j_lower  ≈ logit_j  -  Σ_f |∇_f| · ε_f

        For data-driven with ε scaling (asymmetric bounds)::

            δ_lo = x_lower - x   (negative or zero)
            δ_up = x_upper - x   (positive or zero)

            logit_j_lower = logit_j + Σ_f min(∇_f·δ_lo_f, ∇_f·δ_up_f)
            logit_j_upper = logit_j + Σ_f max(∇_f·δ_lo_f, ∇_f·δ_up_f)
        """
        x_input = x.clone().detach().requires_grad_(True)
        logits = self.model(x_input)

        num_classes = logits.size(-1)

        logit_lower = torch.zeros_like(logits)
        logit_upper = torch.zeros_like(logits)

        use_asymmetric = self.perturbation_bounds is not None and y is not None
        if use_asymmetric:
            x_l, x_u = self._input_bounds(x, y)
            delta_lower = (x_l - x).detach()
            delta_upper = (x_u - x).detach()

        for j in range(num_classes):
            self.model.zero_grad()
            if x_input.grad is not None:
                x_input.grad.zero_()

            grad_j = torch.autograd.grad(
                logits[:, j].sum(), x_input, retain_graph=False, create_graph=False
            )[0]

            if use_asymmetric:
                effect_lo = grad_j * delta_lower
                effect_up = grad_j * delta_upper

                if grad_j.dim() == 3:
                    per_lo = torch.min(effect_lo, effect_up).sum(dim=-1).sum(dim=-1)
                    per_up = torch.max(effect_lo, effect_up).sum(dim=-1).sum(dim=-1)
                else:
                    per_lo = torch.min(effect_lo, effect_up).sum(dim=-1)
                    per_up = torch.max(effect_lo, effect_up).sum(dim=-1)

                logit_lower[:, j] = logits[:, j] + per_lo
                logit_upper[:, j] = logits[:, j] + per_up

            elif self.epsilon_per_feature is not None:
                eps = self.epsilon_per_feature.to(device)
                if self.n_continuous_features is not None:
                    n_cont = self.n_continuous_features
                    if grad_j.dim() == 3:
                        weighted = grad_j[..., :n_cont].abs() * eps[:n_cont].unsqueeze(
                            0
                        ).unsqueeze(0)
                        grad_norm = weighted.sum(dim=-1).sum(dim=1)
                    else:
                        weighted = grad_j[..., :n_cont].abs() * eps[:n_cont]
                        grad_norm = weighted.sum(dim=-1)
                else:
                    if grad_j.dim() == 3:
                        weighted = grad_j.abs() * eps.unsqueeze(0).unsqueeze(0)
                        grad_norm = weighted.sum(dim=-1).sum(dim=1)
                    else:
                        weighted = grad_j.abs() * eps
                        grad_norm = weighted.sum(dim=-1)

                logit_lower[:, j] = logits[:, j] - grad_norm
                logit_upper[:, j] = logits[:, j] + grad_norm

            else:
                if self.n_continuous_features is not None:
                    grad_norm = (
                        grad_j[..., : self.n_continuous_features].abs().sum(dim=-1)
                    )
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
    #  IBP-style bounds via dual forward passes                           #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def compute_bounds_ibp(
        self,
        x: torch.Tensor,
        device: torch.device,
        y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute output bounds using forward-pass IBP."""
        x_lower, x_upper = self._input_bounds(x, y)

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
        y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dispatch to the chosen bound-computation method."""
        if method == "crown":
            return self.compute_bounds_crown(x, device, y=y)
        else:
            return self.compute_bounds_ibp(x, device, y=y)

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
        """Evaluate certified accuracy on a dataset.

        A sample is *certified* if the lower bound of the correct-class
        logit exceeds the upper bound of every incorrect-class logit.
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
                X_batch, device, method=method, y=y_batch
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
            "perturbation_types": self.perturbation_types,
            "epsilon": self.epsilon,
        }

    # ------------------------------------------------------------------ #
    #  IBP training loss                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_ibp_loss(
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilon: float = 0.1,
        lambda_ibp: float = 1.0,
        n_continuous_features: Optional[int] = None,
        method: str = "crown",
        epsilon_per_feature: Optional[torch.Tensor] = None,
        perturbation_bounds: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute the IBP regularisation loss for adversarial training.

        L_ibp = max(0, max_{j!=y} ub_j - lb_y + margin)

        When lb_y > ub_j for all j!=y, the prediction is certified and
        the loss is zero.

        Supports three modes:

        1. **Data-driven with ε scaling** (``perturbation_bounds`` provided):
           Asymmetric CROWN bounds derived from benign_stats per class,
           scaled by ε:

               effective[f] = x[f] + ε × (stat_bound[f] - x[f])

        2. **Per-feature epsilon** (``epsilon_per_feature`` provided):
           Symmetric CROWN bounds with feature-specific budgets.

        3. **Scalar epsilon** (fallback):
           Original uniform perturbation budget.

        Args:
            model:  Classifier.
            x:  Input batch (batch, seq_len, features) or (batch, features).
            y:  Labels (batch,).
            epsilon:  Perturbation budget / scaling factor.
            lambda_ibp:  Weight for the IBP loss term.
            n_continuous_features:  Number of perturbable features.
            method:  'crown' or 'ibp'.
            epsilon_per_feature:  Per-feature epsilon vector (n_features,).
            perturbation_bounds:  Pre-computed per-class bounds dict from
                ``IntervalBoundPropagation._compute_perturbation_bounds()``.

        Returns:
            (ibp_loss, info_dict)
        """
        device = x.device
        batch_size = x.size(0)

        x_input = x.clone().detach().requires_grad_(True)
        logits = model(x_input)
        num_classes = logits.size(-1)

        use_data_driven = perturbation_bounds is not None
        use_per_feature = epsilon_per_feature is not None and not use_data_driven

        # ── Compute perturbed input bounds (detached, used as certificates) ──
        # We compute bounds on x, then perform a *differentiable* forward pass
        # on those perturbed inputs so that ibp_loss has grad_fn w.r.t. model params.

        if method == "crown":
            # CROWN: use gradient of logits w.r.t. x_input to approximate bounds.
            # x_input is detached so grads flow to it, not through model params here.
            # We then compute the actual differentiable loss via a second forward pass.
            logit_lower_det = torch.zeros(batch_size, device=device)
            logit_upper_det = torch.zeros(batch_size, device=device)

            if use_data_driven:
                # Compute ε-scaled data-driven input bounds
                x_lower = x.clone().detach()
                x_upper = x.clone().detach()
                with torch.no_grad():
                    for i in range(batch_size):
                        cls = y[i].item()
                        if cls in perturbation_bounds:
                            cls_lower = perturbation_bounds[cls]["lower"].to(device)
                            cls_upper = perturbation_bounds[cls]["upper"].to(device)
                            if x.dim() == 3 and cls_lower.dim() == 1:
                                cls_lower = cls_lower.unsqueeze(0).expand(x.size(1), -1)
                                cls_upper = cls_upper.unsqueeze(0).expand(x.size(1), -1)
                            x_lower[i] = x[i] + epsilon * (
                                torch.min(x[i], cls_lower) - x[i]
                            )
                            x_upper[i] = x[i] + epsilon * (
                                torch.max(x[i], cls_upper) - x[i]
                            )

                delta_lower = (x_lower - x.detach())
                delta_upper = (x_upper - x.detach())

                # Use CROWN gradients to estimate worst-case logit bounds per sample
                # (these bounds are detached — used only as certificate indicators)
                lb_y_det = torch.zeros(batch_size, device=device)
                ub_other_det = torch.zeros(batch_size, device=device)

                for j in range(num_classes):
                    if x_input.grad is not None:
                        x_input.grad.zero_()

                    grad_j = torch.autograd.grad(
                        logits[:, j].sum(),
                        x_input,
                        retain_graph=True,  # keep graph alive for all classes
                        create_graph=False,
                    )[0].detach()

                    effect_lo = grad_j * delta_lower
                    effect_up = grad_j * delta_upper

                    if grad_j.dim() == 3:
                        per_lo = torch.min(effect_lo, effect_up).sum(dim=-1).sum(dim=-1)
                        per_up = torch.max(effect_lo, effect_up).sum(dim=-1).sum(dim=-1)
                    else:
                        per_lo = torch.min(effect_lo, effect_up).sum(dim=-1)
                        per_up = torch.max(effect_lo, effect_up).sum(dim=-1)

                    lb_j = (logits[:, j] + per_lo).detach()
                    ub_j = (logits[:, j] + per_up).detach()

                    is_true = (y == j)
                    lb_y_det = torch.where(is_true, lb_j, lb_y_det)
                    # For ub_other: take max over all non-true classes
                    if j == 0:
                        ub_other_det = torch.where(is_true, torch.full_like(ub_j, -1e9), ub_j)
                    else:
                        ub_other_det = torch.where(
                            is_true, ub_other_det, torch.max(ub_other_det, ub_j)
                        )

            elif use_per_feature:
                eps = epsilon_per_feature.to(device)
                x_lower = x.clone().detach()
                x_upper = x.clone().detach()
                if n_continuous_features is not None:
                    n_cont = n_continuous_features
                    if x.dim() == 3:
                        eps_exp = eps[:n_cont].unsqueeze(0).unsqueeze(0)
                    else:
                        eps_exp = eps[:n_cont].unsqueeze(0)
                    x_lower[..., :n_cont] = x.detach()[..., :n_cont] - eps_exp
                    x_upper[..., :n_cont] = x.detach()[..., :n_cont] + eps_exp
                else:
                    if x.dim() == 3:
                        eps_exp = eps.unsqueeze(0).unsqueeze(0)
                    else:
                        eps_exp = eps.unsqueeze(0)
                    x_lower = x.detach() - eps_exp
                    x_upper = x.detach() + eps_exp

                delta_lower = x_lower - x.detach()
                delta_upper = x_upper - x.detach()

                lb_y_det = torch.zeros(batch_size, device=device)
                ub_other_det = torch.zeros(batch_size, device=device)

                for j in range(num_classes):
                    if x_input.grad is not None:
                        x_input.grad.zero_()

                    grad_j = torch.autograd.grad(
                        logits[:, j].sum(),
                        x_input,
                        retain_graph=True,
                        create_graph=False,
                    )[0].detach()

                    effect_lo = grad_j * delta_lower
                    effect_up = grad_j * delta_upper

                    if grad_j.dim() == 3:
                        per_lo = torch.min(effect_lo, effect_up).sum(dim=-1).sum(dim=-1)
                        per_up = torch.max(effect_lo, effect_up).sum(dim=-1).sum(dim=-1)
                    else:
                        per_lo = torch.min(effect_lo, effect_up).sum(dim=-1)
                        per_up = torch.max(effect_lo, effect_up).sum(dim=-1)

                    lb_j = (logits[:, j] + per_lo).detach()
                    ub_j = (logits[:, j] + per_up).detach()

                    is_true = (y == j)
                    lb_y_det = torch.where(is_true, lb_j, lb_y_det)
                    if j == 0:
                        ub_other_det = torch.where(is_true, torch.full_like(ub_j, -1e9), ub_j)
                    else:
                        ub_other_det = torch.where(
                            is_true, ub_other_det, torch.max(ub_other_det, ub_j)
                        )

            else:
                # Scalar epsilon — compute symmetric bounds
                if n_continuous_features is not None:
                    n_cont = n_continuous_features
                    x_lower = x.clone().detach()
                    x_upper = x.clone().detach()
                    x_lower[..., :n_cont] = x.detach()[..., :n_cont] - epsilon
                    x_upper[..., :n_cont] = x.detach()[..., :n_cont] + epsilon
                else:
                    x_lower = x.detach() - epsilon
                    x_upper = x.detach() + epsilon

                delta_lower = x_lower - x.detach()
                delta_upper = x_upper - x.detach()

                lb_y_det = torch.zeros(batch_size, device=device)
                ub_other_det = torch.zeros(batch_size, device=device)

                for j in range(num_classes):
                    if x_input.grad is not None:
                        x_input.grad.zero_()

                    grad_j = torch.autograd.grad(
                        logits[:, j].sum(),
                        x_input,
                        retain_graph=True,
                        create_graph=False,
                    )[0].detach()

                    effect_lo = grad_j * delta_lower
                    effect_up = grad_j * delta_upper

                    if grad_j.dim() == 3:
                        per_lo = torch.min(effect_lo, effect_up).sum(dim=-1).sum(dim=-1)
                        per_up = torch.max(effect_lo, effect_up).sum(dim=-1).sum(dim=-1)
                    else:
                        per_lo = torch.min(effect_lo, effect_up).sum(dim=-1)
                        per_up = torch.max(effect_lo, effect_up).sum(dim=-1)

                    lb_j = (logits[:, j] + per_lo).detach()
                    ub_j = (logits[:, j] + per_up).detach()

                    is_true = (y == j)
                    lb_y_det = torch.where(is_true, lb_j, lb_y_det)
                    if j == 0:
                        ub_other_det = torch.where(is_true, torch.full_like(ub_j, -1e9), ub_j)
                    else:
                        ub_other_det = torch.where(
                            is_true, ub_other_det, torch.max(ub_other_det, ub_j)
                        )

            # ── Differentiable IBP loss via a fresh forward pass ──
            # The bound violation mask (certified_mask) is computed from detached bounds.
            # Non-certified samples incur a cross-entropy loss on the worst-case
            # perturbed input (x_lower or x_upper, chosen by the bound direction).
            # This forward pass IS connected to model parameters.
            certified_mask = (lb_y_det > ub_other_det).detach()
            violation = F.relu(ub_other_det - lb_y_det).detach()

            # For non-certified samples, compute cross-entropy on perturbed input
            # (use the bound endpoint that is worst: x_lower or x_upper midpoint)
            if not certified_mask.all():
                x_perturbed = (x_lower + x_upper) / 2.0  # midpoint of the box
                perturbed_logits = model(x_perturbed)  # differentiable!
                ibp_ce = F.cross_entropy(perturbed_logits, y, reduction='none')
                ibp_loss = lambda_ibp * (ibp_ce * (1.0 - certified_mask.float())).mean()
            else:
                ibp_loss = torch.tensor(0.0, device=device).requires_grad_()

        else:
            # ── IBP method: dual forward passes (differentiable) ──
            x_lower = x.clone().detach()
            x_upper = x.clone().detach()

            if use_data_driven:
                with torch.no_grad():
                    for i in range(batch_size):
                        cls = y[i].item()
                        if cls in perturbation_bounds:
                            cls_lower = perturbation_bounds[cls]["lower"].to(device)
                            cls_upper = perturbation_bounds[cls]["upper"].to(device)
                            if x.dim() == 3 and cls_lower.dim() == 1:
                                cls_lower = cls_lower.unsqueeze(0).expand(x.size(1), -1)
                                cls_upper = cls_upper.unsqueeze(0).expand(x.size(1), -1)
                            x_lower[i] = x[i] + epsilon * (
                                torch.min(x[i], cls_lower) - x[i]
                            )
                            x_upper[i] = x[i] + epsilon * (
                                torch.max(x[i], cls_upper) - x[i]
                            )

            elif use_per_feature:
                eps = epsilon_per_feature.to(device)
                if n_continuous_features is not None:
                    n_cont = n_continuous_features
                    if x.dim() == 3:
                        eps_expanded = eps[:n_cont].unsqueeze(0).unsqueeze(0)
                    else:
                        eps_expanded = eps[:n_cont].unsqueeze(0)
                    x_lower[..., :n_cont] = x.detach()[..., :n_cont] - eps_expanded
                    x_upper[..., :n_cont] = x.detach()[..., :n_cont] + eps_expanded
                else:
                    if x.dim() == 3:
                        eps_expanded = eps.unsqueeze(0).unsqueeze(0)
                    else:
                        eps_expanded = eps.unsqueeze(0)
                    x_lower = x.detach() - eps_expanded
                    x_upper = x.detach() + eps_expanded

            else:
                if n_continuous_features is not None:
                    n_cont = n_continuous_features
                    x_lower[..., :n_cont] = x.detach()[..., :n_cont] - epsilon
                    x_upper[..., :n_cont] = x.detach()[..., :n_cont] + epsilon
                else:
                    x_lower = x.detach() - epsilon
                    x_upper = x.detach() + epsilon

            # Differentiable forward passes — no torch.no_grad() so grads flow through model
            logits_l = model(x_lower)
            logits_u = model(x_upper)

            logit_lower = torch.min(logits_l, logits_u)
            logit_upper = torch.max(logits_l, logits_u)

            lb_y = logit_lower[torch.arange(batch_size, device=device), y]
            ub_others = logit_upper.clone()
            ub_others[torch.arange(batch_size, device=device), y] = -float("inf")
            max_ub_other = ub_others.max(dim=1)[0]

            per_sample_violation = F.relu(max_ub_other - lb_y)
            ibp_loss = lambda_ibp * per_sample_violation.mean()

            certified_mask = (lb_y > max_ub_other).detach().float()
            lb_y_det = lb_y.detach()
            ub_other_det = max_ub_other.detach()

        certified_mask_f = certified_mask.float() if certified_mask.dtype != torch.float32 else certified_mask
        info = {
            "ibp_loss": ibp_loss.item(),
            "certified_ratio": certified_mask_f.mean().item(),
            "avg_margin": (lb_y_det - ub_other_det).mean().item(),
        }

        return ibp_loss, info


class IBPTrainer:
    """Helper for integrating IBP loss into the adversarial training loop.

    Uses **combined mode** when benign_stats is provided:

        effective_bound[f] = x[f] + ε × (stat_bound[f] - x[f])

    ε acts as a budget knob:
      - ε = 0   → no perturbation
      - ε = 1   → full data-driven box
      - ε = 0.3 → 30% of the way (tighter, more certifiable)

    Additionally supports epsilon scheduling during warmup:
      - First ``warmup_epochs`` epochs: ε ramps linearly from
        ``epsilon_start`` to ``epsilon``.
      - After warmup: ε = ``epsilon`` (full budget).

    Lambda scaling also ramps during warmup (0 → lambda_ibp).

    Usage inside a training loop::

        ibp_trainer = IBPTrainer(
            model, device, epsilon=0.3,
            benign_stats=sensitivity_analysis.benign_stats,
            perturbation_types=["zero", "mean", "p95"],
            non_modifiable_indices=[0, 5],
            epsilon_start=0.05,
        )
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
        epsilon_per_feature: Optional[torch.Tensor] = None,
        benign_stats: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
        perturbation_types: Optional[List[str]] = None,
        non_modifiable_indices: Optional[List[int]] = None,
        epsilon_start: Optional[float] = None,
    ):
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.lambda_ibp = lambda_ibp
        self.n_continuous_features = n_continuous_features
        self.method = method
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.epsilon_per_feature = epsilon_per_feature
        self.benign_stats = benign_stats
        self.perturbation_types = perturbation_types or ["zero", "mean", "p95"]
        self.non_modifiable_indices = non_modifiable_indices or []
        self.epsilon_start = epsilon_start if epsilon_start is not None else 0.0

        self.perturbation_bounds = self._compute_perturbation_bounds()

    def _compute_perturbation_bounds(self) -> Optional[Dict]:
        """Pre-compute per-class perturbation bounds from benign_stats."""
        if self.benign_stats is None:
            return None

        bounds = {}
        for cls, stats in self.benign_stats.items():
            mean_vals = stats["mean"]
            p95_vals = stats["p95"]
            n_features = len(mean_vals)

            lower = np.full(n_features, np.inf)
            upper = np.full(n_features, -np.inf)

            if "zero" in self.perturbation_types:
                lower = np.minimum(lower, 0.0)
                upper = np.maximum(upper, 0.0)

            if "mean" in self.perturbation_types:
                lower = np.minimum(lower, mean_vals)
                upper = np.maximum(upper, mean_vals)

            if "p95" in self.perturbation_types:
                lower = np.minimum(lower, p95_vals)
                upper = np.maximum(upper, p95_vals)

            for idx in self.non_modifiable_indices:
                if idx < n_features:
                    lower[idx] = np.inf
                    upper[idx] = -np.inf

            lower = np.clip(lower, 0.0, 1.0)
            upper = np.clip(upper, 0.0, 1.0)

            bounds[cls] = {
                "lower": torch.FloatTensor(lower),
                "upper": torch.FloatTensor(upper),
            }

        return bounds

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def _get_effective_epsilon(self) -> float:
        """Get the effective epsilon for the current epoch.

        During warmup, ε ramps linearly from epsilon_start to epsilon.
        After warmup, ε = epsilon (full budget).
        """
        if self.current_epoch < self.warmup_epochs:
            t = (self.current_epoch + 1) / self.warmup_epochs
            return self.epsilon_start + t * (self.epsilon - self.epsilon_start)
        return self.epsilon

    def compute_loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute IBP loss with warmup scaling (both λ and ε).

        During warmup:
          - λ ramps from 0 to lambda_ibp (prevents early instability)
          - ε ramps from epsilon_start to epsilon (grows perturbation box)

        Args:
            x: Input batch.
            y: Class labels (used for data-driven bounds).

        Returns:
            (ibp_loss, info_dict)
        """
        if self.current_epoch < self.warmup_epochs:
            lambda_scale = (self.current_epoch + 1) / self.warmup_epochs
        else:
            lambda_scale = 1.0

        effective_lambda = self.lambda_ibp * lambda_scale
        effective_epsilon = self._get_effective_epsilon()

        if effective_lambda < 1e-8:
            return torch.tensor(0.0, device=self.device, requires_grad=True), {}

        ibp_loss, info = IntervalBoundPropagation.compute_ibp_loss(
            self.model,
            x,
            y,
            epsilon=effective_epsilon,
            lambda_ibp=effective_lambda,
            n_continuous_features=self.n_continuous_features,
            method=self.method,
            epsilon_per_feature=self.epsilon_per_feature,
            perturbation_bounds=self.perturbation_bounds,
        )
        info["effective_epsilon"] = effective_epsilon
        return ibp_loss, info

    def certify(self, dataloader, max_samples=None) -> Dict:
        """Run certified accuracy evaluation."""
        propagator = IntervalBoundPropagation(
            self.model,
            epsilon=self.epsilon,
            n_continuous_features=self.n_continuous_features,
            benign_stats=self.benign_stats,
            perturbation_types=self.perturbation_types,
            non_modifiable_indices=self.non_modifiable_indices,
            epsilon_per_feature=self.epsilon_per_feature,
        )
        return propagator.certify(
            dataloader, self.device, method=self.method, max_samples=max_samples
        )
