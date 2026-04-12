import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

# =============================================================================
# COUCHE 1 — DÉFENSE D'ENTRÉE (tue Padding_x10 avant même le modèle)
# =============================================================================

class InputDefenseLayer(nn.Module):
    """
    Première ligne de défense : preprocessing défensif.
    
    Tue Padding_x10 : clip les valeurs aberrantes.
    Réduit Zero/Mimic : lissage temporel exponentiel.
    """
    def __init__(self, clip_min: float = -3.5, clip_max: float = 3.5,
                 smooth_alpha: float = 0.25):
        super().__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.alpha = smooth_alpha  # 0 = pas de lissage, 1 = remplace tout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, self.clip_min, self.clip_max)
        smoothed = x.clone()
        for t in range(1, x.shape[1]):
            smoothed[:, t] = (self.alpha * x[:, t]
                              + (1 - self.alpha) * smoothed[:, t - 1])
        return smoothed


# =============================================================================
# COUCHE 2 — WORST-OF-K MULTI-FEATURES (résiste à la recherche greedy)
# =============================================================================

class MultiFeatureWorstOfK:
    """
    Génère des exemples adversariaux en combinant plusieurs features perturbées
    simultanément.
    """
    STRATEGIES = ['Zero', 'Mimic_Mean', 'Mimic_95th', 'Padding_x10']

    def __init__(self, feature_stats: Dict):
        self.stats = feature_stats  # {feat_idx: {'mean': float, 'p95': float}}

    def apply_strategy(self, X: np.ndarray, feat_idx: int,
                       strategy: str) -> np.ndarray:
        X = X.copy()
        if strategy == 'Zero':
            X[:, :, feat_idx] = 0.0
        elif strategy == 'Mimic_Mean':
            X[:, :, feat_idx] = self.stats.get(feat_idx, {}).get('mean', 0.0)
        elif strategy == 'Mimic_95th':
            X[:, :, feat_idx] = self.stats.get(feat_idx, {}).get('p95', 1.0)
        elif strategy == 'Padding_x10':
            X[:, :, feat_idx] = np.clip(X[:, :, feat_idx] * 10, -5.0, 5.0)
        return X

    def generate(self, X: np.ndarray, top_vulnerable: List[Tuple[int, str]],
                 k_features: int = 3) -> np.ndarray:
        X_adv = X.copy()
        for feat_idx, strategy in top_vulnerable[:k_features]:
            X_adv = self.apply_strategy(X_adv, feat_idx, strategy)
        return X_adv

    def worst_of_k_batch(self, X: np.ndarray,
                         top_vulnerable: List[Tuple[int, str]],
                         k_values: List[int] = [1, 2, 3]) -> np.ndarray:
        candidates = [self.generate(X, top_vulnerable, k) for k in k_values]
        return candidates[-1]


# =============================================================================
# COUCHE 3 — AFD : ADVERSARIAL FEATURE DEFENSE
# =============================================================================

class AFDLoss(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int,
                 lambda_intra: float = 1.0, lambda_inter: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_intra = lambda_intra
        self.lambda_inter = lambda_inter
        self.register_buffer('centers', torch.zeros(num_classes, feature_dim))
        self.momentum = 0.9

    @torch.no_grad()
    def update_centers(self, features: torch.Tensor, labels: torch.Tensor):
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                feat_mean = features[mask].mean(0)
                # First time initialization: if the center is at the origin
                if (self.centers[c] == 0).all():
                    self.centers[c] = feat_mean
                else:
                    self.centers[c] = (self.momentum * self.centers[c]
                                       + (1 - self.momentum) * feat_mean)

    def forward(self, features: torch.Tensor, features_adv: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, p=2, dim=1)
        features_adv = F.normalize(features_adv, p=2, dim=1)
        
        self.update_centers(features.detach(), labels)
        centers_batch = self.centers[labels]
        intra_clean = (features - centers_batch).norm(dim=1).mean()
        intra_adv = (features_adv - centers_batch).norm(dim=1).mean()

        inter_loss = torch.tensor(0.0, device=features.device)
        if self.num_classes > 1:
            for i in range(self.num_classes):
                for j in range(i + 1, self.num_classes):
                    dist = (self.centers[i] - self.centers[j]).norm()
                    inter_loss += F.relu(1.0 - dist)
            inter_loss /= (self.num_classes * (self.num_classes - 1) / 2)

        return (self.lambda_intra * (intra_clean + intra_adv)
                - self.lambda_inter * inter_loss)


# =============================================================================
# COUCHE 4 — TRADES : robustesse gradient-based (boule Linf)
# =============================================================================

def trades_loss(model: nn.Module, x_nat: torch.Tensor, y: torch.Tensor,
                optimizer, epsilon: float, step_size: float,
                perturb_steps: int, beta: float,
                defense_layer: Optional[InputDefenseLayer] = None
                ) -> torch.Tensor:
    model.eval()
    x_adv = x_nat.detach() + 0.001 * torch.randn_like(x_nat)

    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        x_input = defense_layer(x_adv) if defense_layer else x_adv
        with torch.enable_grad():
            loss_kl = F.kl_div(
                F.log_softmax(model(x_input), dim=1),
                F.softmax(model(defense_layer(x_nat)
                               if defense_layer else x_nat), dim=1),
                reduction='batchmean'
            )
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x_nat + epsilon),
                          x_nat - epsilon).detach()

    model.train()
    x_input_nat = defense_layer(x_nat) if defense_layer else x_nat
    x_input_adv = defense_layer(x_adv) if defense_layer else x_adv

    loss_nat = F.cross_entropy(model(x_input_nat), y)
    loss_rob = F.kl_div(
        F.log_softmax(model(x_input_adv), dim=1),
        F.softmax(model(x_input_nat), dim=1),
        reduction='batchmean'
    )
    return loss_nat + beta * loss_rob


# =============================================================================
# COUCHE 5 — RÉGULARISATION DE DIVERSITÉ DES FEATURES
# =============================================================================

def feature_diversity_loss(model: nn.Module, x: torch.Tensor,
                           y: torch.Tensor,
                           top_k: int = 5) -> torch.Tensor:
    """Feature diversity regularisation via output-entropy concentration.

    Avoids double-backward (create_graph=True) which crashes LSTM/CuDNN on CUDA.
    Uses a single forward pass — fully differentiable w.r.t. model parameters.

    Penalises over-concentration on few output dimensions by measuring the
    entropy of the mean softmax distribution across the batch:
      - high entropy  → diverse predictions  → small penalty
      - low entropy   → collapsed predictions → large penalty
    Also adds a batch-variance term to encourage spread across samples.
    """
    # Single forward pass – no double-backward needed
    out = model(x)                                   # [B, C]
    probs = F.softmax(out, dim=1)                    # [B, C]

    # Mean distribution across batch
    mean_probs = probs.mean(dim=0)                   # [C]
    # Negative entropy (we want to maximise entropy → minimise neg-entropy)
    entropy = -(mean_probs * (mean_probs + 1e-8).log()).sum()
    max_entropy = torch.log(torch.tensor(float(out.shape[1]),
                                         device=out.device, dtype=out.dtype))
    # Concentration in [0, 1]: 0 = max diversity, 1 = fully collapsed
    concentration = 1.0 - entropy / (max_entropy + 1e-8)
    return concentration


# =============================================================================
# CURRICULUM D'ENTRAÎNEMENT — 4 PHASES
# =============================================================================

def get_phase_config(epoch: int) -> Dict:
    """
    Retourne la configuration de la phase courante.
    """
    if epoch <= 15:
        return {
            'phase': 0, 'name': 'Fondation propre',
            'epsilon': 0.0, 'trades_beta': 0.0, 'trades_steps': 0,
            'afd_lambda': 0.5, 'diversity_lambda': 0.05,
            'label_smoothing': 0.05, 'worst_k': 0, 'cutmix_prob': 0.0,
        }
    elif epoch <= 35:
        t = (epoch - 16) / 19
        eps = 0.01 + t * 0.04
        return {
            'phase': 1, 'name': 'Robustesse douce',
            'epsilon': eps, 'trades_beta': 1.0, 'trades_steps': 7,
            'step_size': eps / 4, 'afd_lambda': 0.2,
            'diversity_lambda': 0.10, 'label_smoothing': 0.08,
            'worst_k': 1, 'cutmix_prob': 0.2,
        }
    elif epoch <= 55:
        t = (epoch - 36) / 19
        eps = 0.05 + t * 0.05
        return {
            'phase': 2, 'name': 'Robustesse forte',
            'epsilon': eps, 'trades_beta': 2.0, 'trades_steps': 10,
            'step_size': eps / 4, 'afd_lambda': 0.1,
            'diversity_lambda': 0.15, 'label_smoothing': 0.10,
            'worst_k': 3, 'cutmix_prob': 0.3,
        }
    else:
        return {
            'phase': 3, 'name': 'Consolidation',
            'epsilon': 0.10, 'trades_beta': 2.0, 'trades_steps': 10,
            'step_size': 0.025, 'afd_lambda': 0.05,
            'diversity_lambda': 0.10, 'label_smoothing': 0.10,
            'worst_k': 3, 'cutmix_prob': 0.3,
        }

def combined_loss_normalized(losses: Dict[str, torch.Tensor],
                              weights: Dict[str, float]) -> torch.Tensor:
    total = sum(weights[k] * losses[k] for k in losses if k in weights)
    w_sum = sum(weights[k] for k in losses if k in weights)
    return total / (w_sum + 1e-8)


class AdversarialEarlyStopping:
    def __init__(self, max_gap: float = 0.60, max_adv_loss: float = 10.0,
                 patience: int = 5):
        self.max_gap = max_gap
        self.max_adv_loss = max_adv_loss
        self.patience = patience
        self._violations = 0

    def check(self, benign_acc: float, adv_acc: float,
              adv_loss: float, current_eps: float) -> Tuple[bool, float]:
        gap = benign_acc - adv_acc
        if gap > self.max_gap or adv_loss > self.max_adv_loss:
            self._violations += 1
            new_eps = current_eps * 0.7
            if self._violations >= self.patience:
                return True, new_eps
            return False, new_eps
        self._violations = 0
        return False, current_eps

def compute_feature_stats(X_train: np.ndarray) -> Dict[int, Dict[str, float]]:
    X_flat = X_train.reshape(-1, X_train.shape[-1])
    stats = {}
    for i in range(X_flat.shape[1]):
        stats[i] = {
            'mean': float(X_flat[:, i].mean()),
            'p95':  float(np.percentile(X_flat[:, i], 95)),
        }
    return stats

