"""
TRADES: TRadeoff-inspired Adversarial DEfenseS

Implementation of TRADES (Zhang et al., 2019) for IoT Device Identification.

Reference:
    "Theoretically Principled Trade-off between Robustness and Accuracy"
    https://arxiv.org/abs/1901.08573

Key idea:
    - Minimize standard loss on clean examples
    - Maximize consistency between predictions on clean and adversarial examples
    - Control trade-off with lambda parameter

Usage:
    trainer = TRADESTrainer(model, device, lambda_trades=6.0)
    history = trainer.fit(train_loader, val_loader, epochs=30)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import copy


class TRADESAttack:
    """
    PGD-like attack for TRADES: generates adversarial examples that maximize
    KL divergence from clean predictions.
    """

    def __init__(
        self,
        epsilon: float = 0.03,
        alpha: float = 0.01,
        num_steps: int = 10,
        projection_fn=None,
    ):
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.projection_fn = projection_fn

    def generate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate adversarial examples that maximize KL divergence from clean predictions.

        Args:
            model: the target model
            x: clean inputs (batch_size, seq_len, features)
            y: true labels (not used in TRADES attack, but kept for interface consistency)
            device: torch device

        Returns:
            x_adv: adversarial examples
        """
        was_training = model.training
        x_adv = x.clone().detach()

        # Random initialization within epsilon ball
        delta = torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        x_adv = x + delta

        # Get clean predictions (detach to avoid affecting gradients)
        with torch.no_grad():
            clean_logits = model(x)
            clean_probs = F.softmax(clean_logits, dim=1).detach()

        # PGD iterations require model in training mode for cuDNN RNN backward
        model.train()

        for _ in range(self.num_steps):
            x_adv.requires_grad_(True)

            adv_logits = model(x_adv)
            adv_log_probs = F.log_softmax(adv_logits, dim=1)

            # KL divergence: KL(clean || adv) = sum(clean * log(clean/adv))
            kl_loss = F.kl_div(adv_log_probs, clean_probs, reduction="batchmean")

            # Maximize KL (minimize negative)
            grad = torch.autograd.grad(-kl_loss, x_adv)[0]

            # Take step in gradient direction
            x_adv = x_adv.detach() + self.alpha * grad.sign()

            # Project back to epsilon ball around x
            delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = x + delta

            # Apply custom projection if provided (e.g., for feature constraints)
            if self.projection_fn is not None:
                x_adv_np = x_adv.cpu().numpy()
                x_adv_np = self.projection_fn(x_adv_np)
                x_adv = torch.FloatTensor(x_adv_np).to(device)

        if not was_training:
            model.eval()

        return x_adv.detach()


class TRADESTrainer:
    """
    TRADES Adversarial Training.

    Loss = CE(x, y) + lambda * KL(f(x) || f(x_adv))

    Args:
        model: neural network model
        device: torch device
        lambda_trades: trade-off parameter (default 6.0, typical range 1-10)
        epsilon: perturbation budget for adversarial examples
        alpha: step size for PGD
        num_steps: number of PGD steps
        projection_fn: optional function to project perturbed samples back to valid range
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lambda_trades: float = 6.0,
        epsilon: float = 0.05,
        alpha: float = 0.01,
        num_steps: int = 7,
        projection_fn=None,
        verbose: bool = False,
    ):
        self.model = model.to(device)
        self.device = device
        self.lambda_trades = lambda_trades
        self.verbose = verbose

        self.attack = TRADESAttack(
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps,
            projection_fn=projection_fn,
        )

        self.history = {
            "train_loss": [],
            "train_ce_loss": [],
            "train_kl_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "robust_acc": [],
        }

    def train_epoch(
        self,
        train_loader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epoch: int,
    ) -> Tuple[float, float, float, float]:
        """
        Train for one epoch using TRADES loss.

        Returns:
            (total_loss, ce_loss, kl_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_kl_loss = 0.0
        correct = 0
        total = 0

        desc = f"Epoch {epoch} [TRADES]"
        for batch_idx, (X_batch, y_batch) in enumerate(
            tqdm(train_loader, desc=desc, leave=False, disable=not self.verbose)
        ):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Generate adversarial examples
            with torch.no_grad():
                X_adv = self.attack.generate(self.model, X_batch, y_batch, self.device)

            optimizer.zero_grad()

            # Forward pass on clean examples
            clean_logits = self.model(X_batch)
            ce_loss = criterion(clean_logits, y_batch)

            # Forward pass on adversarial examples
            adv_logits = self.model(X_adv)

            # KL divergence: make predictions consistent
            # Detach clean_probs - gradients should only flow through adv_logits
            clean_probs = F.softmax(clean_logits, dim=1).detach()
            adv_log_probs = F.log_softmax(adv_logits, dim=1)
            kl_loss = F.kl_div(adv_log_probs, clean_probs, reduction="batchmean")

            # Total TRADES loss
            loss = ce_loss + self.lambda_trades * kl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_kl_loss += kl_loss.item()

            _, predicted = clean_logits.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()

        n = len(train_loader)
        return total_loss / n, total_ce_loss / n, total_kl_loss / n, correct / total

    def evaluate(
        self,
        dataloader,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """Evaluate on clean data."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in tqdm(
                dataloader, desc="Evaluating", leave=False, disable=not self.verbose
            ):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()

        return total_loss / len(dataloader), correct / total

    def evaluate_robust(
        self,
        dataloader,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """Evaluate on adversarial examples."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in tqdm(
            dataloader, desc="Robust Eval", leave=False, disable=not self.verbose
        ):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Generate adversarial examples
            X_adv = self.attack.generate(self.model, X_batch, y_batch, self.device)

            with torch.no_grad():
                outputs = self.model(X_adv)
                loss = criterion(outputs, y_batch)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()

        return total_loss / len(dataloader), correct / total

    def fit(
        self,
        train_loader,
        val_loader,
        test_loader=None,
        epochs: int = 30,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 7,
        save_path: str = None,
        use_class_weights: bool = False,
    ) -> Dict:
        """
        Full TRAINES training with early stopping.

        Args:
            train_loader: training dataloader
            val_loader: validation dataloader
            test_loader: optional test dataloader for robustness evaluation
            epochs: number of training epochs
            lr: learning rate
            weight_decay: weight decay for optimizer
            patience: early stopping patience
            save_path: path to save best model
            use_class_weights: whether to use class weights

        Returns:
            training history dict
        """
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )

        # Class weights
        if use_class_weights:
            y_all = []
            for _, yb in train_loader:
                y_all.extend(yb.numpy())
            y_all = np.array(y_all)
            classes = np.array(sorted(set(y_all)))
            from sklearn.utils.class_weight import compute_class_weight

            cw = compute_class_weight("balanced", classes=classes, y=y_all)
            class_weights = torch.FloatTensor(cw).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"  [TRADES] Class weights enabled: {cw[:5]}...")
        else:
            criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        patience_counter = 0

        print(f"\n{'=' * 60}")
        print(f"  TRADES Adversarial Training")
        print(f"{'=' * 60}")
        print(f"  Lambda: {self.lambda_trades}")
        print(f"  Epsilon: {self.attack.epsilon}")
        print(f"  PGD steps: {self.attack.num_steps}")
        print(f"  Epochs: {epochs}")
        print(f"  Early stopping patience: {patience}")
        print(f"{'=' * 60}\n")

        for epoch in range(1, epochs + 1):
            # Train
            train_loss, ce_loss, kl_loss, train_acc = self.train_epoch(
                train_loader, optimizer, criterion, epoch
            )

            # Validate
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            scheduler.step(val_loss)

            # Robustness evaluation (optional)
            robust_acc = 0.0
            if test_loader is not None and epoch % 5 == 0:
                _, robust_acc = self.evaluate_robust(test_loader, criterion)

            # Log history
            self.history["train_loss"].append(train_loss)
            self.history["train_ce_loss"].append(ce_loss)
            self.history["train_kl_loss"].append(kl_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["robust_acc"].append(robust_acc)

            # Print progress
            msg = (
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} (CE: {ce_loss:.4f}, KL: {kl_loss:.4f}) | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )
            if robust_acc > 0:
                msg += f" | Robust Acc: {robust_acc:.4f}"
            print(msg)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0

                if save_path:
                    import os

                    os.makedirs(save_path, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": best_state,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                            "lambda_trades": self.lambda_trades,
                        },
                        f"{save_path}/best_model_trades.pt",
                    )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        # Load best model
        self.model.load_state_dict(best_state)
        print(f"\n  Best model loaded (val_loss: {best_val_loss:.4f})")

        return self.history

    def test_robustness(
        self,
        test_loader,
        criterion: nn.Module,
        feature_attack=None,
        sequence_attack=None,
        sensitivity_results=None,
        eval_batch_size: int = 32,
    ) -> Dict:
        """
        Comprehensive robustness evaluation on test set.

        Returns metrics on:
        - Clean test data
        - TRADES adversarial examples (PGD)
        - Feature-level mimicry attack (if provided)
        - Sequence-level adversarial search (if provided)
        """
        self.model.eval()
        results = {}

        # 1. Clean accuracy
        clean_loss, clean_acc = self.evaluate(test_loader, criterion)
        results["clean"] = {
            "loss": clean_loss,
            "accuracy": clean_acc,
        }
        print(f"  [Clean] Loss: {clean_loss:.4f}, Acc: {clean_acc:.4f}")

        # 2. TRADES robust accuracy
        trades_loss, trades_acc = self.evaluate_robust(test_loader, criterion)
        results["trades_adversarial"] = {
            "loss": trades_loss,
            "accuracy": trades_acc,
            "robustness_ratio": trades_acc / max(clean_acc, 1e-8),
        }
        print(
            f"  [TRADES Adv] Loss: {trades_loss:.4f}, Acc: {trades_acc:.4f}, RR: {trades_acc / max(clean_acc, 1e-8):.4f}"
        )

        # 3. Feature-level attack (if provided)
        if feature_attack is not None:
            import numpy as np
            from torch.utils.data import DataLoader, Dataset

            # Get batch
            X_test, y_test = [], []
            for Xb, yb in test_loader:
                X_test.append(Xb.numpy())
                y_test.append(yb.numpy())
                if len(X_test) * test_loader.batch_size >= 1000:
                    break
            X_test = np.concatenate(X_test)[:1000]
            y_test = np.concatenate(y_test)[:1000]

            # Generate feature-level adversarial
            X_adv_feat = feature_attack.generate_batch(X_test, y_test)

            # Evaluate
            class SimpleDataset(Dataset):
                def __init__(self, X, y):
                    self.X = torch.FloatTensor(X)
                    self.y = torch.LongTensor(y)

                def __len__(self):
                    return len(self.X)

                def __getitem__(self, idx):
                    return self.X[idx], self.y[idx]

            adv_loader = DataLoader(
                SimpleDataset(X_adv_feat, y_test), batch_size=eval_batch_size
            )
            feat_loss, feat_acc = self.evaluate(adv_loader, criterion)

            results["feature_attack"] = {
                "loss": feat_loss,
                "accuracy": feat_acc,
                "robustness_ratio": feat_acc / max(clean_acc, 1e-8),
            }
            print(
                f"  [Feature Adv] Loss: {feat_loss:.4f}, Acc: {feat_acc:.4f}, RR: {feat_acc / max(clean_acc, 1e-8):.4f}"
            )

        # 4. Sequence-level attack (if provided)
        if sequence_attack is not None and sensitivity_results is not None:
            X_adv_seq = sequence_attack.generate_batch(
                X_test, y_test, sensitivity_results=sensitivity_results
            )
            adv_loader = DataLoader(
                SimpleDataset(X_adv_seq, y_test), batch_size=eval_batch_size
            )
            seq_loss, seq_acc = self.evaluate(adv_loader, criterion)

            results["sequence_attack"] = {
                "loss": seq_loss,
                "accuracy": seq_acc,
                "robustness_ratio": seq_acc / max(clean_acc, 1e-8),
            }
            print(
                f"  [Seq Adv] Loss: {seq_loss:.4f}, Acc: {seq_acc:.4f}, RR: {seq_acc / max(clean_acc, 1e-8):.4f}"
            )

        return results


class FeatureAttackGenerator:
    """
    Torch-native feature-level adversarial attack generator.

    Applies Zero / Mimic_Mean / Mimic_95th / Padding_x10 strategies
    directly on GPU tensors without numpy conversion.

    Unlike SensitivityAnalysis which operates per-feature with ranking,
    this class applies a chosen strategy to ALL modifiable features at once,
    producing an x_adv tensor compatible with the TRADES training loop.
    """

    VALID_STRATEGIES = ["Zero", "Mimic_Mean", "Mimic_95th", "Padding_x10"]

    def __init__(
        self,
        benign_stats: Dict[int, Dict[str, np.ndarray]],
        feature_names: List[str],
        num_classes: int,
        n_continuous_features: Optional[int] = None,
        non_modifiable: Optional[List[str]] = None,
        dependent_pairs: Optional[Dict[str, str]] = None,
        strategies: Optional[List[str]] = None,
    ):
        self.num_classes = num_classes
        self.n_continuous_features = n_continuous_features
        self.feature_names = feature_names
        self.strategies = strategies or self.VALID_STRATEGIES

        if non_modifiable is not None:
            self.non_modifiable = non_modifiable
        else:
            has_pkt_dir = any(f.startswith("pkt_dir_") for f in feature_names)
            if has_pkt_dir:
                self.non_modifiable = ["protocolIdentifier"] + [
                    f"pkt_dir_{i}" for i in range(8)
                ]
            else:
                self.non_modifiable = ["ipProto"]

        if dependent_pairs is not None:
            self.dependent_pairs = dependent_pairs
        else:
            has_pkt_dir = any(f.startswith("pkt_dir_") for f in feature_names)
            if has_pkt_dir:
                self.dependent_pairs = {
                    "reversePacketTotalCount": "packetTotalCount",
                    "reverseOctetTotalCount": "octetTotalCount",
                    "reverseAverageInterarrivalTime": "averageInterarrivalTime",
                }
            else:
                self.dependent_pairs = {
                    "inPacketCount": "outPacketCount",
                    "inByteCount": "outByteCount",
                    "inAvgIAT": "outAvgIAT",
                    "inAvgPacketSize": "outAvgPacketSize",
                }

        self.modifiable_indices = self._get_modifiable_indices()
        self.dependent_indices = self._get_dependent_indices()

        self._benign_means_np = {}
        self._benign_p95_np = {}
        for cls, stats in benign_stats.items():
            self._benign_means_np[cls] = stats["mean"].astype(np.float32)
            self._benign_p95_np[cls] = stats["p95"].astype(np.float32)

        self._benign_means_torch: Optional[torch.Tensor] = None
        self._benign_p95_torch: Optional[torch.Tensor] = None
        self._device: Optional[torch.device] = None

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

    def _ensure_torch_stats(self, device: torch.device):
        if self._device == device and self._benign_means_torch is not None:
            return
        means = np.zeros((self.num_classes, len(self.feature_names)), dtype=np.float32)
        p95s = np.zeros((self.num_classes, len(self.feature_names)), dtype=np.float32)
        for cls in range(self.num_classes):
            if cls in self._benign_means_np:
                means[cls] = self._benign_means_np[cls]
                p95s[cls] = self._benign_p95_np[cls]
        self._benign_means_torch = torch.from_numpy(means).to(device)
        self._benign_p95_torch = torch.from_numpy(p95s).to(device)
        self._mod_idx_torch = (
            torch.from_numpy(self.modifiable_indices).long().to(device)
        )
        self._device = device

    @torch.no_grad()
    def generate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        device: torch.device,
        strategy: str = "Mimic_Mean",
    ) -> torch.Tensor:
        """
        Generate adversarial examples by applying a feature-level strategy
        to ALL modifiable features simultaneously.

        Args:
            model: target model (used only for interface consistency)
            x: clean inputs (batch, seq_len, features) or (batch, features)
            y: true labels (batch,)
            device: torch device
            strategy: one of "Zero", "Mimic_Mean", "Mimic_95th", "Padding_x10"

        Returns:
            x_adv: adversarial examples, same shape as x
        """
        self._ensure_torch_stats(device)
        x_adv = x.clone()

        mod_idx = self._mod_idx_torch

        if strategy == "Zero":
            x_adv[..., mod_idx] = 0.0
        elif strategy == "Mimic_Mean":
            means = self._benign_means_torch[y]
            if x_adv.ndim == 3:
                means = means.unsqueeze(1)
            x_adv[..., mod_idx] = means[..., mod_idx]
        elif strategy == "Mimic_95th":
            p95s = self._benign_p95_torch[y]
            if x_adv.ndim == 3:
                p95s = p95s.unsqueeze(1)
            x_adv[..., mod_idx] = p95s[..., mod_idx]
        elif strategy == "Padding_x10":
            x_adv[..., mod_idx] = x_adv[..., mod_idx] * 10.0
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Valid: {self.VALID_STRATEGIES}"
            )

        x_adv = self._project(x_adv)

        return x_adv.detach()

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        """Project adversarial examples to valid feature ranges (torch-native)."""
        n_cont = self.n_continuous_features
        if n_cont is not None:
            x[..., :n_cont] = torch.clamp(x[..., :n_cont], -3.0, 3.0)
            x[..., n_cont:] = torch.clamp(torch.round(x[..., n_cont:]), 0.0, 1.0)
        else:
            x = torch.clamp(x, -3.0, 3.0)

        for indep_idx, dep_idx in self.dependent_indices:
            if self.n_continuous_features is not None and (
                dep_idx >= self.n_continuous_features
                or indep_idx >= self.n_continuous_features
            ):
                continue
            ratio = torch.abs(x[..., dep_idx]) / (torch.abs(x[..., indep_idx]) + 1e-8)
            x[..., dep_idx] = x[..., indep_idx] * torch.clamp(ratio, 0.5, 2.0)

        return x


class MultiAttackTRADES:
    """
    Worst-of-K adversarial attack generator for TRADES training.

    Generates adversarial examples using multiple attack methods (PGD + feature-level
    strategies) and selects the one that maximizes KL divergence from clean predictions.
    This ensures the model is trained against the strongest available attack at each step.

    Usage:
        multi_attack = MultiAttackTRADES(
            trades_attack=trades_attack,
            feature_attack=feature_attack,
            strategies=["Zero", "Mimic_Mean", "Mimic_95th", "Padding_x10"],
        )
        x_adv = multi_attack.generate(model, x, y, device)
        # x_adv is the adversarial example that maximizes KL(clean || adv)
    """

    def __init__(
        self,
        trades_attack: TRADESAttack,
        feature_attack: Optional[FeatureAttackGenerator] = None,
        strategies: Optional[List[str]] = None,
        projection_fn=None,
    ):
        self.trades_attack = trades_attack
        self.feature_attack = feature_attack
        self.strategies = strategies or FeatureAttackGenerator.VALID_STRATEGIES
        self.projection_fn = projection_fn
        self.selection_counts: Dict[str, int] = {}

    def generate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Generate adversarial examples using worst-of-K selection.

        Generates x_adv with each attack method, computes KL(clean || x_adv)
        for each, and returns the one that maximizes KL divergence.

        Args:
            model: target model
            x: clean inputs (batch, seq_len, features) or (batch, features)
            y: true labels (batch,)
            device: torch device

        Returns:
            x_adv_worst: the adversarial example that maximizes KL divergence
            batch_counts: dict mapping attack name -> count of samples where it won
        """
        was_training = model.training

        with torch.no_grad():
            clean_logits = model(x)
            clean_probs = F.softmax(clean_logits, dim=1).detach()

        candidates = {}

        pgd_adv = self.trades_attack.generate(model, x, y, device)
        candidates["PGD"] = pgd_adv

        if self.feature_attack is not None:
            for strat in self.strategies:
                feat_adv = self.feature_attack.generate(
                    model, x, y, device, strategy=strat
                )
                candidates[strat] = feat_adv

        best_kl = torch.full((x.size(0),), -float("inf"), device=device)
        best_adv = pgd_adv.clone()
        batch_counts = {name: 0 for name in candidates}

        model.eval()
        with torch.no_grad():
            for name, x_adv_i in candidates.items():
                adv_logits = model(x_adv_i)
                adv_log_probs = F.log_softmax(adv_logits, dim=1)
                kl_per_sample = F.kl_div(
                    adv_log_probs, clean_probs, reduction="none"
                ).sum(dim=1)

                mask = kl_per_sample > best_kl
                best_adv[mask] = x_adv_i[mask]
                best_kl[mask] = kl_per_sample[mask]

                batch_counts[name] = mask.sum().item()

        for name, count in batch_counts.items():
            self.selection_counts[name] = self.selection_counts.get(name, 0) + count

        if was_training:
            model.train()

        return best_adv.detach(), batch_counts

    def get_selection_stats(self) -> Dict[str, Dict[str, float]]:
        """Return selection frequency statistics for each attack."""
        total = sum(self.selection_counts.values()) or 1
        return {
            name: {
                "count": count,
                "frequency": count / total,
            }
            for name, count in self.selection_counts.items()
        }

    def reset_epoch_stats(self):
        """Reset selection counts for a new epoch."""
        self.selection_counts = {}


def create_trades_projection_fn(feature_attack, n_continuous_features: int):
    """
    Create a projection function for TRADES that respects feature constraints.

    This ensures adversarial examples stay within valid ranges:
    - Continuous features: clipped to [-3, 3] (standardized range)
    - Binary features: rounded to 0 or 1
    - Dependent features: maintain relationships
    """

    def projection_fn(X):
        return feature_attack.projection(X)

    return projection_fn
