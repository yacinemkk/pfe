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
