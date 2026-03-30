"""
PHASE 5: Entraînement - Phase 1 (Standard)
Per docs/important.md

Configuration d'Entraînement:
- Loss: CrossEntropyLoss
- Optimiseur: Adam ou AdamW
- Learning rate scheduler
- Early stopping
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
import yaml
import json


class IoTSequenceDataset(Dataset):
    """Dataset pour les séquences IoT."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EarlyStopping:
    """Early stopping pour éviter le surapprentissage."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class Trainer:
    """
    Entraîneur pour les modèles IoT.

    Per docs/important.md §5.1:
    - CrossEntropyLoss
    - Adam/AdamW
    - Learning rate scheduler
    - Early stopping
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        config_path: str = "config/config.yaml",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        training_config = self.config["training"]
        self.batch_size = training_config.get("batch_size", 64)
        self.num_epochs = training_config.get("num_epochs", 30)
        self.learning_rate = training_config.get("learning_rate", 1e-3)
        self.weight_decay = training_config.get("weight_decay", 1e-4)
        self.patience = training_config.get("early_stopping_patience", 5)
        self.min_delta = training_config.get("early_stopping_min_delta", 0.001)

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """Entraîne une époque."""
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        for X_batch, y_batch in tqdm(dataloader, desc="Training", leave=False):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()

        return total_loss / len(dataloader), correct / total

    def evaluate(
        self, dataloader: DataLoader, criterion: nn.Module
    ) -> Tuple[float, float]:
        """Évalue le modèle."""
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()

        return total_loss / len(dataloader), correct / total

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
        save_path: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Entraîne le modèle.

        Args:
            train_loader: DataLoader d'entraînement
            val_loader: DataLoader de validation
            epochs: Nombre d'époques
            lr: Learning rate
            weight_decay: Décroissance des poids
            save_path: Chemin de sauvegarde
            verbose: Afficher les informations

        Returns:
            Historique d'entraînement
        """
        epochs = epochs or self.num_epochs
        lr = lr or self.learning_rate
        weight_decay = weight_decay or self.weight_decay

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        criterion = nn.CrossEntropyLoss()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5
        )

        early_stopping = EarlyStopping(
            patience=self.patience, min_delta=self.min_delta, mode="max"
        )

        best_val_acc = 0

        if verbose:
            print(f"\n[TRAINING] Entraînement sur {self.device}")
            print(
                f"  Epochs: {epochs}, LR: {lr}, Batch size: {train_loader.batch_size}"
            )

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.evaluate(val_loader, criterion)

            scheduler.step(val_acc)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if save_path:
                    self.save_checkpoint(save_path, epoch, optimizer, val_acc)

            if early_stopping(val_acc):
                if verbose:
                    print(f"  Early stopping à l'époque {epoch + 1}")
                break

        return self.history

    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Prédictions sur un dataset."""
        self.model.eval()
        predictions, labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                _, predicted = outputs.max(1)

                predictions.extend(predicted.cpu().numpy())
                labels.extend(y_batch.numpy())

        return np.array(predictions), np.array(labels)

    def predict_proba(self, dataloader: DataLoader) -> np.ndarray:
        """Probabilités de prédiction."""
        self.model.eval()
        probabilities = []

        with torch.no_grad():
            for X_batch, _ in dataloader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                probs = torch.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())

        return np.array(probabilities)

    def save_checkpoint(
        self,
        save_path: str,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        val_acc: float,
    ):
        """Sauvegarde un checkpoint."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            },
            save_path / "best_model.pt",
        )

    def load_checkpoint(self, checkpoint_path: str):
        """Charge un checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    def save_history(self, save_path: str):
        """Sauvegarde l'historique."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / "history.json", "w") as f:
            json.dump(self.history, f)


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Crée les DataLoaders."""
    train_dataset = IoTSequenceDataset(X_train, y_train)
    val_dataset = IoTSequenceDataset(X_val, y_val)
    test_dataset = IoTSequenceDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import numpy as np
    from lstm import LSTMClassifier

    X_train = np.random.randn(1000, 10, 36).astype(np.float32)
    y_train = np.random.randint(0, 5, 1000)
    X_val = np.random.randn(200, 10, 36).astype(np.float32)
    y_val = np.random.randint(0, 5, 200)
    X_test = np.random.randn(200, 10, 36).astype(np.float32)
    y_test = np.random.randint(0, 5, 200)

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64
    )

    model = LSTMClassifier(input_size=36, num_classes=5)
    trainer = Trainer(model)

    history = trainer.fit(train_loader, val_loader, epochs=5)
    print(f"\nBest val accuracy: {max(history['val_acc']):.4f}")
