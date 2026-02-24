"""
LSTM Model for IoT Device Identification
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import *


class IoTSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, num_classes, config=None):
        super().__init__()

        config = config or LSTM_CONFIG
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        bidirectional = config["bidirectional"]
        dropout = config["dropout"]

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)

        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        output = self.classifier(hidden)
        return output


class Trainer:
    def __init__(self, model, device="auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        for X_batch, y_batch in tqdm(dataloader, desc="Training", leave=False):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()

        return total_loss / len(dataloader), correct / total

    def evaluate(self, dataloader, criterion):
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
        train_loader,
        val_loader,
        epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        save_path=None,
    ):

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3
        )

        best_val_acc = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.evaluate(val_loader, criterion)

            scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if save_path:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_acc": val_acc,
                        },
                        Path(save_path) / "best_model.pt",
                    )

        return self.history

    def predict(self, dataloader):
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


def train_lstm(data_path, save_path, epochs=NUM_EPOCHS):
    data_path = Path(data_path)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    X_train = np.load(data_path / "X_train.npy")
    X_val = np.load(data_path / "X_val.npy")
    X_test = np.load(data_path / "X_test.npy")
    y_train = np.load(data_path / "y_train.npy")
    y_val = np.load(data_path / "y_val.npy")
    y_test = np.load(data_path / "y_test.npy")

    input_size = X_train.shape[2]
    num_classes = len(np.unique(y_train))

    print(f"Input size: {input_size}, Classes: {num_classes}")

    train_dataset = IoTSequenceDataset(X_train, y_train)
    val_dataset = IoTSequenceDataset(X_val, y_val)
    test_dataset = IoTSequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = LSTMClassifier(input_size, num_classes)
    trainer = Trainer(model)

    print(f"\nTraining on: {trainer.device}")
    history = trainer.fit(train_loader, val_loader, epochs=epochs, save_path=save_path)

    test_loss, test_acc = trainer.evaluate(test_loader, nn.CrossEntropyLoss())
    print(f"\nTest Accuracy: {test_acc:.4f}")

    with open(save_path / "history.json", "w") as f:
        json.dump(history, f)

    results = {
        "model": "LSTM",
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "best_val_accuracy": max(history["val_acc"]),
        "num_classes": num_classes,
        "input_size": input_size,
    }

    with open(save_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return model, history


if __name__ == "__main__":
    train_lstm(PROCESSED_DATA_DIR, RESULTS_DIR / "models" / "lstm")
