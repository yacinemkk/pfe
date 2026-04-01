"""
PHASE 5.3 & 7.1: Évaluation et Comparaison des Modèles
Per docs/important.md

Métriques:
- Globales: Macro F1-Score, Accuracy
- Par classe: Précision, Rappel, F1-Score
- Matrice de confusion
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns


class Evaluator:
    """
    Évaluateur pour les modèles IoT.

    Per docs/important.md §5.3:
    - Métriques globales: Macro F1-Score, Accuracy
    - Métriques par classe: Précision, Rappel, F1-Score
    - Matrice de confusion
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.results: Dict[str, Dict] = {}

    def predict(
        self, dataloader: torch.utils.data.DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
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

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        dataset_name: str = "test",
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Évalue le modèle sur un dataset.

        Returns:
            Dict avec accuracy, macro_f1, precision, recall
        """
        y_pred, y_true = self.predict(dataloader)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "macro_precision": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "macro_recall": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
        }

        self.results[dataset_name] = {
            "metrics": metrics,
            "y_true": y_true,
            "y_pred": y_pred,
        }

        if verbose:
            print(f"\n[{dataset_name.upper()}] Résultats:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Macro F1: {metrics['macro_f1']:.4f}")
            print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
            print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
            print(f"  Macro Recall: {metrics['macro_recall']:.4f}")

        return metrics

    def get_per_class_metrics(self, dataset_name: str = "test") -> pd.DataFrame:
        """
        Retourne les métriques par classe.

        Returns:
            DataFrame avec precision, recall, f1-score par classe
        """
        if dataset_name not in self.results:
            raise ValueError(f"Dataset {dataset_name} non évalué")

        y_true = self.results[dataset_name]["y_true"]
        y_pred = self.results[dataset_name]["y_pred"]

        labels = np.unique(np.concatenate([y_true, y_pred]))

        report = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )

        rows = []
        for label in labels:
            label_str = str(label)
            if label_str in report:
                if self.class_names and label < len(self.class_names):
                    class_name = self.class_names[label]
                else:
                    class_name = f"class_{label}"

                rows.append(
                    {
                        "class": class_name,
                        "precision": report[label_str]["precision"],
                        "recall": report[label_str]["recall"],
                        "f1_score": report[label_str]["f1-score"],
                        "support": report[label_str]["support"],
                    }
                )

        return pd.DataFrame(rows)

    def get_confusion_matrix(self, dataset_name: str = "test") -> np.ndarray:
        """Retourne la matrice de confusion."""
        if dataset_name not in self.results:
            raise ValueError(f"Dataset {dataset_name} non évalué")

        y_true = self.results[dataset_name]["y_true"]
        y_pred = self.results[dataset_name]["y_pred"]

        return confusion_matrix(y_true, y_pred)

    def plot_confusion_matrix(
        self,
        dataset_name: str = "test",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        normalize: bool = False,
    ) -> plt.Figure:
        """
        Affiche et sauvegarde la matrice de confusion.

        Args:
            dataset_name: Nom du dataset
            save_path: Chemin de sauvegarde
            figsize: Taille de la figure
            normalize: Normaliser par ligne
        """
        cm = self.get_confusion_matrix(dataset_name)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=figsize)

        n_classes = cm.shape[0]
        if self.class_names and len(self.class_names) == n_classes:
            labels = self.class_names
        else:
            labels = [f"C{i}" for i in range(n_classes)]

        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )

        ax.set_xlabel("Prédit")
        ax.set_ylabel("Vrai")
        ax.set_title(f"Matrice de Confusion - {dataset_name}")

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path / f"confusion_matrix_{dataset_name}.png", dpi=150)

        return fig

    def save_results(self, save_dir: str, dataset_name: str = "test"):
        """Sauvegarde les résultats dans des fichiers."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if dataset_name not in self.results:
            raise ValueError(f"Dataset {dataset_name} non évalué")

        metrics = self.results[dataset_name]["metrics"]
        with open(save_dir / f"metrics_{dataset_name}.json", "w") as f:
            json.dump(metrics, f, indent=2)

        per_class = self.get_per_class_metrics(dataset_name)
        per_class.to_csv(save_dir / f"per_class_{dataset_name}.csv", index=False)

        self.plot_confusion_matrix(dataset_name, save_dir, normalize=True)
        plt.close()


class ModelComparator:
    """
    Compare les performances de plusieurs modèles.

    Per docs/important.md §7.1:
    - Tableau comparatif
    - Graphiques F1-Score
    - Identification du meilleur modèle
    """

    def __init__(self):
        self.models_results: Dict[str, Dict] = {}

    def add_model_results(
        self,
        model_name: str,
        metrics: Dict[str, float],
        phase: str = "standard",
    ):
        """
        Ajoute les résultats d'un modèle.

        Args:
            model_name: Nom du modèle
            metrics: Dictionnaire des métriques
            phase: "standard" ou "adversarial"
        """
        if model_name not in self.models_results:
            self.models_results[model_name] = {}

        self.models_results[model_name][phase] = metrics

    def get_comparison_table(self) -> pd.DataFrame:
        """
        Génère un tableau comparatif.

        Returns:
            DataFrame avec comparaison par modèle et phase
        """
        rows = []

        for model_name, phases in self.models_results.items():
            row = {"model": model_name}

            if "standard" in phases:
                row["std_accuracy"] = phases["standard"].get("accuracy", 0)
                row["std_macro_f1"] = phases["standard"].get("macro_f1", 0)

            if "adversarial" in phases:
                row["adv_accuracy"] = phases["adversarial"].get("accuracy", 0)
                row["adv_macro_f1"] = phases["adversarial"].get("macro_f1", 0)

                if "standard" in phases:
                    std_f1 = phases["standard"].get("macro_f1", 0)
                    adv_f1 = phases["adversarial"].get("macro_f1", 0)
                    row["f1_drop"] = std_f1 - adv_f1
                    row["robustness_ratio"] = adv_f1 / std_f1 if std_f1 > 0 else 0

            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values("std_macro_f1", ascending=False)

        return df

    def plot_comparison(
        self,
        metric: str = "macro_f1",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """
        Graphique comparatif des modèles.

        Args:
            metric: Métrique à comparer
            save_path: Chemin de sauvegarde
        """
        df = self.get_comparison_table()

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(df))
        width = 0.35

        if "std_macro_f1" in df.columns:
            bars1 = ax.bar(
                x - width / 2,
                df["std_macro_f1"],
                width,
                label="Standard",
                color="steelblue",
            )

        if "adv_macro_f1" in df.columns:
            bars2 = ax.bar(
                x + width / 2,
                df["adv_macro_f1"],
                width,
                label="Adversarial",
                color="coral",
            )

        ax.set_xlabel("Modèle")
        ax.set_ylabel("Macro F1-Score")
        ax.set_title("Comparaison des Modèles - Macro F1-Score")
        ax.set_xticks(x)
        ax.set_xticklabels(df["model"], rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path / "model_comparison.png", dpi=150)

        return fig

    def get_best_model(self, phase: str = "standard") -> Tuple[str, float]:
        """
        Identifie le meilleur modèle.

        Args:
            phase: "standard" ou "adversarial"

        Returns:
            (model_name, f1_score)
        """
        best_model = None
        best_f1 = 0

        for model_name, phases in self.models_results.items():
            if phase in phases:
                f1 = phases[phase].get("macro_f1", 0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model_name

        return best_model, best_f1

    def save_report(self, save_dir: str):
        """Sauvegarde le rapport complet."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        comparison_table = self.get_comparison_table()
        comparison_table.to_csv(save_dir / "comparison_table.csv", index=False)

        self.plot_comparison(save_path=save_dir)
        plt.close()

        best_std = self.get_best_model("standard")
        best_adv = self.get_best_model("adversarial")

        report = {
            "best_standard_model": {
                "name": best_std[0],
                "macro_f1": best_std[1],
            },
            "best_adversarial_model": {
                "name": best_adv[0],
                "macro_f1": best_adv[1],
            },
            "num_models": len(self.models_results),
        }

        with open(save_dir / "comparison_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n[COMPARISON] Rapport sauvegardé dans {save_dir}")
        print(f"  Meilleur modèle (standard): {best_std[0]} (F1={best_std[1]:.4f})")
        print(f"  Meilleur modèle (adversarial): {best_adv[0]} (F1={best_adv[1]:.4f})")


class CrashTestEvaluator:
    """
    Évalue la robustesse face aux attaques adverses.

    Per docs/important.md §5.4 (Crash Test 1) et §6.3 (Crash Test 2):
    - Test 1: Données bénignes (réservées, jamais utilisées pour générer des attaques)
    - Test 2: Données adverses (générées à partir d'un sous-ensemble séparé du test)
    - Test 3: Mélange bénignes + adverses (reflète un trafic réel)
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.test_results: Dict[str, Dict] = {}

    def run_crash_test(
        self,
        X_clean: np.ndarray,
        y_clean: np.ndarray,
        X_adv: np.ndarray,
        y_adv: np.ndarray,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Exécute les 3 tests de crash.

        Args:
            X_clean: Données bénignes réservées (jamais utilisées pour l'attaque)
            y_clean: Labels des données bénignes
            X_adv: Données adverses (générées à partir d'un sous-ensemble séparé)
            y_adv: Labels des données adverses (correspondent aux sources d'attaque)
        """
        results = {}

        if verbose:
            print("\n[CRASH TEST] Exécution des 3 tests...")
            print(f"  Test 1 - Clean réservé: {len(X_clean):,} échantillons")
            print(f"  Test 2 - Adversarial: {len(X_adv):,} échantillons")
            print(
                f"  Test 3 - Mixte (trafic réel): {len(X_clean) // 2 + len(X_adv) // 2:,} échantillons"
            )

        results["test1_clean"] = self._evaluate_dataset(
            X_clean, y_clean, batch_size, "Test 1: Bénignes (réservées)", verbose
        )

        results["test2_adversarial"] = self._evaluate_dataset(
            X_adv, y_adv, batch_size, "Test 2: Adverses", verbose
        )

        n_clean_mix = min(len(X_clean) // 2, len(X_adv) // 2)
        n_adv_mix = min(len(X_adv) // 2, len(X_clean) // 2)
        X_mix = np.vstack([X_clean[:n_clean_mix], X_adv[:n_adv_mix]])
        y_mix = np.concatenate([y_clean[:n_clean_mix], y_adv[:n_adv_mix]])

        results["test3_mixed"] = self._evaluate_dataset(
            X_mix, y_mix, batch_size, "Test 3: Mixte (trafic réel)", verbose
        )

        if verbose:
            print("\n[CRASH TEST] Résultats:")
            print(
                f"  Test 1 (Bénignes réservées):    F1={results['test1_clean']['macro_f1']:.4f}"
            )
            print(
                f"  Test 2 (Adverses):              F1={results['test2_adversarial']['macro_f1']:.4f}"
            )
            print(
                f"  Test 3 (Mixte - trafic réel):   F1={results['test3_mixed']['macro_f1']:.4f}"
            )

            f1_drop = (
                results["test1_clean"]["macro_f1"]
                - results["test2_adversarial"]["macro_f1"]
            )
            print(f"\n  Chute F1 (Test1 → Test2): {f1_drop:.4f} ({f1_drop * 100:.1f}%)")

        self.test_results = results
        return results

    def _evaluate_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        name: str,
        verbose: bool,
    ) -> Dict[str, float]:
        """Évalue un dataset."""
        self.model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                X_batch = torch.FloatTensor(X[i : i + batch_size]).to(self.device)
                y_batch = torch.LongTensor(y[i : i + batch_size]).to(self.device)

                outputs = self.model(X_batch)
                _, predicted = outputs.max(1)

                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        accuracy = correct / total if total > 0 else 0
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        return {"accuracy": accuracy, "macro_f1": macro_f1}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.lstm import LSTMClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMClassifier(input_size=36, num_classes=5).to(device)

    X_test = np.random.randn(200, 10, 36).astype(np.float32)
    y_test = np.random.randint(0, 5, 200)

    from torch.utils.data import TensorDataset, DataLoader

    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=64)

    evaluator = Evaluator(model, device, class_names=["A", "B", "C", "D", "E"])
    metrics = evaluator.evaluate(test_loader, "test")
    print(f"\nPer-class metrics:\n{evaluator.get_per_class_metrics('test')}")
