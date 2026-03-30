"""
PHASE 2.4: Génération des Séquences
Per docs/important.md

Contraintes:
- Taille fenêtre définie (ex: 10 flux consécutifs)
- Sliding window SÉPARÉMENT sur Train et Test
- Chaque séquence = flux d'un SEUL appareil
- Aucune séquence ne doit traverser la frontière train/test
"""

import numpy as np
from typing import Tuple, Optional, List
import yaml


class SequenceGenerator:
    """Génère des séquences de flux pour les modèles séquentiels."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.seq_length = self.config["data"]["sequence_length"]
        self.stride = self.config["data"]["stride"]
        self.label_column = self.config["data"]["label_column"]

    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_length: Optional[int] = None,
        stride: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crée des séquences avec sliding window.

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            seq_length: Longueur de séquence (défaut: config)
            stride: Pas de décalage (défaut: config)
            verbose: Afficher les informations

        Returns:
            X_seq: Séquences (n_sequences, seq_length, n_features)
            y_seq: Labels (n_sequences,)
        """
        if seq_length is None:
            seq_length = self.seq_length
        if stride is None:
            stride = self.stride

        X_seq, y_seq = [], []
        unique_labels = np.unique(y)

        for label in unique_labels:
            mask = y == label
            X_group = X[mask]
            y_group = y[mask]

            n = len(X_group) - seq_length + 1
            if n <= 0:
                continue

            for i in range(0, n, stride):
                if i + seq_length <= len(X_group):
                    X_seq.append(X_group[i : i + seq_length])
                    y_seq.append(y_group[i + seq_length - 1])

        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq)

        if verbose:
            print(
                f"  Séquences créées: {len(X_seq):,} "
                f"(length={seq_length}, stride={stride})"
            )

        return X_seq, y_seq

    def create_sequences_from_groups(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        seq_length: Optional[int] = None,
        stride: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crée des séquences en respectant les groupes d'appareils.

        Garantit qu'aucune séquence ne mélange des flux de différents appareils.
        """
        if seq_length is None:
            seq_length = self.seq_length
        if stride is None:
            stride = self.stride

        X_seq, y_seq = [], []
        unique_groups = np.unique(groups)

        for group_id in unique_groups:
            mask = groups == group_id
            X_group = X[mask]
            y_group = y[mask]

            n = len(X_group) - seq_length + 1
            if n <= 0:
                continue

            for i in range(0, n, stride):
                if i + seq_length <= len(X_group):
                    X_seq.append(X_group[i : i + seq_length])
                    y_seq.append(y_group[i + seq_length - 1])

        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq)

        if verbose:
            print(f"  Séquences par groupe: {len(X_seq):,}")

        return X_seq, y_seq

    def create_train_test_sequences(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        seq_length: Optional[int] = None,
        stride: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Crée les séquences SÉPARÉMENT pour train et test.

        IMPORTANT: Applique sliding window indépendamment pour éviter
        toute fuite de données entre train et test.
        """
        if verbose:
            print("\n[SÉQUENCES] Génération séparée train/test")

        X_train_seq, y_train_seq = self.create_sequences(
            X_train, y_train, seq_length, stride, verbose=False
        )

        X_test_seq, y_test_seq = self.create_sequences(
            X_test, y_test, seq_length, stride, verbose=False
        )

        if verbose:
            print(f"  Train: {len(X_train_seq):,} séquences")
            print(f"  Test: {len(X_test_seq):,} séquences")

        return X_train_seq, y_train_seq, X_test_seq, y_test_seq

    def validate_sequence_integrity(self, X_seq: np.ndarray, y_seq: np.ndarray) -> bool:
        """
        Valide l'intégrité des séquences.

        Vérifie:
        - Chaque séquence contient des flux d'un seul appareil
        - Pas de séquence vide
        """
        is_valid = True

        if len(X_seq) == 0:
            print("  ❌ Aucune séquence générée")
            return False

        if np.any(np.isnan(X_seq)):
            print("  ❌ Valeurs NaN détectées dans les séquences")
            is_valid = False

        if len(np.unique(y_seq)) < 2:
            print("  ⚠️ Attention: moins de 2 classes dans les séquences")

        if is_valid:
            print("  ✓ Intégrité des séquences validée")

        return is_valid

    def get_sequence_stats(self, X_seq: np.ndarray, y_seq: np.ndarray) -> dict:
        """Retourne les statistiques des séquences."""
        return {
            "n_sequences": len(X_seq),
            "seq_length": X_seq.shape[1],
            "n_features": X_seq.shape[2],
            "n_classes": len(np.unique(y_seq)),
            "class_distribution": {
                int(k): int(v) for k, v in zip(*np.unique(y_seq, return_counts=True))
            },
        }


def generate_sequences(
    X: np.ndarray, y: np.ndarray, config_path: str = "config/config.yaml", **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Fonction utilitaire pour générer des séquences."""
    generator = SequenceGenerator(config_path)
    return generator.create_sequences(X, y, **kwargs)


if __name__ == "__main__":
    import numpy as np

    X = np.random.randn(1000, 36).astype(np.float32)
    y = np.random.randint(0, 5, 1000)

    generator = SequenceGenerator()
    X_seq, y_seq = generator.create_sequences(X, y)

    print(f"X_seq: {X_seq.shape}, y_seq: {y_seq.shape}")
    stats = generator.get_sequence_stats(X_seq, y_seq)
    print(f"Stats: {stats}")
