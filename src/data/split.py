"""
PHASE 2: Pipeline Anti-Data Leakage
Per docs/important.md

Contraintes CRITIQUES:
1. Pas de chevauchement entre train et test
2. Pas de mélange temporel (train = passé, test = futur)
3. Pas de mélange entre appareils dans une même séquence
4. Normalisation calculée sur train, appliquée sur test
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import yaml


class TemporalSplitter:
    """
    Split temporel anti-data leakage.

    Garantit:
    - Train = Passé
    - Test = Futur
    - Aucun chevauchement
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.test_size = self.config["split"]["test_size"]
        self.val_size = self.config["split"]["val_size"]
        self.label_column = self.config["data"]["label_column"]

    def group_by_device(self, df: pd.DataFrame) -> dict:
        """
        Etape 2.1: Regroupement par Appareil.

        Groupe les flux par identifiant appareil (label ou MAC).
        """
        groups = {}
        for device, group in df.groupby(self.label_column, sort=False):
            groups[device] = group.copy()

        print(f"  {len(groups)} groupes d'appareils créés")
        return groups

    def sort_chronologically(self, groups: dict, timestamp_col: str = "start") -> dict:
        """
        Etape 2.2: Tri Chronologique.

        Pour chaque groupe: trier par flow_start (ordre croissant).
        """
        sorted_groups = {}

        for device, group in groups.items():
            if timestamp_col in group.columns:
                sorted_groups[device] = group.sort_values(timestamp_col)
            else:
                sorted_groups[device] = group.reset_index(drop=True)

        print(f"  Groupes triés chronologiquement")
        return sorted_groups

    def split_per_device(
        self, groups: dict, train_ratio: float = 0.8, verbose: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Etape 2.3: Split Temporel 80/20.

        Pour chaque appareil: premiers 80% → Train, derniers 20% → Test.
        Garantit: Train = Passé, Test = Futur.
        """
        train_parts = []
        test_parts = []

        for device, group in groups.items():
            n = len(group)
            split_idx = max(1, int(n * train_ratio))

            train_parts.append(group.iloc[:split_idx])
            test_parts.append(group.iloc[split_idx:])

        df_train = pd.concat(train_parts, ignore_index=True)
        df_test = pd.concat(test_parts, ignore_index=True)

        if verbose:
            print(
                f"  Split temporel → Train: {len(df_train):,} | Test: {len(df_test):,}"
            )
            self._validate_no_overlap(df_train, df_test)

        return df_train, df_test

    def _validate_no_overlap(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        """Vérifie qu'il n'y a pas de chevauchement."""
        if "start" in df_train.columns and "start" in df_test.columns:
            train_max = df_train["start"].max()
            test_min = df_test["start"].min()

            if train_max >= test_min:
                print(
                    f"  ⚠️ Attention: chevauchement possible "
                    f"(train max: {train_max}, test min: {test_min})"
                )
            else:
                print(f"  ✓ Pas de chevauchement temporel")

    def temporal_split(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "start",
        train_ratio: float = 0.8,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Pipeline complet de split temporel.

        Args:
            df: DataFrame à splitter
            timestamp_col: Colonne de timestamp
            train_ratio: Ratio pour le train (défaut: 0.8)
            verbose: Afficher les informations

        Returns:
            df_train, df_test: DataFrames de train et test
        """
        if verbose:
            print("\n[ANTI-LEAKAGE] Split temporel par appareil")

        groups = self.group_by_device(df)
        sorted_groups = self.sort_chronologically(groups, timestamp_col)
        df_train, df_test = self.split_per_device(sorted_groups, train_ratio, verbose)

        return df_train, df_test

    def validate_integrity(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> bool:
        """
        Valide l'intégrité du split anti-leakage.

        Vérifie:
        1. Pas de chevauchement temporel
        2. Train contient le passé
        3. Test contient le futur
        """
        is_valid = True

        if "start" in df_train.columns and "start" in df_test.columns:
            train_max = df_train.groupby(self.label_column)["start"].max()
            test_min = df_test.groupby(self.label_column)["start"].min()

            for device in train_max.index:
                if device in test_min.index:
                    if train_max[device] >= test_min[device]:
                        print(f"  ❌ Leakage détecté pour {device}")
                        is_valid = False

        if is_valid:
            print("  ✓ Intégrité anti-leakage validée")

        return is_valid


def split_temporal(
    df: pd.DataFrame, config_path: str = "config/config.yaml", **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fonction utilitaire pour le split temporel."""
    splitter = TemporalSplitter(config_path)
    return splitter.temporal_split(df, **kwargs)


if __name__ == "__main__":
    from loader import DataLoader

    loader = DataLoader()
    df = loader.load_iot_ipfix_home(max_files=2)

    splitter = TemporalSplitter()
    df_train, df_test = splitter.temporal_split(df)

    print(f"\nTrain: {len(df_train):,} | Test: {len(df_test):,}")
    splitter.validate_integrity(df_train, df_test)
