"""
PHASE 1.1: Chargement et Exploration des Datasets
Per docs/important.md

Charge les datasets:
- IoT IPFIX Home (47 jours, 12 foyers, 24 types)
- IPFIX Records (3 mois, 26 appareils, 9M+ enregistrements)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import yaml
import json


class DataLoader:
    """Charge et explore les datasets IoT IPFIX."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.base_dir = Path(self.config["paths"]["base_dir"])
        self.raw_data_dir = self.base_dir / self.config["paths"]["raw_data_dir"]
        self.label_column = self.config["data"]["label_column"]

    def load_iot_ipfix_home(
        self, max_files: Optional[int] = None, verbose: bool = True
    ) -> pd.DataFrame:
        """
        Charge le dataset IoT IPFIX Home (home*_labeled.csv).

        Args:
            max_files: Nombre max de fichiers à charger (None = tous)
            verbose: Afficher les informations de chargement

        Returns:
            DataFrame combiné de tous les fichiers
        """
        csv_files = sorted(self.raw_data_dir.glob("home*_labeled.csv"))

        if not csv_files:
            raise FileNotFoundError(
                f"Aucun fichier CSV trouvé dans {self.raw_data_dir}. "
                f"Attendu: home*_labeled.csv"
            )

        if max_files:
            csv_files = csv_files[:max_files]

        dfs = []
        for f in csv_files:
            if verbose:
                print(f"  Chargement: {f.name}")
            df = pd.read_csv(f)
            df["source_file"] = f.stem
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)

        if verbose:
            print(
                f"  Total: {len(combined):,} lignes depuis {len(csv_files)} fichier(s)"
            )
            self._print_dataset_info(combined)

        return combined

    def load_ipfix_records(
        self, data_dir: Optional[str] = None, verbose: bool = True
    ) -> pd.DataFrame:
        """
        Charge le dataset IPFIX Records (JSON).

        Args:
            data_dir: Répertoire des données JSON
            verbose: Afficher les informations de chargement

        Returns:
            DataFrame des enregistrements IPFIX
        """
        if data_dir is None:
            data_dir = self.base_dir / "data/pcap/IPFIX Records (UNSW IoT Analytics)"
        else:
            data_dir = Path(data_dir)

        json_files = list(data_dir.glob("**/*.json"))

        if not json_files:
            raise FileNotFoundError(f"Aucun fichier JSON trouvé dans {data_dir}")

        dfs = []
        for f in json_files:
            if verbose:
                print(f"  Chargement: {f.name}")
            with open(f, "r") as file:
                data = json.load(file)
            if isinstance(data, list):
                dfs.append(pd.DataFrame(data))
            elif isinstance(data, dict):
                dfs.append(pd.DataFrame([data]))

        combined = pd.concat(dfs, ignore_index=True)

        if verbose:
            print(f"  Total: {len(combined):,} enregistrements")

        return combined

    def analyze_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Analyse la distribution des classes et identifie le déséquilibre.

        Args:
            df: DataFrame à analyser

        Returns:
            Dictionnaire avec les statistiques de distribution
        """
        label_col = self.label_column
        if label_col not in df.columns:
            possible_labels = ["label", "name", "device_label", "device_class"]
            for col in possible_labels:
                if col in df.columns:
                    label_col = col
                    break

        class_counts = df[label_col].value_counts()

        stats = {
            "num_classes": len(class_counts),
            "total_samples": len(df),
            "class_distribution": class_counts.to_dict(),
            "min_samples": class_counts.min(),
            "max_samples": class_counts.max(),
            "imbalance_ratio": class_counts.max() / class_counts.min()
            if class_counts.min() > 0
            else float("inf"),
            "minority_classes": class_counts[
                class_counts < class_counts.median()
            ].index.tolist(),
        }

        return stats

    def get_flow_attributes(self, df: pd.DataFrame) -> List[str]:
        """
        Documente les attributs de flux disponibles.

        Args:
            df: DataFrame à analyser

        Returns:
            Liste des attributs de flux
        """
        attributes = df.columns.tolist()
        print(f"\nAttributs de flux disponibles ({len(attributes)}):")
        for i, attr in enumerate(attributes, 1):
            print(f"  {i}. {attr}")
        return attributes

    def _print_dataset_info(self, df: pd.DataFrame):
        """Affiche les informations du dataset."""
        print(f"\n  Shape: {df.shape}")
        print(f"  Colonnes: {len(df.columns)}")

        label_col = self.label_column
        if label_col in df.columns:
            unique_labels = df[label_col].nunique()
            print(f"  Classes uniques: {unique_labels}")

        if "start" in df.columns:
            print(f"  Timestamps: {df['start'].min()} à {df['start'].max()}")


def load_data(config_path: str = "config/config.yaml", **kwargs) -> pd.DataFrame:
    """Fonction utilitaire pour charger les données."""
    loader = DataLoader(config_path)
    return loader.load_iot_ipfix_home(**kwargs)


if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_iot_ipfix_home(max_files=2)
    stats = loader.analyze_distribution(df)
    print(f"\nDéséquilibre ratio: {stats['imbalance_ratio']:.2f}")
