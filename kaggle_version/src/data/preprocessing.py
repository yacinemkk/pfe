"""
PHASE 1.2-1.5: Prétraitement des Données
Per docs/important.md

Etapes:
1.2: Filtrage Orienté SDN
1.3: Nettoyage
1.4: Gestion du Déséquilibre des Classes
1.5: Normalisation et Standardisation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import BorderlineSMOTE
import yaml
import warnings

warnings.filterwarnings("ignore")


class Preprocessor:
    """Prétraitement des données IoT selon le pipeline 4 étapes."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names: Optional[List[str]] = None
        self.num_classes: int = 0

    def sdn_filter(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Etape 1.2: Filtrage Orienté SDN.

        Supprime les colonnes interdites:
        - IP source, IP destination
        - MAC source, MAC destination
        - Port source, Port destination

        Conserve uniquement:
        - Inter-arrival time
        - Taille moyenne paquets entrants/sortants
        - Type protocole IP
        - Durée flux
        - Nombre paquets
        """
        df = df.copy()

        sdn_excluded = self.config["data"]["sdn_excluded_columns"]
        existing_drop = [c for c in sdn_excluded if c in df.columns]
        df = df.drop(columns=existing_drop, errors="ignore")

        protocols_to_filter = self.config["data"].get("protocols_to_filter", [])
        if "ipProto" in df.columns and protocols_to_filter:
            proto_map = {1: "ICMP", 17: "UDP", 6: "TCP"}
            if "ARP" in protocols_to_filter and "arp" in df.columns:
                df = df[df["arp"] == 0]
            if "DHCP" in protocols_to_filter and "dhcp" in df.columns:
                df = df[df["dhcp"] == 0]
            if "ICMP" in protocols_to_filter and "ipProto" in df.columns:
                df = df[df["ipProto"] != 1]

        if verbose:
            print(f"  Après filtrage SDN: {len(df):,} lignes")

        return df

    def clean(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Etape 1.3: Nettoyage.

        - Supprime les doublons
        - Gère les valeurs manquantes
        - Valide l'intégrité des données
        """
        initial_len = len(df)

        df = df.drop_duplicates()
        df = df.dropna(subset=[self.config["data"]["label_column"]])

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        if verbose:
            print(
                f"  Nettoyage: {initial_len:,} → {len(df):,} lignes "
                f"({initial_len - len(df):,} supprimées)"
            )

        return df

    def filter_classes(
        self,
        df: pd.DataFrame,
        valid_classes: Optional[List[str]] = None,
        min_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Filtre les classes selon les paramètres.
        """
        label_col = self.config["data"]["label_column"]

        if valid_classes is None:
            valid_classes = self.config["data"]["iot_ipfix_home_classes"]

        valid_labels_lower = {c.lower() for c in valid_classes}
        df = df[df[label_col].str.lower().isin(valid_labels_lower)]

        if min_samples is None:
            min_samples = self.config["data"]["min_samples_per_class"]

        class_counts = df[label_col].value_counts()
        valid_classes_final = class_counts[class_counts >= min_samples].index
        df = df[df[label_col].isin(valid_classes_final)]

        if verbose:
            print(f"  Classes retenues: {len(valid_classes_final)}")
            print(f"  Lignes: {len(df):,}")

        return df

    def balance_classes(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = "smote",
        contamination: float = 0.05,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Etape 1.4: Gestion du Déséquilibre des Classes.

        Options:
        - smote: Borderline-SMOTE pour suréchantillonnage
        - undersample: Sous-échantillonnage des classes majoritaires
        - none: Pas d'équilibrage
        """
        if verbose:
            print(f"  Avant équilibrage: {len(X):,} échantillons")

        if method == "smote":
            X_resampled, y_resampled = self._apply_smote(X, y, verbose)
        elif method == "undersample":
            X_resampled, y_resampled = self._apply_undersample(X, y, verbose)
        else:
            X_resampled, y_resampled = X, y

        X_filtered, y_filtered = self._remove_outliers(
            X_resampled, y_resampled, contamination, verbose
        )

        if verbose:
            print(f"  Après équilibrage: {len(X_filtered):,} échantillons")

        return X_filtered, y_filtered

    def _apply_smote(
        self, X: np.ndarray, y: np.ndarray, verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Applique Borderline-SMOTE."""
        try:
            smote = BorderlineSMOTE(
                kind="borderline-1",
                random_state=self.config["split"]["random_state"],
                k_neighbors=5,
            )
            X_resampled, y_resampled = smote.fit_resample(X, y)
            if verbose:
                print(f"    SMOTE: {len(X):,} → {len(X_resampled):,}")
        except Exception as e:
            if verbose:
                print(f"    SMOTE échoué ({e}), conservation données originales")
            X_resampled, y_resampled = X, y

        return X_resampled, y_resampled

    def _apply_undersample(
        self, X: np.ndarray, y: np.ndarray, verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sous-échantillonne les classes majoritaires."""
        unique, counts = np.unique(y, return_counts=True)
        min_count = counts.min()

        indices = []
        for cls in unique:
            cls_indices = np.where(y == cls)[0]
            if len(cls_indices) > min_count:
                cls_indices = np.random.choice(cls_indices, min_count, replace=False)
            indices.extend(cls_indices)

        if verbose:
            print(f"    Undersample: {len(X):,} → {len(indices):,}")

        return X[indices], y[indices]

    def _remove_outliers(
        self, X: np.ndarray, y: np.ndarray, contamination: float, verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Supprime les outliers avec Isolation Forest et LOF."""
        if verbose:
            print("    Isolation Forest...")

        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=self.config["split"]["random_state"],
            n_jobs=-1,
        )
        outliers_if = iso_forest.fit_predict(X)
        mask_if = outliers_if == 1

        X_filtered = X[mask_if]
        y_filtered = y[mask_if]

        if verbose:
            print(
                f"      Après IF: {len(X_filtered):,} ({np.sum(~mask_if):,} supprimés)"
            )
            print("    Local Outlier Factor...")

        try:
            lof = LocalOutlierFactor(
                n_neighbors=20, contamination=contamination, n_jobs=-1
            )
            outliers_lof = lof.fit_predict(X_filtered)
            mask_lof = outliers_lof == 1

            X_final = X_filtered[mask_lof]
            y_final = y_filtered[mask_lof]

            if verbose:
                print(
                    f"      Après LOF: {len(X_final):,} ({np.sum(~mask_lof):,} supprimés)"
                )
        except Exception as e:
            if verbose:
                print(f"      LOF échoué ({e})")
            X_final, y_final = X_filtered, y_filtered

        return X_final, y_final

    def normalize(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        fit: bool = True,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Etape 1.5: Normalisation StandardScaler.

        IMPORTANT: Calculer les paramètres UNIQUEMENT sur train,
        appliquer sur train ET test.
        """
        if verbose:
            print("  StandardScaler (moyenne=0, variance=1)...")

        if fit:
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled.astype(np.float32), X_test_scaled.astype(np.float32)

    def encode_labels(
        self, y_train: np.ndarray, y_test: np.ndarray, verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode les labels en entiers."""
        y_train_enc = self.label_encoder.fit_transform(y_train)

        y_test_enc = np.array(
            [
                self.label_encoder.transform([lbl])[0]
                if lbl in self.label_encoder.classes_
                else -1
                for lbl in y_test
            ]
        )

        self.num_classes = len(self.label_encoder.classes_)

        if verbose:
            print(f"  {self.num_classes} classes encodées")

        return y_train_enc, y_test_enc

    def extract_features(
        self, df: pd.DataFrame, verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extrait les features et labels du DataFrame."""
        features = self.config["data"]["features_to_keep"]
        available_features = [f for f in features if f in df.columns]

        X = df[available_features].fillna(0).values.astype(np.float32)
        y = df[self.config["data"]["label_column"]].values

        self.feature_names = available_features

        if verbose:
            print(f"  Features: {len(available_features)} | Samples: {len(X):,}")

        return X, y

    def process(
        self, df: pd.DataFrame, apply_balancing: bool = True, verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pipeline complet de prétraitement.

        Returns:
            X, y: Features et labels prétraités
        """
        if verbose:
            print("\n[PREPROCESSING] Pipeline 4 étapes")

        if verbose:
            print("\n[ETAPE 1.2] Filtrage SDN...")
        df = self.sdn_filter(df, verbose)

        if verbose:
            print("\n[ETAPE 1.3] Nettoyage...")
        df = self.clean(df, verbose)

        if verbose:
            print("\n[Filtrage classes]")
        df = self.filter_classes(df, verbose=verbose)

        if verbose:
            print("\n[Extraction features]")
        X, y = self.extract_features(df, verbose)

        return X, y


if __name__ == "__main__":
    from loader import DataLoader

    loader = DataLoader()
    df = loader.load_iot_ipfix_home(max_files=2)

    preprocessor = Preprocessor()
    X, y = preprocessor.process(df)
    print(f"\nOutput: X={X.shape}, y={y.shape}")
