"""
Data Preprocessing for IoT Device Identification — CSV Pipeline (IPFIX ML Instances)

Pipeline en 4 etapes (docs/pretraitement):

Etape 1: Filtrage et adaptation au SDN
  - Supprimer les adresses IP (source/destination) et ports (source/destination)
  - Conserver uniquement les attributs statistiques accessibles via les API SDN:
    * Temps moyen entre les arrivees (inter-arrival time)
    * Taille moyenne des paquets entrants/sortants
    * Protocole IP
  - Eliminer les doublons et les valeurs manquantes

Etape 2: Equilibrage et filtrage du bruit
  - Borderline-SMOTE pour suréchantillonner les classes minoritaires
  - Isolation Forest et Local Outlier Factor (LOF) pour supprimer les valeurs aberrantes

Etape 3: Selection hybride des caracteristiques
  - XGBoost pour evaluation preliminaire de l'importance
  - Test du Chi-carre (Chi2) pour pertinence statistique
  - Information Mutuelle pour dependances non lineaires

Etape 4: Normalisation (Min-Max uniquement)
  - Min-Max Scaling (0-1)
  - NOTE: Pas de standardisation après Min-Max (redondant)

Anti-Leakage Temporal Split (docs/general):
  1. Group flows by device label
  2. Sort each group chronologically by 'start' timestamp
  3. Per-device 80/20 temporal split
  4. Build sliding-window sequences INDEPENDENTLY on train and test
  5. Fit scalers ONLY on training data
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import BorderlineSMOTE
import xgboost as xgb
import pickle
import warnings
import sys

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
CATEGORICAL_FEATURES_CSV = ["ipProto"]

from config.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FEATURES_TO_KEEP,
    FEATURES_TO_DROP,
    LABEL_COLUMN,
    IOT_IPFIX_HOME_CLASSES,
    IOT_DEVICE_CLASSES,
    SEQUENCE_LENGTH,
    STRIDE,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
    MIN_SAMPLES_PER_CLASS,
)


class IoTDataProcessor:
    """CSV-native preprocessor for the IPFIX ML Instances dataset (18 IoT classes).

    Pipeline en 4 etapes:
    Etape 1: Filtrage SDN - supprimer IP/ports, garder caracteristiques statistiques
    Etape 2: Equilibrage (Borderline-SMOTE) + Filtrage bruit (Isolation Forest, LOF)
    Etape 3: Selection hybride des caracteristiques (XGBoost, Chi2, MI)
    Etape 4: Normalisation (MinMax) puis Standardisation

    Anti-leakage:
    - Groups flows by device
    - Sorts each device group by 'start' timestamp
    - Applies 80/20 temporal split per device
    - Creates sequences separately for train and test
    - Scalers fitted on training data only
    """

    def __init__(self):
        self.minmax_scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.selected_feature_indices = None
        self.num_classes = 0

    # ─── Step 1: Load data ───────────────────────────────────────────────────

    def load_all_data(self, max_files=None, data_dir=None):
        """Load all home*_labeled.csv files."""
        if data_dir is None:
            data_dir = RAW_DATA_DIR
        dfs = []
        csv_files = sorted(Path(data_dir).glob("home*_labeled.csv"))

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {data_dir}. "
                "Expected files matching: home*_labeled.csv"
            )

        if max_files:
            csv_files = csv_files[:max_files]

        for f in csv_files:
            print(f"  Loading {f.name}...")
            df = pd.read_csv(f)
            df["source_file"] = f.stem
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        print(f"  Total rows loaded: {len(combined):,} from {len(csv_files)} file(s)")
        return combined

    # ─── Etape 1: Filtrage SDN ──────────────────────────────────────────────

    def sdn_filter(self, df):
        """
        Etape 1: Filtrage et adaptation au SDN.

        Supprime les colonnes non accessibles via API SDN:
        - Adresses IP (source/destination)
        - Adresses MAC (source/destination)
        - Ports (source/destination)

        Conserve uniquement les caracteristiques statistiques:
        - Temps moyen entre arrivees (inter-arrival time)
        - Taille moyenne des paquets entrants/sortants
        - Protocole IP
        - Duree, compteurs de paquets/bytes
        """
        df = df.drop_duplicates()
        df = df.dropna(subset=[LABEL_COLUMN])

        # Colonnes a supprimer (IP, MAC, ports - non accessibles SDN)
        # NOTE: "start" is PRESERVED here for temporal splitting.
        # It is dropped AFTER the temporal split in process_all().
        sdn_excluded = [
            "srcMac",
            "destMac",
            "srcIP",
            "destIP",
            "srcPort",
            "destPort",
            "device",
        ]
        existing_drop = [c for c in sdn_excluded if c in df.columns]
        df = df.drop(columns=existing_drop, errors="ignore")

        # Conserver uniquement les classes IoT specifiees (18 classes)
        # Essayer d'abord les noms exacts, puis kebab-case
        valid_labels = set(IOT_IPFIX_HOME_CLASSES) | set(IOT_DEVICE_CLASSES)
        df = df[df[LABEL_COLUMN].isin(valid_labels)]

        # Supprimer les classes avec trop peu d'echantillons
        class_counts = df[LABEL_COLUMN].value_counts()
        valid_classes = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index
        df = df[df[LABEL_COLUMN].isin(valid_classes)].copy()

        print(
            f"  Apres filtrage SDN: {len(df):,} lignes | {len(valid_classes)} classes"
        )
        return df

    # ─── Etape 2: Equilibrage et filtrage du bruit ───────────────────────────

    def balance_and_filter_noise(self, X, y, contamination=0.05):
        """
        Etape 2: Equilibrage et filtrage du bruit.

        1. Borderline-SMOTE: suréchantillonnage des classes minoritaires
        2. Isolation Forest: detection des anomalies
        3. Local Outlier Factor (LOF): filtrage du bruit residual
        """
        print(f"  Avant equilibrage: {len(X):,} echantillons")

        # 2.1 Borderline-SMOTE pour equilibrer les classes
        print("  Application de Borderline-SMOTE...")
        try:
            smote = BorderlineSMOTE(
                kind="borderline-1", random_state=RANDOM_STATE, k_neighbors=5
            )
            X_resampled, y_resampled = smote.fit_resample(X, y)
            print(f"  Apres SMOTE: {len(X_resampled):,} echantillons")
        except Exception as e:
            print(f"  SMOTE echoue ({e}), conservation des donnees originales")
            X_resampled, y_resampled = X, y

        # 2.2 Isolation Forest pour detecter les anomalies
        print("  Application d'Isolation Forest...")
        iso_forest = IsolationForest(
            contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1
        )
        outliers_if = iso_forest.fit_predict(X_resampled)
        mask_if = outliers_if == 1  # 1 = inlier, -1 = outlier

        X_filtered = X_resampled[mask_if]
        y_filtered = y_resampled[mask_if]
        print(
            f"  Apres Isolation Forest: {len(X_filtered):,} echantillons "
            f"({np.sum(~mask_if):,} supprimes)"
        )

        # 2.3 Local Outlier Factor pour filtrage supplementaire
        print("  Application de Local Outlier Factor...")
        try:
            lof = LocalOutlierFactor(
                n_neighbors=20, contamination=contamination, n_jobs=-1
            )
            outliers_lof = lof.fit_predict(X_filtered)
            mask_lof = outliers_lof == 1

            X_final = X_filtered[mask_lof]
            y_final = y_filtered[mask_lof]
            print(
                f"  Apres LOF: {len(X_final):,} echantillons "
                f"({np.sum(~mask_lof):,} supprimes)"
            )
        except Exception as e:
            print(f"  LOF echoue ({e}), conservation des donnees filtrees")
            X_final, y_final = X_filtered, y_filtered

        return X_final, y_final

    # ─── Etape 3: Selection hybride des caracteristiques ─────────────────────

    @staticmethod
    def find_elbow_k(scores: np.ndarray) -> int:
        """Trouve le k optimal via la methode du coude (elbow).

        Calcule la distance perpendiculaire maximale entre chaque point
        de la courbe triee (descendant) et la ligne reliant le premier
        et le dernier point. Le point le plus eloigne est le coude.
        """
        n = len(scores)
        if n <= 2:
            return n

        sorted_scores = np.sort(scores)[::-1]
        x = np.arange(n, dtype=float)

        # Ligne du premier au dernier point
        p1 = np.array([x[0], sorted_scores[0]])
        p2 = np.array([x[-1], sorted_scores[-1]])

        # Distance perpendiculaire de chaque point a la ligne p1-p2
        line_vec = p2 - p1
        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq < 1e-12:
            return n // 2

        distances = np.zeros(n)
        for i in range(n):
            pt = np.array([x[i], sorted_scores[i]])
            distances[i] = abs(np.cross(line_vec, pt - p1)) / np.sqrt(line_len_sq)

        elbow_idx = int(np.argmax(distances))
        return max(1, elbow_idx + 1)

    def hybrid_feature_selection(self, X, y, feature_names, top_k=None):
        """
        Etape 3: Selection hybride des caracteristiques.

        1. XGBoost: importance des caracteristiques
        2. Chi2: pertinence statistique
        3. Information Mutuelle: dependances non lineaires

        Combine les trois scores pour selectionner les meilleures caracteristiques.
        Si top_k est None, utilise la methode du coude automatiquement.
        """
        n_features = X.shape[1]
        print(f"  Selection parmi {n_features} caracteristiques...")

        # 3.1 XGBoost Feature Importance
        print("  3.1 XGBoost importance...")
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="mlogloss",
            verbosity=0,
        )
        xgb_clf.fit(X, y)
        xgb_importance = xgb_clf.feature_importances_

        # 3.2 Chi2 test (besoin de valeurs positives)
        print("  3.2 Chi2 test...")
        X_positive = X - X.min() + 1e-6  # Rendre positif pour Chi2
        chi2_scores, _ = chi2(X_positive, y)
        chi2_scores = chi2_scores / (chi2_scores.max() + 1e-10)  # Normaliser

        # 3.3 Mutual Information
        print("  3.3 Information Mutuelle...")
        mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
        mi_scores = mi_scores / (mi_scores.max() + 1e-10)  # Normaliser

        # Combiner les scores (moyenne ponderee)
        xgb_norm = xgb_importance / (xgb_importance.max() + 1e-10)
        combined_scores = 0.4 * xgb_norm + 0.3 * chi2_scores + 0.3 * mi_scores

        # Determiner k : elbow method ou valeur fixee
        if top_k is None:
            top_k = self.find_elbow_k(combined_scores)
            print(f"  Elbow method → k={top_k} (sur {n_features} features)")
        else:
            top_k = min(top_k, n_features)
            print(f"  k fixe → {top_k}")

        # Enforce 15-20 feature cap per docs/featureselection
        top_k = max(15, min(20, top_k))
        print(
            f"  Cap enforced: k={top_k} (within 15-20 range per docs/featureselection)"
        )

        selected_indices = np.argsort(combined_scores)[-top_k:]
        selected_indices = np.sort(selected_indices)

        selected_features = [feature_names[i] for i in selected_indices]
        print(f"  {len(selected_indices)} caracteristiques selectionnees:")
        for i, idx in enumerate(selected_indices[-10:]):  # Afficher top 10
            print(f"    {i + 1}. {feature_names[idx]}: {combined_scores[idx]:.4f}")

        self.selected_feature_indices = selected_indices
        return selected_indices, selected_features

    # ─── Step 3: Per-device temporal 80/20 split ─────────────────────────────

    def temporal_split_per_device(self, df, train_ratio=0.8, val_ratio=0.0):
        """
        Group → sort by timestamp → split train/val/test per device.

        Returns (df_train, df_val, df_test). If val_ratio=0, df_val is None.
        'start' column is used for sorting; falls back to row order if absent.
        """
        train_parts, val_parts, test_parts = [], [], []
        timestamp_col = "start" if "start" in df.columns else None

        for device, group in df.groupby(LABEL_COLUMN, sort=False):
            if timestamp_col:
                group = group.sort_values(timestamp_col)

            n = len(group)
            split_idx_train = max(1, int(n * train_ratio))
            train_parts.append(group.iloc[:split_idx_train])

            if val_ratio > 0:
                split_idx_val = max(
                    split_idx_train + 1, int(n * (train_ratio + val_ratio))
                )
                val_parts.append(group.iloc[split_idx_train:split_idx_val])
                test_parts.append(group.iloc[split_idx_val:])
            else:
                test_parts.append(group.iloc[split_idx_train:])

        df_train = pd.concat(train_parts, ignore_index=True)
        df_val = pd.concat(val_parts, ignore_index=True) if val_parts else None
        df_test = pd.concat(test_parts, ignore_index=True)

        if df_val is not None:
            print(
                f"  Temporal split → Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}"
            )
        else:
            print(
                f"  Temporal split → Train: {len(df_train):,} | Test: {len(df_test):,}"
            )
        return df_train, df_val, df_test

    # ─── Step 4: Feature extraction ─────────────────────────────────────────

    def select_features(self, df):
        """Return feature matrices and label array, separating categorical features."""
        all_features = [c for c in FEATURES_TO_KEEP if c in df.columns]
        categorical_features = [
            f for f in CATEGORICAL_FEATURES_CSV if f in all_features
        ]
        continuous_features = [f for f in all_features if f not in categorical_features]

        X_continuous = df[continuous_features].fillna(0).values.astype(np.float32)
        X_categorical = (
            df[categorical_features].fillna(0).values.astype(np.float32)
            if categorical_features
            else np.empty((len(df), 0), dtype=np.float32)
        )
        y = df[LABEL_COLUMN].values

        self.feature_names = all_features
        self.continuous_feature_names = continuous_features
        self.categorical_feature_names = categorical_features
        return X_continuous, X_categorical, y

    # ─── Step 5: Sequence creation ───────────────────────────────────────────

    def create_sequences(self, X, y, seq_length=None, stride=None):
        """
        Create sliding-window sequences grouped by device label.

        Each sequence contains flows from the SAME device only.
        Generated separately on train and test data to prevent leakage.
        """
        if seq_length is None:
            seq_length = SEQUENCE_LENGTH
        if stride is None:
            stride = STRIDE

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

        return np.array(X_seq), np.array(y_seq)

    def create_sequences_with_categorical(
        self, X_continuous, X_categorical, y, seq_length=None, stride=None
    ):
        """
        Create sequences combining continuous and categorical features.
        Categorical features are kept as raw integers for BPE tokenization.
        Final feature order: [continuous | categorical]
        """
        if seq_length is None:
            seq_length = SEQUENCE_LENGTH
        if stride is None:
            stride = STRIDE

        X_combined = np.concatenate([X_continuous, X_categorical], axis=1)

        X_seq, y_seq = [], []
        unique_labels = np.unique(y)

        for label in unique_labels:
            mask = y == label
            X_group = X_combined[mask]
            y_group = y[mask]

            n = len(X_group) - seq_length + 1
            if n <= 0:
                continue
            for i in range(0, n, stride):
                if i + seq_length <= len(X_group):
                    X_seq.append(X_group[i : i + seq_length])
                    y_seq.append(y_group[i + seq_length - 1])

        return np.array(X_seq), np.array(y_seq)

    # ─── Main pipeline ───────────────────────────────────────────────────────

    def process_all(
        self,
        max_files=None,
        data_dir=None,
        save_path=None,
        seq_length=None,
        stride=None,
        apply_balancing=True,
        apply_feature_selection=True,
        top_k_features=None,
    ):
        """
            Full pipeline en 4 etapes avec anti-leakage:

            Etape 1: Filtrage SDN (supprimer IP/ports, garder caracteristiques statistiques)
            Etape 2: Equilibrage (Borderline-SMOTE) + Filtrage bruit (Isolation Forest, LOF)
            Etape 3: Selection hybride des caracteristiques (XGBoost, Chi2, MI)
        Etape 4: Normalisation (MinMax uniquement)

            Anti-leakage:
            - Per-device temporal 80/20 split (docs/general)
            - Sequences generees separement sur train et test
            - Scalers ajustes uniquement sur train

            Returns:
                (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if seq_length is None:
            seq_length = SEQUENCE_LENGTH
        if stride is None:
            stride = STRIDE

        print("=" * 70)
        print("CSV-NATIVE IPFIX ML INSTANCES PREPROCESSOR (Pipeline 4 Etapes)")
        print("=" * 70)

        # ─── Etape 1: Chargement et Filtrage SDN ──────────────────────────────
        print("\n[ETAPE 1] Filtrage et adaptation au SDN...")
        print("  1.1 Chargement des fichiers CSV...")
        df = self.load_all_data(max_files=max_files, data_dir=data_dir)

        print("  1.2 Filtrage SDN (suppression IP/MAC/ports)...")
        df = self.sdn_filter(df)

        # ─── Split temporel ANTI-LEAKAGE (train/val/test) ─────────────────────
        val_ratio_for_split = VAL_SIZE  # e.g. 0.1
        print("\n[PRE-SPLIT] Application du split temporel 72/18/10 par appareil...")
        df_train_raw, df_val_raw, df_test_raw = self.temporal_split_per_device(
            df,
            train_ratio=1.0 - TEST_SIZE - val_ratio_for_split,
            val_ratio=val_ratio_for_split,
        )
        del df

        # Drop 'start' now that temporal split is done (not needed as a feature)
        if "start" in df_train_raw.columns:
            df_train_raw = df_train_raw.drop(columns=["start"])
        if df_val_raw is not None and "start" in df_val_raw.columns:
            df_val_raw = df_val_raw.drop(columns=["start"])
        if "start" in df_test_raw.columns:
            df_test_raw = df_test_raw.drop(columns=["start"])

        # ─── Extraction des caracteristiques ─────────────────────────────────
        print("\n  Extraction des caracteristiques...")
        X_train_cont_raw, X_train_cat_raw, y_train_str = self.select_features(
            df_train_raw
        )
        X_val_cont_raw, X_val_cat_raw, y_val_str = (
            self.select_features(df_val_raw)
            if df_val_raw is not None
            else (None, None, None)
        )
        X_test_cont_raw, X_test_cat_raw, y_test_str = self.select_features(df_test_raw)
        del df_train_raw, df_test_raw
        if df_val_raw is not None:
            del df_val_raw

        # ─── Encodage des labels ─────────────────────────────────────────────
        print("\n  Encodage des labels...")
        self.label_encoder.fit(y_train_str)
        y_train_enc = self.label_encoder.transform(y_train_str)
        y_val_enc = (
            self.label_encoder.transform(y_val_str) if y_val_str is not None else None
        )
        y_test_enc = np.array(
            [
                self.label_encoder.transform([lbl])[0]
                if lbl in self.label_encoder.classes_
                else 0
                for lbl in y_test_str
            ]
        )
        self.num_classes = len(self.label_encoder.classes_)
        print(f"  Classes: {self.num_classes}")

        # ─── Etape 2: Equilibrage et Filtrage du Bruit (TRAIN UNIQUEMENT) ───
        if apply_balancing:
            print("\n[ETAPE 2] Equilibrage et filtrage du bruit (train only)...")
            X_train_cont_balanced, y_train_balanced = self.balance_and_filter_noise(
                X_train_cont_raw, y_train_enc, contamination=0.05
            )
        else:
            print("\n[ETAPE 2] Equilibrage desactive.")
            X_train_cont_balanced = X_train_cont_raw
            y_train_balanced = y_train_enc

        # ─── Etape 3: Selection Hybride des Caracteristiques ─────────────────
        if apply_feature_selection and self.continuous_feature_names:
            print("\n[ETAPE 3] Selection hybride des caracteristiques...")
            selected_indices, selected_features = self.hybrid_feature_selection(
                X_train_cont_balanced,
                y_train_balanced,
                self.continuous_feature_names,
                top_k=top_k_features,
            )
            X_train_cont_selected = X_train_cont_balanced[:, selected_indices]
            X_val_cont_selected = (
                X_val_cont_raw[:, selected_indices]
                if X_val_cont_raw is not None
                else None
            )
            X_test_cont_selected = X_test_cont_raw[:, selected_indices]
            self.continuous_feature_names = selected_features
            self.feature_names = selected_features + self.categorical_feature_names
        else:
            print("\n[ETAPE 3] Selection desactivee.")
            X_train_cont_selected = X_train_cont_balanced
            X_val_cont_selected = X_val_cont_raw
            X_test_cont_selected = X_test_cont_raw

        # ─── Etape 4: Normalisation (Min-Max uniquement) ───────────────────────
        # IMPORTANT: Categorical features (ipProto) are NOT scaled.
        # They remain as raw integers for the BPE tokenizer to convert to
        # human-readable labels (tcp, udp, icmp, etc.).
        print(
            "\n[ETAPE 4] Normalisation Min-Max (fit sur train only, continuous only)..."
        )
        print("  4.1 Min-Max Scaling (0-1) — continuous features only...")
        X_train_cont_scaled = self.minmax_scaler.fit_transform(
            X_train_cont_selected
        ).astype(np.float32)
        X_val_cont_scaled = (
            self.minmax_scaler.transform(X_val_cont_selected).astype(np.float32)
            if X_val_cont_selected is not None
            else None
        )
        X_test_cont_scaled = self.minmax_scaler.transform(X_test_cont_selected).astype(
            np.float32
        )

        del (
            X_train_cont_raw,
            X_val_cont_raw,
            X_test_cont_raw,
            X_train_cont_balanced,
            X_train_cont_selected,
            X_val_cont_selected,
            X_test_cont_selected,
        )

        # ─── Creation des sequences (SEPARATEMENT pour train, val, test) ──────
        print(f"\n[SEQUENCES] Creation (length={seq_length}, stride={stride})...")
        X_train_seq, y_train_seq = self.create_sequences_with_categorical(
            X_train_cont_scaled, X_train_cat_raw, y_train_balanced, seq_length, stride
        )
        X_val_seq, y_val_seq = (
            self.create_sequences_with_categorical(
                X_val_cont_scaled, X_val_cat_raw, y_val_enc, seq_length, stride
            )
            if X_val_cont_scaled is not None
            else (np.array([]), np.array([]))
        )
        X_test_seq, y_test_seq = self.create_sequences_with_categorical(
            X_test_cont_scaled, X_test_cat_raw, y_test_enc, seq_length, stride
        )
        del (
            X_train_cont_scaled,
            X_val_cont_scaled,
            X_test_cont_scaled,
            X_train_cat_raw,
            X_val_cat_raw,
            X_test_cat_raw,
        )
        print(
            f"  Train sequences: {len(X_train_seq):,} | Val sequences: {len(X_val_seq):,} | Test sequences: {len(X_test_seq):,}"
        )

        # ─── Sauvegarde ─────────────────────────────────────────────────────
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)

            np.save(save_path / "X_train.npy", X_train_seq)
            np.save(save_path / "X_val.npy", X_val_seq)
            np.save(save_path / "X_test.npy", X_test_seq)
            np.save(save_path / "y_train.npy", y_train_seq)
            np.save(save_path / "y_val.npy", y_val_seq)
            np.save(save_path / "y_test.npy", y_test_seq)

            with open(save_path / "preprocessor.pkl", "wb") as f:
                pickle.dump(
                    {
                        "minmax_scaler": self.minmax_scaler,
                        "scaler": self.scaler,
                        "label_encoder": self.label_encoder,
                        "feature_names": self.feature_names,
                        "selected_feature_indices": self.selected_feature_indices,
                        "num_classes": self.num_classes,
                    },
                    f,
                )
            print(f"\n  Donnees sauvegardees dans {save_path}")

        return (
            X_train_seq,
            X_val_seq,
            X_test_seq,
            y_train_seq,
            y_val_seq,
            y_test_seq,
            self.feature_names,
            self.scaler,
            self.label_encoder,
        )


if __name__ == "__main__":
    processor = IoTDataProcessor()
    processor.process_all(save_path=PROCESSED_DATA_DIR)
