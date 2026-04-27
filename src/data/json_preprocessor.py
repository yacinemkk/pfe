"""
JSON-Native Preprocessor for IPFIX Records Dataset

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

Etape 4: Normalisation (StandardScaler)
  - StandardScaler (moyenne=0, ecart-type=1)

17 classes cibles (IPFIX Records):
  Qrio Hub, Philips Hue Light Bulb, Planex Pan–Tilt Camera 1, JVC Kenwood Camera,
  Planex Pan–Tilt Camera 2, Google Home, Apple HomePod, Sony Bravia TV,
  Wansview Camera, Qwatch Camera, Fredi Camera, Planex Outdoor Camera,
  Powerlec Wi-Fi Plug, LINE Clova Speaker, Sony Smart Speaker,
  Amazon Echo, Amazon Echo Show
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import BorderlineSMOTE
import xgboost as xgb
import pickle
import gc
import warnings
import sys

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import (
    DATA_DIR,
    PROCESSED_DATA_DIR,
    IPFIX_RECORDS_CLASSES,
    SEQUENCE_LENGTH,
    STRIDE,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
    MIN_SAMPLES_PER_CLASS,
)

CATEGORICAL_FEATURES_JSON = ["protocolIdentifier"]

# ─── MAC-to-Device Mapping pour IPFIX Records ────────────────────────────────
# Primary source: config/config.yaml (mac_mapping section)
# Fallback: hardcoded dict below (kept for backwards compatibility)


def _load_mac_mapping_from_config() -> dict:
    """Load MAC-to-device mapping from config/config.yaml."""
    import yaml

    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            mapping = cfg.get("mac_mapping", {}).get("devices", {})
            if mapping:
                return mapping
        except Exception:
            pass
    return MAC_TO_DEVICE_FALLBACK


MAC_TO_DEVICE_FALLBACK = {
    "38:d5:47:0c:25:d4": "DEFAULT GATEWAY",
    "80:c5:f2:0b:aa:a9": "Qrio Hub",
    "00:17:88:47:20:f2": "Philips Hue Light Bulb",
    "e0:b9:4d:b9:eb:e9": "Planex Pan–Tilt Camera 1",
    "e0:b9:4d:5c:cf:c5": "JVC Kenwood Camera",
    "e0:b9:4d:5c:cf:c6": "Planex Pan–Tilt Camera 2",
    "d8:6c:63:47:54:dc": "Google Home",
    "d4:90:9c:da:0d:f0": "Apple HomePod",
    "04:5d:4b:a4:d0:2e": "Sony Bravia TV",
    "e0:09:bf:54:68:47": "Wansview Camera",
    "34:76:c5:7f:91:07": "Qwatch Camera",
    "20:32:33:86:f7:0f": "Fredi Camera",
    "00:22:cf:fd:c1:08": "Planex Outdoor Camera",
    "ec:f0:0e:55:25:39": "Powerlec Wi-Fi Plug",
    "a8:1e:84:e8:cc:c3": "LINE Clova Speaker",
    "6c:5a:b5:56:39:3e": "Sony Smart Speaker",
    "4c:ef:c0:17:e0:42": "Amazon Echo",
    "14:0a:c5:f1:e5:52": "Amazon Echo Show",
}

MAC_TO_DEVICE = _load_mac_mapping_from_config()
GATEWAY_MAC = MAC_TO_DEVICE.get("gateway", "38:d5:47:0c:25:d4")

# 17 classes cibles pour IPFIX Records (docs/pretraitement)
TARGET_CLASSES = [
    "Qrio Hub",
    "Philips Hue Light Bulb",
    "Planex Pan–Tilt Camera 1",
    "JVC Kenwood Camera",
    "Planex Pan–Tilt Camera 2",
    "Google Home",
    "Apple HomePod",
    "Sony Bravia TV",
    "Wansview Camera",
    "Qwatch Camera",
    "Fredi Camera",
    "Planex Outdoor Camera",
    "Powerlec Wi-Fi Plug",
    "LINE Clova Speaker",
    "Sony Smart Speaker",
    "Amazon Echo",
    "Amazon Echo Show",
]

# ─── Issue 4: Columns to drop (prevent data leakage) ────────────────────────

COLUMNS_TO_DROP = [
    "sourceMacAddress",
    "destinationMacAddress",
    "sourceIPv4Address",
    "destinationIPv4Address",
    "sourceTransportPort",
    "destinationTransportPort",
    "tcpSequenceNumber",
    "reverseTcpSequenceNumber",
    "collectorName",
    "observationDomainId",
    "vlanId",
    "ingressInterface",
    "egressInterface",
    "flowAttributes",
    "reverseFlowAttributes",
    "flowStartMilliseconds",  # used for ordering, then dropped
    "flowEndMilliseconds",
    "flowEndReason",
    "silkAppLabel",
    "ipClassOfService",
    "active_timeout",
    "idle_timeout",
]

# ─── Issue 4: Behavioral features to keep ────────────────────────────────────

FEATURES_TO_KEEP_JSON = [
    # --- Temps et Protocoles ---
    "flowDurationMilliseconds",
    "protocolIdentifier",
    # --- Métriques Globales ---
    "packetTotalCount",
    "octetTotalCount",
    "reversePacketTotalCount",
    "reverseOctetTotalCount",
    # --- Drapeaux TCP (visibles par OpenFlow) ---
    "initialTCPFlags",
    "unionTCPFlags",
    "reverseInitialTCPFlags",
    "reverseUnionTCPFlags",
    # --- Métriques Détaillées Forward ---
    "tcpUrgTotalCount",
    "smallPacketCount",
    "nonEmptyPacketCount",
    "dataByteCount",
    "averageInterarrivalTime",
    "firstNonEmptyPacketSize",
    "largePacketCount",
    "maxPacketSize",
    "standardDeviationPayloadLength",
    "standardDeviationInterarrivalTime",
    "bytesPerPacket",
    # --- Métriques Détaillées Reverse ---
    "reverseTcpUrgTotalCount",
    "reverseSmallPacketCount",
    "reverseNonEmptyPacketCount",
    "reverseDataByteCount",
    "reverseAverageInterarrivalTime",
    "reverseFirstNonEmptyPacketSize",
    "reverseLargePacketCount",
    "reverseMaxPacketSize",
    "reverseStandardDeviationPayloadLength",
    "reverseStandardDeviationInterarrivalTime",
]

# Remove duplicates while preserving order
FEATURES_TO_KEEP_JSON = list(dict.fromkeys(FEATURES_TO_KEEP_JSON))

# Packet direction columns (binary, decoded from hex)
PKT_DIR_COLS = [f"pkt_dir_{i}" for i in range(8)]

# Total feature count: 28 continuous + 8 binary = 36
N_CONTINUOUS = len(FEATURES_TO_KEEP_JSON)
N_BINARY = 8
N_TOTAL_FEATURES = N_CONTINUOUS + N_BINARY

# ─── Default JSON data directory ─────────────────────────────────────────────

JSON_DATA_DIR = DATA_DIR / "pcap" / "IPFIX Records (UNSW IoT Analytics)"


# ─── Issue 5: Hex decoding ───────────────────────────────────────────────────


def decode_packet_directions(hex_str) -> list:
    """
    Decode hex-encoded packet directions into 8-element binary list.

    Args:
        hex_str: Hex string like "02", "0e", "ff", or None/NaN

    Returns:
        List of 8 integers (0 or 1), MSB first.
        1 = outbound, 0 = inbound
    """
    if hex_str is None or (isinstance(hex_str, float) and np.isnan(hex_str)):
        return [0] * 8

    hex_str = str(hex_str).strip()
    if not hex_str:
        return [0] * 8

    # Zero-pad to 2 chars if needed
    hex_str = hex_str.zfill(2)

    try:
        value = int(hex_str, 16)
    except ValueError:
        return [0] * 8

    return [(value >> (7 - i)) & 1 for i in range(8)]


# ─── Issue 3: Bidirectional flow labeling ────────────────────────────────────


def label_flow(flow: dict, mac_to_device: dict = None) -> str:
    """
    Label a flow record using dual-MAC lookup.

    Checks both sourceMacAddress and destinationMacAddress to correctly
    label bidirectional flows. Prefers non-gateway IoT device.

    Args:
        flow: Dict with 'sourceMacAddress' and 'destinationMacAddress'
        mac_to_device: MAC-to-device name mapping (defaults to MAC_TO_DEVICE)

    Returns:
        Device name string, or None if no IoT device found
    """
    if mac_to_device is None:
        mac_to_device = MAC_TO_DEVICE

    src_mac = flow.get("sourceMacAddress", "").strip().rstrip(":")
    dst_mac = flow.get("destinationMacAddress", "").strip().rstrip(":")

    src_device = mac_to_device.get(src_mac)
    dst_device = mac_to_device.get(dst_mac)

    # Skip flows internal to gateway only
    if src_device == "DEFAULT GATEWAY" and dst_device == "DEFAULT GATEWAY":
        return None

    # Prefer the non-gateway IoT device
    if src_device and src_device != "DEFAULT GATEWAY":
        return src_device
    if dst_device and dst_device != "DEFAULT GATEWAY":
        return dst_device

    # Neither MAC is a known IoT device
    return None


# ─── Main Preprocessor Class ─────────────────────────────────────────────────


class JsonIoTDataProcessor:
    """
    Native JSON preprocessor for the IPFIX Records dataset.

    Pipeline en 4 etapes:
    Etape 1: Filtrage SDN - supprimer IP/ports, garder caracteristiques statistiques
    Etape 2: Equilibrage (Borderline-SMOTE) + Filtrage bruit (Isolation Forest, LOF)
    Etape 3: Selection hybride des caracteristiques (XGBoost, Chi2, MI)
    Etape 4: Normalisation (StandardScaler per docs/pretraitement)

    17 classes cibles pour IPFIX Records.
    """

    def __init__(self):
        self.standard_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.selected_feature_indices = None
        self.num_classes = 0
        self.continuous_feature_names = list(FEATURES_TO_KEEP_JSON)
        self.categorical_feature_names = list(CATEGORICAL_FEATURES_JSON)
        self.binary_feature_names = list(PKT_DIR_COLS)

    def load_json_files(
        self, data_dir: Path = None, chunk_size: int = 100_000, max_records: int = None
    ) -> pd.DataFrame:
        """
        Stream-parse JSON files line-by-line and return labeled DataFrame.

        Each line is a JSON object: {"flows": {...}}

        Args:
            data_dir: Directory containing JSON subdirectories
            chunk_size: Number of records to accumulate before creating DataFrame chunk
            max_records: Maximum total records to load (None = all)
        """
        if data_dir is None:
            data_dir = JSON_DATA_DIR
        else:
            data_dir = Path(data_dir)

        # Find all JSON files
        json_files = sorted(data_dir.rglob("*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"No JSON files found in {data_dir}. "
                f"Expected files like ipfix_*.json in subdirectories."
            )

        print(f"Found {len(json_files)} JSON file(s):")
        for f in json_files:
            size_gb = f.stat().st_size / (1024**3)
            print(f"  {f.relative_to(data_dir)} ({size_gb:.1f} GB)")

        all_chunks = []
        total_records = 0
        total_labeled = 0
        total_unlabeled = 0

        for json_file in json_files:
            print(f"\nProcessing {json_file.name}...")
            chunk_records = []

            with open(json_file, "r") as fh:
                for line_num, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Extract the flows dict
                    flow = record.get("flows", record)

                    # Issue 3: Label using bidirectional MAC lookup
                    device_label = label_flow(flow)
                    if device_label is None:
                        total_unlabeled += 1
                        continue

                    # Extract behavioral features
                    row = {}
                    for feat in FEATURES_TO_KEEP_JSON:
                        val = flow.get(feat, 0)
                        try:
                            row[feat] = float(val) if val is not None else 0.0
                        except (ValueError, TypeError):
                            row[feat] = 0.0

                    # Issue 5: Decode hex packet directions
                    hex_dirs = flow.get("firstEightNonEmptyPacketDirections", "00")
                    bits = decode_packet_directions(hex_dirs)
                    for i, bit in enumerate(bits):
                        row[f"pkt_dir_{i}"] = bit

                    # Temporal ordering key
                    row["flow_start"] = flow.get("flowStartMilliseconds", "")

                    # Label
                    row["label"] = device_label

                    chunk_records.append(row)
                    total_labeled += 1
                    total_records += 1

                    # Flush chunk
                    if len(chunk_records) >= chunk_size:
                        chunk_df = pd.DataFrame(chunk_records)
                        all_chunks.append(chunk_df)
                        print(
                            f"  Loaded {total_records:,} records "
                            f"({total_labeled:,} labeled, {total_unlabeled:,} skipped)..."
                        )
                        chunk_records = []
                        gc.collect()

                    # Check max records
                    if max_records and total_records >= max_records:
                        break

            # Flush remaining records
            if chunk_records:
                chunk_df = pd.DataFrame(chunk_records)
                all_chunks.append(chunk_df)

            if max_records and total_records >= max_records:
                print(f"  Reached max_records limit ({max_records:,})")
                break

        if not all_chunks:
            raise ValueError(
                "No labeled records found. Check MAC mapping and JSON structure."
            )

        # Concatenate all chunks
        print(f"\nConcatenating {len(all_chunks)} chunks...")
        df = pd.concat(all_chunks, ignore_index=True)
        del all_chunks
        gc.collect()

        print(f"\nTotal records loaded: {len(df):,}")
        print(f"  Labeled: {total_labeled:,}")
        print(f"  Skipped (no matching MAC): {total_unlabeled:,}")
        print(f"  Classes found: {df['label'].nunique()}")
        print(f"\nClass distribution:")
        print(df["label"].value_counts().to_string())

        return df

    def filter_classes(self, df: pd.DataFrame, min_samples: int = None) -> pd.DataFrame:
        """Filter to only keep the 17 target classes specified for the project."""
        initial_samples = len(df)
        df = df[df["label"].isin(TARGET_CLASSES)].copy()
        print(
            f"\nKept {df['label'].nunique()} target classes, {len(df):,} samples "
            f"(filtered out {initial_samples - len(df):,} samples)"
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
                kind="borderline-1",
                random_state=RANDOM_STATE,
                k_neighbors=min(5, min(pd.Series(y).value_counts()) - 1),
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
        mask_if = outliers_if == 1

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

        p1 = np.array([x[0], sorted_scores[0]])
        p2 = np.array([x[-1], sorted_scores[-1]])

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
        X_positive = X - X.min() + 1e-6
        chi2_scores, _ = chi2(X_positive, y)
        chi2_scores = chi2_scores / (chi2_scores.max() + 1e-10)

        # 3.3 Mutual Information
        print("  3.3 Information Mutuelle...")
        mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
        mi_scores = mi_scores / (mi_scores.max() + 1e-10)

        # Combiner les scores
        xgb_norm = xgb_importance / (xgb_importance.max() + 1e-10)
        combined_scores = 0.4 * xgb_norm + 0.3 * chi2_scores + 0.3 * mi_scores

        # Determiner k : elbow method ou valeur fixee
        if top_k is None:
            top_k = self.find_elbow_k(combined_scores)
            print(f"  Elbow method → k={top_k} (sur {n_features} features)")
        else:
            top_k = min(top_k, n_features)
            print(f"  k fixe → {top_k}")

        selected_indices = np.argsort(combined_scores)[-top_k:]
        selected_indices = np.sort(selected_indices)

        selected_features = [feature_names[i] for i in selected_indices]
        print(f"  {len(selected_indices)} caracteristiques selectionnees:")
        for i, idx in enumerate(selected_indices[-10:]):
            print(f"    {i + 1}. {feature_names[idx]}: {combined_scores[idx]:.4f}")

        self.selected_feature_indices = selected_indices
        return selected_indices, selected_features

    def prepare_features(self, df: pd.DataFrame):
        """
        Extract features, separating continuous, categorical, and binary.

        CRITICAL: Categorical features (protocolIdentifier) are NOT scaled.
        They are kept as raw integers for the BPE tokenizer to convert
        to human-readable labels (tcp, udp, icmp, etc.).

        CRITICAL: StandardScaler is applied ONLY to the 28 continuous features.
        The 8 binary packet direction bits are NOT scaled.
        """
        all_cols = [c for c in FEATURES_TO_KEEP_JSON if c in df.columns]

        categorical_cols = [c for c in CATEGORICAL_FEATURES_JSON if c in df.columns]
        continuous_cols = [c for c in all_cols if c not in categorical_cols]

        X_continuous = df[continuous_cols].values.astype(np.float32)
        X_categorical = (
            df[categorical_cols].values.astype(np.float32)
            if categorical_cols
            else np.empty((len(df), 0), dtype=np.float32)
        )

        binary_cols = [c for c in PKT_DIR_COLS if c in df.columns]
        X_binary = df[binary_cols].values.astype(np.float32)

        y = df["label"].values

        flow_start = df["flow_start"].values if "flow_start" in df.columns else None

        self.continuous_feature_names = continuous_cols
        self.categorical_feature_names = categorical_cols
        self.binary_feature_names = binary_cols
        self.feature_names = continuous_cols + categorical_cols + binary_cols

        return X_continuous, X_categorical, X_binary, y, flow_start

    def create_sequences(
        self,
        X_continuous: np.ndarray,
        X_binary: np.ndarray,
        y: np.ndarray,
        labels_str: np.ndarray,
        seq_length: int = None,
        stride: int = None,
    ):
        """
        Create sliding-window sequences, grouped by device (label).

        Sorts flows by flow_start within each device, then creates
        overlapping windows of seq_length flows.

        Args:
            X_continuous: Scaled continuous features (N, 28)
            X_binary: Binary direction features (N, 8) — NOT scaled
            y: Encoded integer labels
            labels_str: String labels for grouping
            seq_length: Sequence window length
            stride: Step between windows
        """
        if seq_length is None:
            seq_length = SEQUENCE_LENGTH
        if stride is None:
            stride = STRIDE

        # Combine features: [continuous | binary]
        X = np.concatenate([X_continuous, X_binary], axis=1)

        X_seq, y_seq = [], []

        unique_labels = np.unique(labels_str)
        for label in unique_labels:
            mask = labels_str == label
            X_group = X[mask]
            y_group = y[mask]

            n_samples = len(X_group) - seq_length + 1
            if n_samples <= 0:
                continue

            for i in range(0, n_samples, stride):
                if i + seq_length <= len(X_group):
                    X_seq.append(X_group[i : i + seq_length])
                    y_seq.append(y_group[i + seq_length - 1])

        return np.array(X_seq), np.array(y_seq)

    def process_all(
        self,
        data_dir: Path = None,
        save_path: Path = None,
        max_records: int = None,
        seq_length: int = None,
        stride: int = None,
        min_samples: int = None,
        apply_balancing: bool = True,
        apply_feature_selection: bool = True,
        top_k_features: int = None,
    ):
        """
        Full pipeline en 4 etapes avec anti-leakage pour IPFIX Records.

        Etape 1: Filtrage SDN (supprimer IP/ports, garder caracteristiques statistiques)
        Etape 2: Equilibrage (Borderline-SMOTE) + Filtrage bruit (Isolation Forest, LOF)
        Etape 3: Selection hybride des caracteristiques (XGBoost, Chi2, MI)
        Etape 4: Normalisation Standardisation

        Anti-leakage:
        - Per-device temporal 80/20 split
        - Sequences generees separement sur train et test
        - Scalers ajustes uniquement sur train

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test,
                      feature_names, scaler, label_encoder)
        """
        if seq_length is None:
            seq_length = SEQUENCE_LENGTH
        if stride is None:
            stride = STRIDE

        print("=" * 70)
        print("JSON-NATIVE IPFIX RECORDS PREPROCESSOR (Pipeline 4 Etapes)")
        print("17 classes cibles: Qrio Hub, Philips Hue Light Bulb, etc.")
        print("=" * 70)

        # ─── Etape 1: Chargement et Filtrage SDN ──────────────────────────────
        print("\n[ETAPE 1] Filtrage et adaptation au SDN...")
        print("  1.1 Chargement des donnees JSON...")
        df = self.load_json_files(data_dir, max_records=max_records)

        print("  1.2 Filtrage aux 17 classes cibles...")
        df = self.filter_classes(df, min_samples)

        # ─── Extraction des caracteristiques ─────────────────────────────────
        print("\n  Extraction des caracteristiques...")
        X_continuous, X_categorical, X_binary, y_str, flow_start = (
            self.prepare_features(df)
        )
        del df
        gc.collect()

        # ─── Encodage des labels ─────────────────────────────────────────────
        print("\n  Encodage des labels...")
        self.label_encoder.fit(y_str)
        y_encoded = self.label_encoder.transform(y_str)
        self.num_classes = len(self.label_encoder.classes_)
        print(f"  Classes: {self.num_classes}")
        for i, cls in enumerate(self.label_encoder.classes_):
            count = int(np.sum(y_encoded == i))
            print(f"    {i}: {cls} ({count:,} samples)")

        # ─── Split temporel ANTI-LEAKAGE (train/val/test) ─────────────────────
        val_ratio = VAL_SIZE  # e.g. 0.1
        train_ratio = 1.0 - TEST_SIZE - val_ratio  # e.g. 0.7
        print(
            f"\n[PRE-SPLIT] Application du split temporel {int(train_ratio * 100)}/{int(val_ratio * 100)}/{int(TEST_SIZE * 100)} par appareil..."
        )
        train_mask = np.zeros(len(y_str), dtype=bool)
        val_mask = np.zeros(len(y_str), dtype=bool)

        for device in np.unique(y_str):
            dev_indices = np.where(y_str == device)[0]
            if flow_start is not None:
                sort_order = np.argsort(flow_start[dev_indices])
                dev_indices = dev_indices[sort_order]
            n = len(dev_indices)
            split_idx_train = max(1, int(n * train_ratio))
            split_idx_val = max(split_idx_train + 1, int(n * (train_ratio + val_ratio)))
            train_mask[dev_indices[:split_idx_train]] = True
            val_mask[dev_indices[split_idx_train:split_idx_val]] = True

        test_mask = ~train_mask & ~val_mask

        X_cont_train = X_continuous[train_mask].copy()
        X_cat_train = X_categorical[train_mask].copy()
        X_bin_train = X_binary[train_mask].copy()
        y_enc_train = y_encoded[train_mask].copy()
        y_str_train = y_str[train_mask].copy()

        X_cont_val = X_continuous[val_mask].copy()
        X_cat_val = X_categorical[val_mask].copy()
        X_bin_val = X_binary[val_mask].copy()
        y_enc_val = y_encoded[val_mask].copy()
        y_str_val = y_str[val_mask].copy()

        X_cont_test = X_continuous[test_mask].copy()
        X_cat_test = X_categorical[test_mask].copy()
        X_bin_test = X_binary[test_mask].copy()
        y_enc_test = y_encoded[test_mask].copy()
        y_str_test = y_str[test_mask].copy()

        print(
            f"  Train rows: {len(y_enc_train):,} | Val rows: {len(y_enc_val):,} | Test rows: {len(y_enc_test):,}"
        )

        del (
            X_continuous,
            X_categorical,
            X_binary,
            y_encoded,
            y_str,
            flow_start,
            train_mask,
            val_mask,
            test_mask,
        )
        gc.collect()

        # Combiner features pour les etapes 2 et 3
        # Continuous + binary for SMOTE/outlier/feature selection
        X_train_combined = np.concatenate([X_cont_train, X_bin_train], axis=1)
        X_val_combined = np.concatenate([X_cont_val, X_bin_val], axis=1)
        X_test_combined = np.concatenate([X_cont_test, X_bin_test], axis=1)
        all_feature_names = self.continuous_feature_names + self.binary_feature_names

        # ─── Etape 2: Equilibrage et Filtrage du Bruit (TRAIN UNIQUEMENT) ───
        if apply_balancing:
            print("\n[ETAPE 2] Equilibrage et filtrage du bruit (train only)...")
            X_train_balanced, y_train_balanced = self.balance_and_filter_noise(
                X_train_combined, y_enc_train, contamination=0.05
            )
        else:
            print("\n[ETAPE 2] Equilibrage desactive.")
            X_train_balanced = X_train_combined
            y_train_balanced = y_enc_train

        # ─── Etape 3: Selection Hybride des Caracteristiques ─────────────────
        if apply_feature_selection:
            print("\n[ETAPE 3] Selection hybride des caracteristiques...")
            selected_indices, selected_features = self.hybrid_feature_selection(
                X_train_balanced,
                y_train_balanced,
                all_feature_names,
                top_k=top_k_features,
            )
            X_train_selected = X_train_balanced[:, selected_indices]
            X_val_selected = X_val_combined[:, selected_indices]
            X_test_selected = X_test_combined[:, selected_indices]
            self.feature_names = selected_features

            # Mettre a jour les indices des features continues/binaires
            n_continuous = len(self.continuous_feature_names)
            cont_mask = selected_indices < n_continuous
            bin_mask = selected_indices >= n_continuous

            self.selected_continuous_indices = selected_indices[cont_mask]
            self.selected_binary_indices = selected_indices[bin_mask] - n_continuous
        else:
            print("\n[ETAPE 3] Selection desactivee.")
            X_train_selected = X_train_balanced
            X_val_selected = X_val_combined
            X_test_selected = X_test_combined

        del X_train_combined, X_val_combined, X_test_combined, X_train_balanced
        gc.collect()

        # ─── Etape 4: Normalisation (StandardScaler per docs/pretraitement) ─────
        # IMPORTANT: Categorical features (protocolIdentifier) are NOT scaled.
        # They remain as raw integers for the BPE tokenizer to convert to
        # human-readable labels (tcp, udp, icmp, etc.).
        print(
            "\n[ETAPE 4] StandardScaler (moyenne=0, variance=1) — continuous features only (fit sur train only)..."
        )
        X_train_cont_scaled = self.standard_scaler.fit_transform(X_train_selected).astype(
            np.float32
        )
        X_val_cont_scaled = self.standard_scaler.transform(X_val_selected).astype(
            np.float32
        )
        X_test_cont_scaled = self.standard_scaler.transform(X_test_selected).astype(
            np.float32
        )

        del X_train_selected, X_val_selected, X_test_selected
        gc.collect()

        # ─── Creation des sequences (SEPARATEMENT pour train, val, test) ──────
        print(f"\n[SEQUENCES] Creation (length={seq_length}, stride={stride})...")

        X_train_seq, y_train_seq = self.create_sequences_with_categorical(
            X_train_cont_scaled,
            X_cat_train,
            X_bin_train,
            y_train_balanced,
            y_str_train if not apply_balancing else None,
            seq_length,
            stride,
        )
        X_val_seq, y_val_seq = self.create_sequences_with_categorical(
            X_val_cont_scaled,
            X_cat_val,
            X_bin_val,
            y_enc_val,
            y_str_val,
            seq_length,
            stride,
        )
        X_test_seq, y_test_seq = self.create_sequences_with_categorical(
            X_test_cont_scaled,
            X_cat_test,
            X_bin_test,
            y_enc_test,
            y_str_test,
            seq_length,
            stride,
        )

        print(f"  Train sequences: {len(X_train_seq):,}")
        print(f"  Val   sequences: {len(X_val_seq):,}")
        print(f"  Test  sequences: {len(X_test_seq):,}")
        if len(X_train_seq) > 0:
            print(f"  Sequence shape:  {X_train_seq.shape}")

        del (
            X_train_cont_scaled,
            X_val_cont_scaled,
            X_test_cont_scaled,
            X_cat_train,
            X_cat_val,
            X_cat_test,
            X_bin_train,
            X_bin_val,
            X_bin_test,
        )
        gc.collect()

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
                        "standard_scaler": self.standard_scaler,
                        "label_encoder": self.label_encoder,
                        "feature_names": self.feature_names,
                        "selected_feature_indices": self.selected_feature_indices,
                        "num_classes": self.num_classes,
                    },
                    f,
                )
            print(f"\nDonnees sauvegardees dans {save_path}")

        return (
            X_train_seq,
            X_val_seq,
            X_test_seq,
            y_train_seq,
            y_val_seq,
            y_test_seq,
            self.feature_names,
            self.standard_scaler,
            self.label_encoder,
        )

    def create_sequences_with_categorical(
        self,
        X_continuous: np.ndarray,
        X_categorical: np.ndarray,
        X_binary: np.ndarray,
        y: np.ndarray,
        labels_str: np.ndarray = None,
        seq_length: int = None,
        stride: int = None,
    ):
        """
        Create sequences combining continuous, categorical, and binary features.

        Continuous features are scaled (already done before this call).
        Categorical features are kept as raw integers for BPE tokenization.
        Binary features are kept as-is.

        Final feature order: [continuous | categorical | binary]
        """
        if seq_length is None:
            seq_length = SEQUENCE_LENGTH
        if stride is None:
            stride = STRIDE

        X_combined = np.concatenate([X_continuous, X_categorical, X_binary], axis=1)

        X_seq, y_seq = [], []

        if labels_str is not None:
            unique_labels = np.unique(labels_str)
            for label in unique_labels:
                mask = labels_str == label
                X_group = X_combined[mask]
                y_group = y[mask]
                n_samples = len(X_group) - seq_length + 1
                if n_samples <= 0:
                    continue
                for i in range(0, n_samples, stride):
                    if i + seq_length <= len(X_group):
                        X_seq.append(X_group[i : i + seq_length])
                        y_seq.append(y_group[i + seq_length - 1])
        else:
            n_samples = len(X_combined) - seq_length + 1
            for i in range(0, n_samples, stride):
                if i + seq_length <= len(X_combined):
                    X_seq.append(X_combined[i : i + seq_length])
                    y_seq.append(y[i + seq_length - 1])

        return np.array(X_seq), np.array(y_seq)

    def create_sequences_simple(
        self,
        X: np.ndarray,
        y: np.ndarray,
        labels_str: np.ndarray = None,
        seq_length: int = None,
        stride: int = None,
    ):
        """Create sequences without time ordering (simplified for balanced data)."""
        if seq_length is None:
            seq_length = SEQUENCE_LENGTH
        if stride is None:
            stride = STRIDE

        X_seq, y_seq = [], []

        if labels_str is not None:
            unique_labels = np.unique(labels_str)
            for label in unique_labels:
                mask = labels_str == label
                X_group = X[mask]
                y_group = y[mask]
                n_samples = len(X_group) - seq_length + 1
                if n_samples <= 0:
                    continue
                for i in range(0, n_samples, stride):
                    if i + seq_length <= len(X_group):
                        X_seq.append(X_group[i : i + seq_length])
                        y_seq.append(y_group[i + seq_length - 1])
        else:
            n_samples = len(X) - seq_length + 1
            for i in range(0, n_samples, stride):
                if i + seq_length <= len(X):
                    X_seq.append(X[i : i + seq_length])
                    y_seq.append(y[i + seq_length - 1])

        return np.array(X_seq), np.array(y_seq)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process IPFIX Records JSON dataset - Pipeline 4 Etapes"
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Maximum records to load (for testing)",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=SEQUENCE_LENGTH,
        help="Sequence length for sliding window",
    )
    parser.add_argument(
        "--stride", type=int, default=STRIDE, help="Stride for sliding window"
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=MIN_SAMPLES_PER_CLASS,
        help="Minimum samples per class",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=str(PROCESSED_DATA_DIR / "json_native"),
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--no_balancing",
        action="store_true",
        help="Disable SMOTE + outlier filtering",
    )
    parser.add_argument(
        "--no_feature_selection",
        action="store_true",
        help="Disable hybrid feature selection",
    )
    parser.add_argument(
        "--top_k_features",
        type=int,
        default=25,
        help="Number of features to select (default: 25)",
    )
    args = parser.parse_args()

    processor = JsonIoTDataProcessor()
    processor.process_all(
        save_path=Path(args.save_path),
        max_records=args.max_records,
        seq_length=args.seq_length,
        stride=args.stride,
        min_samples=args.min_samples,
        apply_balancing=not args.no_balancing,
        apply_feature_selection=not args.no_feature_selection,
        top_k_features=args.top_k_features,
    )
