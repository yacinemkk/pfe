"""
Data Preprocessing for IoT Device Identification — CSV Pipeline (IPFIX ML Instances)

Anti-Leakage Temporal Split (docs/general §Étape 3 & 4):
  1. Group flows by device label (name column)
  2. Sort each group chronologically by 'start' timestamp
  3. Per-device 80/20 temporal split (first 80% → train, last 20% → test)
  4. Build sliding-window sequences INDEPENDENTLY on train and test parts
  5. Fit StandardScaler ONLY on training data
  6. Val set is carved from the training sequences (stratified 10% split)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import pickle
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FEATURES_TO_KEEP,
    FEATURES_TO_DROP,
    LABEL_COLUMN,
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

    Implements a strict anti-leakage temporal pipeline:
    - Groups flows by device
    - Sorts each device group by 'start' timestamp
    - Applies 80/20 temporal split per device
    - Creates sequences separately for train and test
    - Scaler fitted on training data only
    """

    def __init__(self):
        self.minmax_scaler = MinMaxScaler(feature_range=(0,1))
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
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

    # ─── Step 2: Clean / filter ──────────────────────────────────────────────

    def clean_data(self, df):
        """Remove duplicates, drop identifying columns, filter to IoT device classes."""
        df = df.drop_duplicates()
        df = df.dropna(subset=[LABEL_COLUMN])

        # Drop identifying columns (MACs, IPs, ports…)
        existing_drop = [c for c in FEATURES_TO_DROP if c in df.columns]
        df = df.drop(columns=existing_drop, errors="ignore")

        # Keep only the 18 IoT device classes (docs/pretraitement §3)
        df = df[df[LABEL_COLUMN].isin(IOT_DEVICE_CLASSES)]

        # Drop classes with very few samples
        class_counts = df[LABEL_COLUMN].value_counts()
        valid_classes = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index
        df = df[df[LABEL_COLUMN].isin(valid_classes)].copy()

        print(f"  After cleaning: {len(df):,} rows | {len(valid_classes)} classes")
        return df

    # ─── Step 3: Per-device temporal 80/20 split ─────────────────────────────

    def temporal_split_per_device(self, df, train_ratio=0.8):
        """
        Group → sort by timestamp → split 80/20 per device.

        Returns two DataFrames (train, test) with no future-data leakage.
        'start' column is used for sorting; falls back to row order if absent.
        """
        train_parts = []
        test_parts = []

        timestamp_col = "start" if "start" in df.columns else None

        for device, group in df.groupby(LABEL_COLUMN, sort=False):
            if timestamp_col:
                group = group.sort_values(timestamp_col)
            else:
                # Row order as proxy for time (already roughly chronological in CSVs)
                pass

            n = len(group)
            split_idx = max(1, int(n * train_ratio))

            train_parts.append(group.iloc[:split_idx])
            test_parts.append(group.iloc[split_idx:])

        df_train = pd.concat(train_parts, ignore_index=True)
        df_test  = pd.concat(test_parts,  ignore_index=True)

        print(f"  Temporal split → Train: {len(df_train):,} | Test: {len(df_test):,}")
        return df_train, df_test

    # ─── Step 4: Feature extraction ─────────────────────────────────────────

    def select_features(self, df):
        """Return feature matrix X and label array y."""
        features = [c for c in FEATURES_TO_KEEP if c in df.columns]
        X = df[features].fillna(0).values.astype(np.float32)
        y = df[LABEL_COLUMN].values
        self.feature_names = features
        return X, y

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
                    X_seq.append(X_group[i: i + seq_length])
                    y_seq.append(y_group[i + seq_length - 1])

        return np.array(X_seq), np.array(y_seq)

    # ─── Main pipeline ───────────────────────────────────────────────────────

    def process_all(self, max_files=None, data_dir=None, save_path=None,
                    seq_length=None, stride=None):
        """
        Full anti-leakage pipeline:
        1. Load CSVs
        2. Clean & filter to 18 IoT classes
        3. Per-device temporal 80/20 split (docs/general §Étape 3)
        4. Extract features & encode labels
        5. Fit scaler on TRAINING data only
        6. Create sequences separately on train and test (docs/general §Étape 4)
        7. Carve val from train (stratified 10%)
        8. Save

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if seq_length is None:
            seq_length = SEQUENCE_LENGTH
        if stride is None:
            stride = STRIDE

        print("=" * 60)
        print("CSV-NATIVE IPFIX ML INSTANCES PREPROCESSOR (Anti-Leakage)")
        print("=" * 60)

        # 1. Load
        print("\n[1/7] Loading CSV files...")
        df = self.load_all_data(max_files=max_files, data_dir=data_dir)

        # 2. Clean
        print("\n[2/7] Cleaning data...")
        df = self.clean_data(df)

        # 3. Temporal split per device
        print("\n[3/7] Applying per-device temporal 80/20 split...")
        df_train_raw, df_test_raw = self.temporal_split_per_device(df)
        del df

        # 4. Feature extraction
        print("\n[4/7] Extracting features...")
        X_train_raw, y_train_str = self.select_features(df_train_raw)
        X_test_raw,  y_test_str  = self.select_features(df_test_raw)
        del df_train_raw, df_test_raw

        # Encode labels
        print("\n[5/7] Encoding labels...")
        self.label_encoder.fit(y_train_str)
        y_train_enc = self.label_encoder.transform(y_train_str)
        # Test set may contain unseen classes (very rare) — map to nearest
        y_test_enc = np.array([
            self.label_encoder.transform([lbl])[0]
            if lbl in self.label_encoder.classes_
            else 0
            for lbl in y_test_str
        ])
        self.num_classes = len(self.label_encoder.classes_)
        print(f"  Classes: {self.num_classes}")

        # 5. Scale — fit on train ONLY (MinMax then Standard)
        print("\n[5b/7] Scaling (fit on train only)...")
        # Step A: Min-Max (0 to 1)
        X_train_minmax = self.minmax_scaler.fit_transform(X_train_raw)
        X_test_minmax = self.minmax_scaler.transform(X_test_raw)
        
        # Step B: Standardisation (mean 0, std 1)
        X_train_scaled = self.scaler.fit_transform(X_train_minmax).astype(np.float32)
        X_test_scaled  = self.scaler.transform(X_test_minmax).astype(np.float32)
        del X_train_raw, X_test_raw

        # 6. Create sequences per device
        print(f"\n[6/7] Creating sequences (length={seq_length}, stride={stride})...")
        X_train_seq, y_train_seq = self.create_sequences(
            X_train_scaled, y_train_enc, seq_length, stride)
        X_test_seq,  y_test_seq  = self.create_sequences(
            X_test_scaled,  y_test_enc,  seq_length, stride)
        del X_train_scaled, X_test_scaled
        print(f"  Train sequences: {len(X_train_seq):,} | Test sequences: {len(X_test_seq):,}")

        # 7. Carve validation from training sequences
        print("\n[7/7] Splitting validation from training data...")
        val_ratio = VAL_SIZE / (1 - TEST_SIZE)  # 10% of original → ~12.5% of train-split
        X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(
            X_train_seq, y_train_seq,
            test_size=val_ratio,
            random_state=RANDOM_STATE,
            stratify=y_train_seq,
        )
        print(f"  Train: {len(X_train_seq):,} | Val: {len(X_val_seq):,} | Test: {len(X_test_seq):,}")

        # 8. Save
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)

            np.save(save_path / "X_train.npy", X_train_seq)
            np.save(save_path / "X_val.npy",   X_val_seq)
            np.save(save_path / "X_test.npy",  X_test_seq)
            np.save(save_path / "y_train.npy", y_train_seq)
            np.save(save_path / "y_val.npy",   y_val_seq)
            np.save(save_path / "y_test.npy",  y_test_seq)

            with open(save_path / "preprocessor.pkl", "wb") as f:
                pickle.dump(
                    {
                        "minmax_scaler": self.minmax_scaler,
                        "scaler": self.scaler,
                        "label_encoder": self.label_encoder,
                        "feature_names": self.feature_names,
                        "num_classes": self.num_classes,
                    },
                    f,
                )
            print(f"\n  Data saved to {save_path}")

        return X_train_seq, X_val_seq, X_test_seq, y_train_seq, y_val_seq, y_test_seq, \
               self.feature_names, self.scaler, self.label_encoder


if __name__ == "__main__":
    processor = IoTDataProcessor()
    processor.process_all(save_path=PROCESSED_DATA_DIR)
