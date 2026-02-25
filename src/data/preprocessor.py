"""
Data Preprocessing for IoT Device Identification
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import *


class IoTDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.num_classes = 0

    def load_all_data(self, max_files=None):
        dfs = []
        csv_files = sorted(RAW_DATA_DIR.glob("home*_labeled.csv"))

        if max_files:
            csv_files = csv_files[:max_files]

        for f in csv_files:
            print(f"Loading {f.name}...")
            df = pd.read_csv(f)
            df["source_file"] = f.stem
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    def clean_data(self, df):
        df = df.drop_duplicates()
        df = df.dropna(subset=[LABEL_COLUMN])

        existing_cols = [c for c in FEATURES_TO_DROP if c in df.columns]
        df = df.drop(columns=existing_cols, errors="ignore")

        class_counts = df[LABEL_COLUMN].value_counts()
        valid_classes = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index
        df = df[df[LABEL_COLUMN].isin(valid_classes)]

        print(f"Classes: {len(valid_classes)} | Samples: {len(df)}")
        return df

    def select_features(self, df):
        features = [c for c in FEATURES_TO_KEEP if c in df.columns]
        X = df[features].values
        y = df[LABEL_COLUMN].values
        self.feature_names = features
        return X, y

    def create_sequences(self, X, y, source_groups=None):
        X_seq, y_seq = [], []

        if source_groups is None:
            n_samples = len(X) - SEQUENCE_LENGTH
            for i in range(0, n_samples, STRIDE):
                X_seq.append(X[i : i + SEQUENCE_LENGTH])
                y_seq.append(y[i + SEQUENCE_LENGTH - 1])
        else:
            for group_id in np.unique(source_groups):
                mask = source_groups == group_id
                X_group = X[mask]
                y_group = y[mask]

                n_samples = len(X_group) - SEQUENCE_LENGTH
                for i in range(0, max(1, n_samples), STRIDE):
                    if i + SEQUENCE_LENGTH <= len(X_group):
                        X_seq.append(X_group[i : i + SEQUENCE_LENGTH])
                        y_seq.append(y_group[i + SEQUENCE_LENGTH - 1])

        return np.array(X_seq), np.array(y_seq)

    def fit_transform(self, X):
        return self.scaler.fit_transform(X)

    def transform(self, X):
        return self.scaler.transform(X)

    def encode_labels(self, y):
        y_encoded = self.label_encoder.fit_transform(y)
        self.num_classes = len(self.label_encoder.classes_)
        return y_encoded

    def process_all(self, max_files=None, save_path=None):
        print("=" * 50)
        print("Loading data...")
        df = self.load_all_data(max_files)

        print("\nCleaning data...")
        df = self.clean_data(df)

        source_groups = (
            df["source_file"].values if "source_file" in df.columns else None
        )

        print("\nSelecting features...")
        X, y = self.select_features(df)

        print("\nEncoding labels...")
        y_encoded = self.encode_labels(y)

        print("\nNormalizing features...")
        X_normalized = self.fit_transform(X)

        print("\nCreating sequences...")
        X_seq, y_seq = self.create_sequences(X_normalized, y_encoded, source_groups)

        print(f"\nTotal sequences: {len(X_seq)}")
        print(f"Sequence shape: {X_seq.shape}")
        print(f"Number of classes: {self.num_classes}")

        X_train, X_temp, y_train, y_temp = train_test_split(
            X_seq, y_seq, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_seq
        )

        val_ratio = VAL_SIZE / (1 - TEST_SIZE)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=(1 - val_ratio),
            random_state=RANDOM_STATE,
            stratify=y_temp,
        )

        print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)

            np.save(save_path / "X_train.npy", X_train)
            np.save(save_path / "X_val.npy", X_val)
            np.save(save_path / "X_test.npy", X_test)
            np.save(save_path / "y_train.npy", y_train)
            np.save(save_path / "y_val.npy", y_val)
            np.save(save_path / "y_test.npy", y_test)

            with open(save_path / "preprocessor.pkl", "wb") as f:
                pickle.dump(
                    {
                        "scaler": self.scaler,
                        "label_encoder": self.label_encoder,
                        "feature_names": self.feature_names,
                        "num_classes": self.num_classes,
                    },
                    f,
                )

            print(f"\nData saved to {save_path}")

        return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    processor = IoTDataProcessor()
    processor.process_all(save_path=PROCESSED_DATA_DIR)
