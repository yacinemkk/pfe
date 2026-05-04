"""
Categorical Feature Encoder for IoT Pipelines

Rules per model type:
1. Transformer / Hybrid (CNN-BiLSTM-Transformer):
   - NO one-hot encoding
   - Categorical features passed as raw string labels → BPE tokenizer → Embedding
   - Self-attention dynamically evaluates importance

2. DL models (LSTM, CNN-LSTM, CNN-BiLSTM):
   - Label Encoding (integer mapping)
   - One-hot is acceptable but less efficient

3. XGBoost:
   - Label Encoding preferred (trees handle integers well)
   - One-hot creates sparse matrices that slow down tree models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path

# Categorical features in IoT datasets
CATEGORICAL_FEATURES_CSV = ["ipProto"]
CATEGORICAL_FEATURES_JSON = ["protocolIdentifier"]

# Protocol value mappings for human-readable tokenization
PROTO_MAP = {
    1: "icmp",
    6: "tcp",
    17: "udp",
    47: "gre",
    50: "esp",
    51: "ah",
    89: "ospf",
    132: "sctp",
}


class CategoricalFeatureEncoder:
    """
    Manages categorical feature encoding per model type.

    For Transformer: returns string labels (for BPE tokenization)
    For DL models: returns label-encoded integers
    For XGBoost: returns label-encoded integers
    """

    def __init__(self):
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []
        self.is_fit = False

    def fit(
        self, X: np.ndarray, feature_names: List[str], categorical_features: List[str]
    ):
        """
        Fit label encoders for categorical features.

        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: Names of all features
            categorical_features: Subset of feature_names that are categorical
        """
        self.feature_names = feature_names
        self.is_fit = True

        for feat_name in categorical_features:
            if feat_name not in feature_names:
                continue
            idx = feature_names.index(feat_name)
            le = LabelEncoder()
            col_values = X[:, idx]
            if np.issubdtype(col_values.dtype, np.number):
                col_str = np.array(
                    [PROTO_MAP.get(int(v), f"proto_{int(v)}") for v in col_values]
                )
            else:
                col_str = col_values.astype(str)
            le.fit(col_str)
            self.encoders[feat_name] = le

    def transform_for_transformer(
        self, X: np.ndarray, feature_names: List[str], categorical_features: List[str]
    ) -> np.ndarray:
        """
        For Transformer: return string labels for categorical features.
        The BPE tokenizer will handle them as pre-defined tokens.

        Returns:
            X with categorical columns as strings (object dtype)
        """
        X_out = X.copy().astype(object)
        for feat_name in categorical_features:
            if feat_name not in feature_names:
                continue
            idx = feature_names.index(feat_name)
            col_values = X[:, idx]
            if np.issubdtype(col_values.dtype, np.number):
                X_out[:, idx] = np.array(
                    [PROTO_MAP.get(int(v), f"proto_{int(v)}") for v in col_values],
                    dtype=object,
                )
            else:
                X_out[:, idx] = col_values.astype(str)
        return X_out

    def transform_for_dl(
        self, X: np.ndarray, feature_names: List[str], categorical_features: List[str]
    ) -> np.ndarray:
        """
        For DL models (LSTM, CNN, BiLSTM): label encode categorical features.

        Returns:
            X with categorical columns as integer-encoded values
        """
        X_out = X.copy().astype(np.float32)
        for feat_name in categorical_features:
            if feat_name not in feature_names:
                continue
            idx = feature_names.index(feat_name)
            if feat_name not in self.encoders:
                raise ValueError(
                    f"Encoder not fitted for '{feat_name}'. Call fit() first."
                )
            col_values = X[:, idx]
            if np.issubdtype(col_values.dtype, np.number):
                col_str = np.array(
                    [PROTO_MAP.get(int(v), f"proto_{int(v)}") for v in col_values]
                )
            else:
                col_str = col_values.astype(str)
            encoded = self.encoders[feat_name].transform(col_str)
            X_out[:, idx] = encoded.astype(np.float32)
        return X_out

    def transform_for_xgboost(
        self, X: np.ndarray, feature_names: List[str], categorical_features: List[str]
    ) -> np.ndarray:
        """
        For XGBoost: label encode categorical features (same as DL).

        Returns:
            X with categorical columns as integer-encoded values
        """
        return self.transform_for_dl(X, feature_names, categorical_features)

    def get_categorical_string_values(
        self, X: np.ndarray, feature_names: List[str], categorical_features: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Extract categorical features as human-readable strings for BPE tokenization.

        Returns:
            Dict mapping feature_name -> array of string values
        """
        result = {}
        for feat_name in categorical_features:
            if feat_name not in feature_names:
                continue
            idx = feature_names.index(feat_name)
            col_values = X[:, idx]
            if np.issubdtype(col_values.dtype, np.number):
                result[feat_name] = np.array(
                    [PROTO_MAP.get(int(v), f"proto_{int(v)}") for v in col_values]
                )
            else:
                result[feat_name] = col_values.astype(str)
        return result

    def save(self, path: str):
        """Save fitted encoders."""
        with open(path, "wb") as f:
            pickle.dump(
                {"encoders": self.encoders, "feature_names": self.feature_names}, f
            )

    def load(self, path: str):
        """Load fitted encoders."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.encoders = data["encoders"]
        self.feature_names = data["feature_names"]
        self.is_fit = True


def get_categorical_features_for_dataset(dataset_type: str) -> List[str]:
    """Return the list of categorical feature names for the given dataset."""
    if dataset_type == "csv":
        return CATEGORICAL_FEATURES_CSV
    elif dataset_type == "json":
        return CATEGORICAL_FEATURES_JSON
    return []


def separate_continuous_and_categorical(
    X: np.ndarray,
    feature_names: List[str],
    categorical_features: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Split feature matrix into continuous and categorical parts.

    Returns:
        X_continuous, X_categorical, continuous_names, categorical_names
    """
    cat_indices = [
        feature_names.index(f) for f in categorical_features if f in feature_names
    ]
    cont_indices = [i for i in range(len(feature_names)) if i not in cat_indices]

    X_continuous = (
        X[:, cont_indices] if cont_indices else np.empty((X.shape[0], 0), dtype=X.dtype)
    )
    X_categorical = (
        X[:, cat_indices] if cat_indices else np.empty((X.shape[0], 0), dtype=X.dtype)
    )

    continuous_names = [feature_names[i] for i in cont_indices]
    categorical_names = [feature_names[i] for i in cat_indices]

    return X_continuous, X_categorical, continuous_names, categorical_names
