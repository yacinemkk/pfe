"""
JSON-Native Preprocessor for IPFIX Records Dataset

Addresses the following roadmap issues:
- Issue 1: Uses JSON dataset natively (not joined with CSV)
- Issue 2: Labels flows via MAC-to-device mapping
- Issue 3: Bidirectional flow labeling (checks both src and dst MACs)
- Issue 4: Drops all identifying columns to prevent data leakage
- Issue 5: Decodes hex-encoded packet directions into 8-bit binary arrays

This module is INDEPENDENT from preprocessor.py (CSV pipeline).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import gc
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import (
    DATA_DIR,
    PROCESSED_DATA_DIR,
    SEQUENCE_LENGTH,
    STRIDE,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
    MIN_SAMPLES_PER_CLASS,
)

# ─── Issue 2: MAC-to-Device Mapping (from audit findings) ───────────────────

MAC_TO_DEVICE = {
    "38:d5:47:0c:25:d4": "DEFAULT GATEWAY",
    "00:24:e4:62:68:2e": "Nokia body",
    "bc:c3:42:dc:24:78": "Panasonic doorphone",
    "80:c5:f2:0b:aa:a9": "Qrio hub",
    "00:17:88:47:20:f2": "Philips Hue lightbulb",
    "78:11:dc:55:76:4c": "Xiaomi LED",
    "ec:3d:fd:39:6f:98": "Planex UCA01A camera",
    "e0:b9:4d:b9:eb:e9": "Planex pantilt camera",
    "e0:b9:4d:5c:cf:c5": "JVC Kenwood camera",
    "60:01:94:54:6b:e8": "Nature remote control",
    "70:88:6b:10:22:83": "Bitfinder aware sensor",
    "d8:6c:63:47:54:dc": "Google Home",
    "d4:90:9c:da:0d:f0": "Apple Homepod",
    "04:5d:4b:a4:d0:2e": "Sony Bravia TV",
    "c0:e4:34:4b:89:fc": "iRobot roomba",
    "38:56:10:00:1d:8c": "Sesame access point",
    "00:a2:b2:b9:09:87": "JVC Kenwood hub",
    "e0:09:bf:54:68:47": "Wansview camera",
    "34:76:c5:7f:91:07": "Qwatch camera",
    "20:32:33:86:f7:0f": "Fredi camera",
    "00:22:cf:fd:c1:08": "Planex outdoor camera",
    "ec:f0:0e:55:25:39": "PowerElec WIFI plug",
    "a8:1e:84:e8:cc:c3": "Line Clova speaker",
    "6c:5a:b5:56:39:3e": "Sony smart speaker",
    "4c:ef:c0:17:e0:42": "Amazon Echo",
    "14:0a:c5:f1:e5:52": "Amazon Echo Show",
    "aa:1e:84:06:1c:b4": "MCJ room hub",
}

GATEWAY_MAC = "38:d5:47:0c:25:d4"

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
    # TCP flag strings — not numeric, drop them
    "initialTCPFlags",
    "unionTCPFlags",
    "reverseInitialTCPFlags",
    "reverseUnionTCPFlags",
]

# ─── Issue 4: Behavioral features to keep ────────────────────────────────────

FEATURES_TO_KEEP_JSON = [
    "flowDurationMilliseconds",
    "reverseFlowDeltaMilliseconds",
    "protocolIdentifier",
    "packetTotalCount",
    "octetTotalCount",
    "reversePacketTotalCount",
    "reverseOctetTotalCount",
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
    "bytesPerPacket",
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

    src_mac = flow.get("sourceMacAddress", "").strip().rstrip(':')
    dst_mac = flow.get("destinationMacAddress", "").strip().rstrip(':')

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
    
    This class:
    1. Stream-parses JSON files line-by-line (memory efficient)
    2. Labels flows via dual-MAC lookup
    3. Decodes hex packet directions to 8 binary features
    4. Drops identifying columns (prevents data leakage)
    5. Scales only continuous features (not binary direction bits)
    6. Creates temporal sequences with sliding window
    7. Filters underrepresented classes
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.num_classes = 0
        self.continuous_feature_names = list(FEATURES_TO_KEEP_JSON)
        self.binary_feature_names = list(PKT_DIR_COLS)

    def load_json_files(self, data_dir: Path = None, chunk_size: int = 100_000,
                         max_records: int = None) -> pd.DataFrame:
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

        # Find all JSON files
        json_files = sorted(data_dir.rglob("*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"No JSON files found in {data_dir}. "
                f"Expected files like ipfix_*.json in subdirectories."
            )

        print(f"Found {len(json_files)} JSON file(s):")
        for f in json_files:
            size_gb = f.stat().st_size / (1024 ** 3)
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
                        print(f"  Loaded {total_records:,} records "
                              f"({total_labeled:,} labeled, {total_unlabeled:,} skipped)...")
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

    def filter_classes(self, df: pd.DataFrame,
                        min_samples: int = None) -> pd.DataFrame:
        """Filter underrepresented classes."""
        if min_samples is None:
            min_samples = MIN_SAMPLES_PER_CLASS

        class_counts = df["label"].value_counts()
        valid_classes = class_counts[class_counts >= min_samples].index.tolist()
        removed_classes = class_counts[class_counts < min_samples].index.tolist()

        if removed_classes:
            print(f"\nRemoving {len(removed_classes)} underrepresented classes "
                  f"(< {min_samples} samples):")
            for cls in removed_classes:
                print(f"  - {cls}: {class_counts[cls]} samples")

        df = df[df["label"].isin(valid_classes)].copy()
        print(f"\nKept {len(valid_classes)} classes, {len(df):,} samples")
        return df

    def prepare_features(self, df: pd.DataFrame):
        """
        Extract and scale features.
        
        CRITICAL: StandardScaler is applied ONLY to the 28 continuous features.
        The 8 binary packet direction bits are NOT scaled.
        """
        # Continuous features
        continuous_cols = [c for c in FEATURES_TO_KEEP_JSON if c in df.columns]
        X_continuous = df[continuous_cols].values.astype(np.float32)

        # Binary direction features
        binary_cols = [c for c in PKT_DIR_COLS if c in df.columns]
        X_binary = df[binary_cols].values.astype(np.float32)

        # Labels
        y = df["label"].values

        # Temporal ordering
        flow_start = df["flow_start"].values if "flow_start" in df.columns else None

        self.continuous_feature_names = continuous_cols
        self.binary_feature_names = binary_cols
        self.feature_names = continuous_cols + binary_cols

        return X_continuous, X_binary, y, flow_start

    def create_sequences(self, X_continuous: np.ndarray, X_binary: np.ndarray,
                          y: np.ndarray, labels_str: np.ndarray,
                          seq_length: int = None, stride: int = None):
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
                    X_seq.append(X_group[i: i + seq_length])
                    y_seq.append(y_group[i + seq_length - 1])

        return np.array(X_seq), np.array(y_seq)

    def process_all(self, data_dir: Path = None, save_path: Path = None,
                     max_records: int = None, seq_length: int = None,
                     stride: int = None, min_samples: int = None):
        """
        Full processing pipeline for JSON IPFIX Records.

        Args:
            data_dir: JSON data directory
            save_path: Where to save processed numpy arrays
            max_records: Limit total records loaded (for debugging)
            seq_length: Sequence window length
            stride: Step between windows
            min_samples: Minimum samples per class

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test,
                      feature_names, scaler, label_encoder)
        """
        if seq_length is None:
            seq_length = SEQUENCE_LENGTH
        if stride is None:
            stride = STRIDE

        print("=" * 60)
        print("JSON-NATIVE IPFIX RECORDS PREPROCESSOR")
        print("=" * 60)

        # Step 1: Load and label JSON data
        print("\n[1/6] Loading JSON data...")
        df = self.load_json_files(data_dir, max_records=max_records)

        # Step 2: Filter underrepresented classes
        print("\n[2/6] Filtering classes...")
        df = self.filter_classes(df, min_samples)

        # Step 3: Extract features
        print("\n[3/6] Extracting features...")
        X_continuous, X_binary, y_str, flow_start = self.prepare_features(df)

        # Sort by flow_start within each device for temporal ordering
        if flow_start is not None:
            print("  Sorting by flow_start for temporal ordering...")
            sort_idx = np.argsort(flow_start)
            X_continuous = X_continuous[sort_idx]
            X_binary = X_binary[sort_idx]
            y_str = y_str[sort_idx]

        del df
        gc.collect()

        # Step 4: Encode labels
        print("\n[4/6] Encoding labels...")
        y_encoded = self.label_encoder.fit_transform(y_str)
        self.num_classes = len(self.label_encoder.classes_)
        print(f"  Classes: {self.num_classes}")
        for i, cls in enumerate(self.label_encoder.classes_):
            count = np.sum(y_encoded == i)
            print(f"    {i}: {cls} ({count:,} samples)")

        # Step 5: Scale continuous features ONLY
        print("\n[5/6] Scaling continuous features (binary features preserved)...")
        X_continuous_scaled = self.scaler.fit_transform(X_continuous)
        print(f"  Continuous features: {X_continuous_scaled.shape[1]}")
        print(f"  Binary features: {X_binary.shape[1]} (NOT scaled)")

        # Step 6: Create sequences
        print(f"\n[6/6] Creating sequences (length={seq_length}, stride={stride})...")
        X_seq, y_seq = self.create_sequences(
            X_continuous_scaled, X_binary, y_encoded, y_str,
            seq_length, stride
        )
        print(f"  Total sequences: {len(X_seq):,}")
        print(f"  Sequence shape: {X_seq.shape}")
        print(f"  Features per timestep: {X_seq.shape[2]} "
              f"({len(self.continuous_feature_names)} continuous + "
              f"{len(self.binary_feature_names)} binary)")

        del X_continuous, X_continuous_scaled, X_binary, y_encoded, y_str
        gc.collect()

        # Train/Val/Test split with stratification
        print("\nSplitting data...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_seq, y_seq, test_size=TEST_SIZE,
            random_state=RANDOM_STATE, stratify=y_seq
        )

        val_ratio = VAL_SIZE / (1 - TEST_SIZE)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio),
            random_state=RANDOM_STATE, stratify=y_temp
        )

        del X_seq, y_seq, X_temp, y_temp
        gc.collect()

        print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

        # Save if path provided
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
                        "continuous_feature_names": self.continuous_feature_names,
                        "binary_feature_names": self.binary_feature_names,
                        "num_classes": self.num_classes,
                        "n_continuous": len(self.continuous_feature_names),
                        "n_binary": len(self.binary_feature_names),
                    },
                    f,
                )

            print(f"\nData saved to {save_path}")

        return (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            self.feature_names, self.scaler, self.label_encoder,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process IPFIX Records JSON dataset")
    parser.add_argument("--max_records", type=int, default=None,
                        help="Maximum records to load (for testing)")
    parser.add_argument("--seq_length", type=int, default=SEQUENCE_LENGTH,
                        help="Sequence length for sliding window")
    parser.add_argument("--stride", type=int, default=STRIDE,
                        help="Stride for sliding window")
    parser.add_argument("--min_samples", type=int, default=MIN_SAMPLES_PER_CLASS,
                        help="Minimum samples per class")
    parser.add_argument("--save_path", type=str,
                        default=str(PROCESSED_DATA_DIR / "json_native"),
                        help="Output directory for processed data")
    args = parser.parse_args()

    processor = JsonIoTDataProcessor()
    processor.process_all(
        save_path=Path(args.save_path),
        max_records=args.max_records,
        seq_length=args.seq_length,
        stride=args.stride,
        min_samples=args.min_samples,
    )
