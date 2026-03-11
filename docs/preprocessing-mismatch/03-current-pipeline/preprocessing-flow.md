# Preprocessing Flow

## Step-by-Step Transformations

This document traces the data through each preprocessing step.

---

## Step 1: Load CSV Files

**Source**: `src/data/preprocessor.py` - `load_all_data()`

**Input**: 12 CSV files (`home1_labeled.csv` through `home12_labeled.csv`)

**Process**:
- Read each CSV file
- Add `source_file` column for grouping
- Concatenate all files into single DataFrame

**Output**: Single DataFrame with all flows

**Shape**: (~16M rows, ~44 columns)

---

## Step 2: Clean Data

**Source**: `src/data/preprocessor.py` - `clean_data()`

**Transformations**:

| Transformation | Purpose |
|----------------|---------|
| Drop duplicates | Remove duplicate flow records |
| Drop N/A labels | Ensure valid device labels |
| Filter to 18 IoT classes | Focus on target devices |
| Remove classes < 500 samples | Ensure enough training data |

**Features dropped** (identifying information):
- `start`, `srcMac`, `destMac` (timing/addresses)
- `srcIP`, `destIP` (would memorize devices)
- `srcPort`, `destPort` (would memorize services)
- `device` (redundant with `name`)

**Output**: Cleaned DataFrame

**Shape**: (~16M rows, fewer columns)

---

## Step 3: Select Features

**Source**: `src/data/preprocessor.py` - `select_features()`

**Process**:
- Extract 37 features from `FEATURES_TO_KEEP`
- Separate X (features) and y (labels)

**Features selected**:

| Category | Count | Examples |
|----------|-------|----------|
| Duration/Protocol | 2 | `duration`, `ipProto` |
| Packet counts | 4 | `outPacketCount`, `inPacketCount` |
| Byte counts | 2 | `outByteCount`, `inByteCount` |
| Packet size stats | 14 | `outAvgPacketSize`, `outMaxPktSize` |
| IAT stats | 4 | `outAvgIAT`, `outStdevIAT` |
| Protocol flags | 8 | `http`, `https`, `dns`, `tcp`, `udp` |
| Network flags | 3 | `lan`, `wan`, `deviceInitiated` |

**Output**: X array, y array

**Shape**: X = (~16M, 37), y = (~16M,)

---

## Step 4: Normalize Features

**Source**: `src/data/preprocessor.py` - `fit_transform()`

**Method**: StandardScaler (z-score normalization)

**Formula**: `z = (x - mean) / std`

**Effect**:
- Mean = 0 for each feature
- Std = 1 for each feature
- Values typically in [-3, 3] range

**Why**: Neural networks train better with normalized inputs

**Output**: Normalized X array

**Shape**: (~16M, 37)

---

## Step 5: Encode Labels

**Source**: `src/data/preprocessor.py` - `encode_labels()`

**Method**: LabelEncoder from sklearn

**Process**:
- Map device name strings to integers
- Example: "echo-dot" → 0, "google-nest-mini" → 1

**Output**: Encoded y array

**Shape**: (~16M,)

---

## Step 6: Create Sequences

**Source**: `train_adversarial.py` - `create_sequences_with_stride()`

**Process**:
- Use sliding window of length 10
- Stride of 5 (50% overlap)
- Respect source file boundaries (don't mix flows from different homes)

**Input**: X = (~16M, 37), y = (~16M,)

**Output**: 
- X_seq = (n_sequences, 10, 37)
- y_seq = (n_sequences,)

**Label**: The label is taken from the **last flow** in each sequence

**Shape calculation**:
- ~16M flows / stride of 5 ≈ 3.2M sequences

---

## Step 7: Train/Val/Test Split

**Source**: `train_adversarial.py` - `preprocess_and_split()`

**Method**: Stratified train_test_split

**Ratios**:
- Train: 70%
- Validation: 10%
- Test: 20%

**Why stratified**: Ensure all device classes are represented in each split

**Output**:
- X_train, X_val, X_test
- y_train, y_val, y_test

---

## Step 8: Create DataLoaders

**Source**: `train_adversarial.py` - `create_dataloaders()`

**Process**:
- Wrap arrays in PyTorch Dataset (`IoTSequenceDataset`)
- Create DataLoader with batch_size=64
- Shuffle training data

**Output**:
- train_loader, val_loader, test_loader
- Each yields (X_batch, y_batch) where X_batch.shape = (64, 10, 37)

---

## What Happens to the Data

### Before Preprocessing

```
Raw CSV row:
{
  "duration": 1500,
  "outPacketCount": 50,
  "outByteCount": 12800,
  "outAvgIAT": 0.03,
  "name": "echo-dot"
}
```

### After Preprocessing

```
Sequence element (one timestep):
[0.23, -0.45, 1.12, 0.87, ...]  # 37 normalized values

Full sequence (10 timesteps):
[[0.23, -0.45, ...],   # flow 1
 [0.11, 0.32, ...],    # flow 2
 ...
 [0.45, -0.21, ...]]   # flow 10

Label: 5 (integer for "echo-dot")
```

### What's Lost

- **Packet timing**: Only average IAT remains
- **Packet sizes**: Only average/max remains
- **Packet directions**: Completely absent
- **Flow relationships**: Treated as independent timesteps

---

Return to [Pipeline Index](./README.md) | Return to [Main Index](../README.md)
