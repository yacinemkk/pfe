# Current Pipeline

## Overview

This section documents how the current preprocessing and training pipeline works, from raw data to model input.

---

## Pipeline Summary

```
CSV Files (home*_labeled.csv)
        ↓
    Load & Concatenate
        ↓
    Filter to 18 IoT classes
        ↓
    Select 37 features
        ↓
    StandardScaler normalization
        ↓
    LabelEncoder for device names
        ↓
    Create sequences (sliding window)
        ↓
    Train/Val/Test split (70/10/20)
        ↓
    PyTorch DataLoaders
        ↓
    LSTM / Transformer / CNN-LSTM
```

---

## Documents in This Section

| Document | Description |
|----------|-------------|
| [preprocessing-flow.md](./preprocessing-flow.md) | Step-by-step data transformations |
| [sequence-creation.md](./sequence-creation.md) | How sequences are built (the flaw) |
| [model-inputs.md](./model-inputs.md) | What models actually receive |

---

## Key Components

### 1. Data Loading (`src/data/preprocessor.py`)

The `IoTDataProcessor` class handles:
- Loading CSV files from Google Drive
- Cleaning (drop duplicates, filter classes)
- Feature selection
- Normalization
- Label encoding

### 2. Sequence Creation (`train_adversarial.py`)

The `create_sequences_with_stride()` function:
- Takes normalized feature matrix
- Creates sliding windows of length 10
- Uses stride of 5
- Respects source file boundaries

### 3. Model Training (`train_adversarial.py`)

The training script:
- Supports LSTM, Transformer, CNN-LSTM
- Implements 3-phase adversarial training
- Evaluates against multiple attack types

---

## Input/Output Shapes

| Stage | Shape |
|-------|-------|
| Raw CSV | (n_flows, ~44 columns) |
| After filtering | (n_flows, 37 features) |
| After normalization | (n_flows, 37) standardized |
| After sequence creation | (n_sequences, 10, 37) |
| Model input | (batch_size, 10, 37) |

---

## The Fundamental Assumption

The pipeline assumes that **a sequence of 10 flow records** captures temporal patterns that can distinguish IoT devices.

The problem: Each flow record is already an aggregate of many packets. The "temporal patterns" are already lost in aggregation.

---

Return to [Main Index](../README.md)
