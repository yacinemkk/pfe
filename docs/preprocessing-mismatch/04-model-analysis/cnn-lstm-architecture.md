# CNN-LSTM Architecture

## Architecture Overview

The CNN-LSTM model combines 1D convolution for local pattern extraction with LSTM for temporal modeling.

---

## Architecture Details

```
Input: (batch_size, seq_len=10, features=37)
       ↓
Permute: (batch_size, features=37, seq_len=10)
       ↓
1D CNN:
  Conv1d(37 → 64, kernel=3) → BatchNorm → ReLU → Dropout
  Conv1d(64 → 128, kernel=3) → BatchNorm → ReLU → Dropout
       ↓
Permute: (batch_size, seq_len=8, channels=128)  # Note: seq_len reduced by conv
       ↓
Bidirectional LSTM (hidden_size=128, num_layers=2)
       ↓
Attention: 
  Linear(256 → 128) → Tanh → Linear(128 → 1) → Softmax
  Weighted sum of LSTM outputs
       ↓
Linear(256 → 128) → ReLU → Dropout
       ↓
Linear(128 → 18)
       ↓
Output: (batch_size, 18) logits
```

---

## The 1D Convolution Component

### What 1D CNN Does

Applies filters across the sequence dimension:

```
Input: [Flow_1, Flow_2, Flow_3, ..., Flow_10]
            ↓ (kernel size 3)
Filter: extracts patterns from [Flow_1, Flow_2, Flow_3]
        extracts patterns from [Flow_2, Flow_3, Flow_4]
        ...
```

### What It Captures

- Local patterns in the flow sequence
- Feature combinations (37 features → 128 channels)
- Short-range dependencies (kernel size 3 = 3 consecutive flows)

### What It Cannot Capture

- Long-range dependencies (covered by LSTM)
- Packet-level patterns (flows are already aggregated)

---

## The LSTM Component

### What LSTM Does

After CNN extracts local features, LSTM models temporal dependencies:

```
CNN output: [Feature_1, Feature_2, ..., Feature_8]  (8 timesteps after conv)
     ↓
LSTM: models relationships between these features
```

### Sequence Length Reduction

- Input: 10 flows
- After Conv1d(k=3): 10 - 3 + 1 = 8 positions (after first conv)
- After second Conv1d(k=3): 8 - 3 + 1 = 6 positions

The LSTM operates on 6-8 positions (depending on padding).

---

## The Attention Component

### What Attention Does

Computes weights for each LSTM output position:

```
Attention weights: [w_1, w_2, ..., w_n] where sum = 1
Context vector: w_1 * h_1 + w_2 * h_2 + ... + w_n * h_n
```

### What It Captures

- Which positions in the CNN-LSTM output are most important
- Adaptive weighting based on the sequence content

---

## What CNN-LSTM Learns

### Strengths of This Architecture

| Component | What It Contributes |
|-----------|---------------------|
| CNN | Local feature patterns, translation invariance |
| LSTM | Long-range temporal dependencies |
| Attention | Position importance weighting |

### The Combined Effect

```
Flows → CNN (local patterns) → LSTM (temporal) → Attention (weighting) → Classification
```

### What's Still Missing

| What We Want | What We Get |
|--------------|-------------|
| Packet burst patterns | Flow-level burst patterns |
| Inter-packet timing | Inter-flow timing (average) |
| Packet direction sequences | Not captured |

Even with CNN + LSTM + Attention, the fundamental limitation remains: **the input is aggregated flow statistics, not packets**.

---

## Parameter Count

| Layer | Parameters |
|-------|------------|
| Conv1d layer 1 | 37 × 64 × 3 = 7,104 |
| Conv1d layer 2 | 64 × 128 × 3 = 24,576 |
| LSTM (bidirectional) | ~395,000 |
| Attention layers | ~35,000 |
| Classifier | ~35,000 |
| **Total** | **~760,000** |

---

## Why CNN-LSTM Doesn't Solve the Problem

The CNN operates on **sequence positions**, not packets:

```
Correct (if we had packets):
  [Pkt_1, Pkt_2, Pkt_3] → CNN → local packet pattern
  
Current (with flows):
  [Flow_1, Flow_2, Flow_3] → CNN → local flow pattern
```

Each "position" in the current input is already an aggregate. The CNN's local patterns are patterns of aggregates, not patterns of packets.

---

Return to [Model Analysis Index](./README.md) | Return to [Main Index](../README.md)
