# Option B: Packet Embeddings

## Overview

Extract packet direction sequences from JSON and create embeddings for temporal modeling.

---

## What This Fixes

- Provides packet-level temporal information
- Enables true sequence modeling
- Improves adversarial robustness
- Leverages existing but unused data

## What This Doesn't Fix

- Still using 8 packets per flow (limited)
- No raw packet timing (directions only)
- No payload information

---

## The Key Resource

### `firstEightNonEmptyPacketDirections`

From JSON files, each flow has:

```
firstEightNonEmptyPacketDirections: [1, 0, 1, 0, 1, 1, 0, 1]
```

Where:
- 1 = outbound packet
- 0 = inbound packet

This provides the **direction sequence** of the first 8 packets.

---

## Proposed Architecture

### New Input Representation

```
Current:  (batch, 10, 37)  # 10 flows, 37 features each
Proposed: (batch, 10, 37 + 8*embedding_dim)  # Flow stats + packet embeddings
```

### Architecture Flow

```
Packet Directions [d1, d2, d3, d4, d5, d6, d7, d8]  (per flow)
        ↓
Direction Embedding Layer (learned, e.g., 16-dim)
        ↓
Direction Sequence Encoder (LSTM or Transformer)
        ↓
Direction Representation Vector (e.g., 32-dim)
        ↓
Concatenate with Flow Statistics (37-dim)
        ↓
[Direction_Vector | Flow_Stats]  (e.g., 32 + 37 = 69-dim)
        ↓
Existing Model (LSTM/Transformer)
```

---

## Implementation Steps

### Phase 1: Data Extraction

| Step | Description |
|------|-------------|
| 1.1 | Parse JSON files (JSON Lines format) |
| 1.2 | Extract `firstEightNonEmptyPacketDirections` |
| 1.3 | Match flows to CSV using IP/port/timestamp |
| 1.4 | Save extracted data to efficient format (Parquet) |

### Phase 2: Model Modifications

| Step | Description |
|------|-------------|
| 2.1 | Create direction embedding layer |
| 2.2 | Create direction sequence encoder |
| 2.3 | Modify existing models to accept new input |
| 2.4 | Update training pipeline |

### Phase 3: Attack Modifications

| Step | Description |
|------|-------------|
| 3.1 | Design packet direction attack |
| 3.2 | Implement constrained perturbation |
| 3.3 | Update adversarial training |

---

## New Components

### `src/data/json_extractor.py`

- Parse JSON files
- Extract packet directions
- Match to CSV flows

### `src/models/packet_embedding.py`

- `PacketDirectionEmbedding` class
- Direction sequence encoder

### `src/models/combined_model.py`

- Combines packet embeddings with flow stats
- Modified LSTM/Transformer

---

## Expected Outcomes

### Model Performance

| Metric | Current | After Fix |
|--------|---------|-----------|
| Clean accuracy | 90% | 90-92% |
| Feature attack accuracy | 46% | 50-55% |
| Sequence attack accuracy | 70% | 55-65% |

### Adversarial Robustness

With packet directions:
- Attackers must modify direction patterns (harder)
- Temporal patterns provide additional defense
- Model has more information to learn robust features

---

## Risks and Mitigations

### Risk 1: JSON Matching Failure

**Risk**: Cannot reliably match JSON flows to CSV flows

**Mitigation**: 
- Use multiple fields (IP, port, timestamp, duration)
- Accept partial matches
- Document match rate

### Risk 2: Limited Packet Directions

**Risk**: Only 8 packets per flow may be insufficient

**Mitigation**:
- 8 directions still capture significant temporal pattern
- Can augment with additional features

### Risk 3: Data Size

**Risk**: JSON files are 19GB, parsing may be slow

**Mitigation**:
- One-time extraction to Parquet
- Process incrementally
- Cache extracted data

---

## Recommendation

This is the **recommended option** because:

1. Uses data we already have
2. Provides significant improvement
3. Moderate effort
4. Backward compatible

---

Return to [Proposed Solutions Index](./README.md) | Return to [Main Index](../README.md)
