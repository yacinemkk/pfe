# Gap 2: Feature Embeddings

## The Problem

The current system uses **raw numerical features** without learned embeddings.

---

## What Are Feature Embeddings?

### Current Approach

Features are fed directly to the model:

```
[outPacketCount=50, outByteCount=12800, outAvgIAT=0.03, ...]
       ↓
   StandardScaler (normalize)
       ↓
   Model (LSTM/Transformer)
```

Each feature is treated as an independent numerical value.

### With Embeddings

Features are transformed into learned representations:

```
outPacketCount=50 → Embedding layer → [0.1, 0.3, -0.2, ...]
outByteCount=12800 → Embedding layer → [0.5, -0.1, 0.4, ...]
...
       ↓
   Concatenate embeddings
       ↓
   Model
```

---

## Why Embeddings Help

### 1. Capture Feature Relationships

Some features are related:
- `outPacketCount` and `outByteCount` (more packets = more bytes)
- `outAvgPacketSize` = `outByteCount / outPacketCount`

Embeddings can learn these relationships implicitly.

### 2. Handle Different Feature Types

Current features include:
- **Continuous**: `duration`, `outAvgIAT`
- **Discrete**: `ipProto` (6=TCP, 17=UDP)
- **Binary flags**: `http`, `https`, `dns`

Embeddings can handle each type appropriately.

### 3. Enable Better Generalization

Raw numerical features force the model to learn:
- What packet counts are "high" vs "low"
- How to interpret protocol numbers

Embeddings can encode this knowledge.

---

## Current Implementation

### Feature Types

| Type | Features | Current Handling |
|------|----------|------------------|
| Continuous | `duration`, `outAvgIAT`, `outAvgPacketSize` | StandardScaler |
| Discrete | `ipProto` | Treated as continuous |
| Binary | `http`, `https`, `dns`, `tcp`, `udp` | Treated as continuous |
| Counts | `outPacketCount`, `outByteCount` | StandardScaler |

### Problems

1. **Protocol number** (`ipProto=6` vs `ipProto=17`) has no inherent ordering, but is treated as continuous
2. **Binary flags** don't need normalization, but get it anyway
3. **Counts** span different ranges and may have long tails

---

## Proposed Embedding Approach

### For Continuous Features

```
Continuous feature → Bucketize → Embedding
```

Example:
```
outPacketCount=50 → bucket "medium" → embedding
```

Or use:
```
Continuous feature → Normalize → Linear projection
```

### For Discrete Features

```
Discrete feature → One-hot → Embedding
```

Example:
```
ipProto=6 (TCP) → [1, 0, 0, ...] → embedding_tcp
ipProto=17 (UDP) → [0, 1, 0, ...] → embedding_udp
```

### For Binary Flags

```
Binary flags → Concatenate → Linear projection
```

Or use directly (already 0/1).

---

## Why This Gap Matters Less

### Current Performance is Acceptable

- Clean accuracy: ~90%
- Models do learn from raw features

### The Bigger Problem is Data Granularity

Even with perfect embeddings, the model cannot learn packet-level patterns because:
- Packets are aggregated into flows
- Temporal information is lost

### Priority

Fixing **Gap 1 (Data Granularity)** is more important than this gap.

Embeddings would be an optimization after fixing the fundamental issue.

---

## Recommendation

1. **First**: Fix Gap 1 (extract packet directions from JSON)
2. **Then**: Add embeddings for packet direction sequences
3. **Later**: Consider embeddings for flow-level features

---

Return to [Gap Analysis Index](./README.md) | Return to [Main Index](../README.md)
