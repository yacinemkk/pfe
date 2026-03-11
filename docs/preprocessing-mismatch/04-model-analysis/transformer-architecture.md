# Transformer Architecture

## Architecture Overview

The Transformer model uses self-attention to process sequences of flow records.

---

## Architecture Details

```
Input: (batch_size, seq_len=10, features=37)
       ↓
Linear Projection: (batch_size, seq_len=10, d_model=64)
       ↓
Add Positional Encoding (sinusoidal)
       ↓
TransformerEncoder (3 layers, 4 heads, d_ff=256, GELU activation)
       ↓
Mean Pooling over sequence: (batch_size, 64)
       ↓
LayerNorm
       ↓
Linear(64 → 128) → GELU → Dropout(0.2)
       ↓
Linear(128 → 18)
       ↓
Output: (batch_size, 18) logits
```

---

## Positional Encoding

### What It Does

Positional encoding adds position information to each flow in the sequence:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### What It Means

- Flow at position 0 gets different encoding than position 9
- The model can distinguish which flow came first
- But it doesn't know the actual timing between flows

---

## Self-Attention Mechanism

### How Attention Works

For each flow position, the model computes attention weights over all positions:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

### What Gets Attended

The Q, K, V are derived from **projected flow statistics**:

```
Flow_1 → Linear → [q1, k1, v1]
Flow_2 → Linear → [q2, k2, v2]
...
Flow_10 → Linear → [q10, k10, v10]
```

The attention score between Flow_i and Flow_j depends on the similarity of their feature vectors.

### What This Captures

- Which flows are most relevant for classification
- Relationships between flow statistics at different positions
- Not: packet-level patterns

---

## Aggregation: Mean Pooling

After the Transformer encoder, the model uses **mean pooling**:

```
output = mean([Flow_1_encoded, Flow_2_encoded, ..., Flow_10_encoded])
```

This means:
- All positions contribute equally
- No special "CLS" token
- Position importance is learned through attention

---

## What Transformer Learns

### Flow-Level Attention

| Can Learn | Cannot Learn |
|-----------|--------------|
| "Flow 3 is most important for this device" | "Packets 5-7 in Flow 3 are important" |
| "Flows with certain stats should be weighted more" | "The pattern of packet directions matters" |
| "Early flows vs late flows have different importance" | "The timing between packets matters" |

### The Attention Paradox

The Transformer's attention is powerful for:
- NLP: Attend to important words in context
- Vision: Attend to important patches in an image
- Time series: Attend to important timesteps

But for **aggregated flow sequences**, attention can only operate on:
- The 37 aggregated features
- The relative position (1st, 2nd, ... 10th flow)

It cannot attend to **what happened within each flow**.

---

## Comparison with NLP Transformers

| Aspect | NLP | Our Transformer |
|--------|-----|-----------------|
| Token | Word/subword | Flow record |
| Token content | Word embedding | 37 aggregated features |
| Sequence meaning | Sentence semantics | Device activity over time |
| What's attended | Word relationships | Flow relationships |
| Lost information | None (token is atomic) | Packet-level timing |

---

## Parameter Count

| Layer | Parameters |
|-------|------------|
| Input projection | 37 × 64 = 2,368 |
| Transformer encoder (3 layers) | ~150,000 |
| Linear 1 (64 → 128) | 8,192 |
| Linear 2 (128 → 18) | 2,322 |
| **Total** | **~163,000** |

---

## What Would Be Needed for True Packet Attention

To enable attention over packets:

1. **Packet-level tokens**: Each packet is a token
2. **Packet embeddings**: Direction, size, timing
3. **Hierarchical attention**: Packets → Flows → Device

The `firstEightNonEmptyPacketDirections` field could provide 8 packet tokens per flow.

---

Return to [Model Analysis Index](./README.md) | Return to [Main Index](../README.md)
