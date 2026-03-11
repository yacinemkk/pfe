# Model Inputs

## What Models Actually Receive

This document describes the exact input format that LSTM, Transformer, and CNN-LSTM models receive.

---

## Input Shape

All models receive the same input:

```
Shape: (batch_size, sequence_length, n_features)
       (64,           10,              37)
```

| Dimension | Value | Meaning |
|-----------|-------|---------|
| batch_size | 64 | Number of sequences processed together |
| sequence_length | 10 | Number of flow records per sequence |
| n_features | 37 | Features per flow record |

---

## Feature Vector (37 dimensions)

Each timestep in the sequence is a 37-dimensional vector:

### Index Mapping

| Index Range | Feature Type | Example Features |
|-------------|--------------|------------------|
| 0 | duration | Flow duration |
| 1 | ipProto | Protocol number |
| 2-5 | Packet counts | outPacketCount, inPacketCount |
| 6-7 | Byte counts | outByteCount, inByteCount |
| 8-21 | Packet size stats | avgPacketSize, maxPktSize, stdev |
| 22-25 | IAT stats | avgIAT, stdevIAT |
| 26-33 | Protocol flags | http, https, dns, tcp, udp |
| 34-36 | Network flags | lan, wan, deviceInitiated |

### Values After Normalization

After StandardScaler:
- Mean = 0
- Std = 1
- Most values in [-3, 3]

---

## What LSTM Receives

### Processing

```
Input: (64, 10, 37)
       ↓
nn.LSTM(input_size=37, hidden_size=128, bidirectional=True)
       ↓
Hidden states: (64, 10, 256)  # 128 * 2 for bidirectional
       ↓
Take last hidden state: (64, 256)
       ↓
Classifier: Linear → ReLU → Linear
       ↓
Output: (64, 18) logits
```

### What LSTM Learns

- Temporal relationships between consecutive flows
- Patterns in how flow statistics change over time
- Not: packet-level patterns (not available)

---

## What Transformer Receives

### Processing

```
Input: (64, 10, 37)
       ↓
Linear projection: (64, 10, 64)  # d_model=64
       ↓
Add positional encoding
       ↓
TransformerEncoder (3 layers, 4 heads)
       ↓
Mean pooling: (64, 64)
       ↓
Classifier: Linear → GELU → Linear
       ↓
Output: (64, 18) logits
```

### Attention Over What?

The Transformer's self-attention operates on **projected flow statistics**:

```
Attention(Q, K, V) where Q, K, V are derived from:
  [Flow1_features, Flow2_features, ..., Flow10_features]
```

Each "token" is a flow record, not a packet.

### What Attention Learns

- Which flows in the sequence are most relevant
- Relationships between flow statistics at different positions
- Not: which packets are most relevant (packets don't exist)

---

## What CNN-LSTM Receives

### Processing

```
Input: (64, 10, 37)
       ↓
Permute: (64, 37, 10)  # CNN expects (batch, channels, length)
       ↓
1D CNN: Conv1d(37→64, k=3) → Conv1d(64→128, k=3)
       ↓
Permute: (64, 10, 128)
       ↓
LSTM: bidirectional, hidden=128
       ↓
Attention: weighted sum over timesteps
       ↓
Classifier
       ↓
Output: (64, 18) logits
```

### What CNN Learns

- Local patterns in the sequence of flows
- Feature combinations via convolution
- Not: packet-level patterns

---

## The Common Problem

All three models receive the same input: **aggregated flow statistics**.

| Model | What It Could Learn | What It Actually Learns |
|-------|---------------------|-------------------------|
| LSTM | Packet-level sequences | Flow-level sequences |
| Transformer | Attend to important packets | Attend to important flows |
| CNN-LSTM | Local packet patterns | Local flow patterns |

The "temporal" processing is applied to **sequences of aggregates**, not sequences of packets.

---

## Information Flow Diagram

```
Raw Packets (not available)
       ↓
   IPFIX Collector (aggregates into flows)
       ↓
   CSV Files (flow statistics)
       ↓
   Preprocessing (select 37 features, normalize)
       ↓
   Sequence Creation (10 flows per sequence)
       ↓
   Model Input: (batch, 10, 37)
       ↓
   LSTM / Transformer / CNN-LSTM
       ↓
   Device Classification
```

The **aggregation happens before** the model sees any data. The model has no access to packet-level information.

---

## What's Missing

To enable true temporal learning, the input should include:

| Desired Input | Current Status |
|---------------|----------------|
| Packet sizes | Aggregated to average |
| Packet timings | Aggregated to average IAT |
| Packet directions | Not in CSV at all |
| Payload patterns | Not captured |

The only way to get packet-level information is to use the JSON files, which contain `firstEightNonEmptyPacketDirections`.

---

Return to [Pipeline Index](./README.md) | Return to [Main Index](../README.md)
