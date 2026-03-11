# Model Analysis

## Overview

This section analyzes how each model architecture processes the input and what temporal patterns they can actually learn.

---

## Key Findings

1. **All models receive the same input**: Flow-level statistics, not packet-level data
2. **Temporal processing is superficial**: Models process sequences of aggregates, not temporal packet patterns
3. **No learned embeddings**: Raw numerical features are fed directly to models
4. **Attention operates on flows**: Transformer attention is over flow positions, not packet positions

---

## Documents in This Section

| Document | Description |
|----------|-------------|
| [lstm-architecture.md](./lstm-architecture.md) | LSTM: sequential processing of flows |
| [transformer-architecture.md](./transformer-architecture.md) | Transformer: attention on flow statistics |
| [cnn-lstm-architecture.md](./cnn-lstm-architecture.md) | CNN-LSTM: hybrid approach |

---

## Architecture Comparison

| Aspect | LSTM | Transformer | CNN-LSTM |
|--------|------|-------------|----------|
| **Input processing** | Sequential | Parallel | Local → Sequential |
| **Temporal modeling** | Hidden states | Positional encoding + attention | CNN features → LSTM |
| **Sequence aggregation** | Last hidden state | Mean pooling | Attention over LSTM outputs |
| **Feature transformation** | None (raw) | Linear projection | CNN channels |
| **Parameters** | ~600K | ~163K | ~760K |

---

## The Common Limitation

All three models are designed to process **sequences**. However, the "sequence" they receive is:

```
[Flow_1_stats, Flow_2_stats, ..., Flow_10_stats]
```

Where each `Flow_X_stats` is a **37-dimensional vector of aggregated statistics**.

This is fundamentally different from:

```
[Packet_1, Packet_2, ..., Packet_N]
```

Where each `Packet_X` is an **individual packet** with timing and direction.

---

## What Models Could Learn vs What They Learn

| Model Type | Could Learn (with packets) | Learns (with flows) |
|------------|---------------------------|---------------------|
| LSTM | Packet-level patterns, timing sequences | Flow-to-flow correlations |
| Transformer | Attend to important packets | Attend to important flows |
| CNN-LSTM | Local packet burst patterns | Local flow feature patterns |

---

Return to [Main Index](../README.md)
