# LSTM Architecture

## Architecture Overview

The LSTM model processes sequences of flow records using a bidirectional LSTM.

---

## Architecture Details

```
Input: (batch_size, seq_len=10, features=37)
       ↓
Bidirectional LSTM (input_size=37, hidden_size=128, num_layers=2, dropout=0.3)
       ↓
Concatenate final hidden states: (batch_size, 256)
       ↓
Linear(256 → 128) → ReLU → Dropout(0.3)
       ↓
Linear(128 → 18)
       ↓
Output: (batch_size, 18) logits
```

---

## How LSTM Processes Sequences

### Bidirectional Processing

The LSTM processes the sequence in both directions:

```
Forward:  Flow_1 → Flow_2 → Flow_3 → ... → Flow_10
Backward: Flow_10 → Flow_9 → Flow_8 → ... → Flow_1
```

### Hidden State Evolution

At each timestep, the LSTM updates its hidden state:

```
h_1 = LSTM(Flow_1, h_0)
h_2 = LSTM(Flow_2, h_1)
...
h_10 = LSTM(Flow_10, h_9)
```

The final hidden state `h_10` (forward) and `h_1` (backward) are concatenated to form the sequence representation.

---

## What LSTM Learns

### Flow-Level Patterns

- How flow statistics evolve over time
- Correlations between consecutive flows
- Device-specific flow patterns

### What's Captured

| Pattern | Can Learn? | Example |
|---------|------------|---------|
| Flow count trends | Yes | "Device A has increasing packet counts" |
| Flow duration patterns | Yes | "Device B has short bursts" |
| Protocol usage over time | Yes | "Device C uses DNS then HTTP" |

### What's NOT Captured

| Pattern | Can Learn? | Reason |
|---------|------------|--------|
| Packet burst patterns | No | Packets are aggregated |
| Request-response timing | No | Only average IAT |
| Packet direction sequences | No | Not in input |

---

## The Temporal Mismatch

### What LSTM Expects

LSTM is designed for sequences where each element has temporal meaning:

```
Example (NLP): ["The", "cat", "sat", "on", "the", "mat"]
Example (Speech): [audio_frame_1, audio_frame_2, ...]
Example (Video): [frame_1, frame_2, frame_3, ...]
```

### What It Gets

```
[Flow_1_aggregates, Flow_2_aggregates, ..., Flow_10_aggregates]
```

Each flow aggregate is a **summary of many packets**. The "temporal" relationship between Flow_1 and Flow_2 is already lost - it's just that Flow_1 happened before Flow_2, but we don't know what happened within either flow.

---

## Parameter Count

| Layer | Parameters |
|-------|------------|
| LSTM (bidirectional, 2 layers) | ~525,000 |
| Linear 1 (256 → 128) | 32,768 |
| Linear 2 (128 → 18) | 2,322 |
| **Total** | **~600,000** |

---

## What Would Be Needed for True Temporal Learning

To learn packet-level patterns, the LSTM would need:

1. **Packet-level input**: Each timestep is a packet, not a flow
2. **Timing information**: Actual inter-packet intervals, not averages
3. **Direction information**: Which packets are inbound/outbound

The JSON field `firstEightNonEmptyPacketDirections` could provide direction information for a limited number of packets.

---

Return to [Model Analysis Index](./README.md) | Return to [Main Index](../README.md)
