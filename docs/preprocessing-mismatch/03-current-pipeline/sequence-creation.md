# Sequence Creation

## The Core Mechanism

Sequence creation is where the preprocessing pipeline attempts to create "temporal" input from flow records.

---

## How Sequences Are Created

### Sliding Window Approach

```
Flow records:  [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, ...]
                                                   
Sequence 1:    [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10]  → Label: F10's device
                 ↓ (stride=5)
Sequence 2:            [F6, F7, F8, F9, F10, F11, F12, F13, F14, F15] → Label: F15's device
```

### Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `seq_length` | 10 | Number of flows per sequence |
| `stride` | 5 | Step between sequences |
| Label | Last flow | Device type of last flow in sequence |

### Code Logic

```python
def create_sequences_with_stride(X, y, seq_length, stride, source_groups):
    for i in range(0, n_samples, stride):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length-1])  # Last element's label
```

---

## What Each Sequence Element Represents

### A Flow Record Contains:

| Field | Meaning | Example |
|-------|---------|---------|
| `outPacketCount` | Number of outbound packets in flow | 50 |
| `outByteCount` | Total outbound bytes | 12800 |
| `outAvgIAT` | Average inter-arrival time | 0.03s |
| `outAvgPacketSize` | Average packet size | 256 bytes |

### What's Missing:

- **Which packets went first?** No packet order
- **How were packets spaced?** Only average timing
- **What was the communication pattern?** No direction sequence

### An Example:

A flow with `outPacketCount=50` could represent:
1. 50 small packets in quick succession (sensor burst)
2. 50 large packets spread over time (video stream)
3. A mix of both

The current representation **cannot distinguish** between these cases.

---

## The Flaw

### Assumption

The pipeline assumes that **10 consecutive flows** capture a temporal pattern that distinguishes devices.

### Reality

Each flow is already a **summary of many packets**. The sequence [Flow1, Flow2, ..., Flow10] represents:

```
Flow 1: summary of packets 1-50
Flow 2: summary of packets 51-100
...
Flow 10: summary of packets 451-500
```

The "temporal" pattern within each flow is **already lost**.

### What Models Actually Learn

Given a sequence of flows, models learn:
- Device A typically has flows with X packet counts
- Device B typically has flows with Y byte counts
- The sequence helps, but only because flows are correlated

They **cannot** learn:
- Device A sends bursty traffic
- Device B has a request-response pattern
- Device C streams continuously

---

## Source File Boundaries

### Why It Matters

The code respects `source_groups` (which home the flow came from):

```python
for group_id in np.unique(source_groups):
    mask = source_groups == group_id
    X_group = X[mask]
    # Create sequences within this group only
```

### What This Prevents

Without this boundary:
- Sequence could contain flows from different homes
- Model might learn home-specific patterns instead of device patterns

### What This Doesn't Fix

The boundary doesn't help with the fundamental problem:
- Each flow is still an aggregate
- Temporal packet patterns are still lost

---

## Comparison: What Could Be

### Current (Flow Sequences)

```
Input: [Flow1_stats, Flow2_stats, ..., Flow10_stats]
Each Flow_stats = (37 aggregated features)
Shape: (batch, 10, 37)
```

### Alternative (Packet Sequences)

```
Input: [Pkt1_dir, Pkt2_dir, ..., Pkt8_dir, Flow_stats]
Each Pkt_dir = (in/out direction)
Flow_stats = (37 aggregated features)
Shape: (batch, 8, embedding_dim) + (batch, 37)
```

### Why Alternative is Better

1. **Actual temporal pattern**: The sequence of packet directions is real temporal data
2. **Harder to spoof**: An attacker would need to change communication pattern, not just statistics
3. **More interpretable**: We can see which packet patterns the model attends to

---

Return to [Pipeline Index](./README.md) | Return to [Main Index](../README.md)
