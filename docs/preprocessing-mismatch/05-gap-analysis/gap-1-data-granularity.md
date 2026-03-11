# Gap 1: Data Granularity

## The Problem

The current system operates on **flow-level aggregates** rather than **packet-level sequences**.

---

## What is Flow-Level Aggregation?

### Original Data: Packets

A network packet contains:
- Timestamp
- Source/destination IP and port
- Protocol
- Payload size
- Direction (inbound/outbound)

### After IPFIX Aggregation: Flow

An IPFIX flow aggregates many packets:

| Metric | How It's Computed | Example |
|--------|-------------------|---------|
| `outPacketCount` | Count of outbound packets | 50 |
| `outByteCount` | Sum of outbound bytes | 12800 |
| `outAvgPacketSize` | Mean packet size | 256 |
| `outAvgIAT` | Mean inter-arrival time | 0.03s |
| `outStdevIAT` | Std dev of inter-arrival times | 0.01s |

### What's Lost

| Original Packet Info | Flow Representation | What's Lost |
|---------------------|---------------------|-------------|
| Packet 1: 100 bytes, t=0s | | |
| Packet 2: 100 bytes, t=0.01s | outPacketCount=3 | Order of sizes |
| Packet 3: 500 bytes, t=0.05s | outAvgPacketSize=233 | Burst pattern |
| | outStdevIAT=0.02 | Exact timing |

---

## Why This Matters for Device Identification

### Different Devices Have Different Packet Patterns

| Device | Packet Pattern | Flow Stats (Current) |
|--------|---------------|---------------------|
| Security camera | Bursts of large packets (video frames) | High byte count, large avg size |
| Smart plug | Small periodic packets (status) | Low byte count, regular timing |
| Motion sensor | Event-triggered bursts | Variable counts |
| Smart speaker | Streaming packets (audio) | Medium size, continuous |

### The Problem

Two devices can have **similar flow statistics** but **different packet patterns**:

```
Device A (camera streaming):
  Packet sequence: [1000, 1000, 1000, 1000, 1000] bytes
  Flow stats: outAvgPacketSize=1000, outPacketCount=5

Device B (sensor burst):
  Packet sequence: [5000] bytes (one large packet)
  Flow stats: outAvgPacketSize=5000, outPacketCount=1
```

If Device A sends 5 packets of 200 bytes each:
```
Device A (reconfigured):
  Packet sequence: [200, 200, 200, 200, 200] bytes
  Flow stats: outAvgPacketSize=200, outPacketCount=5
```

These are very different communication patterns, but flow statistics can look similar.

---

## Why This Matters for Adversarial Attacks

### Current Attack Surface

An attacker can modify **flow statistics** to fool the classifier:
- Reduce `outAvgPacketSize` to look like a different device
- Adjust `outPacketCount` and `outByteCount`
- Change `outAvgIAT` to match target device's timing

### What Would Be Harder to Attack

If the model learned **packet sequences**, an attacker would need to:
- Change the actual sequence of packet directions
- Maintain realistic inter-packet timing
- Preserve the communication pattern of the target device

This is much harder than modifying aggregate statistics.

---

## The Missing Data: Packet Directions

The JSON files contain `firstEightNonEmptyPacketDirections`:

```
Example: [1, 0, 1, 0, 1, 1, 0, 1]
Where: 1 = outbound, 0 = inbound
```

This provides a **packet-level sequence** for each flow.

### What This Captures

| Pattern | Device Type |
|---------|-------------|
| `[1, 1, 1, 1, 1, 1, 1, 1]` | Streaming device (camera, speaker) |
| `[1, 0, 1, 0, 1, 0, 1, 0]` | Request-response (plug, sensor) |
| `[0, 0, 0, 0, 0, 0, 0, 0]` | Receiving device (display) |

### Current Status

| Aspect | Status |
|--------|--------|
| Available in JSON | Yes |
| Available in CSV | No |
| Currently used | No |

---

## The Fix

### Option A: Use Packet Directions from JSON

- Extract `firstEightNonEmptyPacketDirections` from JSON
- Create embeddings for packet direction sequences
- Combine with existing flow statistics

### Option B: Require Raw Packets

- Would need PCAP files (not available)
- Would need to rebuild entire pipeline
- Not feasible with current data

### Recommendation

**Option A** is the practical fix. It uses data we already have.

---

Return to [Gap Analysis Index](./README.md) | Return to [Main Index](../README.md)
