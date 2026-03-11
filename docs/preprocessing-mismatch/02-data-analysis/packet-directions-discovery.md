# Packet Directions Discovery

## The Key Finding

The JSON data contains a field called `firstEightNonEmptyPacketDirections` that provides actual packet-level temporal information.

---

## What is `firstEightNonEmptyPacketDirections`?

This field encodes the **direction** (inbound or outbound) of the **first 8 non-empty packets** in each flow.

### Format

```
Type: Array of integers
Values: [1, 0, 1, 0, 1, 1, 0, 1]
Where: 1 = outbound (device → network)
       0 = inbound (network → device)
```

### Examples

| Pattern | Interpretation | Device Type Example |
|---------|----------------|---------------------|
| `[1, 1, 1, 1, 1, 1, 1, 1]` | All outbound | Device streaming data (camera) |
| `[0, 0, 0, 0, 0, 0, 0, 0]` | All inbound | Device receiving data (display) |
| `[1, 0, 1, 0, 1, 0, 1, 0]` | Alternating | Request-response (smart plug) |
| `[1, 1, 0, 0, 1, 1, 0, 0]` | Burst pattern | Bursty traffic (sensor) |

---

## Why This Matters

### Current Situation

The current pipeline uses **aggregated statistics** like:
- `outPacketCount = 50` (total count)
- `outAvgPacketSize = 256` (average)
- `outAvgIAT = 0.1` (average timing)

These are single numbers that summarize many packets. The model has no idea about the **sequence** or **pattern** of packets.

### What Packet Directions Provide

With `firstEightNonEmptyPacketDirections`, the model can learn:

1. **Communication patterns**: Is the device primarily a sender or receiver?
2. **Request-response behavior**: Does it alternate in/out?
3. **Burst patterns**: Are packets clustered or spread out?
4. **Device fingerprint**: Different IoT devices have characteristic patterns

### Example Device Signatures

| Device Type | Expected Pattern | Why |
|-------------|------------------|-----|
| Security camera | `[1, 1, 1, 1, 1, 1, 1, 1]` | Streaming video outbound |
| Smart speaker | `[0, 0, 0, 0, 0, 0, 0, 0]` | Receiving audio stream |
| Smart plug | `[1, 0, 1, 0, 1, 0, 1, 0]` | Status request/response |
| Motion sensor | `[1, 0, 0, 0, 1, 0, 0, 0]` | Event-triggered burst |

---

## How to Use This

### Embedding Approach

Each packet direction sequence can be treated as a sequence of tokens:

```
Input: [1, 0, 1, 0, 1, 1, 0, 1]
       ↓
Embedding Layer (learned)
       ↓
Sequence representation (8 x embedding_dim)
       ↓
LSTM or Transformer encoder
       ↓
Packet direction representation vector
```

### Combined with Flow Statistics

The final input to the classifier would be:

```
[packet_direction_vector] concat [flow_statistics_vector]
```

This gives the model both:
1. Temporal packet patterns (from directions)
2. Volume and size information (from statistics)

---

## Current Status

| Aspect | Status |
|--------|--------|
| Field exists in JSON | ✅ Yes |
| Field in CSV | ❌ No |
| Currently extracted | ❌ No |
| Currently used | ❌ No |

The field is completely unused because:
1. The pipeline only reads CSV files
2. The CSV files don't contain this field
3. The JSON files are 19GB and were never integrated

---

## Technical Details

### Source

Field name: `firstEightNonEmptyPacketDirections`
Source: IPFIX Records (JSON)
Location: `gdrive:PFE/IPFIX_Records/*.json`

### JSON Example

```json
{
  "sourceIPv4Address": "192.168.1.100",
  "destinationIPv4Address": "8.8.8.8",
  "packetTotalCount": 15,
  "firstEightNonEmptyPacketDirections": [1, 0, 1, 0, 1, 1, 0, 1],
  ...
}
```

### Extraction Method

1. Parse JSON files line by line
2. Extract `firstEightNonEmptyPacketDirections` for each flow
3. Match flows to device labels using MAC address mapping
4. Store as additional input alongside flow statistics

---

## Implications for Adversarial Training

### Current Problem

Sequence-level adversarial attacks are weak because:
- They perturb flow statistics uniformly
- There's no true temporal pattern to attack

### With Packet Directions

An attacker would need to:
1. Change the packet direction pattern (much harder than changing statistics)
2. Maintain semantic validity (the pattern must still be realizable)
3. Match the target device's communication style

This makes adversarial attacks much more constrained and harder to execute.

---

## Related Fields

Other JSON fields that could provide temporal information:

| Field | Description | Potential Value |
|-------|-------------|-----------------|
| `firstEightNonEmptyPacketDirections` | Direction sequence | High - used in this doc |
| `standardDeviationInterarrivalTime` | Timing variance | Medium - already captured |
| `bytesPerPacket` | Bytes per packet ratio | Low - derived from existing |
| `firstNonEmptyPacketSize` | Size of first packet | Low - single value |

---

Return to [Data Analysis Index](./README.md) | Return to [Main Index](../README.md)
