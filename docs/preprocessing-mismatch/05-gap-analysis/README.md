# Gap Analysis

## Overview

This section identifies the specific gaps between what the system should do and what it currently does.

---

## Gap Summary Table

| Gap | Current State | Required State | Impact |
|-----|---------------|----------------|--------|
| Data granularity | Flow-level aggregates | Packet-level sequences | Models cannot learn temporal patterns |
| Feature embeddings | None (raw numerical) | Learned embeddings | Cannot capture feature relationships |
| Sequence attacks | Uniform perturbation | Position-aware, adaptive | Weak adversarial robustness |
| JSON data usage | Ignored completely | Extract packet directions | Missing valuable temporal info |

---

## Documents in This Section

| Document | Description |
|----------|-------------|
| [gap-1-data-granularity.md](./gap-1-data-granularity.md) | Flow-level vs packet-level |
| [gap-2-feature-embeddings.md](./gap-2-feature-embeddings.md) | No learned representations |
| [gap-3-sequence-attack.md](./gap-3-sequence-attack.md) | Why adversarial training fails |
| [gap-4-json-unused.md](./gap-4-json-unused.md) | JSON data completely ignored |

---

## Gap Impact on Project Goals

### Goal: Device Identification

| Goal | Current Status | With Fix |
|------|----------------|----------|
| Accuracy on clean data | ~90% | Similar |
| Accuracy on adversarial data | ~46-70% | More robust |
| Interpretability | Limited | Better temporal attribution |

### Goal: Adversarial Robustness

| Attack Type | Current Robustness | Expected with Fix |
|-------------|-------------------|-------------------|
| Feature-level | Poor (46% accuracy) | Better |
| Sequence-level | Moderate (70% accuracy) | Strong (harder to attack) |
| Hybrid | Poor (59% accuracy) | Better |

---

## Root Cause Analysis

All gaps trace back to a single root cause:

**The data pipeline discards temporal packet information before models see it.**

```
Root Cause: Aggregation at IPFIX collection level
     ↓
Gap 1: Flow-level data (not packet-level)
     ↓
Gap 2: No embeddings needed (just raw stats)
     ↓
Gap 3: Sequence attacks are superficial
     ↓
Gap 4: JSON (with packet directions) unused
```

---

## Fix Priority

| Priority | Gap | Effort | Impact |
|----------|-----|--------|--------|
| 1 | JSON unused | Medium | High (enables packet directions) |
| 2 | Data granularity | High | High (fundamental fix) |
| 3 | Sequence attack | Low | Medium (if gap 1-2 fixed) |
| 4 | Feature embeddings | Low | Low (optimization) |

---

Return to [Main Index](../README.md)
