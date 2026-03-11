# Proposed Solutions

## Overview

This section presents three options for fixing the preprocessing mismatch, ranging from minimal changes to a full redesign.

---

## Solutions Comparison Matrix

| Aspect | Option A | Option B | Option C |
|--------|----------|----------|----------|
| **Name** | Minimal Fix | Packet Embeddings | Full Redesign |
| **Effort** | Low | Medium | High |
| **Data change** | None | Add JSON extraction | Full JSON pipeline |
| **Model change** | Attack only | Model + Attack | New architecture |
| **Impact** | Moderate | High | Highest |
| **Risk** | Low | Medium | High |

---

## Documents in This Section

| Document | Description |
|----------|-------------|
| [option-a-minimal-fix.md](./option-a-minimal-fix.md) | Keep CSV, fix attacks only |
| [option-b-packet-embeddings.md](./option-b-packet-embeddings.md) | Extract packet directions from JSON |
| [option-c-full-redesign.md](./option-c-full-redesign.md) | True packet-level pipeline |

---

## Recommendation

**Option B: Packet Embeddings** is recommended because:

1. **Leverages existing data**: JSON files contain `firstEightNonEmptyPacketDirections`
2. **Moderate effort**: One-time extraction + model modifications
3. **High impact**: Enables true temporal modeling
4. **Low risk**: Backward compatible with existing pipeline

---

## Decision Factors

### Choose Option A If:
- Limited time/resources
- Want quick improvement
- Can't access JSON data

### Choose Option B If:
- Can invest moderate effort
- Want significant improvement
- Have access to JSON data

### Choose Option C If:
- Starting fresh
- Need best possible solution
- Have raw packet data (PCAP)

---

Return to [Main Index](../README.md)
