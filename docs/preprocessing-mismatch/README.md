# Preprocessing Mismatch Documentation

## Executive Summary

The current implementation has a fundamental mismatch between the model architecture and the data representation.

| Component | Current State | Expected Behavior |
|-----------|---------------|-------------------|
| **MODELS** | LSTM/Transformer designed for temporal sequences | Meaningful flow dynamics between sequence elements |
| **DATA** | Aggregated flow statistics (37 features per flow) | Temporal packet-level information |
| **ATTACKS** | Uniform perturbations across all timesteps | Position-aware, temporally-adaptive attacks |

### The Core Problem

Sequence-level adversarial attacks are ineffective because:

1. Each "sequence element" is an **aggregated flow record**, not an individual packet
2. Perturbations are applied **uniformly across all timesteps** without considering position
3. **Temporal dynamics between packets are lost** during flow aggregation
4. The Transformer's attention mechanism operates on **pre-aggregated statistics**, not raw temporal patterns

### Key Discovery

The JSON data in `IPFIX_Records/` (currently unused) contains `firstEightNonEmptyPacketDirections` - a field with actual packet-level direction sequences that could enable true temporal modeling. This field is **not present in the CSV files** currently being used.

---

## Document Structure

```
preprocessing-mismatch/
├── README.md                          ← You are here
├── 01-problem-statement.md            What's wrong and why it matters
├── 02-data-analysis/                  Deep dive into available data
├── 03-current-pipeline/               How the current code works
├── 04-model-analysis/                 Architecture analysis
├── 05-gap-analysis/                   Identified gaps
├── 06-adversarial-analysis/           Why attacks fail
├── 07-proposed-solutions/             How to fix it
├── 08-implementation-plan/            What to change (⚠️ partially superseded)
└── 09-audit-findings/                 🆕 Critical corrections from deep audit
```

---

## Quick Navigation

| Question | Document |
|----------|----------|
| What exactly is the problem? | [01-problem-statement.md](./01-problem-statement.md) |
| What data do we have available? | [02-data-analysis/README.md](./02-data-analysis/README.md) |
| How does the current pipeline work? | [03-current-pipeline/README.md](./03-current-pipeline/README.md) |
| How do the models process input? | [04-model-analysis/README.md](./04-model-analysis/README.md) |
| What are the specific gaps? | [05-gap-analysis/README.md](./05-gap-analysis/README.md) |
| Why don't adversarial attacks work properly? | [06-adversarial-analysis/README.md](./06-adversarial-analysis/README.md) |
| What are the proposed solutions? | [07-proposed-solutions/README.md](./07-proposed-solutions/README.md) |
| What files need to be changed? | [08-implementation-plan/README.md](./08-implementation-plan/README.md) |
| **🆕 What did the deep audit find?** | [**09-audit-findings/README.md**](./09-audit-findings/README.md) |
| **🆕 What is the corrected fix plan?** | [**🗺️ roadmap.md**](../../roadmap.md) |

---

## ⚠️ Important Update (2026-03-11)

A [deep audit](./09-audit-findings/README.md) revealed that several assumptions in sections 07 and 08 were **incorrect**:

- The CSV and JSON datasets are from **completely different data collection campaigns** and cannot be joined
- The JSON files require native processing with a MAC-to-device mapping (not CSV enrichment)
- `firstEightNonEmptyPacketDirections` is hex-encoded, not an array

**The corrected implementation plan is in [`roadmap.md`](../../roadmap.md) at the project root.**

---

## Key Findings Summary

### Data Findings
- CSV files are preprocessed from JSON by UNSW team using YAF (Yet Another Flowmeter)
- JSON contains 50+ fields; CSV only exposes ~44 fields
- `firstEightNonEmptyPacketDirections` exists in JSON but not in CSV
- Current pipeline uses CSV exclusively; JSON data is completely ignored
- 🆕 **CSV and JSON are different datasets** — see [audit finding](./09-audit-findings/csv-json-dataset-separation.md)
- 🆕 **JSON requires MAC → device labeling** — see [MAC mapping](./09-audit-findings/mac-device-mapping.md)

### Pipeline Findings
- Sequences are created from flow records using sliding windows
- Each sequence element = one aggregated flow (statistics over many packets)
- No packet-level information is preserved
- No learned embeddings - raw numerical features are fed directly to models
- 🆕 **Both src and dst MAC must be checked** — see [bidirectional labeling](./09-audit-findings/bidirectional-labeling.md)
- 🆕 **Identifying columns will cause data leakage** — see [leakage risk](./09-audit-findings/data-leakage-risk.md)

### Model Findings
- LSTM uses final hidden state (bidirectional concatenation)
- Transformer uses mean pooling across all positions
- Neither model receives true temporal packet dynamics

### Attack Findings
- Feature-level attacks work on flow statistics
- Sequence-level attacks apply same perturbation to all timesteps
- The `preserve_temporal_structure` function only limits perturbation magnitude, not shape
- 🆕 **PGD/FGSM cannot apply to binary packet directions** — see [discrete features](./09-audit-findings/adversarial-discrete-features.md)

---

## Impact Assessment

| Area | Current Performance | Expected with Fix |
|------|---------------------|-------------------|
| Clean accuracy | ~90% | Similar or better |
| Feature-level attack robustness | Drops to ~46% | Should improve |
| Sequence-level attack robustness | Drops to ~70% (weak) | Should be harder |
| Model interpretability | Limited | Better temporal attribution |

---

## Recommended Path Forward

~~After analyzing all options (see [07-proposed-solutions/README.md](./07-proposed-solutions/README.md)), the recommended approach was Option B (Packet Embeddings from JSON Data).~~

**Updated recommendation (2026-03-11):** Build a **native JSON pipeline** that processes the IPFIX Records dataset directly using MAC-to-device labeling, without any CSV joining. See the full corrected plan in [`roadmap.md`](../../roadmap.md).
