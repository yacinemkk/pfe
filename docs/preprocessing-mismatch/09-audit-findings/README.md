# Audit Findings (2026-03-11)

## Overview

This section documents critical findings discovered during a deep audit of the preprocessing pipeline conducted on March 11, 2026. These findings reveal fundamental misunderstandings in the existing migration plan and introduce corrections that must be addressed before any implementation begins.

---

## Key Insight

**The CSV and JSON datasets are from two completely different data collection campaigns.**
They cannot be joined. The JSON dataset must be processed natively using a MAC-to-device mapping table.

---

## Documents in This Section

| Document | Severity | Description |
|----------|----------|-------------|
| [csv-json-dataset-separation.md](./csv-json-dataset-separation.md) | 🔴 Critical | CSVs and JSONs are different datasets — cannot be joined |
| [mac-device-mapping.md](./mac-device-mapping.md) | 🔴 Critical | The ground-truth MAC → device label mapping for JSON |
| [bidirectional-labeling.md](./bidirectional-labeling.md) | 🔴 Critical | Must check both src and dst MAC when labeling flows |
| [hex-directions-encoding.md](./hex-directions-encoding.md) | 🟡 High | `firstEightNonEmptyPacketDirections` is hex, not an array |
| [data-leakage-risk.md](./data-leakage-risk.md) | 🔴 Critical | Identifying columns must be dropped after labeling |
| [adversarial-discrete-features.md](./adversarial-discrete-features.md) | 🟡 High | PGD attacks cannot be applied to binary packet directions |

---

## Impact on Existing Plan

These findings **invalidate** several assumptions in the original [migration checklist](../08-implementation-plan/migration-checklist.md):

| Original Assumption | Reality |
|---------------------|---------|
| JSON enriches the CSV data | JSON is a separate dataset entirely |
| Match JSON to CSV by IP/port/timestamp | Impossible — different homes, different devices |
| 18 IoT device classes | JSON dataset has 26 devices (17 after filtering) |
| `firstEightNonEmptyPacketDirections` is an array | It's a hex-encoded bitmask string |

---

Return to [Main Index](../README.md)
