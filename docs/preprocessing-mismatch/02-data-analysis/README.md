# Data Analysis

## Overview

This section documents all available data sources, their relationships, and the differences between what we have and what we use.

### Key Findings

1. **Two data sources exist**: CSV files (currently used) and JSON files (unused)
2. **CSV is derived from JSON**: The IPFIX ML Instances (CSV) were preprocessed from IPFIX Records (JSON) by the UNSW team
3. **Important fields are missing from CSV**: The JSON contains packet direction sequences that are not in the CSV
4. **No raw packet data exists**: Both formats are flow-level aggregates; raw PCAP files are not available

---

## Data Sources Summary

| Source | Format | Size | Currently Used |
|--------|--------|------|----------------|
| IPFIX ML Instances | CSV | ~13 GB | Yes |
| IPFIX Records | JSON | ~19 GB | No |

---

## Documents in This Section

| Document | Description |
|----------|-------------|
| [csv-vs-json.md](./csv-vs-json.md) | Detailed comparison of the two data formats |
| [available-features.md](./available-features.md) | Complete feature matrix showing what's in each format |
| [packet-directions-discovery.md](./packet-directions-discovery.md) | The key finding: packet direction sequences in JSON |

---

## Quick Reference: Data Locations

### Google Drive Paths

```
gdrive:PFE/
├── IPFIX_ML_Instances/          # CSV files (currently used)
│   ├── home1_labeled.csv
│   ├── home2_labeled.csv
│   └── ... (12 files total)
│
└── IPFIX_Records/               # JSON files (unused)
    ├── 20-01-31(1)/ipfix_202001_fixed.json
    ├── 20-03-31/ipfix_202003.json
    └── 20-04-30/ipfix_202004.json
```

### Local Processing

The CSV files are loaded from Google Drive in the Colab notebook. The JSON files have never been processed.

---

## Data Origin

Both datasets are from **UNSW IoT Analytics** (https://iotanalytics.unsw.edu.au):

| Dataset | Paper | Year |
|---------|-------|------|
| IPFIX Records (JSON) | IEEE IoT-J | 2022 |
| IPFIX ML Instances (CSV) | IEEE IoT-J | 2023 |

The CSV files were created from the JSON records using **YAF (Yet Another Flowmeter)** for feature extraction, with additional labeling for ML tasks.

---

## Critical Discovery

The JSON files contain `firstEightNonEmptyPacketDirections` - a field that encodes the direction (inbound/outbound) of the first 8 packets in each flow. This is **actual packet-level temporal information** that could enable proper sequence modeling.

See [packet-directions-discovery.md](./packet-directions-discovery.md) for details.

---

Return to [Main Index](../README.md)
