# Gap 4: JSON Data Unused

## The Problem

The JSON data files (`IPFIX_Records/`) contain valuable information that is not being used.

---

## Current Data Usage

| Data Source | Size | Used? |
|-------------|------|-------|
| IPFIX ML Instances (CSV) | ~13 GB | Yes |
| IPFIX Records (JSON) | ~19 GB | No |

The pipeline only reads CSV files. The JSON files are completely ignored.

---

## What JSON Contains That CSV Doesn't

### Critical Field: `firstEightNonEmptyPacketDirections`

This field provides **packet-level temporal information**:

```
Type: Array of 8 integers
Values: [1, 0, 1, 0, 1, 1, 0, 1]
Where: 1 = outbound, 0 = inbound
```

This is the only field that contains actual packet-level sequencing.

### Other JSON-Only Fields

| Field | Description | Potential Value |
|-------|-------------|-----------------|
| `tcpUrgTotalCount` | Urgent packet count | Protocol behavior |
| `flowEndReason` | Why flow ended | Connection patterns |
| `vlanId` | VLAN identifier | Network segmentation |
| `ipClassOfService` | ToS/DSCP field | QoS patterns |

---

## Why JSON Is Not Used

### Technical Reasons

1. **Pipeline built for CSV**: All preprocessing code expects CSV format
2. **Labeling complexity**: JSON requires separate device mapping (MAC → device name)
3. **File size**: JSON is 19GB vs 13GB for CSV
4. **Parsing overhead**: JSON is slower to parse than CSV

### Historical Reasons

1. **CSV was ready for ML**: Labels already attached
2. **JSON is raw**: Requires additional processing
3. **Documentation mentioned CSV**: The IEEE IoT-J 2023 paper uses CSV

---

## What Using JSON Would Require

### Data Processing

1. Parse JSON files (JSON Lines format)
2. Match flows to device labels using MAC address mapping
3. Extract `firstEightNonEmptyPacketDirections`
4. Join with or replace CSV data

### Pipeline Changes

1. New JSON loader module
2. Device label mapping logic
3. Data joining/merging logic
4. Updated feature extraction

---

## The Key Value: Packet Directions

### What Packet Directions Enable

| Current | With Packet Directions |
|---------|----------------------|
| Model sees flow statistics | Model sees packet direction patterns |
| Sequence = 10 flows | Sequence = 10 flows × 8 packets |
| Temporal patterns lost | Temporal patterns preserved |
| Weak sequence attacks | Strong temporal attack surface |

### Example Device Signatures

| Device | Direction Pattern | Interpretation |
|--------|------------------|----------------|
| Camera | `[1, 1, 1, 1, 1, 1, 1, 1]` | Streaming outbound |
| Speaker | `[0, 0, 0, 0, 0, 0, 0, 0]` | Receiving audio |
| Smart plug | `[1, 0, 1, 0, 1, 0, 1, 0]` | Request-response |

---

## Integration Options

### Option 1: Replace CSV with JSON

- Use JSON as primary data source
- Extract all fields including packet directions
- Create device label mapping

**Effort**: High
**Benefit**: Full access to all JSON fields

### Option 2: Augment CSV with JSON

- Keep using CSV for flow statistics
- Extract only `firstEightNonEmptyPacketDirections` from JSON
- Join on flow identifiers (IP, port, timestamp)

**Effort**: Medium
**Benefit**: Adds packet directions without rebuilding pipeline

### Option 3: Create Supplemental File

- Extract packet directions from JSON once
- Save as separate file (e.g., `packet_directions.parquet`)
- Load alongside CSV during training

**Effort**: Low (one-time extraction)
**Benefit**: Minimal pipeline changes

---

## Recommendation

**Option 3** is the most practical:

1. One-time extraction of `firstEightNonEmptyPacketDirections` from JSON
2. Save to efficient format (Parquet or HDF5)
3. Load during preprocessing alongside CSV
4. Minimal changes to existing pipeline

This bridges Gap 1 (data granularity) with minimal effort.

---

## File Locations

### JSON Files

```
gdrive:PFE/IPFIX_Records/
├── 20-01-31(1)/ipfix_202001_fixed.json  (7.1 GB)
├── 20-03-31/ipfix_202003.json           (6.1 GB)
└── 20-04-30/ipfix_202004.json           (5.7 GB)
```

### CSV Files

```
gdrive:PFE/IPFIX_ML_Instances/
├── home1_labeled.csv
├── home2_labeled.csv
└── ... (12 files)
```

---

Return to [Gap Analysis Index](./README.md) | Return to [Main Index](../README.md)
