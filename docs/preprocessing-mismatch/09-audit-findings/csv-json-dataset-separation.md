# CSV and JSON Are Separate Datasets

## Severity: 🔴 Critical

## Summary

The CSV files (`IPFIX ML Instances/`) and the JSON files (`IPFIX Records/`) originate from **two completely different data collection campaigns** with different devices, different homes, and different time periods. They cannot be joined.

---

## Evidence

### CSV Dataset: "IoT IPFIX Home"

- **Source**: 12 households in Japan
- **Devices**: 24 types (18 after filtering)
- **Collection period**: 47 days
- **Reference**: Pashamokhtari et al., 2023
- **Files**: `home1_labeled.csv` through `home12_labeled.csv`
- **Labels**: Pre-joined into a `name` column

Sample CSV devices: `eclear`, `sleep`, `esensor`, `hub-plus`, `humidifier`, `smart-power-strip`, `echo-dot`, `fire-tv-stick-4k`, `atom-cam`, `kasa-camera-pro`...

### JSON Dataset: "IPFIX Records"

- **Source**: Single residential testbed
- **Devices**: 26 types (17 after filtering)
- **Collection period**: 3 months (Jan–Apr 2020)
- **Reference**: Pashamokhtari et al., 2022
- **Files**: `ipfix_202001_fixed.json`, `ipfix_202003.json`, `ipfix_202004.json`
- **Labels**: Must be derived by mapping MAC addresses to device names

Sample JSON devices: `Philips Hue lightbulb`, `Qwatch camera`, `Google Home`, `Apple Homepod`, `Sony Bravia TV`, `Amazon Echo`, `iRobot roomba`...

---

## The Overlap

| Aspect | CSV (IoT IPFIX Home) | JSON (IPFIX Records) |
|--------|---------------------|---------------------|
| Number of devices | 24 (18 kept) | 26 (17 kept) |
| Device types | Smart plugs, Echo Dot, Fire TV, Kasa cameras | Philips Hue, Google Home, Apple Homepod, Sony TV |
| Geography | 12 homes in Japan | 1 testbed |
| Time period | 47 days | 3 months (2020) |
| Format | Pre-labeled CSV | Raw JSON lines, no labels |
| Feature count | ~44 columns | ~50+ fields per record |
| **Common devices** | **Almost none** | **Almost none** |

The only device name that appears in both Tables is "Qrio Hub."

---

## What This Means

### ❌ The Original Plan Was Wrong

The [migration checklist](../08-implementation-plan/migration-checklist.md) Step 1.3 says:

> *"Match JSON to CSV: Implement flow matching logic. Use IP/port/timestamp as match keys."*

This is **impossible**. The records come from different physical networks, different devices, different time periods. There are no common keys.

### ✅ The Correct Approach

Use the JSON dataset **natively** with the [MAC-to-device mapping](./mac-device-mapping.md):

1. Parse JSON files line-by-line
2. For each flow, extract `sourceMacAddress` and `destinationMacAddress`
3. Look up both MACs in the mapping table
4. If a match is found, label the flow with the device name
5. Drop identifying features (MAC, IP, port)
6. Use the remaining flow statistics and `firstEightNonEmptyPacketDirections` as model input

---

## Source

- Manuscript Table 7: "Device names in IoT IPFIX Home vs. IPFIX Records"
- Direct inspection of CSV headers (`head -n 1 home1_labeled.csv`)
- Direct inspection of JSON records (`head -n 5 ipfix_202001_fixed.json`)

---

Return to [Audit Findings Index](./README.md) | Return to [Main Index](../README.md)
