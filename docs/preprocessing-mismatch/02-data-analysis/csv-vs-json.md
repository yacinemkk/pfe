# CSV vs JSON Data Formats

## Relationship

The CSV files (`IPFIX_ML_Instances/`) are **derived from** the JSON files (`IPFIX_Records/`). Both come from the UNSW IoT Analytics research group.

| Aspect | IPFIX Records (JSON) | IPFIX ML Instances (CSV) |
|--------|---------------------|-------------------------|
| **Publication** | IEEE IoT-J 2022 | IEEE IoT-J 2023 |
| **Purpose** | Raw flow telemetry | Ready for ML training |
| **Processing** | Original | Preprocessed from JSON |
| **Tool** | Direct from collector | YAF (Yet Another Flowmeter) |
| **Labels** | Requires mapping | Already labeled |

---

## Format Comparison

### CSV Format (IPFIX ML Instances)

**Structure**: One row per flow record

**Files**: 12 CSV files, one per home network (`home1_labeled.csv` through `home12_labeled.csv`)

**Total Size**: ~13 GB

**Columns**: ~44 fields including:
- Flow identification: `start`, `duration`, `srcMac`, `destMac`, `srcIP`, `destIP`, `srcPort`, `destPort`
- Flow statistics: `outPacketCount`, `outByteCount`, `inPacketCount`, `inByteCount`
- Packet size stats: `outAvgPacketSize`, `outMaxPktSize`, `outSmallPktCount`, `outLargePktCount`
- Timing stats: `outAvgIAT`, `outStdevIAT`, `inAvgIAT`, `inStdevIAT`
- Protocol flags: `http`, `https`, `dns`, `ntp`, `tcp`, `udp`, `ssdp`, `smb`
- Network flags: `lan`, `wan`, `deviceInitiated`
- Labels: `device`, `name` (IoT device type)

**Pros**:
- Ready for ML training
- Labels already attached
- Smaller file size
- Faster to load

**Cons**:
- Loses some fields from JSON
- No packet direction sequences
- Aggregated already

---

### JSON Format (IPFIX Records)

**Structure**: One JSON object per line (JSON Lines format)

**Files**: 3 JSON files by date (`ipfix_202001_fixed.json`, `ipfix_202003.json`, `ipfix_202004.json`)

**Total Size**: ~19 GB

**Fields**: 50+ fields including all CSV fields plus:
- Advanced timing: `averageInterarrivalTime`, `standardDeviationInterarrivalTime`
- Packet sequences: `firstEightNonEmptyPacketDirections` (KEY FIELD)
- TCP details: `tcpUrgTotalCount`, `tcpSequenceNumber`
- Flow metadata: `flowEndReason`, `flowAttributes`
- QoS info: `vlanId`, `ipClassOfService`
- Collector info: `collectorName`, `observationDomainId`

**Pros**:
- More fields available
- Contains packet direction sequences
- Rawer data (less preprocessed)

**Cons**:
- Larger file size
- No labels (need device mapping)
- Slower to parse

---

## Key Differences Table

| Feature | CSV | JSON |
|---------|-----|------|
| Packet counts | Total count only | Count + direction sequence |
| Inter-arrival time | Average + std dev | Same |
| Packet sizes | Min/max/avg | Same + first non-empty size |
| TCP flags | Initial + union | More detailed |
| Packet directions | NOT AVAILABLE | `firstEightNonEmptyPacketDirections` |
| Labels | Included | Requires separate mapping |

---

## The Missing Field: `firstEightNonEmptyPacketDirections`

This is the most significant difference. The JSON contains a field that encodes the direction of the first 8 non-empty packets in each flow:

```
Example value: [1, 0, 1, 0, 1, 1, 0, 1]
Where: 1 = outbound, 0 = inbound
```

This represents the actual bidirectional pattern of communication:
- `[1, 1, 1, 1, 1, 1, 1, 1]` = all outbound (device sending data)
- `[0, 0, 0, 0, 0, 0, 0, 0]` = all inbound (device receiving data)
- `[1, 0, 1, 0, 1, 0, 1, 0]` = alternating (request-response pattern)

**This field is NOT present in the CSV files.**

See [packet-directions-discovery.md](./packet-directions-discovery.md) for why this matters.

---

## Data Processing Pipeline

```
Original PCAP (not available)
        ↓
   IPFIX Collector
        ↓
   IPFIX Records (JSON) ← Contains firstEightNonEmptyPacketDirections
        ↓
   YAF Processing + Labeling
        ↓
   IPFIX ML Instances (CSV) ← Lost firstEightNonEmptyPacketDirections
        ↓
   Current Pipeline
```

---

Return to [Data Analysis Index](./README.md) | Return to [Main Index](../README.md)
