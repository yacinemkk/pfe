# Data Leakage Risk

## Severity: 🔴 Critical

## Summary

The raw JSON files contain highly identifying network-layer features (`sourceMacAddress`, `sourceIPv4Address`, `sourceTransportPort`, etc.) that, if left in the feature matrix, would allow models to achieve trivially perfect accuracy by memorizing addresses rather than learning behavioral flow patterns.

---

## The Risk

### How the CSVs Handled This

The CSV dataset (`IoT IPFIX Home`) was pre-processed by the dataset authors:
- IP addresses, MAC addresses, and ports were retained as metadata for labeling
- But the ML pipeline ([preprocessor.py](../../../src/data/preprocessor.py)) explicitly drops them via `FEATURES_TO_KEEP`

### What the JSON Contains

Every JSON flow record has these identifying fields:

| Field | Example Value | Risk |
|-------|---------------|------|
| `sourceMacAddress` | `34:76:c5:7f:91:07` | **Direct device identifier** — uniquely maps to label |
| `destinationMacAddress` | `38:d5:47:0c:25:d4` | Leaks device identity for inbound flows |
| `sourceIPv4Address` | `192.168.1.230` | Near-unique per device on LAN |
| `destinationIPv4Address` | `192.168.1.1` | Correlation with device |
| `sourceTransportPort` | `40709` | Port ranges correlate with device |
| `destinationTransportPort` | `53965` | Same |
| `tcpSequenceNumber` | `0xb456b1ee` | Connection-specific identifier |
| `reverseTcpSequenceNumber` | `0x3468ffd3` | Same |
| `collectorName` | `C1` | Infrastructure identifier |
| `vlanId` | `0x000` | Network configuration |

---

## Why This Causes 100% Accuracy (And Why That's Bad)

If `sourceMacAddress` is left as a feature:
- The model learns: `"34:76:c5:7f:91:07"` → `"Qwatch camera"` (direct lookup)
- Accuracy: 100%
- What the model actually learned: **Nothing about behavior**
- Adversarial robustness: **Zero** — model is trivially foolable by any different traffic from the same MAC, and the adversarial training pipeline becomes meaningless

---

## Fields to DROP After Labeling

These fields must be used for labeling (via the [MAC mapping](./mac-device-mapping.md)) and then **immediately removed** before any feature scaling, sequence creation, or model training:

```python
COLUMNS_TO_DROP = [
    "sourceMacAddress",
    "destinationMacAddress",
    "sourceIPv4Address",
    "destinationIPv4Address",
    "sourceTransportPort",
    "destinationTransportPort",
    "tcpSequenceNumber",
    "reverseTcpSequenceNumber",
    "collectorName",
    "observationDomainId",
    "vlanId",
    "ingressInterface",
    "egressInterface",
    "flowAttributes",
    "reverseFlowAttributes",
    "flowStartMilliseconds",   # use for ordering only, then drop
    "flowEndMilliseconds",     # same
    "flowEndReason",
    "silkAppLabel",
    "ipClassOfService",
    "active_timeout",
    "idle_timeout",
]
```

### Fields to KEEP (behavioral flow statistics)

These are the SDN-accessible features per the manuscript:

```python
FEATURES_TO_KEEP = [
    "flowDurationMilliseconds",
    "reverseFlowDeltaMilliseconds",
    "protocolIdentifier",
    "packetTotalCount",
    "octetTotalCount",
    "reversePacketTotalCount",
    "reverseOctetTotalCount",
    "tcpUrgTotalCount",
    "smallPacketCount",
    "nonEmptyPacketCount",
    "dataByteCount",
    "averageInterarrivalTime",
    "firstNonEmptyPacketSize",
    "largePacketCount",
    "maxPacketSize",
    "standardDeviationPayloadLength",
    "standardDeviationInterarrivalTime",
    "bytesPerPacket",
    "reverseTcpUrgTotalCount",
    "reverseSmallPacketCount",
    "reverseNonEmptyPacketCount",
    "reverseDataByteCount",
    "reverseAverageInterarrivalTime",
    "reverseFirstNonEmptyPacketSize",
    "reverseLargePacketCount",
    "reverseMaxPacketSize",
    "reverseStandardDeviationPayloadLength",
    "reverseStandardDeviationInterarrivalTime",
    # "reverseBytesPerPacket",  # only present in some records
    "firstEightNonEmptyPacketDirections",  # hex → decoded to 8 binary features
]
```

---

## Comparison with CSV Pipeline

| Aspect | CSV Pipeline | JSON Pipeline (proposed) |
|--------|-------------|-------------------------|
| Identifying columns | Dropped by `FEATURES_TO_KEEP` | Must be explicitly dropped after MAC labeling |
| Risk of leakage | Low (well-handled) | **High if not explicitly handled** |
| TCP flags | Encoded as strings, dropped | Must be encoded or dropped |

---

Return to [Audit Findings Index](./README.md) | Return to [Main Index](../README.md)
