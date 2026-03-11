# Available Features Matrix

## Complete Feature Comparison

This document lists all features available in each data format and whether they are currently used in the pipeline.

---

## Feature Categories

### 1. Flow Identification (Dropped in Preprocessing)

| Field | CSV | JSON | Used | Notes |
|-------|-----|------|------|-------|
| `start` | ✅ | ✅ | ❌ | Dropped (identifying) |
| `srcMac` | ✅ | ✅ | ❌ | Dropped (identifying) |
| `destMac` | ✅ | ✅ | ❌ | Dropped (identifying) |
| `srcIP` | ✅ | ✅ | ❌ | Dropped (identifying) |
| `destIP` | ✅ | ✅ | ❌ | Dropped (identifying) |
| `srcPort` | ✅ | ✅ | ❌ | Dropped (identifying) |
| `destPort` | ✅ | ✅ | ❌ | Dropped (identifying) |

These are correctly dropped as they would allow the model to memorize device identities rather than learn traffic patterns.

---

### 2. Flow Duration and Protocol

| Field | CSV | JSON | Used | Notes |
|-------|-----|------|------|-------|
| `duration` | ✅ | ✅ | ✅ | Flow duration in ms |
| `ipProto` | ✅ | ✅ | ✅ | Protocol number |
| `protocolIdentifier` | - | ✅ | - | Same as ipProto |

---

### 3. Packet and Byte Counts

| Field | CSV | JSON | Used | Notes |
|-------|-----|------|------|-------|
| `outPacketCount` | ✅ | ✅ | ✅ | Outbound packets |
| `outByteCount` | ✅ | ✅ | ✅ | Outbound bytes |
| `inPacketCount` | ✅ | ✅ | ✅ | Inbound packets |
| `inByteCount` | ✅ | ✅ | ✅ | Inbound bytes |
| `packetTotalCount` | - | ✅ | - | Total packets |
| `octetTotalCount` | - | ✅ | - | Total bytes |
| `reversePacketTotalCount` | - | ✅ | - | Reverse flow packets |
| `reverseOctetTotalCount` | - | ✅ | - | Reverse flow bytes |

---

### 4. Packet Size Statistics

| Field | CSV | JSON | Used | Notes |
|-------|-----|------|------|-------|
| `outSmallPktCount` | ✅ | ✅ | ✅ | Small outbound packets |
| `outLargePktCount` | ✅ | ✅ | ✅ | Large outbound packets |
| `outNonEmptyPktCount` | ✅ | ✅ | ✅ | Non-empty outbound packets |
| `outDataByteCount` | ✅ | ✅ | ✅ | Outbound data bytes |
| `outAvgPacketSize` | ✅ | ✅ | ✅ | Average outbound packet size |
| `outMaxPktSize` | ✅ | ✅ | ✅ | Max outbound packet size |
| `outFirstNonEmptyPktSize` | ✅ | ✅ | ✅ | First non-empty packet size |
| `outStdevPayloadSize` | ✅ | ✅ | ✅ | Std dev of payload size |
| `inSmallPktCount` | ✅ | ✅ | ✅ | Small inbound packets |
| `inLargePktCount` | ✅ | ✅ | ✅ | Large inbound packets |
| `inNonEmptyPktCount` | ✅ | ✅ | ✅ | Non-empty inbound packets |
| `inDataByteCount` | ✅ | ✅ | ✅ | Inbound data bytes |
| `inAvgPacketSize` | ✅ | ✅ | ✅ | Average inbound packet size |
| `inMaxPktSize` | ✅ | ✅ | ✅ | Max inbound packet size |
| `inFirstNonEmptyPktSize` | ✅ | ✅ | ✅ | First non-empty packet size |
| `inStdevPayloadSize` | ✅ | ✅ | ✅ | Std dev of payload size |
| `maxPacketSize` | - | ✅ | - | Overall max packet size |
| `smallPacketCount` | - | ✅ | - | Overall small packet count |
| `largePacketCount` | - | ✅ | - | Overall large packet count |

---

### 5. Inter-Arrival Time Statistics

| Field | CSV | JSON | Used | Notes |
|-------|-----|------|------|-------|
| `outAvgIAT` | ✅ | ✅ | ✅ | Average outbound IAT |
| `outStdevIAT` | ✅ | ✅ | ✅ | Std dev outbound IAT |
| `inAvgIAT` | ✅ | ✅ | ✅ | Average inbound IAT |
| `inStdevIAT` | ✅ | ✅ | ✅ | Std dev inbound IAT |
| `averageInterarrivalTime` | - | ✅ | - | Overall average IAT |
| `standardDeviationInterarrivalTime` | - | ✅ | - | Overall std dev IAT |

---

### 6. Protocol Flags

| Field | CSV | JSON | Used | Notes |
|-------|-----|------|------|-------|
| `http` | ✅ | - | ✅ | HTTP traffic flag |
| `https` | ✅ | - | ✅ | HTTPS traffic flag |
| `dns` | ✅ | - | ✅ | DNS traffic flag |
| `ntp` | ✅ | - | ✅ | NTP traffic flag |
| `tcp` | ✅ | - | ✅ | TCP traffic flag |
| `udp` | ✅ | - | ✅ | UDP traffic flag |
| `ssdp` | ✅ | - | ✅ | SSDP traffic flag |
| `smb` | ✅ | - | ✅ | SMB traffic flag |

Note: These flags appear to be derived fields in the CSV, not present as-is in the JSON.

---

### 7. Network Location Flags

| Field | CSV | JSON | Used | Notes |
|-------|-----|------|------|-------|
| `lan` | ✅ | - | ✅ | LAN traffic flag |
| `wan` | ✅ | - | ✅ | WAN traffic flag |
| `deviceInitiated` | ✅ | - | ✅ | Device initiated connection |

---

### 8. TCP-Specific Fields

| Field | CSV | JSON | Used | Notes |
|-------|-----|------|------|-------|
| `initialTCPFlags` | - | ✅ | - | TCP flags at flow start |
| `unionTCPFlags` | - | ✅ | - | All TCP flags seen |
| `reverseInitialTCPFlags` | - | ✅ | - | Reverse flow initial flags |
| `reverseUnionTCPFlags` | - | ✅ | - | Reverse flow all flags |
| `tcpSequenceNumber` | - | ✅ | - | TCP sequence number |
| `reverseTcpSequenceNumber` | - | ✅ | - | Reverse TCP sequence |
| `tcpUrgTotalCount` | - | ✅ | - | Urgent packet count |

---

### 9. Packet Direction Sequence (CRITICAL)

| Field | CSV | JSON | Used | Notes |
|-------|-----|------|------|-------|
| `firstEightNonEmptyPacketDirections` | ❌ | ✅ | ❌ | **KEY FIELD - NOT USED** |

This field contains the direction of the first 8 non-empty packets in the flow. This is the only field that provides actual packet-level temporal information.

---

### 10. Flow Metadata

| Field | CSV | JSON | Used | Notes |
|-------|-----|------|------|-------|
| `flowStartMilliseconds` | - | ✅ | - | Flow start timestamp |
| `flowEndMilliseconds` | - | ✅ | - | Flow end timestamp |
| `flowDurationMilliseconds` | - | ✅ | - | Flow duration |
| `flowEndReason` | - | ✅ | - | Why flow ended |
| `flowAttributes` | - | ✅ | - | Flow attributes |

---

### 11. QoS and Network Info

| Field | CSV | JSON | Used | Notes |
|-------|-----|------|------|-------|
| `vlanId` | - | ✅ | - | VLAN ID |
| `ipClassOfService` | - | ✅ | - | ToS/DSCP field |
| `ingressInterface` | - | ✅ | - | Ingress interface |
| `egressInterface` | - | ✅ | - | Egress interface |

---

### 12. Labels

| Field | CSV | JSON | Used | Notes |
|-------|-----|------|------|-------|
| `device` | ✅ | - | ❌ | Device identifier |
| `name` | ✅ | - | ✅ | Device type label |
| `devices.xlsx/txt` | - | Separate | - | Device mapping for JSON |

---

## Summary

### Features Used (37 total)

All from CSV, all are aggregated flow statistics:
- 1 duration field
- 1 protocol field  
- 4 packet/byte count fields
- 14 packet size statistic fields
- 4 inter-arrival time fields
- 8 protocol flags
- 3 network location flags

### Features Available but Not Used

| Source | Count | Notable Fields |
|--------|-------|----------------|
| JSON only | ~15 | `firstEightNonEmptyPacketDirections`, TCP details, QoS info |
| CSV only | ~5 | Protocol flags, network flags (derived) |

### Critical Gap

**`firstEightNonEmptyPacketDirections`** is the only field that provides packet-level temporal information. It exists in JSON but not in CSV, and is not used in the current pipeline.

---

Return to [Data Analysis Index](./README.md) | Return to [Main Index](../README.md)
