# Bidirectional Flow Labeling

## Severity: 🔴 Critical

## Summary

Network flows are bidirectional. Every flow record has a `sourceMacAddress` and a `destinationMacAddress`. The labeling logic must check **both** MAC addresses against the [MAC-to-device mapping](./mac-device-mapping.md) to correctly label every flow.

---

## The Problem

A naive implementation might only check `sourceMacAddress`:

```python
# ❌ WRONG — misses ~50% of flows
label = MAC_TO_DEVICE.get(flow["sourceMacAddress"], None)
```

This **silently drops** all flows where the IoT device is the **destination** (inbound traffic), which represents roughly half of all flows.

---

## Example

Consider a flow between the `Qwatch camera` and an external server via the gateway:

### Outbound flow (camera → server)
```json
{
    "sourceMacAddress": "34:76:c5:7f:91:07",
    "destinationMacAddress": "38:d5:47:0c:25:d4",
    ...
}
```
- `sourceMacAddress` = Qwatch camera ✅ → label = "Qwatch camera"

### Inbound flow (server → camera)
```json
{
    "sourceMacAddress": "38:d5:47:0c:25:d4",
    "destinationMacAddress": "34:76:c5:7f:91:07",
    ...
}
```
- `sourceMacAddress` = DEFAULT GATEWAY ❌ → label = None (flow dropped!)
- `destinationMacAddress` = Qwatch camera ✅ → **this is the correct label**

---

## Correct Labeling Algorithm

```python
GATEWAY_MAC = "38:d5:47:0c:25:d4"

def label_flow(flow, mac_to_device):
    src_mac = flow["sourceMacAddress"]
    dst_mac = flow["destinationMacAddress"]

    src_device = mac_to_device.get(src_mac)
    dst_device = mac_to_device.get(dst_mac)

    # Skip flows internal to gateway only
    if src_device == "DEFAULT GATEWAY" and dst_device == "DEFAULT GATEWAY":
        return None

    # Prefer the non-gateway IoT device
    if src_device and src_device != "DEFAULT GATEWAY":
        return src_device
    if dst_device and dst_device != "DEFAULT GATEWAY":
        return dst_device

    # Neither MAC is a known IoT device
    return None
```

---

## Edge Cases

| Scenario | sourceMac | destinationMac | Correct Label |
|----------|-----------|----------------|---------------|
| IoT → Gateway | IoT device | Gateway | IoT device |
| Gateway → IoT | Gateway | IoT device | IoT device |
| IoT → IoT | IoT device A | IoT device B | **Ambiguous** — assign to source |
| IoT → External | IoT device | Unknown | IoT device |
| External → IoT | Unknown | IoT device | IoT device |
| Gateway → External | Gateway | Unknown | Skip (no IoT device) |
| Unknown → Unknown | Unknown | Unknown | Skip (no IoT device) |

---

## Impact If Not Fixed

- ~50% of flows could be mislabeled or dropped entirely
- Model will only learn **outbound** behavior, not **inbound** behavior
- Devices that primarily **receive** data (displays, speakers) will be catastrophically underrepresented
- Class imbalance will worsen artificially

---

Return to [Audit Findings Index](./README.md) | Return to [Main Index](../README.md)
