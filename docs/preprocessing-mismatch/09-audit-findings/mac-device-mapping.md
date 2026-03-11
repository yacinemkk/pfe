# MAC-to-Device Mapping for IPFIX Records (JSON Dataset)

## Severity: 🔴 Critical

## Summary

The JSON dataset (`IPFIX Records`) does **not** contain device labels. Labeling must be performed by mapping `sourceMacAddress` or `destinationMacAddress` to the device name using the ground-truth table below.

---

## Ground-Truth Mapping

This mapping was extracted from the original dataset documentation (Table 7 of the manuscript / dataset specifications).

| # | Device Name | MAC Address |
|---|-------------|-------------|
| 0 | DEFAULT GATEWAY | `38:d5:47:0c:25:d4` |
| 1 | Nokia body | `00:24:e4:62:68:2e` |
| 2 | Panasonic doorphone | `bc:c3:42:dc:24:78` |
| 3 | Qrio hub | `80:c5:f2:0b:aa:a9` |
| 4 | Philips Hue lightbulb | `00:17:88:47:20:f2` |
| 5 | Xiaomi LED | `78:11:dc:55:76:4c` |
| 6 | Planex UCA01A camera | `ec:3d:fd:39:6f:98` |
| 7 | Planex pantilt camera | `e0:b9:4d:b9:eb:e9` |
| 8 | JVC Kenwood camera | `e0:b9:4d:5c:cf:c5` |
| 9 | Nature remote control | `60:01:94:54:6b:e8` |
| 10 | Bitfinder aware sensor | `70:88:6b:10:22:83` |
| 11 | Google Home | `d8:6c:63:47:54:dc` |
| 12 | Apple Homepod | `d4:90:9c:da:0d:f0` |
| 13 | Sony Bravia TV | `04:5d:4b:a4:d0:2e` |
| 14 | iRobot roomba | `c0:e4:34:4b:89:fc` |
| 15 | Sesame access point | `38:56:10:00:1d:8c` |
| 16 | JVC Kenwood hub | `00:a2:b2:b9:09:87` |
| 17 | Wansview camera | `e0:09:bf:54:68:47` |
| 18 | Qwatch camera | `34:76:c5:7f:91:07` |
| 19 | Fredi camera | `20:32:33:86:f7:0f` |
| 20 | Planex outdoor camera | `00:22:cf:fd:c1:08` |
| 21 | PowerElec WIFI plug | `ec:f0:0e:55:25:39` |
| 22 | Line Clova speaker | `a8:1e:84:e8:cc:c3` |
| 23 | Sony smart speaker | `6c:5a:b5:56:39:3e` |
| 24 | Amazon Echo | `4c:ef:c0:17:e0:42` |
| 25 | Amazon Echo Show | `14:0a:c5:f1:e5:52` |
| 26 | MCJ room hub | `aa:1e:84:06:1c:b4` |

---

## Usage Notes

### The Gateway

`DEFAULT GATEWAY` (`38:d5:47:0c:25:d4`) is **not** an IoT device class. It is the network router. However, it is critical for determining flow directionality:

- If `sourceMacAddress == GATEWAY` and `destinationMacAddress == IoT device` → **inbound** flow to the IoT device
- If `sourceMacAddress == IoT device` and `destinationMacAddress == GATEWAY` → **outbound** flow from the IoT device

### Filtering After the Manuscript

Per the manuscript, the authors reduced to **17 classes** by removing underrepresented devices. The exact classes removed are not specified, but data-driven filtering (e.g., `MIN_SAMPLES_PER_CLASS`) should be applied.

### Code-Ready Dictionary

```python
MAC_TO_DEVICE = {
    "38:d5:47:0c:25:d4": "DEFAULT GATEWAY",
    "00:24:e4:62:68:2e": "Nokia body",
    "bc:c3:42:dc:24:78": "Panasonic doorphone",
    "80:c5:f2:0b:aa:a9": "Qrio hub",
    "00:17:88:47:20:f2": "Philips Hue lightbulb",
    "78:11:dc:55:76:4c": "Xiaomi LED",
    "ec:3d:fd:39:6f:98": "Planex UCA01A camera",
    "e0:b9:4d:b9:eb:e9": "Planex pantilt camera",
    "e0:b9:4d:5c:cf:c5": "JVC Kenwood camera",
    "60:01:94:54:6b:e8": "Nature remote control",
    "70:88:6b:10:22:83": "Bitfinder aware sensor",
    "d8:6c:63:47:54:dc": "Google Home",
    "d4:90:9c:da:0d:f0": "Apple Homepod",
    "04:5d:4b:a4:d0:2e": "Sony Bravia TV",
    "c0:e4:34:4b:89:fc": "iRobot roomba",
    "38:56:10:00:1d:8c": "Sesame access point",
    "00:a2:b2:b9:09:87": "JVC Kenwood hub",
    "e0:09:bf:54:68:47": "Wansview camera",
    "34:76:c5:7f:91:07": "Qwatch camera",
    "20:32:33:86:f7:0f": "Fredi camera",
    "00:22:cf:fd:c1:08": "Planex outdoor camera",
    "ec:f0:0e:55:25:39": "PowerElec WIFI plug",
    "a8:1e:84:e8:cc:c3": "Line Clova speaker",
    "6c:5a:b5:56:39:3e": "Sony smart speaker",
    "4c:ef:c0:17:e0:42": "Amazon Echo",
    "14:0a:c5:f1:e5:52": "Amazon Echo Show",
    "aa:1e:84:06:1c:b4": "MCJ room hub",
}

GATEWAY_MAC = "38:d5:47:0c:25:d4"
```

---

## Verification

We confirmed the mapping by direct inspection of the JSON file:

- Flow records from `sourceMacAddress: "34:76:c5:7f:91:07"` (Qwatch camera) are present in `ipfix_202001_fixed.json`
- Flow records from `sourceMacAddress: "e0:b9:4d:b9:eb:e9"` (Planex pantilt camera) are present
- The gateway MAC `38:d5:47:0c:25:d4` appears frequently as both source and destination

---

Return to [Audit Findings Index](./README.md) | Return to [Main Index](../README.md)
