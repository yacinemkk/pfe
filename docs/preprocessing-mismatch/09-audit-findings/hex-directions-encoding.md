# Hex Encoding of `firstEightNonEmptyPacketDirections`

## Severity: 🟡 High

## Summary

The field `firstEightNonEmptyPacketDirections` in the JSON dataset is encoded as a **hexadecimal string** (e.g., `"02"`, `"0e"`, `"04"`), **not** as a JSON array of integers. The existing documentation in [gap-1-data-granularity.md](../05-gap-analysis/gap-1-data-granularity.md) incorrectly assumes it is an array like `[1, 0, 1, 0, 1, 1, 0, 1]`.

---

## Evidence

Direct inspection of `ipfix_202001_fixed.json`:

```json
{"flows": {"firstEightNonEmptyPacketDirections": "02", ...}}
{"flows": {"firstEightNonEmptyPacketDirections": "0e", ...}}
{"flows": {"firstEightNonEmptyPacketDirections": "04", ...}}
{"flows": {"firstEightNonEmptyPacketDirections": "00", ...}}
```

These are 2-character hexadecimal strings representing an 8-bit bitmask.

---

## Decoding

Each hex string encodes 8 bits, where each bit represents the direction of one of the first eight non-empty packets:

| Hex | Binary | Packet Directions (MSB→LSB) |
|-----|--------|----------------------------|
| `"00"` | `00000000` | All inbound (or no packets) |
| `"02"` | `00000010` | 7 inbound, 1 outbound at position 6 |
| `"04"` | `00000100` | 7 inbound, 1 outbound at position 5 |
| `"0e"` | `00001110` | 5 inbound, 3 outbound |
| `"ff"` | `11111111` | All 8 outbound |

### Decoding Function

```python
def decode_packet_directions(hex_str: str) -> list[int]:
    """
    Decode hex-encoded packet directions into 8-element binary list.

    Args:
        hex_str: Hex string like "02", "0e", "ff"

    Returns:
        List of 8 integers (0 or 1), MSB first.
        1 = outbound, 0 = inbound
    """
    value = int(hex_str, 16)
    return [(value >> (7 - i)) & 1 for i in range(8)]
```

### Examples

```python
decode_packet_directions("02")  # → [0, 0, 0, 0, 0, 0, 1, 0]
decode_packet_directions("0e")  # → [0, 0, 0, 0, 1, 1, 1, 0]
decode_packet_directions("ff")  # → [1, 1, 1, 1, 1, 1, 1, 1]
decode_packet_directions("00")  # → [0, 0, 0, 0, 0, 0, 0, 0]
```

---

## Impact on Pipeline

### Current Documentation Error

[gap-1-data-granularity.md](../05-gap-analysis/gap-1-data-granularity.md) lines 103-106 state:
```
Example: [1, 0, 1, 0, 1, 0, 1, 0]
Where: 1 = outbound, 0 = inbound
```

This is **incorrect for the raw JSON**. The field is a hex string, not an array.

### Required Preprocessing Step

Before feeding `firstEightNonEmptyPacketDirections` to any model:

1. Parse the hex string
2. Convert to 8-element binary array
3. Append as 8 additional binary features per flow record

### Edge Cases

- `"00"` may mean "all inbound" or "no non-empty packets at all" — ambiguous
- Some records have shorter hex strings (e.g., single character) — must zero-pad to 2 chars
- Missing field — fill with `[0, 0, 0, 0, 0, 0, 0, 0]` or mark as missing

---

Return to [Audit Findings Index](./README.md) | Return to [Main Index](../README.md)
