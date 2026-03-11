# Feature-Level Attack

## How It Works

The feature-level attack targets individual flow statistics.

---

## Attack Strategy

### From `src/adversarial/attacks.py`: `FeatureLevelAttack`

The attack works by:

1. **Find target class**: Identify a device to impersonate (usually a "nearby" class)
2. **Compute perturbation**: Move flow statistics toward target class centroid
3. **Respect constraints**: Don't modify non-modifiable features (protocols)

### Mathematical Formulation

```
x_adv = Projection[x0 + c * t * mask * sign(μ_target - x0) * |Δ|]
```

Where:
- `x0` = original flow features
- `μ_target` = target class centroid (average features)
- `c` = perturbation magnitude
- `t` = iteration step
- `mask` = which features can be modified
- `Δ` = difference from target

---

## What Gets Attacked

### Modifiable Features

| Feature Type | Can Modify | Example |
|--------------|------------|---------|
| Packet counts | Yes | `outPacketCount`, `inPacketCount` |
| Byte counts | Yes | `outByteCount`, `inByteCount` |
| Packet sizes | Yes | `outAvgPacketSize` |
| Timing stats | Yes | `outAvgIAT` |

### Non-Modifiable Features

| Feature Type | Cannot Modify | Reason |
|--------------|---------------|--------|
| Protocol flags | No | `http`, `https`, `dns`, `tcp`, `udp` |
| Protocol number | No | `ipProto` |
| Network flags | No | `lan`, `wan` |

### Why Non-Modifiable?

These are hard to change without breaking the traffic:
- A TCP connection cannot become UDP
- HTTP traffic cannot become DNS

---

## Constraints Enforced

### 1. L-infinity Bound

Perturbation is clipped to `[-3, 3]` (StandardScaler range).

### 2. Feature Dependencies

Some features are related:
```
inPacketCount ↔ outPacketCount
inByteCount ↔ outByteCount
```

The attack preserves these ratios.

### 3. Projection

After perturbation, features are projected back to valid range.

---

## Why It's Effective

### 1. Direct Modification

The attack directly changes the features the model uses:

```
Model sees: outPacketCount, outByteCount, outAvgIAT, ...
Attack changes: outPacketCount, outByteCount, outAvgIAT, ...
```

### 2. Targeted Approach

The attack moves toward a specific target class:

```
Camera → Smart Plug: Reduce packet size, reduce byte count
Speaker → Sensor: Reduce continuous streaming stats
```

### 3. Respects Constraints

By not modifying protocols, the attack creates "realistic" adversarial examples.

---

## Evaluation Results

| Metric | Value |
|--------|-------|
| Clean accuracy | 90% |
| After feature attack | 46% |
| Robustness ratio | 0.51 |

The attack more than halves the model's accuracy.

---

## Implications

### For Adversarial Training

The model learns to be somewhat robust:
- Without adversarial training: Would drop to ~10-20%
- With adversarial training: Drops to ~46%

### For Security

An attacker who can modify flow statistics (e.g., by padding traffic) can fool the classifier.

### The Missing Piece

If the model relied on packet patterns (not just statistics), attacking statistics wouldn't be enough.

---

Return to [Adversarial Analysis Index](./README.md) | Return to [Main Index](../README.md)
