# Adversarial Attacks on Discrete Features

## Severity: 🟡 High

## Summary

The current adversarial attack pipeline ([attacks.py](../../../src/adversarial/attacks.py)) uses continuous gradient-based methods (PGD/FGSM) that apply fractional perturbations to all features uniformly. Once `firstEightNonEmptyPacketDirections` (a binary bitmask) is integrated as a model feature, these continuous attacks become **semantically invalid** on the discrete portion of the input.

---

## The Problem

### Current Attack Behavior

The `SequenceLevelAttack` class uses PGD:

```python
# From attacks.py - SequenceLevelAttack.generate()
perturbation = self.epsilon * x_adv.grad.sign()
x_adv = x_adv + perturbation  # applies +0.05 to every feature
```

### What Happens with Binary Packet Directions

If the input includes packet direction bits `[0, 0, 0, 0, 0, 0, 1, 0]`:

```
After PGD: [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1.05, 0.05]
```

This is **semantically impossible** in the real world:
- A packet direction is either inbound (0) or outbound (1)
- A value of `0.05` has no physical meaning
- The perturbed flow cannot exist in real network traffic
- Any anomaly detector would trivially flag this as synthetic

---

## Current Code Reference

The `_preserve_temporal_structure` method in [attacks.py](../../../src/adversarial/attacks.py) attempts to limit perturbation magnitude:

```python
def _preserve_temporal_structure(self, x_adv, x_orig):
    temporal_std = torch.std(x_orig, dim=2, keepdim=True)
    perturbation = x_adv - x_orig
    max_perturbation = 0.5 * temporal_std
    perturbation = torch.clamp(perturbation, -max_perturbation, max_perturbation)
    return x_orig + perturbation
```

This still produces non-integer values for binary features (e.g., `0 + 0.05 * std = 0.03`).

---

## Required Fix

### Split the Attack by Feature Type

```
Input tensor: [continuous_features | binary_direction_bits]
                    ↓                        ↓
              PGD attack              Discrete attack
           (gradient-based)        (bit-flip based)
                    ↓                        ↓
           Perturbed continuous    Flipped bits
                    ↓                        ↓
              Concatenate → adversarial example
```

### Discrete Attack Strategy

For the 8 packet direction bits:

1. Compute gradient w.r.t. each direction bit
2. Find which bit flips have the largest gradient magnitude
3. Flip the top-k bits (where k controls attack budget)
4. Ensure the result is still binary: `{0, 1}` only

```python
def attack_discrete(self, x_orig, grad, k=2):
    """Flip the k packet direction bits with highest gradient magnitude."""
    direction_bits = x_orig[:, :, -8:]  # last 8 features
    direction_grad = grad[:, :, -8:]

    # Find top-k positions to flip
    _, topk_idx = torch.topk(direction_grad.abs(), k=k, dim=-1)

    x_flipped = direction_bits.clone()
    x_flipped.scatter_(-1, topk_idx, 1 - direction_bits.gather(-1, topk_idx))

    return x_flipped
```

---

## Impact If Not Fixed

- Adversarial examples will be semantically invalid
- The adversarial training will train on impossible traffic patterns
- The model will learn to "defend" against attacks that can never occur
- Real adversarial robustness will not be tested or improved

---

Return to [Audit Findings Index](./README.md) | Return to [Main Index](../README.md)
