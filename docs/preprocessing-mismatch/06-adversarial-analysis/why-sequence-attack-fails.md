# Why Sequence Attack Fails

## Technical Deep-Dive

This document explains in detail why the sequence-level adversarial attack fails to be effective.

---

## The Architecture Expectation

### What Sequence Models Expect

LSTM and Transformer models are designed for sequences where:

1. **Each timestep has temporal meaning**
   - NLP: Each word contributes to sentence meaning
   - Speech: Each audio frame is part of the signal
   - Video: Each frame is part of the motion

2. **Order matters**
   - "The cat sat on the mat" ≠ "The mat sat on the cat"
   - Changing order changes meaning

3. **Gradients reveal importance**
   - Gradients show which positions matter most
   - Attacking important positions is more effective

### What Our Models Get

```
Input: [Flow_1_stats, Flow_2_stats, ..., Flow_10_stats]
```

Where each `Flow_X_stats` is a **37-dimensional vector of aggregates**.

**Order still matters**, but only because:
- Flows are correlated (same device over time)
- Different flows may have different device behaviors

**Order does NOT matter for temporal patterns**, because:
- Each flow is already an aggregate
- The "temporal" pattern within each flow is lost

---

## Gradient Analysis

### What Gradients Tell Us

The gradient `∂loss/∂X[i,j,k]` tells us:
- How changing feature `k` at position `j` affects the loss
- Which positions and features are important

### What We Observe

If we analyze the gradients:

```
Position 0 gradients: [0.01, 0.02, 0.01, ...]
Position 1 gradients: [0.01, 0.02, 0.01, ...]
...
Position 9 gradients: [0.01, 0.02, 0.01, ...]
```

Gradients are **similar across positions**.

### Why

Because the model doesn't rely on position-specific information:
- Any single flow can identify the device (mostly)
- The sequence just provides more samples
- All positions contribute similarly

---

## Attention Analysis (Transformer)

### What Attention Weights Show

For the Transformer model, we can analyze attention weights:

```
Attention[Flow_i, Flow_j] = how much Flow_i attends to Flow_j
```

### Expected for True Temporal Data

If the data had temporal patterns:
- Early flows might attend to later flows
- Certain patterns would create specific attention structures
- Attention would reveal temporal relationships

### What We Likely See

For flow aggregates:
- Attention is roughly uniform
- Or attention follows statistical similarity (not temporal)
- No clear temporal structure

---

## The `preserve_temporal_structure` Function

### What It Does

From `src/adversarial/attacks.py`:

```python
def _preserve_temporal_structure(self, x_adv, x_orig):
    temporal_std = torch.std(x_orig, dim=2, keepdim=True)
    perturbation = x_adv - x_orig
    max_perturbation = 0.5 * temporal_std
    perturbation = torch.clamp(perturbation, -max_perturbation, max_perturbation)
    return x_orig + perturbation
```

### What It Means

- Limits perturbation to 50% of feature standard deviation
- Per-position constraint
- Tries to "preserve temporal structure"

### The Problem

This function **cannot preserve temporal structure** because:

1. It only limits perturbation magnitude
2. It doesn't preserve temporal relationships between positions
3. There's no temporal structure to preserve in the first place

---

## Why the Attack Can't Improve

### The Optimization Problem

PGD solves:
```
maximize: loss(model(X + δ), y)
subject to: ||δ||_∞ ≤ ε
```

### The Solution

The optimal perturbation is:
```
δ* = ε * sign(∇_X loss)
```

### The Gradient Limitation

If `∇_X loss` is similar across positions, then `δ*` is similar across positions.

There's **no mathematical way** to make the attack more effective without:
1. Different input data (with temporal patterns)
2. A different model (that uses temporal patterns)
3. Different constraints (that respect temporal structure)

---

## The Vicious Cycle

```
No temporal patterns in data
        ↓
Model doesn't learn temporal patterns
        ↓
Gradients don't reveal temporal importance
        ↓
Attack can't exploit temporal patterns
        ↓
Attack is weak
        ↓
Adversarial training on weak attack
        ↓
Model remains vulnerable to real attacks
```

---

## The Solution

### Break the Cycle

The only way to break this cycle is to **fix the data**:

1. **Extract packet directions from JSON**
   - Provides actual temporal information
   - Enables true sequence modeling

2. **Redesign the input**
   - Packets as sequence elements (not flows)
   - Or flow + packet directions

3. **Then the attack will work**
   - Temporal gradients will be meaningful
   - Attention will capture patterns
   - Attack will be constrained by temporal consistency

---

## Summary

| Factor | Current | Required |
|--------|---------|----------|
| Input temporal structure | None (aggregates) | Packet-level patterns |
| Model temporal learning | Limited | Meaningful |
| Gradient temporal info | None | Position-specific |
| Attack effectiveness | Weak | Strong |
| Adversarial training | Suboptimal | Effective |

The attack failure is a **symptom**, not the root cause. The root cause is the data pipeline.

---

Return to [Adversarial Analysis Index](./README.md) | Return to [Main Index](../README.md)
