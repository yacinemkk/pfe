# Gap 3: Sequence-Level Attack

## The Problem

The sequence-level adversarial attack does not truly exploit temporal patterns.

---

## How Sequence-Level Attacks Should Work

### The Intuition

A sequence-level attack should:
1. Identify which timesteps are most important
2. Perturb important timesteps more than unimportant ones
3. Exploit the temporal structure of the model (LSTM hidden states, Transformer attention)

### Example Attack Strategy

```
Sequence: [Flow_1, Flow_2, ..., Flow_10]
                    ↓
Identify: Flow_3, Flow_7 are most important for classification
                    ↓
Perturb: Flow_3 heavily, Flow_7 moderately, others lightly
```

---

## Current Implementation

### From `src/adversarial/attacks.py`

The `SequenceLevelAttack.pgd_attack()` method:

```python
def pgd_attack(self, X, y):
    X_adv = X.clone()
    for step in range(self.num_steps):
        outputs = self.model(X_adv)
        loss = CrossEntropyLoss()(outputs, y)
        loss.backward()
        
        grad = X_adv.grad.data
        X_adv = X_adv + alpha * grad.sign()  # Same perturbation for all timesteps
        
        # Clip to epsilon ball
        X_adv = clamp(X_adv, X - epsilon, X + epsilon)
    
    return X_adv
```

### What This Does

1. Computes gradient of loss w.r.t. **entire input sequence**
2. Applies **same perturbation magnitude** to all timesteps
3. Only constraint is total L-infinity norm

### What This Doesn't Do

| Missing | Why It Matters |
|---------|----------------|
| Timestep importance | Not all flows equally important |
| Temporal gradients | LSTM hidden states, attention not explicitly targeted |
| Position-aware attack | Early vs late flows treated same |

---

## Why The Attack Is Weak

### Evidence from Evaluation

| Attack Type | Accuracy Drop | Effectiveness |
|-------------|---------------|---------------|
| Feature-level | 90% → 46% | Strong |
| Sequence-level (PGD) | 90% → 70% | Weak |
| Sequence-level (FGSM) | 90% → 71% | Weak |

The sequence-level attack is **less effective** than feature-level attack.

### Why

1. **Input is already aggregated**: There's no "temporal pattern" to attack within each flow
2. **Uniform perturbation**: All timesteps get same treatment
3. **Model doesn't rely on temporal structure**: It mostly uses flow statistics

---

## What Would Make Sequence Attacks Effective

### 1. Packet-Level Input

If each timestep were a packet:
- Attack could target specific packets
- Temporal patterns could be disrupted
- Attack would be more constrained (harder to execute, but more effective when successful)

### 2. Position-Aware Perturbation

Attack could learn which positions are most vulnerable:
```
perturbation[important_position] = large
perturbation[unimportant_position] = small
```

### 3. Temporal Constraint Violation

Attack could target temporal constraints:
- Break the sequence pattern
- Make flows look out-of-order
- Disrupt the temporal signature

---

## The Fundamental Issue

The sequence-level attack cannot be truly effective because:

**There is no meaningful temporal structure to attack.**

Each flow record is an aggregate. The "sequence" is just:
```
[Aggregate_1, Aggregate_2, ..., Aggregate_10]
```

There's no temporal dynamics within each aggregate, so attacking the "sequence" is just attacking 10 independent aggregates.

---

## The Fix

### Root Fix

Fix **Gap 1** first: Use packet-level data.

With packet directions from `firstEightNonEmptyPacketDirections`:
- Each flow has an 8-element direction sequence
- The model can learn temporal patterns
- The attack can target these patterns

### Attack Improvement

After fixing data:
- Implement position-aware perturbations
- Target specific attention heads in Transformer
- Attack packet direction embeddings

---

## Current Mitigation (If Data Can't Be Changed)

If stuck with flow-level data:

1. **Increase attack iterations**: More PGD steps might find better perturbations
2. **Use momentum**: Momentum-based PGD for better optimization
3. **Target specific features**: Identify which features matter most per timestep

But these are optimizations, not fundamental fixes.

---

Return to [Gap Analysis Index](./README.md) | Return to [Main Index](../README.md)
