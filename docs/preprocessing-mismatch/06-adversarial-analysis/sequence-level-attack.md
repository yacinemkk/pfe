# Sequence-Level Attack

## How It Works

The sequence-level attack uses gradient-based methods to perturb the entire input sequence.

---

## Attack Strategy

### From `src/adversarial/attacks.py`: `SequenceLevelAttack`

The attack uses Projected Gradient Descent (PGD):

```
for step in range(num_steps):
    1. Forward pass: output = model(X_adv)
    2. Compute loss: loss = CrossEntropyLoss(output, y)
    3. Backward pass: loss.backward()
    4. Get gradient: grad = X_adv.grad
    5. Update: X_adv = X_adv + alpha * sign(grad)
    6. Project: X_adv = clip(X_adv, X - epsilon, X + epsilon)
```

### Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `epsilon` | 0.1 | Maximum perturbation |
| `alpha` | 0.01 | Step size |
| `num_steps` | 10 | Number of PGD iterations |

---

## What Gets Attacked

### The Input Shape

```
X.shape = (batch_size, seq_len=10, features=37)
```

### The Gradient

The gradient `∂loss/∂X` has the same shape:

```
grad.shape = (batch_size, seq_len=10, features=37)
```

### The Perturbation

```
X_adv[i, j, k] = X[i, j, k] + alpha * sign(grad[i, j, k])
```

For all `i` (batch), `j` (sequence position), `k` (feature).

---

## The Problem: Uniform Perturbation

### What Happens

The perturbation is applied **uniformly across all sequence positions**:

```
Position 0: X[0] += perturbation
Position 1: X[1] += perturbation
...
Position 9: X[9] += perturbation
```

All positions get the same treatment.

### What Doesn't Happen

- Position-specific perturbations
- Temporal gradient analysis
- Attention-weighted attacks

---

## Why It's Weak

### 1. No Temporal Targeting

The attack doesn't consider:
- Which positions are most important
- How positions relate to each other
- The LSTM's hidden state evolution
- The Transformer's attention patterns

### 2. The Model Doesn't Rely on Temporal Patterns

Because each flow is an aggregate:
- The model can classify from any single flow's statistics
- The "sequence" provides correlation, not temporal patterns
- Attacking position 0 is as effective as attacking position 9

### 3. The Loss Landscape is Flat

Gradients are similar across positions because the model treats them similarly.

---

## Comparison with Effective Sequence Attacks

### What an Effective Attack Would Do

| Aspect | Current | Effective |
|--------|---------|-----------|
| Position importance | All equal | Weighted by importance |
| Perturbation type | Same for all | Position-specific |
| Temporal targeting | None | Target hidden states, attention |
| Constraints | L-inf only | Temporal consistency |

### Why We Can't Do This

Without packet-level temporal patterns:
- There's no meaningful "temporal importance"
- Hidden states just aggregate statistics
- Attention weights are over flows, not packets

---

## Evaluation Results

| Metric | Value |
|--------|-------|
| Clean accuracy | 90% |
| After PGD attack | 70% |
| After FGSM attack | 71% |
| Robustness ratio | 0.78 |

The attack barely reduces accuracy.

---

## The Fix

### Root Cause

The sequence-level attack is weak because the input doesn't have meaningful temporal patterns.

### Solution

Fix the data first:
1. Extract packet directions from JSON
2. Create packet-level input
3. Then implement proper sequence attacks

### Improved Attack Design

After fixing data:
- Attack packet direction embeddings
- Target specific attention heads
- Disrupt temporal patterns

---

Return to [Adversarial Analysis Index](./README.md) | Return to [Main Index](../README.md)
