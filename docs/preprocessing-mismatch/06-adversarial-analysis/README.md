# Adversarial Analysis

## Overview

This section analyzes why the current adversarial training approach is not effective for sequence-level attacks.

---

## Key Findings

1. **Feature-level attacks work**: They directly modify flow statistics
2. **Sequence-level attacks are weak**: They don't exploit temporal patterns
3. **The 3-phase training doesn't help**: Phase 3 (sequence attacks) is ineffective

---

## Documents in This Section

| Document | Description |
|----------|-------------|
| [feature-level-attack.md](./feature-level-attack.md) | How feature attacks work (and why they're effective) |
| [sequence-level-attack.md](./sequence-level-attack.md) | How sequence attacks work (and why they're weak) |
| [why-sequence-attack-fails.md](./why-sequence-attack-fails.md) | Technical deep-dive into the failure |

---

## Attack Effectiveness Comparison

From the evaluation results:

| Attack Type | Clean Accuracy | Attacked Accuracy | Accuracy Drop |
|-------------|----------------|-------------------|---------------|
| Feature-level | 90% | 46% | 44% drop |
| Sequence-level (PGD) | 90% | 70% | 20% drop |
| Sequence-level (FGSM) | 90% | 71% | 19% drop |
| Hybrid | 90% | 59% | 31% drop |

**Feature-level attacks are more than twice as effective as sequence-level attacks.**

---

## Why This Matters

### The Goal of Adversarial Training

Train the model to be robust against attacks that try to disguise one device as another.

### Expected Outcome

Both feature-level and sequence-level attacks should be difficult to execute on a robust model.

### Current Outcome

- Model is somewhat robust to feature attacks (46% vs 10% random)
- Model is weak to sequence attacks (70% - not much worse than clean)
- This suggests the model doesn't rely on temporal patterns

---

## The Root Cause

Sequence-level attacks are ineffective because:

1. **No temporal patterns to attack**: Each flow is an aggregate
2. **Uniform perturbation**: All timesteps treated equally
3. **Model doesn't use temporal info**: Statistics are sufficient for classification

---

Return to [Main Index](../README.md)
