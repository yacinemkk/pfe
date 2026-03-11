# Problem Statement

## The Core Issue

There is a fundamental mismatch between what the models are designed to do and what the data actually provides.

### What We Claim

The documentation states that the system uses LSTM and Transformer models to identify IoT devices from network traffic sequences. The models are trained with adversarial examples to be robust against attacks that try to mask device identity.

### What Actually Happens

1. **The "sequences" are not packets** - Each element in a sequence is an aggregated flow record containing statistics about many packets
2. **The models see statistics, not temporal dynamics** - A flow record has `outPacketCount=50`, `outByteCount=10000`, `outAvgIAT=0.1s` - these are summaries, not the actual pattern of 50 packets
3. **The adversarial attacks cannot exploit temporal patterns** - Because there are no temporal patterns to exploit

---

## Why This Matters

### For Device Identification

Different IoT devices have distinct traffic patterns:
- A camera sends bursts of large packets (video frames)
- A smart plug sends small, periodic packets (status updates)
- A speaker streams continuous medium-sized packets (audio)

**Current approach**: Uses average packet size, total byte count, average inter-arrival time
**Better approach**: Use the actual sequence of packet sizes and timings

### For Adversarial Robustness

An attacker trying to disguise a camera as a smart plug could:
- **Current defense**: Change statistical summaries (reduce avg packet size, reduce byte count)
- **What attacker cannot easily do**: Change the fundamental bursty pattern of video frames into regular periodic status updates

The current system is vulnerable to attacks that only need to modify statistics. A system that uses actual packet patterns would be much harder to fool.

### For Model Interpretability

When a Transformer model makes a decision, its attention mechanism should reveal which temporal patterns were important:
- "The model attended to the burst pattern in packets 3-7"
- "The model noticed the regular 100ms intervals between packets"

Currently, attention is over flow statistics, which are much harder to interpret meaningfully.

---

## The Specific Flaw in Sequence-Level Attacks

The `SequenceLevelAttack` class in `src/adversarial/attacks.py` implements a PGD-style attack that:

1. Takes a sequence of shape `(batch, seq_len, features)`
2. Computes gradient of loss w.r.t. the entire input
3. Applies perturbation: `X_adv = X + epsilon * sign(gradient)`

**The problem**: The gradient and perturbation are computed uniformly for all timesteps. The attack does not:
- Consider that different timesteps might have different importance
- Model temporal dependencies between consecutive flows
- Exploit the fact that the model processes sequences (LSTM hidden states, Transformer attention)

This is why sequence-level attacks are less effective than feature-level attacks (70% accuracy drop vs 46% accuracy drop). The "sequence" nature of the attack is superficial.

---

## The Unused Resource

The JSON data in `IPFIX_Records/` contains a field called `firstEightNonEmptyPacketDirections`. This field contains the direction (in/out) of the first 8 non-empty packets in each flow.

Example: `[1, 0, 1, 0, 1, 1, 0, 1]` means: out, in, out, in, out, out, in, out

This is actual packet-level temporal information that could be used to create meaningful sequence representations. But it is currently completely ignored because:
1. The pipeline only reads CSV files
2. The CSV files don't contain this field
3. The JSON files are 19GB and were never integrated

---

## Summary

| Aspect | Current Reality | What Was Intended |
|--------|-----------------|-------------------|
| Sequence elements | Aggregated flow statistics | Individual packets or packet patterns |
| Temporal information | Average timing (single number) | Actual timing sequence |
| Model input | 37 numerical features per timestep | Rich temporal patterns |
| Attack target | Feature values | Temporal dynamics |
| Data source | CSV (pre-aggregated) | JSON (contains packet directions) |

The fix requires either:
1. Processing the JSON data to extract packet-level information
2. Redesigning the sequence-level attack to work properly with flow statistics
3. Both

Return to [Main Index](./README.md)
