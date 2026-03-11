# Files to Modify

## Existing Files Requiring Changes

---

## Data Processing Files

### `src/data/preprocessor.py`

| Change | Description | Complexity |
|--------|-------------|------------|
| Add JSON loading | Support loading from JSON | High |
| Add packet direction extraction | Extract `firstEightNonEmptyPacketDirections` | Medium |
| Modify feature selection | Include packet directions | Low |
| Add join logic | Match JSON to CSV records | High |

### `config/config.py`

| Change | Description | Complexity |
|--------|-------------|------------|
| Add JSON paths | Path to IPFIX_Records | Low |
| Add packet direction config | Embedding dimension, encoder type | Low |
| Add feature flags | Enable/disable new pipeline | Low |

---

## Model Files

### `src/models/lstm.py`

| Change | Description | Complexity |
|--------|-------------|------------|
| Accept new input shape | Handle flow stats + packet embeddings | Medium |
| Add embedding layer | For packet directions | Medium |
| Modify forward pass | Combine flow and packet representations | Medium |

### `src/models/transformer.py`

| Change | Description | Complexity |
|--------|-------------|------------|
| Accept new input shape | Handle flow stats + packet embeddings | Medium |
| Add packet attention | Attend to packet sequences | Medium |
| Modify forward pass | Hierarchical encoding | High |

### `src/models/cnn_lstm.py`

| Change | Description | Complexity |
|--------|-------------|------------|
| Accept new input shape | Handle flow stats + packet embeddings | Medium |
| Add packet encoder | Process packet directions | Medium |
| Modify forward pass | Combine representations | Medium |

---

## Training Files

### `train_adversarial.py`

| Change | Description | Complexity |
|--------|-------------|------------|
| Add data loading option | Load from JSON or CSV | Medium |
| Modify sequence creation | Include packet directions | Medium |
| Update model creation | Use modified models | Low |
| Update evaluation | Report new metrics | Low |

---

## Attack Files

### `src/adversarial/attacks.py`

| Change | Description | Complexity |
|--------|-------------|------------|
| Add packet direction attack | Attack direction embeddings | High |
| Modify sequence attack | Position-aware perturbation | Medium |
| Add constrained attack | Respect direction semantics | High |

---

## Summary

| File Category | Files to Modify | Total Changes |
|---------------|-----------------|---------------|
| Data | 2 | 6 |
| Models | 3 | 9 |
| Training | 1 | 4 |
| Attacks | 1 | 3 |
| **Total** | **7** | **22** |

---

## Modification Strategy

### 1. Feature Branch

All changes in a new branch:
```
git checkout -b feature/packet-embeddings
```

### 2. Incremental Commits

One feature per commit:
- Add JSON loader
- Add packet embeddings
- Modify each model
- etc.

### 3. Backward Compatibility

Keep old behavior accessible:
```python
if config.USE_PACKET_EMBEDDINGS:
    # new pipeline
else:
    # old pipeline
```

---

Return to [Implementation Plan Index](./README.md) | Return to [Main Index](../README.md)
