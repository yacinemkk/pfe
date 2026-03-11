# Migration Checklist

## Step-by-Step Execution Plan

---

## Pre-Implementation

- [ ] Create feature branch: `git checkout -b feature/packet-embeddings`
- [ ] Backup current results: `cp -r results results_backup`
- [ ] Document baseline metrics
- [ ] Ensure access to JSON files on Google Drive

---

## Phase 1: Data Extraction

### Step 1.1: Create JSON Extractor

- [ ] Create `src/data/json_extractor.py`
- [ ] Implement `JSONExtractor` class
- [ ] Add JSON streaming for large files
- [ ] Test on single JSON file

### Step 1.2: Extract Packet Directions

- [ ] Add extraction of `firstEightNonEmptyPacketDirections`
- [ ] Handle missing/empty values
- [ ] Validate extraction on sample data
- [ ] Check extraction coverage (% of flows with directions)

### Step 1.3: Match JSON to CSV

- [ ] Implement flow matching logic
- [ ] Use IP/port/timestamp as match keys
- [ ] Calculate match rate
- [ ] Document unmatched flows

### Step 1.4: Save Extracted Data

- [ ] Create Parquet output format
- [ ] Save to `data/packet_directions.parquet`
- [ ] Verify file integrity
- [ ] Document file schema

### Checkpoint 1

- [ ] Verify: Can load Parquet file
- [ ] Verify: Packet directions present
- [ ] Verify: Match rate acceptable (>90%)
- [ ] Commit: "feat: add JSON extraction for packet directions"

---

## Phase 2: Model Modifications

### Step 2.1: Create Packet Embedding Layer

- [ ] Create `src/models/packet_embedding.py`
- [ ] Implement `PacketDirectionEmbedding`
- [ ] Implement `DirectionSequenceEncoder`
- [ ] Test embedding forward pass

### Step 2.2: Create Combined Input Layer

- [ ] Create `src/models/combined_input.py`
- [ ] Implement `CombinedInputLayer`
- [ ] Test concatenation logic
- [ ] Verify output shapes

### Step 2.3: Modify LSTM Model

- [ ] Add packet embedding to `src/models/lstm.py`
- [ ] Modify forward pass
- [ ] Test with sample data
- [ ] Verify backward compatibility

### Step 2.4: Modify Transformer Model

- [ ] Add packet embedding to `src/models/transformer.py`
- [ ] Modify forward pass
- [ ] Test with sample data
- [ ] Verify backward compatibility

### Step 2.5: Update Training Pipeline

- [ ] Modify `train_adversarial.py` for new input
- [ ] Add config option for packet embeddings
- [ ] Test training loop
- [ ] Verify loss convergence

### Checkpoint 2

- [ ] Verify: Models accept new input
- [ ] Verify: Training completes
- [ ] Verify: Loss decreases
- [ ] Commit: "feat: add packet embedding to models"

---

## Phase 3: Attack Modifications

### Step 3.1: Create Packet Direction Attack

- [ ] Create `src/adversarial/packet_attack.py`
- [ ] Implement `PacketDirectionAttack`
- [ ] Implement constrained perturbation
- [ ] Test attack on sample data

### Step 3.2: Update Training with New Attack

- [ ] Modify `train_adversarial.py` for new attack
- [ ] Add to hybrid training mix
- [ ] Test adversarial training
- [ ] Verify robustness improvement

### Checkpoint 3

- [ ] Verify: Attack generates valid perturbations
- [ ] Verify: Attack reduces accuracy
- [ ] Verify: Adversarial training improves robustness
- [ ] Commit: "feat: add packet direction attack"

---

## Phase 4: Testing & Validation

### Step 4.1: Unit Tests

- [ ] Test JSON extractor
- [ ] Test packet embeddings
- [ ] Test modified models
- [ ] Test attacks

### Step 4.2: Integration Tests

- [ ] Test full pipeline
- [ ] Test training from scratch
- [ ] Test evaluation

### Step 4.3: Performance Benchmarks

- [ ] Compare clean accuracy
- [ ] Compare feature attack accuracy
- [ ] Compare sequence attack accuracy
- [ ] Compare training time

### Step 4.4: Documentation

- [ ] Update README
- [ ] Update config documentation
- [ ] Document new CLI options

### Final Checkpoint

- [ ] Verify: All tests pass
- [ ] Verify: Metrics meet targets
- [ ] Verify: No regression on clean data
- [ ] Commit: "feat: complete packet embedding implementation"

---

## Rollback Procedure

If critical issues arise:

```bash
# Revert to main branch
git checkout main

# Restore backup results
rm -rf results
mv results_backup results

# Use original pipeline
export USE_PACKET_EMBEDDINGS=false
```

---

## Post-Implementation

- [ ] Create pull request
- [ ] Request code review
- [ ] Address review comments
- [ ] Merge to main
- [ ] Update project documentation
- [ ] Archive old results

---

Return to [Implementation Plan Index](./README.md) | Return to [Main Index](../README.md)
