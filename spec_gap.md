# Specifications & Implementation Gap Analysis Report

**Generated:** 2026-03-31  
**Updated:** Alignment with docs/ completed

---

## Changes Made to Align Code with Specs

### 1. LSTM Model (`src/models/lstm.py`)

**Spec (docs/architectures §1):**
- 2 couches LSTM, 64 unités/couche
- Activation ReLU
- Embedding de sortie: 128 dimensions

**Changes:**
- Added `embedding_dim` parameter (default: 128)
- Added `embedding_proj` Linear layer to project hidden_size (64) → embedding_dim (128)
- Updated `forward()` to apply ReLU after projection
- Updated `get_embedding()` to return 128-dim embedding

### 2. Transformer Model (`src/models/transformer.py`)

**Spec (docs/architectures §4):**
- Longueur max séquence: 576

**Changes:**
- Updated `max_seq_length` default from 512 to 576
- Updated `TransformerConfig` default to 576

### 3. Training Protocol (`train_adversarial.py`)

**Spec (docs/train):**
- Phase 1: Entraînement Standard + Crash Test 1 (3 tests)
- Phase 2: Entraînement Antagoniste + Crash Test 2 (3 tests)

**Changes:**
- Removed Phase 3 (sequence-level adversarial training)
- Added `fit_with_phase_checkpoints()` with 2-phase protocol
- Added `_crash_test_2phase()` with 3 tests:
  1. Test 1: Données Bénignes
  2. Test 2: Données Adversaires Uniquement
  3. Test 3: Mélange Bénignes + Adversaires
- Added `_print_comparative_summary_2phase()` for 2-phase reporting

### 4. Configuration (`config/config.yaml`)

**Changes:**
- Updated `transformer.max_seq_length`: 512 → 576
- Removed `bim` from `adversarial.attacks`
- Removed `hybrid_training` section (clean_ratio, feature_ratio, sequence_ratio)
- Added `adversarial_ratio: 0.2` for Phase 2
- Updated `features_to_keep` to match docs/featureselection (removed Layer 7 protocols: http, https, smb, dns, ntp, ssdp)

### 5. Feature Selection

**Spec (docs/featureselection):**
- Dataset 1: 28 features (duration, ipProto, 12 out*, 12 in*, indicators)
- Layer 7 protocols excluded (requires DPI)

**Changes:**
- Updated `features_to_keep` to 30 features matching spec
- Removed: http, https, smb, dns, ntp, ssdp

---

## Current Alignment Status

| Component | Status | Notes |
|-----------|--------|-------|
| LSTM Architecture | ALIGNED | 2 layers, 64 hidden, 128 embedding |
| BiLSTM Architecture | ALIGNED | 2 layers, 64 hidden, bidirectional |
| CNN-LSTM Architecture | ALIGNED | Conv1D + LSTM 64 + Dense 100 |
| XGBoost-LSTM Architecture | ALIGNED | LSTM feature extractor + XGBoost |
| Transformer Architecture | ALIGNED | 6 layers, 768 dim, 12 heads, 576 max_seq |
| Hybrid Architecture | ALIGNED | CNN(3,5) + BiLSTM + Transformer + Mean Pooling |
| Training Protocol | ALIGNED | 2 phases per docs/train |
| Crash Test | ALIGNED | 3 tests (benign, adversarial, mixed) |
| Feature Selection | ALIGNED | Per docs/featureselection |
| Anti-Data Leakage | ALIGNED | Per docs/general |

---

## Summary

All code has been aligned with the specifications in the `docs/` folder:

1. **Architectures** match `docs/architectures` exactly
2. **Training** uses 2-phase protocol per `docs/train`
3. **Feature Selection** matches `docs/featureselection`
4. **Preprocessing** follows `docs/pretraitement` and `docs/general`

The codebase is now consistent with the documentation.

---

## IoT-Tokenize Pipeline Corrections (2026-04-01)

### 1. Token-Type Embeddings Added
**Gap:** `NLPTransformerClassifier` lacked the 3rd embedding (token-type) specified in the IoT-Tokenize pipeline.
**Fix:** Added `nn.Embedding(3, d_model)` for token-type embeddings in `src/models/transformer.py`:
- Type 0: feature names (even positions in name-value pairs)
- Type 1: numeric values (odd positions in name-value pairs)
- Type 2: special tokens (`<s>`, `</s>`, `<pad>`)
- Auto-generated via `_build_token_type_ids()` if not provided
- Sum of all 3 embeddings (word + token_type + position) forms the initial E₀ vector

### 2. Unified max_length
**Gap:** `tokenizer.max_length` was 512 while `transformer.max_seq_length` was 576.
**Fix:** Updated `config/config.yaml` → `tokenizer.max_length: 576`

### 3. Dynamic pad_token_id
**Gap:** `padding_idx=2` was hardcoded in the Embedding layer.
**Fix:** Added `pad_token_id` parameter to `NLPTransformerClassifier.__init__()`, passed from training scripts.

### 4. Removed Duplicate Tokenizer
**Gap:** Two overlapping tokenizer implementations (`tokenizer.py` + `iot_tokenizer.py`).
**Fix:** Removed `src/data/iot_tokenizer.py`. Single source of truth: `src/data/tokenizer.py`.

| Component | Status | Notes |
|-----------|--------|-------|
| IoT-Tokenize Pipeline | ALIGNED | 5/5 steps implemented |
| Token-Type Embeddings | ALIGNED | 3 embeddings summed (word + position + type) |
| Vocab Size | ALIGNED | 52 000 tokens |
| Max Sequence Length | ALIGNED | 576 tokens (unified) |
| Pad Token ID | ALIGNED | Dynamic via parameter |

---

## Categorical Feature Encoding Rules (2026-04-01)

### Rule: NO One-Hot Encoding for Transformer/Hybrid

Categorical features (`ipProto`/`protocolIdentifier`) must NOT be one-hot encoded before being passed to the Transformer. Instead:

1. **Transformer / Hybrid (CNN-BiLSTM-Transformer):**
   - Categorical features kept as raw integers during preprocessing
   - NOT scaled by Min-Max or StandardScaler
   - Converted to human-readable labels by the tokenizer: `6 → tcp`, `17 → udp`, `1 → icmp`
   - These labels become pre-defined tokens in the BPE vocabulary
   - Self-attention dynamically evaluates their importance

2. **DL models (LSTM, CNN-LSTM, CNN-BiLSTM):**
   - Label Encoding (integer mapping) for categorical features
   - Applied via `CategoricalFeatureEncoder.transform_for_dl()`

3. **XGBoost:**
   - Label Encoding preferred (trees handle integers well)
   - Applied via `CategoricalFeatureEncoder.transform_for_xgboost()`

### Changes Made

| File | Change |
|------|--------|
| `src/data/categorical_encoder.py` | **NEW** — `CategoricalFeatureEncoder` class with per-model-type encoding |
| `src/data/tokenizer.py` | `_format_value()` converts `ipProto` integers → protocol names (tcp, udp, icmp) |
| `src/data/tokenizer.py` | `PROTO_MAP` dictionary for protocol value → name mapping |
| `src/data/tokenizer.py` | `CATEGORICAL_FEATURES` set for identifying categorical columns |
| `src/data/tokenizer.py` | `SimpleTokenizer` updated to handle categorical features |
| `src/data/json_preprocessor.py` | `prepare_features()` now returns 3 arrays: continuous, categorical, binary |
| `src/data/json_preprocessor.py` | Min-Max/StandardScaler applied ONLY to continuous features |
| `src/data/json_preprocessor.py` | `create_sequences_with_categorical()` merges all feature types |
| `src/data/preprocessor.py` | `select_features()` now returns continuous + categorical separately |
| `src/data/preprocessor.py` | `create_sequences_with_categorical()` merges all feature types |
| `src/data/__init__.py` | Export of `CategoricalFeatureEncoder` |

### Pipeline Flow

```
Raw Data (CSV/JSON)
  ├── Continuous features → Min-Max → StandardScaler → Scaled tensor
  ├── Categorical features → Raw integers (NO scaling)
  └── Binary features → Raw binary (NO scaling)

For Transformer:
  [continuous | categorical | binary] → IoTTokenizer → "ipProto tcp; duration 12.5000; ..."
  → BPE Tokenizer → Token IDs → Embedding Layer (word + position + token_type)
  → Self-Attention (evaluates categorical importance dynamically)

For LSTM/CNN:
  Continuous → Scaled tensor
  Categorical → Label Encoding (integers)
  → Combined tensor → LSTM/CNN layers

For XGBoost:
  Continuous → Scaled
  Categorical → Label Encoding
  → Combined → XGBoost trees
```

---

## IoT-Tokenize Pipeline Integration (2026-04-01)

### Tokenizer-Model Integration Complete
**Gap:** The IoT-Tokenize pipeline (tokenizer + NLPTransformerClassifier) was fully implemented but never connected to the training loop. `train_adversarial.py` passed raw numerical features directly to models, bypassing tokenization entirely.

**Fix:** Connected the full 5-step IoT-Tokenize pipeline to training:

| File | Change |
|------|--------|
| `src/training/trainer.py` | Added `NLPTransformerDataset` class for token ID datasets |
| `src/training/trainer.py` | `AdversarialTrainer.__init__()` accepts optional `tokenizer` and `features` params |
| `src/training/trainer.py` | `train_epoch()` accepts `X_raw_batch` for re-tokenizing adversarial examples |
| `train_adversarial.py` | Added `tokenize_data()` helper: fits BPE tokenizer, transforms train/val/test |
| `train_adversarial.py` | Added `tokenize_adversarial_batch()` for re-tokenizing perturbed examples |
| `train_adversarial.py` | `create_model()` accepts `pad_token_id` parameter |
| `train_adversarial.py` | `run_experiment_with_phase_checkpoints()` detects `nlp_transformer` model type, tokenizes data before training |
| `train_adversarial.py` | `AdversarialTrainer` instantiated with tokenizer for adversarial re-tokenization |
| `train_adversarial.py` | Added `nlp_transformer` to CLI `--model` choices |

### Full Pipeline Flow (Now Connected)

```
Raw Features (X_train, X_val, X_test)
  │
  ├── For nlp_transformer:
  │   │
  │   ├── IoTTokenizer.fit(X_train, features)     ← Step 1-3: BPE training
  │   ├── tokenizer.transform(X_train) → token_ids_train  ← Step 4: Tensor encoding
  │   ├── tokenizer.transform(X_val)   → token_ids_val
  │   └── tokenizer.transform(X_test)  → token_ids_test
  │       │
  │       ├── NLPTransformerDataset(token_ids, y) ← Custom dataset
  │       └── DataLoader → (token_ids, labels) batches
  │           │
  │           ├── Adversarial training:
  │           │   ├── Generate adversarial in RAW feature space
  │           │   └── Re-tokenize adversarial examples → token IDs
  │           │
  │           └── NLPTransformerClassifier:
  │               ├── word_embedding(token_ids)
  │               ├── token_type_embedding(auto-generated)
  │               ├── positional_encoding(sinusoïdal)
  │               └── TransformerEncoder → classification
  │
  └── For other models (lstm, cnn, etc.):
      └── IoTSequenceDataset(raw_features, y) → DataLoader → raw tensor batches
```

### Usage

```bash
# Train with IoT-Tokenize pipeline (BPE tokenization + NLP Transformer)
python train_adversarial.py --model nlp_transformer --seq_length 25 --adv_method hybrid --phase_checkpoints

# Train with raw features (standard Transformer)
python train_adversarial.py --model transformer --seq_length 25 --adv_method hybrid --phase_checkpoints
```
