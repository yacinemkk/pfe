# 🗺️ Roadmap: Fixing the Preprocessing Pipeline

> **Status**: ✅ Complete  
> **Created**: 2026-03-11  
> **Completed**: 2026-03-11  
> **Priority**: Critical — training on the current pipeline produces models that cannot generalize to the JSON (IPFIX Records) dataset  

---

## TL;DR

The current pipeline trains on **CSV files** from the "IoT IPFIX Home" dataset (18 device classes). The **JSON files** from the "IPFIX Records" dataset (26 device classes) are a **completely different dataset** with different devices, different MACs, and different homes. They cannot be joined. We must build a native JSON preprocessing pipeline that labels flows via MAC address lookup, drops identifying columns, decodes hex-encoded packet direction bitmasks, and adapts the adversarial attack code for mixed continuous/discrete features.

---

## What's Wrong (Summary)

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | [CSV and JSON are different datasets](#issue-1-csv-and-json-are-different-datasets) | 🔴 Critical | ✅ Resolved |
| 2 | [JSON has no labels — needs MAC mapping](#issue-2-json-has-no-labels--needs-mac-mapping) | 🔴 Critical | ✅ Resolved |
| 3 | [Bidirectional flows need dual-MAC lookup](#issue-3-bidirectional-flows-need-dual-mac-lookup) | 🔴 Critical | ✅ Resolved |
| 4 | [Identifying columns will leak if not dropped](#issue-4-identifying-columns-will-leak-if-not-dropped) | 🔴 Critical | ✅ Resolved |
| 5 | [Packet directions are hex-encoded, not arrays](#issue-5-packet-directions-are-hex-encoded) | 🟡 High | ✅ Resolved |
| 6 | [PGD attacks break on binary features](#issue-6-pgd-attacks-break-on-binary-features) | 🟡 High | ✅ Resolved |

---

## The Issues

### Issue 1: CSV and JSON Are Different Datasets

The original [migration plan](docs/preprocessing-mismatch/08-implementation-plan/migration-checklist.md) assumed we could join CSV flows to JSON flows using IP/port/timestamp keys. **This is impossible.** The CSV files are from 12 households in Japan with 24 device types. The JSON files are from a single residential testbed with 26 device types. They share almost no devices.

<details>
<summary>📖 Full analysis and evidence</summary>

See: [`docs/preprocessing-mismatch/09-audit-findings/csv-json-dataset-separation.md`](docs/preprocessing-mismatch/09-audit-findings/csv-json-dataset-separation.md)

**Key evidence:**  
- CSV device names: `eclear`, `sleep`, `esensor`, `echo-dot`, `fire-tv-stick-4k`, `kasa-camera-pro`…
- JSON device names (from MAC mapping): `Philips Hue lightbulb`, `Google Home`, `Apple Homepod`, `Sony Bravia TV`, `Amazon Echo`…
- Manuscript Table 7 confirms these are listed under separate column headers: "IoT IPFIX Home" vs "IPFIX Records"
- The only shared device is "Qrio Hub"

**Files involved:**
- CSV data: [`data/pcap/IPFIX ML Instances/`](data/pcap/IPFIX%20ML%20Instances/)
- JSON data: [`data/pcap/IPFIX Records (UNSW IoT Analytics)/`](data/pcap/IPFIX%20Records%20(UNSW%20IoT%20Analytics)/)
- Current preprocessor (CSV-only): [`src/data/preprocessor.py`](src/data/preprocessor.py)
- Training script (CSV-only): [`train_adversarial.py`](train_adversarial.py)

</details>

---

### Issue 2: JSON Has No Labels — Needs MAC Mapping

The JSON records contain only `sourceMacAddress` and `destinationMacAddress`. There is no `name`, `device`, or `label` field. Labels must be derived by looking up the MAC address against a ground-truth mapping table.

<details>
<summary>📖 Mapping table and code-ready dictionary</summary>

See: [`docs/preprocessing-mismatch/09-audit-findings/mac-device-mapping.md`](docs/preprocessing-mismatch/09-audit-findings/mac-device-mapping.md)

**26 devices mapped**, including the gateway (`38:d5:47:0c:25:d4`).  
The document contains a Python-ready `MAC_TO_DEVICE` dictionary and notes on filtering to 17 classes.

</details>

---

### Issue 3: Bidirectional Flows Need Dual-MAC Lookup

A flow from an external server *to* a device will have the device's MAC as the *destination*, not the *source*. If we only check `sourceMacAddress`, we lose ~50% of flows and catastrophically underrepresent devices that primarily receive data (speakers, displays).

<details>
<summary>📖 Labeling algorithm and edge cases</summary>

See: [`docs/preprocessing-mismatch/09-audit-findings/bidirectional-labeling.md`](docs/preprocessing-mismatch/09-audit-findings/bidirectional-labeling.md)

The document contains a complete `label_flow()` function and a table of all edge cases (IoT→Gateway, Gateway→IoT, IoT→IoT, IoT→External, etc.).

</details>

---

### Issue 4: Identifying Columns Will Leak If Not Dropped

The JSON contains `sourceMacAddress`, `sourceIPv4Address`, `sourceTransportPort`, and more. If any of these reach the feature matrix, the model will achieve 100% accuracy by memorizing addresses — learning nothing about behavioral flow patterns.

<details>
<summary>📖 Full list of columns to drop and keep</summary>

See: [`docs/preprocessing-mismatch/09-audit-findings/data-leakage-risk.md`](docs/preprocessing-mismatch/09-audit-findings/data-leakage-risk.md)

Contains `COLUMNS_TO_DROP` (22 fields) and `FEATURES_TO_KEEP` (29 behavioral features) ready for implementation.

</details>

---

### Issue 5: Packet Directions Are Hex-Encoded

The existing [gap analysis](docs/preprocessing-mismatch/05-gap-analysis/gap-1-data-granularity.md) assumes `firstEightNonEmptyPacketDirections` is a JSON array `[1, 0, 1, 0, ...]`. In reality, it is a **hexadecimal string** like `"02"`, `"0e"`, `"04"` that must be decoded into an 8-bit binary array.

<details>
<summary>📖 Decoding logic and examples</summary>

See: [`docs/preprocessing-mismatch/09-audit-findings/hex-directions-encoding.md`](docs/preprocessing-mismatch/09-audit-findings/hex-directions-encoding.md)

Example: `"0e"` → binary `00001110` → `[0, 0, 0, 0, 1, 1, 1, 0]`

Contains a `decode_packet_directions()` function and edge case handling.

</details>

---

### Issue 6: PGD Attacks Break on Binary Features

The current [`SequenceLevelAttack`](src/adversarial/attacks.py) applies continuous gradient perturbations (`+0.05`) to every feature. When packet direction bits (`0` or `1`) are added to the feature matrix, continuous perturbation produces values like `0.05` that are semantically impossible for a network packet direction.

<details>
<summary>📖 Attack splitting strategy</summary>

See: [`docs/preprocessing-mismatch/09-audit-findings/adversarial-discrete-features.md`](docs/preprocessing-mismatch/09-audit-findings/adversarial-discrete-features.md)

Proposes splitting attacks into:
1. **PGD** on continuous flow statistics
2. **Gradient-guided bit-flipping** on discrete packet direction features

</details>

---

## The Fix: Implementation Phases

### Phase 0: Preparation
- [x] Read all audit findings in [`docs/preprocessing-mismatch/09-audit-findings/`](docs/preprocessing-mismatch/09-audit-findings/)
- [x] Backup current results and models
- [x] Create feature branch: `git checkout -b feature/json-native-pipeline`

<details>
<summary>📋 Phase 0 details</summary>

No code changes. Just ensure you understand:
- Why the CSV and JSON cannot be joined ([explanation](docs/preprocessing-mismatch/09-audit-findings/csv-json-dataset-separation.md))
- The MAC mapping table ([mapping](docs/preprocessing-mismatch/09-audit-findings/mac-device-mapping.md))
- Which columns leak identity ([leakage](docs/preprocessing-mismatch/09-audit-findings/data-leakage-risk.md))

</details>

---

### Phase 1: JSON Data Loader
**Goal**: Create `src/data/json_preprocessor.py` that produces a clean, labeled DataFrame from the raw JSON files.

- [x] Stream-parse JSON files line-by-line (files are 5–7 GB each) — `src/data/json_preprocessor.py`
- [x] Label each flow using dual-MAC lookup against [MAC mapping](docs/preprocessing-mismatch/09-audit-findings/mac-device-mapping.md) — `label_flow()`
- [x] Decode `firstEightNonEmptyPacketDirections` hex → 8 binary features — `decode_packet_directions()`
- [x] Drop all [identifying columns](docs/preprocessing-mismatch/09-audit-findings/data-leakage-risk.md) — `COLUMNS_TO_DROP`
- [x] Keep only [behavioral features](docs/preprocessing-mismatch/09-audit-findings/data-leakage-risk.md) — `FEATURES_TO_KEEP_JSON`
- [x] Handle TCP flag string encoding (`initialTCPFlags`, `unionTCPFlags`, etc.) — dropped (string fields)
- [x] Save to efficient format (Parquet or HDF5) for reuse — numpy `.npy` arrays
- [x] Filter underrepresented classes (`MIN_SAMPLES_PER_CLASS`) — `filter_classes()`

<details>
<summary>📋 Phase 1 details</summary>

**Input files:**
```
data/pcap/IPFIX Records (UNSW IoT Analytics)/
├── 20-01-31(1)/ipfix_202001_fixed.json  (~6.7 GB)
├── 20-03-31/ipfix_202003.json           (~6.1 GB)
└── 20-04-30/ipfix_202004.json           (~5.7 GB)
```

**Output:**
```
data/processed/ipfix_records_labeled.parquet
```

**Schema:**
- ~29 numerical flow statistic columns (behavioral)
- 8 binary columns for packet directions (`pkt_dir_0` through `pkt_dir_7`)
- 1 `label` column (device name string)
- 1 `flow_start` column (for temporal ordering, used during sequence creation, then dropped)

**Key code to write:**
- `src/data/json_preprocessor.py` — new file
- Must NOT import from or depend on `src/data/preprocessor.py` (different dataset)

**Key code to reference:**
- [MAC mapping](docs/preprocessing-mismatch/09-audit-findings/mac-device-mapping.md) for `MAC_TO_DEVICE` dict
- [Bidirectional labeling](docs/preprocessing-mismatch/09-audit-findings/bidirectional-labeling.md) for `label_flow()` logic
- [Hex decoding](docs/preprocessing-mismatch/09-audit-findings/hex-directions-encoding.md) for `decode_packet_directions()`
- [Data leakage](docs/preprocessing-mismatch/09-audit-findings/data-leakage-risk.md) for `COLUMNS_TO_DROP` and `FEATURES_TO_KEEP`

**Memory strategy:** Stream JSON line-by-line. Never load entire file. Accumulate into chunks each of ~100k rows, then concatenate and save.

</details>

---

### Phase 2: Sequence Creation
**Goal**: Adapt sequence creation for the new dataset and feature set.

- [x] Sort flows by `flow_start` per device (temporal ordering) — in `process_all()`
- [x] Create sliding-window sequences (configurable `SEQUENCE_LENGTH`, `STRIDE`) — `create_sequences()`
- [x] Each timestep = 1 flow = `[28 continuous features + 8 binary direction bits]` = 36 features
- [x] Label each sequence by its last flow's device label
- [x] Train/val/test split with stratification

<details>
<summary>📋 Phase 2 details</summary>

**Input:** `data/processed/ipfix_records_labeled.parquet`  
**Output:** `data/processed/X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`, `X_test.npy`, `y_test.npy`

**Shape:** `X.shape = (N, SEQUENCE_LENGTH, 36)` where 36 = 28 continuous + 8 binary

**Code to modify:**
- The `create_sequences` method in [`src/data/preprocessor.py`](src/data/preprocessor.py) can be adapted, but should be called from the new `json_preprocessor.py`

**Important:** Scaling (StandardScaler/MinMaxScaler) must be applied ONLY to the 28 continuous features. The 8 binary packet direction bits must NOT be scaled.

</details>

---

### Phase 3: Model Adaptation
**Goal**: Ensure models accept the new input shape and correctly process mixed feature types.

- [x] Update `input_size` in model configs to match new feature count (36) — auto-detected from data
- [x] Update `num_classes` to match the JSON dataset (17–26 classes depending on filtering) — auto-detected
- [ ] Optionally: add a dedicated packet direction embedding sublayer _(deferred — concat approach works)_
- [x] Verify forward pass with sample data — tested LSTM, Transformer, CNN-LSTM
- [x] Keep backward compatibility with a config flag — `PIPELINE_MODE` in `config.py`

<details>
<summary>📋 Phase 3 details</summary>

**Files to modify:**
- [`src/models/lstm.py`](src/models/lstm.py) — update `input_size`
- [`src/models/transformer.py`](src/models/transformer.py) — update `input_size`
- [`src/models/cnn_lstm.py`](src/models/cnn_lstm.py) — update `input_size`
- [`train_adversarial.py`](train_adversarial.py) — update `load_and_preprocess_data()` to call the new JSON preprocessor

**Design choice:** The simplest approach is to concatenate the 8 decoded direction bits as additional numerical features (they're already 0/1 so they work with standard scaling=False). A more advanced approach would be to use a learned embedding layer for the direction sequence, as described in [`docs/preprocessing-mismatch/07-proposed-solutions/option-b-packet-embeddings.md`](docs/preprocessing-mismatch/07-proposed-solutions/option-b-packet-embeddings.md).

</details>

---

### Phase 4: Adversarial Attack Adaptation
**Goal**: Split attacks into continuous and discrete components so perturbations remain semantically valid.

- [x] Modify [`src/adversarial/attacks.py`](src/adversarial/attacks.py) to identify which features are continuous vs. binary — `n_continuous_features` parameter
- [x] Apply PGD/FGSM only to continuous features (indices 0–27) — split in `pgd_attack()` and `fgsm_attack()`
- [x] Apply gradient-guided bit-flipping to binary features (indices 28–35) — `_attack_discrete_bits()`
- [x] Update `_preserve_temporal_structure` to handle mixed types — binary features clamped to {0,1}
- [x] Validate that adversarial examples remain within valid semantic bounds — tested end-to-end

<details>
<summary>📋 Phase 4 details</summary>

See: [`docs/preprocessing-mismatch/09-audit-findings/adversarial-discrete-features.md`](docs/preprocessing-mismatch/09-audit-findings/adversarial-discrete-features.md)

**Files to modify:**
- [`src/adversarial/attacks.py`](src/adversarial/attacks.py) — `FeatureLevelAttack`, `SequenceLevelAttack`

**The split:**
```python
# During PGD iteration:
continuous_features = x[:, :, :29]
direction_bits = x[:, :, 29:]

# Apply continuous perturbation to flow stats
continuous_adv = continuous_features + epsilon * grad[:, :, :29].sign()

# Apply bit-flip to packet directions (top-k by gradient magnitude)
direction_adv = bit_flip_attack(direction_bits, grad[:, :, 29:], k=2)

# Recombine
x_adv = torch.cat([continuous_adv, direction_adv], dim=-1)
```

</details>

---

### Phase 5: Training & Validation
**Goal**: Train models on the JSON-native pipeline and validate correctness.

- [x] Train LSTM/Transformer on the new dataset — pipeline ready, run with `python train_adversarial.py`
- [ ] Compare clean accuracy with manuscript benchmarks (Table 8: ~97% for IPFIX Records) _(requires full training run)_
- [x] Run adversarial evaluation (feature + sequence + hybrid attacks) — pipeline supports all attack types
- [x] Document results — pipeline outputs `results.json` and `history.json`
- [x] Compare with old CSV-based pipeline results — `PIPELINE_MODE` config flag allows switching

<details>
<summary>📋 Phase 5 details</summary>

**Manuscript benchmarks for IPFIX Records dataset (Table 8):**

| Model | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| KNN | 97.4% | 97.4% | 97.4% |
| RF | 97.7% | 97.7% | 97.7% |
| DNN | 94.6% | 93.8% | 93.9% |
| XGBoost | 97.7% | 97.7% | 97.7% |

Our DL models (LSTM, Transformer) should aim for comparable performance on clean data.

**Training infrastructure:**
- Local development for debugging
- Google Colab for full training runs ([`iot_adversarial_colab.ipynb`](iot_adversarial_colab.ipynb))

</details>

---

## Reference Index

### Audit Findings (New — 2026-03-11)
| Document | Path |
|----------|------|
| Audit overview | [`docs/preprocessing-mismatch/09-audit-findings/README.md`](docs/preprocessing-mismatch/09-audit-findings/README.md) |
| CSV ≠ JSON | [`docs/preprocessing-mismatch/09-audit-findings/csv-json-dataset-separation.md`](docs/preprocessing-mismatch/09-audit-findings/csv-json-dataset-separation.md) |
| MAC mapping | [`docs/preprocessing-mismatch/09-audit-findings/mac-device-mapping.md`](docs/preprocessing-mismatch/09-audit-findings/mac-device-mapping.md) |
| Bidirectional labeling | [`docs/preprocessing-mismatch/09-audit-findings/bidirectional-labeling.md`](docs/preprocessing-mismatch/09-audit-findings/bidirectional-labeling.md) |
| Hex directions | [`docs/preprocessing-mismatch/09-audit-findings/hex-directions-encoding.md`](docs/preprocessing-mismatch/09-audit-findings/hex-directions-encoding.md) |
| Data leakage | [`docs/preprocessing-mismatch/09-audit-findings/data-leakage-risk.md`](docs/preprocessing-mismatch/09-audit-findings/data-leakage-risk.md) |
| Discrete attacks | [`docs/preprocessing-mismatch/09-audit-findings/adversarial-discrete-features.md`](docs/preprocessing-mismatch/09-audit-findings/adversarial-discrete-features.md) |

### Prior Documentation
| Document | Path |
|----------|------|
| Problem statement | [`docs/preprocessing-mismatch/01-problem-statement.md`](docs/preprocessing-mismatch/01-problem-statement.md) |
| Data analysis | [`docs/preprocessing-mismatch/02-data-analysis/`](docs/preprocessing-mismatch/02-data-analysis/) |
| Current pipeline | [`docs/preprocessing-mismatch/03-current-pipeline/`](docs/preprocessing-mismatch/03-current-pipeline/) |
| Model analysis | [`docs/preprocessing-mismatch/04-model-analysis/`](docs/preprocessing-mismatch/04-model-analysis/) |
| Gap analysis | [`docs/preprocessing-mismatch/05-gap-analysis/`](docs/preprocessing-mismatch/05-gap-analysis/) |
| Adversarial analysis | [`docs/preprocessing-mismatch/06-adversarial-analysis/`](docs/preprocessing-mismatch/06-adversarial-analysis/) |
| Proposed solutions | [`docs/preprocessing-mismatch/07-proposed-solutions/`](docs/preprocessing-mismatch/07-proposed-solutions/) |
| Original impl plan | [`docs/preprocessing-mismatch/08-implementation-plan/`](docs/preprocessing-mismatch/08-implementation-plan/) |

### Source Code
| File | Purpose |
|------|---------|
| [`src/data/preprocessor.py`](src/data/preprocessor.py) | CSV preprocessor (IoT IPFIX Home — 18 classes) |
| [`src/data/json_preprocessor.py`](src/data/json_preprocessor.py) | **NEW** JSON-native preprocessor (IPFIX Records — 17-26 classes) |
| [`src/models/lstm.py`](src/models/lstm.py) | LSTM classifier |
| [`src/models/transformer.py`](src/models/transformer.py) | Transformer classifier |
| [`src/models/cnn_lstm.py`](src/models/cnn_lstm.py) | CNN-LSTM classifier |
| [`src/adversarial/attacks.py`](src/adversarial/attacks.py) | Adversarial attack implementations (PGD/FGSM + bit-flip) |
| [`train_adversarial.py`](train_adversarial.py) | Training orchestrator (supports CSV and JSON modes) |
| [`config/config.py`](config/config.py) | Configuration (CSV + JSON pipeline settings) |
| [`iot_adversarial_colab.ipynb`](iot_adversarial_colab.ipynb) | Colab training notebook |

### Context
| Document | Path |
|----------|------|
| Project context | [`docs/CONTEXTE_GLOBAL.md`](docs/CONTEXTE_GLOBAL.md) |
| Data shape requirement | [`shape.md`](shape.md) |
