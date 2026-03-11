# New Files Needed

## Modules to Create

---

## Data Processing

### `src/data/json_extractor.py`

**Purpose**: Extract packet directions from JSON files

**Classes**:
- `JSONExtractor`: Main extraction class
- `FlowMatcher`: Match JSON to CSV records

**Functions**:
- `extract_packet_directions()`: Extract `firstEightNonEmptyPacketDirections`
- `match_flows()`: Match JSON flows to CSV records
- `save_to_parquet()`: Save extracted data

**Dependencies**:
- `json` (standard library)
- `pandas`
- `pyarrow` (for Parquet)

---

### `src/data/packet_directions.py`

**Purpose**: Process and encode packet direction sequences

**Classes**:
- `PacketDirectionProcessor`: Preprocess direction sequences
- `DirectionTokenizer`: Convert directions to tokens

**Functions**:
- `pad_directions()`: Pad/truncate to fixed length
- `encode_directions()`: Convert to tensor format

---

## Model Components

### `src/models/packet_embedding.py`

**Purpose**: Embedding layer for packet directions

**Classes**:
- `PacketDirectionEmbedding`: Learnable embedding for directions
- `DirectionSequenceEncoder`: Encode direction sequences (LSTM/Transformer)

**Configuration**:
- Embedding dimension: 16-32
- Encoder type: LSTM or Transformer
- Hidden size: 64

---

### `src/models/combined_input.py`

**Purpose**: Combine flow stats with packet embeddings

**Classes**:
- `CombinedInputLayer`: Concatenate and process combined input

**Functions**:
- `combine_flow_and_packet()`: Merge representations

---

## Attack Components

### `src/adversarial/packet_attack.py`

**Purpose**: Attack packet direction embeddings

**Classes**:
- `PacketDirectionAttack`: Attack direction sequences
- `ConstrainedPerturbation`: Perturbation respecting direction semantics

**Functions**:
- `attack_directions()`: Generate adversarial direction sequences
- `validate_directions()`: Check if perturbation is valid

---

## Utilities

### `src/utils/json_utils.py`

**Purpose**: Helper functions for JSON processing

**Functions**:
- `stream_json()`: Iterate over large JSON files
- `parse_json_line()`: Parse single JSON line
- `get_field()`: Safely extract field

---

### `src/utils/matching.py`

**Purpose**: Flow matching utilities

**Functions**:
- `match_by_key()`: Match flows by composite key
- `fuzzy_match()`: Fuzzy matching for timestamps
- `validate_match()`: Check match quality

---

## Configuration

### `config/packet_embedding_config.py`

**Purpose**: Configuration for packet embeddings

**Constants**:
```python
PACKET_EMBEDDING_DIM = 16
DIRECTION_ENCODER_HIDDEN = 64
MAX_PACKET_DIRECTIONS = 8
DIRECTION_VOCAB_SIZE = 3  # 0=in, 1=out, 2=padding
```

---

## Summary

| Category | File | Purpose |
|----------|------|---------|
| Data | `json_extractor.py` | Extract from JSON |
| Data | `packet_directions.py` | Process directions |
| Model | `packet_embedding.py` | Embedding layer |
| Model | `combined_input.py` | Combine representations |
| Attack | `packet_attack.py` | Attack directions |
| Util | `json_utils.py` | JSON helpers |
| Util | `matching.py` | Matching helpers |
| Config | `packet_embedding_config.py` | Configuration |

**Total new files: 8**

---

## File Creation Order

1. `config/packet_embedding_config.py` - Define constants first
2. `src/utils/json_utils.py` - Needed by extractor
3. `src/utils/matching.py` - Needed by extractor
4. `src/data/json_extractor.py` - Extract data
5. `src/data/packet_directions.py` - Process data
6. `src/models/packet_embedding.py` - Model component
7. `src/models/combined_input.py` - Model component
8. `src/adversarial/packet_attack.py` - Attack component

---

Return to [Implementation Plan Index](./README.md) | Return to [Main Index](../README.md)
