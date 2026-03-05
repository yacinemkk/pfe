# IoT device traffic classification with adversarial learning

**Session ID:** ses_36fe805b8ffeZXUEWPEW60AFzd
**Created:** 2/24/2026, 3:40:16 PM
**Updated:** 2/24/2026, 3:56:29 PM

---

## User

i want to create models that can identify type of IoT device based on its traffic, and robustly, send subagents to research what we have here, then send subagents to research how this can be done, make note that the traffic is a sequence, and doing adversarial learning on it is not clear to me.

---

## Assistant (Plan · z-ai/glm-5:free · 255.5s)

_Thinking:_

The user wants to create models that can identify IoT device types based on traffic. They want me to:
1. Research what's already in their codebase/project
2. Research how this can be done (methodology)
3. Note that traffic is sequential data
4. Note that adversarial learning on sequential data is unclear to them

I'm in plan mode, so I should only read/inspect and delegate to explore agents. Let me send multiple explore agents in parallel to:
1. Explore the current codebase structure and what exists
2. Research IoT device identification methods
3. Research adversarial learning on sequential data

Let me start by exploring the codebase first, then research external methodologies.

I'll launch explore agents to research your codebase and the methodology for IoT device traffic classification with adversarial robustness.

**Tool: task**

**Input:**
```json
{
  "description": "Explore codebase structure",
  "prompt": "Explore the codebase at /home/pc/Desktop/pfe thoroughly. I need to understand:\n1. What is the overall project structure and purpose?\n2. What datasets exist (look for CSV, PCAP, JSON data files)?\n3. What existing models or ML code exists?\n4. What preprocessing or feature extraction code exists?\n5. Any existing adversarial learning implementations?\n6. What frameworks are used (PyTorch, TensorFlow, etc.)?\n\nReturn a comprehensive summary of what exists in this project.",
  "subagent_type": "explore"
}
```

**Output:**
```
task_id: ses_36fe7a496ffeN1sURhNWnPkTW5 (for resuming to continue this task if needed)

<task_result>
Based on my thorough exploration of the codebase at `/home/pc/Desktop/pfe`, here is a comprehensive summary:

---

## 1. Overall Project Structure and Purpose

**Project Name:** PFE (Projet de Fin d'Études) - IoT Device Identification for SDN-NAC

**Purpose:** This is a research project focused on IoT device identification in Software-Defined Networking (SDN) environments with Network Access Control (NAC). The key innovation is developing **robust deep learning models** (LSTM and Transformer) that can:
- Identify IoT devices from network traffic
- Resist adversarial attacks (perturbations designed to fool ML models)
- Provide adaptive defense mechanisms

**Directory Structure:**
```
/home/pc/Desktop/pfe/
├── src/
│   ├── models/
│   │   ├── lstm.py          # PyTorch LSTM classifier
│   │   └── transformer.py   # PyTorch Transformer classifier
│   ├── data/
│   │   └── preprocessor.py  # Data preprocessing pipeline
│   └── utils/               # (empty, placeholder)
├── data/
│   ├── pcap/
│   │   ├── IPFIX ML Instances/     # Main dataset (CSV files)
│   │   └── IPFIX Records (UNSW IoT Analytics)/  # Raw IPFIX JSON data
│   └── flow_csv/            # (empty, placeholder)
├── config/
│   └── config.py            # Central configuration
├── notebooks/
│   └── 01_Data_Preprocessing.ipynb  # Jupyter notebook for Colab
├── results/
│   ├── metrics/             # (empty)
│   ├── models/              # (empty)
│   └── plots/               # (empty)
├── alo/                     # Development notes & Colab scripts
│   ├── train_lstm_colab.py
│   ├── train_transformer_colab.py
│   └── *.txt (French documentation)
└── Manuscript anonymous.pdf  # Research paper draft
```

---

## 2. Datasets

### CSV Files (IPFIX ML Instances - Main Dataset)
**Location:** `/home/pc/Desktop/pfe/data/pcap/IPFIX ML Instances/`

**Labeled CSV Files (32+ million rows total):**
| File | Rows |
|------|------|
| home1_labeled.csv | 1,152,574 |
| home2_labeled.csv | 1,401,332 |
| home3_labeled.csv | 1,707,786 |
| home4_labeled.csv | 8,071,109 |
| home5_labeled.csv | 1,791,773 |
| home6_labeled.csv | 4,643,642 |
| home7_labeled.csv | 2,652,648 |
| home8_labeled.csv | 1,518,574 |
| home9_labeled.csv | 1,137,143 |
| home10_labeled.csv | 5,373,493 |
| home11_labeled.csv | 1,037,094 |
| home12_labeled.csv | 1,630,018 |

**44 Features including:**
- `duration`, `ipProto` 
- `outPacketCount`, `outByteCount`, `inPacketCount`, `inByteCount`
- `outAvgIAT`, `outMaxPktSize`, `outStdevPayloadSize`, `outAvgPacketSize`
- Protocol flags: `http`, `https`, `smb`, `dns`, `ntp`, `tcp`, `udp`, `ssdp`
- Network flags: `lan`, `wan`, `deviceInitiated`
- Labels: `device` (ID), `name` (device type)

**18 IoT Device Classes:**
Eclear, Sleep, Esensor, Hub Plus, Humidifier, Home Unit, Ink Jet Printer, Smart Wi-Fi Plug Mini, Smart Power Strip, Echo Dot, Fire 7 Tablet, Google Nest Mini, Google Chromecast, Atom Cam, Kasa Camera Pro, Kasa Smart LED Lamp, Fire TV Stick 4K, Qrio Hub

### JSON Files (UNSW IoT Analytics - Raw IPFIX)
**Location:** `/home/pc/Desktop/pfe/data/pcap/IPFIX Records (UNSW IoT Analytics)/`
- `ipfix_202001_fixed.json` (7.1 GB)
- `ipfix_202003.json` (6.1 GB)
- `ipfix_202004.json` (5.6 GB)

### Other Data Files
- `devices.xlsx` - Device mapping file for labeling

---

## 3. Existing Models / ML Code

### PyTorch Models (`/home/pc/Desktop/pfe/src/models/`)

**LSTM Model (`lstm.py`):**
- `LSTMClassifier` - Bidirectional LSTM with configurable layers
- `IoTSequenceDataset` - Custom PyTorch Dataset
- `Trainer` class with train/evaluate/predict methods
- Config: hidden_size=128, 2 layers, bidirectional, dropout=0.3
- Uses CrossEntropyLoss with Adam optimizer
- Learning rate scheduling (ReduceLROnPlateau)

**Transformer Model (`transformer.py`):**
- `TransformerClassifier` - Encoder-only Transformer
- `PositionalEncoding` layer
- Config: d_model=64, nhead=4, 3 encoder layers, dim_feedforward=256
- Reuses `Trainer` from LSTM module

### TensorFlow/Keras Models (`/home/pc/Desktop/pfe/alo/`)

**Colab Training Scripts:**
- `train_lstm_colab.py` - Full 2-phase training pipeline with TensorFlow
- `train_transformer_colab.py` - Transformer training with Keras
- Both implement:
  - Phase 1: Training on benign traffic
  - Phase 2: Adversarial training (80% benign + 20% adversarial)

---

## 4. Preprocessing / Feature Extraction Code

**Preprocessor (`/home/pc/Desktop/pfe/src/data/preprocessor.py`):**
- `IoTDataProcessor` class with:
  - `load_all_data()` - Load and concatenate CSV files
  - `clean_data()` - Remove duplicates, NaN, filter rare classes
  - `select_features()` - Extract SDN-compatible features (excludes IP, MAC, ports)
  - `create_sequences()` - Convert to time series with configurable sequence length (10) and stride (5)
  - `fit_transform()` / `transform()` - StandardScaler normalization
  - `encode_labels()` - LabelEncoder for device classes
  - Train/val/test split (70%/10%/20%)

**SDN-Compatible Features (37 features):**
- Excludes: `start`, `srcMac`, `destMac`, `srcIP`, `destIP`, `srcPort`, `destPort`
- Keeps: flow statistics, packet counts, inter-arrival times, protocol flags

**Data Labeling Script (`join_data.py`):**
- Maps device IDs to device names using `devices.xlsx`

---

## 5. Adversarial Learning Implementations

**IoT-SDN Adversarial Attack (`IoTAdversarialAttack` class):**

Found in both `train_lstm_colab.py` and `train_transformer_colab.py`

**Attack Formulation:**
```
x_adv = Projection[x0 + c·t·mask·sign(μ_target - x0)·|Difference(μ_target, x0)|]
```

**Key Components:**
1. **Non-modifiable features:** ipProto, http, https, dns, ntp, tcp, udp, ssdp
2. **Dependent features:** inPacketCount↔outPacketCount, inByteCount↔outByteCount, etc.
3. **L0 Minimization:** Mask-based feature selection (limits number of modified features)
4. **L2 Minimization:** Target selection among 3 nearest classes (minimizes perturbation magnitude)
5. **Projection function:** Maintains semantic constraints, clips values to [-3, 3]

**Adversarial Training Strategy:**
- Generate adversarial examples from training data
- Mix 80% benign + 20% adversarial
- Re-train model on mixed dataset

---

## 6. Frameworks Used

### Primary Framework: **PyTorch** (src/models/)
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
```

### Secondary Framework: **TensorFlow/Keras** (alo/ Colab scripts)
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, MultiHeadAttention
```

### Supporting Libraries:
- **scikit-learn:** StandardScaler, LabelEncoder, train_test_split, classification_report, confusion_matrix
- **pandas/numpy:** Data manipulation
- **matplotlib/seaborn:** Visualization
- **tqdm:** Progress bars
- **pickle:** Model serialization

---

## Summary Table

| Aspect | Details |
|--------|---------|
| **Domain** | IoT Device Identification for SDN-NAC |
| **Dataset** | IPFIX ML Instances (32M+ rows, 12 homes, 18 device classes) |
| **Models** | LSTM, Transformer (PyTorch + TensorFlow/Keras) |
| **Features** | 37 SDN-compatible features (no IP/MAC/ports) |
| **Sequences** | Length=10, Stride=5 |
| **Adversarial Defense** | IoT-SDN specific attack + adversarial training |
| **Frameworks** | PyTorch (main), TensorFlow/Keras (Colab) |
</task_result>
```

**Tool: task**

**Input:**
```json
{
  "description": "Research IoT device classification",
  "prompt": "Research methods for IoT device identification from network traffic. Focus on:\n1. What features are commonly extracted from network traffic (packet sizes, inter-arrival times, protocols, etc.)?\n2. What deep learning architectures work well for sequential traffic data (LSTM, GRU, Transformers, CNNs)?\n3. What datasets are commonly used for IoT device classification?\n4. How is traffic represented as sequences?\n\nReturn a detailed summary of state-of-the-art approaches.",
  "subagent_type": "explore"
}
```

**Output:**
```
task_id: ses_36fe7a47fffe3zOhmu3xJ50fyU (for resuming to continue this task if needed)

<task_result>
Based on my research, here is a comprehensive summary of state-of-the-art approaches for IoT device identification from network traffic:

---

# IoT Device Identification from Network Traffic: State-of-the-Art Approaches

## 1. Features Commonly Extracted from Network Traffic

### 1.1 Packet-Level Features
| Feature Category | Specific Features |
|-----------------|-------------------|
| **Packet Size** | Packet length (bytes), direction-aware packet sizes, payload size |
| **Timing Features** | Inter-arrival times (IAT), timestamps, packet timing patterns |
| **Direction** | Inbound/outbound direction encoded in sequence |
| **Protocol Headers** | TCP/UDP flags, port numbers, IP header fields |

### 1.2 Flow-Level Statistical Features
| Feature Category | Specific Features |
|-----------------|-------------------|
| **Duration & Volume** | Flow duration, total packets, total bytes |
| **Rate Metrics** | Packets/second, bytes/second, burst rates |
| **Statistical Summaries** | Mean, std, min, max of packet sizes and IATs |
| **Port Patterns** | Source/destination ports, port diversity |
| **Protocol Diversity** | DNS queries, TLS cipher suites, HTTP headers |

### 1.3 Directional Packet Length Sequences (DPLS)
A highly effective representation where each value encodes **both size and direction**:
- Positive values = outbound packets
- Negative values = inbound packets  
- Magnitude = packet size in bytes

This approach achieves >99% accuracy on multiple datasets by capturing device-specific communication patterns.

### 1.4 Protocol-Specific Features
- **DNS**: Query patterns, domain names, query frequencies
- **TLS/SSL**: Cipher suites, SNI, certificate info, handshake patterns
- **DHCP**: Hostnames, vendor class identifiers
- **mDNS/SSDP**: Service announcements, device identifiers

### 1.5 Most Discriminative Features (Ranked by Importance)
Research using SHAP analysis and feature importance techniques identifies:
1. **Packet size statistics** (most important)
2. **Connection duration**
3. **Protocol type**
4. **Inter-arrival time patterns**
5. **Port number patterns**
6. **DNS query rates**

---

## 2. Deep Learning Architectures for Sequential Traffic Data

### 2.1 LSTM (Long Short-Term Memory)
**Strengths:**
- Excellent for capturing temporal dependencies in packet sequences
- Handles variable-length sequences naturally
- Effective for modeling long-term context

**Use Cases:**
- Device type classification from packet timing sequences
- Behavioral modeling of device activity cycles
- Anomaly detection in IoT networks

**Performance:** LSTM-based models achieve 95-99% accuracy on device classification tasks.

### 2.2 GRU (Gated Recurrent Unit)
**Characteristics:**
- Simpler than LSTM (fewer parameters)
- Faster training
- Comparable performance for shorter sequences

### 2.3 CNN (Convolutional Neural Networks) - 1D
**Strengths:**
- Extracts local patterns from packet sequences
- Computationally efficient
- Good for detecting short-range dependencies

**Applications:**
- 1D-CNN for directional packet length sequences
- Feature extraction from raw bytes
- Local pattern recognition in packet headers

### 2.4 Hybrid CNN-LSTM Architectatures (State-of-the-Art)
The most successful approach combines CNN and LSTM:

```
Input Sequence → CNN Layers (spatial features) → LSTM Layers (temporal features) → Dense → Output
```

**Key Variants:**
- **LSTM-CNN Cascade**: LSTM first, then CNN for feature refinement
- **CNN-LSTM Parallel**: Both process input, features concatenated
- **Self-Attention CNN-LSTM**: Adds attention mechanism for feature weighting

**Performance Metrics:**
- Accuracy: 98-99.9%
- Precision: 99.89%
- Recall: 99.85%
- False Positive Rate: 0.13%

### 2.5 Transformer-Based Models (Emerging SOTA)

#### Encoder-Only (BERT-style)
| Model | Description |
|-------|-------------|
| **ET-BERT** | Pre-trained on encrypted traffic, achieves 97-99% F1 on various tasks |
| **PERT** | Payload encoding using BERT, treats bytes as words |
| **PEAN** | Packet embedding with attention for intra/inter-packet relationships |

#### Decoder-Only (GPT-style)
| Model | Description |
|-------|-------------|
| **NetGPT** | GPT-2 based for traffic generation and classification |
| **TrafficGPT** | Autoregressive with linear attention for traffic modeling |

#### Key Innovations:
- **Self-supervised pre-training** on unlabeled traffic data
- **Masked Burst Modeling (MBM)**: Analogous to MLM in NLP
- **Same-origin BURST Prediction (SBP)**: Like Next Sentence Prediction
- **Fine-tuning** with small labeled datasets

### 2.6 Attention Mechanisms
**Self-Attention:**
- Captures long-range dependencies between packets
- Learns which packets are most relevant for classification
- Visualizable attention weights for interpretability

**SE (Squeeze-and-Excitation) Module:**
- Channel-wise feature recalibration
- Focuses on informative features
- Suppresses less relevant features

### 2.7 Architecture Comparison Summary
| Architecture | Best For | Accuracy | Training Speed |
|-------------|----------|----------|----------------|
| LSTM | Long temporal dependencies | 95-98% | Medium |
| CNN | Local patterns | 96-99% | Fast |
| CNN-LSTM Hybrid | Combined spatial-temporal | 98-99.9% | Medium |
| Transformer | Long-range dependencies | 97-99.5% | Slow (pre-training) |

---

## 3. Commonly Used Datasets for IoT Device Classification

### 3.1 UNSW IoT Datasets
| Dataset | Devices | Duration | Format | Notes |
|---------|---------|----------|--------|-------|
| **UNSW IoT Traces (2018)** | 28 devices | 6 months | PCAP, CSV | 7GB, flow-based features |
| **UNSW-IoTraffic (2025)** | 27 devices | 203 days | PCAP, flows, protocols | 95.5M packets, 4.9M flows |
| **UNSW HomeNet (2025)** | 105 devices | Extended | Flow-based | >200M data points, 88 features |

**Device Categories:** Cameras, smart plugs, door sensors, health monitors, home assistants, media devices

### 3.2 IoT Sentinel Dataset
- **Devices:** 31 IoT devices
- **Packets:** 102,347
- **Year:** 2017
- **Focus:** Passive fingerprinting based on network traffic properties

### 3.3 YourThings Dataset
- **Devices:** 45 diverse home IoT devices
- **Size:** ~150GB of PCAP traces
- **Categories:** Appliances, cameras, home assistants, home automation, media, network devices
- **Features:** Full-packet captures, security annotations, CVE information

### 3.4 Other Notable Datasets
| Dataset | Devices | Focus |
|---------|---------|-------|
| **D-Link IoT** | 12 devices | Smart home devices |
| **MonIoTr** | Multiple | Mobile IoT traffic |
| **CIC-IoT2023** | 105 devices | IoT attack traffic (DDoS, botnet, Mirai) |
| **BoT-IoT** | Various | Botnet attacks in IoT |
| **ISCX-Tor/VPN** | Various | Encrypted traffic classification |

### 3.5 Dataset Selection Considerations
1. **Device Diversity:** Number and types of devices
2. **Traffic Duration:** Longer captures capture more behavioral patterns
3. **Data Format:** PCAP (raw) vs. CSV (pre-processed)
4. **Annotation Quality:** Labels for device type, state, events
5. **Real-world vs. Lab:** Deployment conditions affect generalization

---

## 4. Traffic Representation as Sequences

### 4.1 Byte-Level Representations

#### Raw Byte Sequences
- Traffic treated as continuous byte stream
- First N bytes (e.g., 784) extracted per packet/flow
- Input to CNN or Transformer

#### N-gram / Byte Pair Encoding (BPE)
```
Raw: 4504008bd0
Bigrams: 4504, 0400, 008b, 8bd0
```
- Creates "vocabulary" of byte pairs
- Enables language model techniques

### 4.2 Packet-Level Sequences

#### Directional Packet Length Sequences
```
Example: [1420, -68, 1420, -68, -1420, 52, ...]
         ↑ outbound    ↑ inbound  ↑ inbound
```
- Each value = size × direction
- Captures communication patterns
- Works well with 1D-CNN

#### Packet Feature Vectors
Each packet represented as vector:
```
[timestamp, size, direction, protocol, src_port, dst_port, flags, ...]
```

### 4.3 Burst-Level Representations

A **burst** = consecutive packets in same direction

**ET-BERT Approach:**
1. Extract bursts from flows
2. Convert to hex strings
3. Apply bi-gram + BPE tokenization
4. Feed to Transformer

### 4.4 Flow-Level Representations

#### Statistical Flow Features
```
Flow = [duration, total_packets, total_bytes, mean_iat, std_iat, 
        mean_pkt_size, std_pkt_size, protocol_counts, ...]
```
- Fixed-length vector per flow
- Suitable for traditional ML and simple neural networks

#### Hierarchical Flow Representation
```
Flow → Packets → Bytes
            ↓
         Fields (protocol-aware)
```
- Multi-level structure preservation
- Protocol semantic representation (PSR)

### 4.5 Graph-Based Representations

**Flow Graph:**
- Packets as nodes
- Edges represent temporal/structural relationships
- Used with Graph Neural Networks (GNNs)

### 4.6 Sequence Encoding Techniques

| Technique | Description | Use Case |
|-----------|-------------|----------|
| **One-hot encoding** | Categorical to binary vectors | Ports, protocols |
| **Embedding layers** | Learned dense representations | All categorical features |
| **Positional encoding** | Adds position information | Transformers |
| **Time encoding** | Encodes timestamp/interval | Temporal patterns |

### 4.7 Input Pipeline Summary

```
Raw PCAP
    ↓
Packet Extraction (headers + payload)
    ↓
Feature Engineering (packet/flow level)
    ↓
Sequence Construction (fixed or variable length)
    ↓
Tokenization (bytes, n-grams, fields)
    ↓
Embedding (learned or fixed)
    ↓
Neural Network Model
```

### 4.8 Practical Recommendations

1. **For quick prototyping:** Use directional packet length sequences with 1D-CNN
2. **For best accuracy:** Use CNN-LSTM hybrid with attention
3. **For limited labeled data:** Use pre-trained Transformer (ET-BERT) with fine-tuning
4. **For real-time deployment:** Use statistical flow features with Random Forest
5. **For interpretability:** Use attention mechanisms to visualize important packets

---

## Key Research Trends (2024-2026)

1. **Foundation Models for Traffic:** Pre-training on large unlabeled traffic corpora, fine-tuning for specific tasks
2. **Self-Supervised Learning:** Reducing dependency on labeled data through masked modeling
3. **Multi-Modal Fusion:** Combining packet sequences, flow statistics, and protocol information
4. **Open-World Classification:** Detecting unknown device types not in training set
5. **Edge Deployment:** Lightweight models for real-time classification on network devices
6. **LLM-based Identification:** Using instruction-tuned LLMs for semantic device fingerprinting from metadata
</task_result>
```

**Tool: task**

**Input:**
```json
{
  "description": "Research adversarial learning on sequences",
  "prompt": "Research adversarial machine learning approaches for sequential data, specifically for network traffic or time series. Focus on:\n1. How is adversarial training done on sequential data (LSTM/GRU/Transformer inputs)?\n2. What attack methods exist for sequences (perturbation strategies)?\n3. How can adversarial robustness be achieved for traffic classification?\n4. Are there specific challenges with adversarial learning on packet sequences?\n5. Any papers or implementations on adversarial IoT traffic classification?\n\nReturn a comprehensive overview of adversarial learning for sequential traffic data.",
  "subagent_type": "explore"
}
```

**Output:**
```
task_id: ses_36fe7a471ffeN5b1lO1yOePaqj (for resuming to continue this task if needed)

<task_result>
Based on my comprehensive research, here is a detailed overview of adversarial machine learning for sequential traffic data:

---

# Comprehensive Overview: Adversarial Machine Learning for Sequential Traffic Data

## 1. Adversarial Training on Sequential Data (LSTM/GRU/Transformer)

### Core Approaches

**Gradient-Based Perturbations for Sequential Models:**
- **FGSM (Fast Gradient Sign Method)** and **PGD (Projected Gradient Descent)** are adapted for sequential inputs by computing gradients through the unrolled time dimension
- For LSTMs/GRUs, backpropagation through time (BPTT) is used to compute gradients w.r.t. input sequences
- Key challenge: Sequential models process inputs step-by-step, requiring perturbations that propagate effectively through recurrent connections

**Adversarial Training Pipeline for Sequences:**
```
For each training batch:
1. Generate adversarial examples: x_adv = x + ε·sign(∇_x L(θ, x, y))
2. For RNNs: Apply perturbations considering temporal dependencies
3. Mix original and adversarial samples
4. Update model parameters
```

**Transformer-Specific Considerations:**
- **Robustness Tokens (2025):** Novel approach that fine-tunes additional private tokens instead of model parameters for adversarial robustness
- Self-attention mechanisms create unique vulnerabilities as perturbations can affect all positions simultaneously
- Positional encodings must be preserved during adversarial perturbation

**Key Paper - PANTS Framework (2024):**
The PANTS (Practical Adversarial Network Traffic Samples) framework integrates:
- AML methods (PGD, ZOO) with SMT (Satisfiability Modulo Theories) solvers
- Ensures generated adversarial samples are **realizable** and **semantics-preserving**
- 70% more effective at finding adversarial inputs compared to baselines
- Achieves 52.7% improvement in MNC robustness through adversarial training

---

## 2. Attack Methods for Sequences (Perturbation Strategies)

### Time Series-Specific Attacks

**1. TSFool (Multi-Objective Attack):**
- Gray-box attack specifically designed for RNN-based time series classification
- Introduces "Camouflage Coefficient" for imperceptibility
- Multi-objective optimization for perturbation quality
- Outperforms FGSM, JSMA, DeepFool, PGD, C&W, and HopSkipJump on UCR datasets

**2. Smooth Perturbations for Time Series:**
- Addresses visual sensitivity of time series to perturbations
- Applies frequency-domain constraints to maintain natural-looking perturbations
- Reference: Pialla et al. (PAKDD 2022, JDSA 2023)

**3. Temporally Unified Adversarial Perturbations (2026):**
- Maintains temporal consistency inherent in time series data
- Prevents divergence issues in forecasting tasks

**4. Black-Box Attacks for Time Series (AAAI 2023):**
- Query-efficient methods for time series classification
- Coordinate descent and importance sampling for gradient estimation

### Network Traffic-Specific Attacks

**1. TANTRA (Timing-Based Attack):**
- Timing-based adversarial network traffic reshaping
- Uses LSTM to learn time differences between benign packets
- Modifies packet timing without changing content
- 99.99% success rate in NIDS evasion

**2. AdvPad, AdvPay, AdvBurst (from ANT paper):**
- **AdvPad:** Injects UAP into packet content
- **AdvPay:** Injects UAP into dummy packet payload
- **AdvBurst:** Injects crafted dummy packets into selected bursts

**3. Deep PackGen:**
- Deep reinforcement learning framework for adversarial packet generation
- Uses RL agent to craft packets that evade detection

**4. TSAF (Time Series-based Adversarial Attack Framework - 2025):**
- Targets temporal characteristics of network traffic
- Generates adversarial samples using time-series features

### Perturbation Constraints for Traffic Data

| Constraint Type | Description | Example |
|-----------------|-------------|---------|
| **Semantic Preservation** | Perturbation must not change flow meaning | VoIP flow must remain usable |
| **Realizability** | Must correspond to valid packet sequences | TCP handshake order preserved |
| **Threat Model** | Within attacker capabilities | End-host can delay packets |
| **Feature Dependencies** | min_iat ≤ max_iat | Statistical feature consistency |

---

## 3. Achieving Adversarial Robustness for Traffic Classification

### Defense Mechanisms

**1. Adversarial Training Strategies:**

| Method | Description | Trade-off |
|--------|-------------|-----------|
| **Standard AT** | Train with PGD-generated samples | Sacrifices accuracy |
| **PANTS Iterative AT** | Realizable + semantic-preserving samples | Preserves accuracy + improves robustness |
| **DBN-LSTM Adversarial** | Deep Belief Network + LSTM hybrid | DDoS detection specific |

**2. Architecture-Based Defenses:**
- **Attention-CNN-LSTM models:** Improved robustness through attention mechanisms
- **TransGraphNet:** Combines Transformer + GNN for encrypted traffic detection
- **Feature denoising layers:** Remove adversarial perturbations before classification

**3. Data Augmentation-Based Defenses (Lightweight - 2025):**
- Five data augmentation methods for time series
- Ensemble method provides better defense than PGD-based AT
- Only 14.07% computational overhead increase
- Less than 1/3 computational resources of PGD-based AT

**4. Input Structure Considerations (2025):**
- Hardware-aware Neural Architecture Search (HW-NAS)
- Different robustness for different input formats (packet bytes vs. statistical features)
- Flow-level vs. packet-level representations affect robustness

**5. Derived Variables Defense (2023):**
- Use derived features instead of raw traffic features
- Harder to craft adversarial examples against derived representations

### Evaluation Metrics for Robustness

```
Robustness = (Accuracy under attack) / (Clean accuracy)

Key metrics:
- Success rate of evasion attacks
- Perturbation distance (L0, L2, L∞ norms)
- Semantic preservation rate
- Realizability of adversarial samples
```

---

## 4. Challenges with Adversarial Learning on Packet Sequences

### Unique Challenges

**1. Feature Interdependencies:**
- Network traffic features have complex dependencies (e.g., min_iat ≤ max_iat)
- Standard AML methods change features independently, creating unrealizable samples
- Example: AML might set min_iat > max_iat, which is impossible

**2. Non-Differentiable Components:**
- Feature engineering modules are often non-invertible
- Cannot directly apply gradient-based methods end-to-end
- Requires SMT solvers or alternative optimization approaches

**3. Semantic Preservation:**
- Unlike images, perturbing traffic can change its meaning/usability
- VoIP flows delayed too much become unusable
- TCP connection semantics must be maintained

**4. Discrete vs. Continuous:**
- Packet sizes, port numbers, flags are discrete
- Timing has continuous properties but practical constraints
- Hybrid optimization required

**5. Real-World Constraints:**
```
End-host attacker: Can delay packets, modify sizes, inject dummy packets
In-network attacker: Can delay, drop, inject (cannot modify encrypted content)
```

**6. Sequence-Level Effects:**
- Perturbations at position t affect hidden states at all subsequent positions
- Attention mechanisms amplify perturbation effects globally
- Long-range dependencies in traffic patterns

**7. Dataset Distribution Shift:**
- Training data may not represent attack scenarios
- Distribution drift in real deployments
- GNN-based NIDS often evaluated on single datasets (overestimating robustness)

### From the NIDS Survey (2024):

| Challenge | Description | Impact |
|-----------|-------------|--------|
| **Structured Data Constraints** | Features have interdependencies | Standard attacks generate invalid samples |
| **Lack of Defense Strategies** | Limited research on NIDS-specific defenses | Vulnerable systems in production |
| **Lack of Real-World Relevance** | Synthetic perturbations in evaluations | Overestimated robustness |
| **Placement of Adversarial Examples** | Location on data manifold affects detectability | Some attacks more detectable than others |

---

## 5. Papers and Implementations on Adversarial IoT Traffic Classification

### Key Papers

**1. IoT Botnet Detection:**

| Paper | Venue | Key Contribution |
|-------|-------|------------------|
| "Robust IoT Botnet Detection Framework Resilient to Gradient Based Adversarial Attacks" | SN Computer Science 2024 | Gradient-based attack resilience |
| "Comprehensive Botnet Detection by Mitigating Adversarial Attacks" | Information Fusion 2024 | Conformal prediction + perturbation distance analysis |
| "Analyzing Adversarial Attacks against Deep Learning for Intrusion Detection in IoT Networks" | IEEE 2020 | First systematic analysis of IoT-specific adversarial attacks |
| "BoT-EnsIDS" | Alexandria Engineering Journal 2025 | Bio-inspired ensemble + hybrid DL for botnet detection |
| "Enhanced Hybrid Deep Learning Approach for Botnet Attacks Detection in IoT Environment" | arXiv 2025 | Hybrid approach for evolving botnet threats |

**2. GNN-Based IoT NIDS:**

| Paper | Contribution |
|-------|--------------|
| "REAL-IoT: Characterizing GNN Intrusion Detection Robustness" (2025) | Unified dataset framework, practical adversarial evaluation |
| "Hierarchical Adversarial Attacks Against GNN-based IoT NIDS" (IEEE 2022) | Multi-level attack strategies |
| "Adversarial Learning Attacks on Graph-based IoT Malware Detection" (IEEE 2019) | Graph structure exploitation |

### Available Implementations

**1. Time Series Adversarial Attacks (GitHub):**
- Repository: `GoktugOcal/time-series-adversarial-attacks`
- Focus: LSTM/CNN variants with FGSM, PGD attacks
- Link: https://github.com/GoktugOcal/time-series-adversarial-attacks

**2. TSFool:**
- Repository: `FlaAI/TSFool`
- Focus: Multi-objective attack for RNN-based TSC
- Features: FGSM, JSMA, DeepFool, PGD, C&W, HopSkipJump benchmarks
- Link: https://github.com/FlaAI/TSFool

**3. Adversarial Attacks on Time Series (LSTM):**
- Repository: `afraarslan/adversarial-attacks`
- Focus: LSTM regression and classification with FGSM, BIM
- Datasets: Energy consumption, EEG, MNIST
- Link: https://github.com/afraarslan/adversarial-attacks

**4. Time Series Robustness:**
- Repository: `ocenandor/ts_robustness`
- Focus: CNN, LSTM, Transformer comparison with DeepFool, SimBA, BIM
- Link: https://github.com/ocenandor/ts_robustness

**5. Adversarial AE-LSTM:**
- Repository: `c-quilo/adversarial-AE-LSTM`
- Focus: Adversarial training for surrogate model forecasts
- Link: https://github.com/c-quilo/adversarial-AE-LSTM

**6. Adversarial Network Traffic (ANT):**
- Repository: `amsadeghzadeh/AdversarialNetworkTraffic`
- Paper: IEEE TNSM 2020
- Features: AdvPad, AdvPay, AdvBurst attacks
- Link: https://github.com/amsadeghzadeh/AdversarialNetworkTraffic

### Benchmarks and Datasets

| Dataset | Type | Use Case |
|---------|------|----------|
| UCR Archive | Time Series Classification | Standard benchmark for adversarial attacks |
| FordA | Univariate, Binary | Industrial sensor data |
| Bot-IoT | Network Traffic | IoT botnet detection |
| NSL-KDD | Network Traffic | Traditional NIDS benchmark |
| CIC-IDS2017/2018 | Network Traffic | Modern attack scenarios |
| IoT-23 | IoT Malware | IoT-specific malware traffic |
| UNSW-NB15 | Network Traffic | Multi-category attacks |

---

## Summary: Best Practices for Adversarial Learning on Sequential Traffic

1. **Use realizable adversarial samples** - Ensure generated samples correspond to valid packet sequences
2. **Preserve semantics** - Perturbations should not change traffic meaning or usability
3. **Consider feature dependencies** - Encode constraints when generating adversarial examples
4. **Combine AML with SMT solvers** - For guaranteed realizability
5. **Iterative adversarial training** - More effective than one-time augmentation
6. **Evaluate on multiple datasets** - Avoid overfitting to specific traffic patterns
7. **Test against diverse attack types** - White-box, black-box, and gray-box scenarios
8. **Consider computational constraints** - Lightweight defenses preferred for IoT/edge deployment
</task_result>
```

---

## Assistant (Plan · z-ai/glm-5:free · 654.3s)

_Thinking:_

Excellent, I have comprehensive research from all three agents. Let me summarize the findings for the user:

**What exists in the codebase:**
- Project is PFE (IoT Device Identification for SDN-NAC)
- Dataset: IPFIX ML Instances (32M+ rows, 12 homes, 18 device classes, 44 features)
- Models: LSTM and Transformer classifiers in PyTorch
- Preprocessing: Creates sequences of length 10 with stride 5, 37 SDN-compatible features
- Adversarial: IoT-SDN specific attack with L0/L2 minimization, adversarial training with 80/20 mix
- Frameworks: PyTorch (main), TensorFlow/Keras (Colab)

**Research findings on IoT device identification:**
- Directional Packet Length Sequences (DPLS) is highly effective
- CNN-LSTM hybrids are state-of-the-art
- Transformer models (ET-BERT) with self-supervised pre-training emerging
- Key features: packet sizes, timing, protocols, ports

**Research findings on adversarial learning for sequences:**
- PANTS framework for realizable, semantic-preserving adversarial samples
- TSFool for time-series specific attacks
- TANTRA for timing-based attacks
- Key challenges: feature interdependencies, semantic preservation, sequence-level effects
- The user's existing implementation seems aligned with some best practices

The user specifically noted:
1. Traffic is a sequence - the research shows this is well-handled by LSTM/Transformer
2. Adversarial learning on sequences is not clear - the research shows several approaches and challenges

Let me present a comprehensive plan to the user.

## Research Summary

### What You Already Have

| Component | Status |
|-----------|--------|
| **Dataset** | IPFIX ML Instances: 32M+ rows, 12 homes, 18 device classes, 44 features |
| **Models** | LSTM (BiLSTM, 128 hidden) and Transformer (d=64, 4 heads) in PyTorch |
| **Preprocessing** | Sequence length=10, stride=5, 37 SDN-compatible features (no IP/MAC/ports) |
| **Adversarial** | Custom IoT-SDN attack with L0/L2 minimization, feature constraints, 80/20 adversarial training |

### Key Research Findings

**For Sequential Traffic Classification:**
- CNN-LSTM hybrids achieve 98-99.9% accuracy (better than LSTM alone)
- Directional Packet Length Sequences (DPLS) is highly effective (>99% accuracy)
- Transformer pre-training (ET-BERT style) reduces label dependency
- Your current sequence approach (length=10) is reasonable, but may be short for capturing full flow patterns

**For Adversarial Robustness on Sequences (Your Concern):**
- **Challenge 1:** Feature interdependencies (e.g., `min_iat ≤ max_iat`) - you already handle this
- **Challenge 2:** Sequence-level effects - perturbations propagate through time steps
- **Challenge 3:** Semantic preservation - traffic must remain valid/usable
- **Best practice:** PANTS framework (2024) combines AML + SMT solvers for *realizable* adversarial samples
- **Alternative:** Data augmentation-based defense (2025) is 3x faster than PGD-AT with comparable robustness

### Gaps in Current Implementation

1. **No CNN-LSTM hybrid** - Research shows this outperforms standalone LSTM/Transformer
2. **Sequence length=10 may be short** - Traffic flows are longer; consider 50-200 packets
3. **No attention visualization** - Attention helps interpretability and robustness
4. **Single attack type** - Research suggests testing against multiple attack strategies (TSFool, TANTRA)
5. **No robustness evaluation on multiple attack types** - Need comprehensive evaluation

---

Before I create a detailed implementation plan, I have questions:

**Tool: question**

**Input:**
```json
{
  "questions": [
    {
      "question": "What is your primary goal for this work?",
      "header": "Goal",
      "options": [
        {
          "label": "Research paper",
          "description": "Produce novel contribution for publication"
        },
        {
          "label": "Production system",
          "description": "Deploy in real SDN/NAC environment"
        },
        {
          "label": "Thesis project",
          "description": "Academic project with comprehensive evaluation"
        }
      ]
    },
    {
      "question": "Which model architectures do you want to implement?",
      "header": "Architecture",
      "options": [
        {
          "label": "CNN-LSTM Hybrid (Recommended)",
          "description": "State-of-the-art accuracy 98-99.9%"
        },
        {
          "label": "Keep LSTM + Transformer",
          "description": "Enhance existing models"
        },
        {
          "label": "All three + compare",
          "description": "Comprehensive comparison study"
        }
      ]
    },
    {
      "question": "How do you want to handle adversarial learning?",
      "header": "Adversarial",
      "options": [
        {
          "label": "Improve current approach",
          "description": "Fix sequence-level issues, add more attack types"
        },
        {
          "label": "PANTS-style framework",
          "description": "SMT solver + AML for realizable samples"
        },
        {
          "label": "Data augmentation defense",
          "description": "Lightweight alternative to adversarial training"
        }
      ]
    },
    {
      "question": "Should we increase sequence length from 10?",
      "header": "Seq Length",
      "options": [
        {
          "label": "Yes, 50-100 packets",
          "description": "Better pattern capture, more compute"
        },
        {
          "label": "Keep at 10",
          "description": "Faster, sufficient for short flows"
        },
        {
          "label": "Test multiple lengths",
          "description": "Experiments to find optimal length"
        }
      ]
    }
  ]
}
```

**Output:**
```
User has answered your questions: "What is your primary goal for this work?"="Thesis project", "Which model architectures do you want to implement?"="All three + compare", "How do you want to handle adversarial learning?"="lets discuss this more as this is the biggest knowledge gap i have, propose to me each approach in pristine yet concise detail, with the tradeoffs", "Should we increase sequence length from 10?"="Test multiple lengths". You can now continue with the user's answers in mind.
```

---

## Assistant (Plan · z-ai/glm-5:free · 63.4s)

_Thinking:_

The user wants:
1. Thesis project - comprehensive evaluation
2. All three architectures (CNN-LSTM, LSTM, Transformer) with comparison
3. More discussion on adversarial learning approaches - they want me to propose each approach with tradeoffs
4. Test multiple sequence lengths

Let me provide a detailed comparison of adversarial learning approaches for their thesis.

## Adversarial Learning Approaches for Sequential Traffic

### Your Current Approach (IoT-SDN Attack)

**How it works:**
```
x_adv = Projection[x₀ + c·t·mask·sign(μ_target - x₀)·|Δ|]
```
- Selects target class (nearest in feature space via L2 minimization)
- Uses mask for L0 constraint (limits number of modified features)
- Projects to valid range, respects feature dependencies

**Strengths:**
- Domain-specific (IoT traffic constraints)
- Respects feature interdependencies (inPacketCount ↔ outPacketCount)
- Non-modifiable features protected (protocols, flags)

**Weaknesses:**
- **Operates on aggregated features, not raw sequences** - perturbation doesn't consider temporal order
- No gradient-based optimization - may miss optimal perturbations
- Single attack strategy - model may overfit to this specific attack

---

### Approach 1: Gradient-Based Adversarial Training (PGD/FGSM for Sequences)

**How it works:**
```
For each sequence x = [x₁, x₂, ..., x_T]:
    x_adv = Project(x + ε·sign(∇_x L(θ, x, y)))
    
Where gradient is computed via Backpropagation Through Time (BPTT)
```

**For LSTM:**
- Gradient flows through recurrent connections
- Perturbation at step t affects all future hidden states
- Captures temporal sensitivity

**For Transformer:**
- Self-attention creates global dependencies
- Perturbation at any position affects all positions
- Must preserve positional encodings

**Trade-offs:**
| Pro | Con |
|-----|-----|
| Optimal perturbations (gradient-guided) | May violate semantic constraints |
| Model-agnostic (works with any DL model) | Requires differentiable pipeline |
| Strong theoretical foundation | Computationally expensive (BPTT for each attack) |
| | Generated samples may be unrealizable |

**Thesis value:** High - standard benchmark for adversarial robustness

---

### Approach 2: PANTS Framework (AML + SMT Solvers)

**How it works:**
```
1. Generate initial adversarial perturbation (PGD/ZOO)
2. Encode traffic constraints as SMT formulas:
   - Feature dependencies: min_iat ≤ max_iat
   - Protocol semantics: TCP flags valid combinations
   - Realizability: valid packet sequences
3. Use SMT solver (Z3) to find valid assignment
4. If unsat, relax constraints and retry
```

**Key Innovation:** Guarantees generated adversarial samples are **realizable** (can actually occur in real traffic)

**Trade-offs:**
| Pro | Con |
|-----|-----|
| Guaranteed realizability | Complex implementation |
| Semantic preservation | Slower than gradient methods (~10x) |
| 70% more effective attack discovery | Requires domain expertise for constraints |
| 52.7% robustness improvement | SMT solver learning curve |

**Thesis value:** Very High - novel contribution with practical relevance

---

### Approach 3: Data Augmentation-Based Defense

**How it works:**
```
Augmentation methods for time series:
1. Jittering: Add Gaussian noise to values
2. Scaling: Random scaling of entire sequence
3. Permutation: Shuffle segments within sequence
4. Magnitude Warping: Smooth scaling curve
5. Time Warping: Smooth temporal distortion
```

Train model on augmented data instead of adversarial examples.

**Trade-offs:**
| Pro | Con |
|-----|-----|
| Only 14% computational overhead | No guarantee against specific attacks |
| No attack model knowledge needed | May not defend against targeted attacks |
| 3x faster than PGD-AT | Less theoretical robustness guarantee |
| Works well for time series | Less effective for feature-level attacks |

**Thesis value:** Medium-High - good as baseline comparison, practical for deployment

---

### Approach 4: Sequence-Aware Adversarial Attack (TSFool/TANTRA Style)

**How it works:**

**TSFool (for classification):**
```
1. Learn temporal smoothness constraint from data
2. Multi-objective optimization:
   - Maximize classification error
   - Minimize perturbation magnitude (L2)
   - Minimize temporal distortion (Camouflage Coefficient)
3. Gray-box: knows model architecture, not weights
```

**TANTRA (timing-based):**
```
1. Learn distribution of inter-arrival times with LSTM
2. Craft timing perturbations that:
   - Stay within natural timing distribution
   - Cause misclassification
   - Are indistinguishable from normal traffic
```

**Trade-offs:**
| Pro | Con |
|-----|-----|
| Designed specifically for time series | More complex than PGD |
| Preserves temporal patterns | Requires training separate LSTM for timing model |
| High success rate (99.99% in NIDS evasion) | Gray-box assumption may not hold |
| Realistic perturbations | Limited to timing features (TANTRA) |

**Thesis value:** High - addresses your exact concern about sequences

---

### Approach 5: Hybrid Approach (Recommended for Thesis)

**Combines your current method with gradient-based sequence attacks:**

```
Phase 1: Feature-level attack (your current IoT-SDN attack)
         - Targets statistical features
         - Respects feature dependencies
         
Phase 2: Sequence-level attack (gradient-based)
         - Targets temporal ordering
         - Uses BPTT for gradients
         - Constrained to preserve semantics
         
Phase 3: Adversarial Training
         - Mix both attack types
         - 60% clean + 20% feature-level + 20% sequence-level
```

**Trade-offs:**
| Pro | Con |
|-----|-----|
| Comprehensive coverage | More implementation work |
| Addresses both feature and sequence aspects | More hyperparameters to tune |
| Novel contribution (not in literature) | Evaluation complexity |

**Thesis value:** Highest - addresses your gap + novel contribution

---

### Recommendation for Your Thesis

| Priority | Approach | Rationale |
|----------|----------|-----------|
| **1** | Hybrid (Approach 5) | Directly addresses your knowledge gap, novel contribution |
| **2** | Gradient-based (Approach 1) | Standard benchmark, good for comparison |
| **3** | Data augmentation (Approach 3) | Lightweight baseline |
| **4** | PANTS (Approach 2) | If time permits, adds realizability guarantee |

Would you like me to proceed with this plan, or do you want to adjust the adversarial learning strategy?

---

