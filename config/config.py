"""
Configuration for IoT Device Identification Project
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RAW_DATA_DIR = DATA_DIR / "pcap" / "IPFIX ML Instances"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

FEATURES_TO_KEEP = [
    # --- SDN Native Features (FlowStats) ---
    "duration",
    "ipProto",
    # --- Trafic Sortant (12) ---
    "outPacketCount",
    "outByteCount",
    "outSmallPktCount",
    "outLargePktCount",
    "outNonEmptyPktCount",
    "outDataByteCount",
    "outAvgIAT",
    "outFirstNonEmptyPktSize",
    "outMaxPktSize",
    "outStdevPayloadSize",
    "outStdevIAT",
    "outAvgPacketSize",
    # --- Trafic Entrant (12) ---
    "inPacketCount",
    "inByteCount",
    "inSmallPktCount",
    "inLargePktCount",
    "inNonEmptyPktCount",
    "inDataByteCount",
    "inAvgIAT",
    "inFirstNonEmptyPktSize",
    "inMaxPktSize",
    "inStdevPayloadSize",
    "inStdevIAT",
    "inAvgPacketSize",
]

FEATURES_TO_DROP = [
    "start",
    "srcMac",
    "destMac",
    "srcIP",
    "destIP",
    "srcPort",
    "destPort",
    "device",
]

LABEL_COLUMN = "name"

SEQUENCE_LENGTH = 10
STRIDE = 5

TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42

BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# docs/architectures §1: 64 hidden units per layer, 2 layers, BiLSTM → 128-dim embedding
LSTM_CONFIG = {
    "hidden_size": 64,  # 64 per layer → 128-dim embedding (BiLSTM)
    "num_layers": 2,
    "bidirectional": True,
    "dropout": 0.3,
}

# docs/architectures §4 (MIND-IoT): 6 encoder layers, GELU, d_model=128 (scaled from 768)
TRANSFORMER_CONFIG = {
    "d_model": 128,
    "nhead": 4,
    "num_encoder_layers": 6,  # 6 per doc (MIND-IoT uses 6 encoder layers)
    "dim_feedforward": 512,
    "dropout": 0.2,
}

MIN_SAMPLES_PER_CLASS = 500

# ─── JSON Pipeline Configuration (IPFIX Records dataset) ────────────────────
# Issue 1: CSV and JSON are completely different datasets — use independently

JSON_DATA_DIR = DATA_DIR / "pcap" / "IPFIX Records (UNSW IoT Analytics)"
JSON_PROCESSED_DATA_DIR = DATA_DIR / "processed" / "json_native"

# JSON-specific behavioral features (Issue 4: identifying columns excluded)
JSON_FEATURES_TO_KEEP = [
    # --- Temps et Protocoles ---
    "flowDurationMilliseconds",
    "protocolIdentifier",
    # --- Métriques Globales ---
    "packetTotalCount",
    "octetTotalCount",
    "reversePacketTotalCount",
    "reverseOctetTotalCount",
    # --- Drapeaux TCP (visibles par OpenFlow) ---
    "initialTCPFlags",
    "unionTCPFlags",
    "reverseInitialTCPFlags",
    "reverseUnionTCPFlags",
    # --- Métriques Détaillées Forward ---
    "tcpUrgTotalCount",
    "smallPacketCount",
    "nonEmptyPacketCount",
    "dataByteCount",
    "averageInterarrivalTime",
    "firstNonEmptyPacketSize",
    "largePacketCount",
    "maxPacketSize",
    "standardDeviationPayloadLength",
    "standardDeviationInterarrivalTime",
    "bytesPerPacket",
    # --- Métriques Détaillées Reverse ---
    "reverseTcpUrgTotalCount",
    "reverseSmallPacketCount",
    "reverseNonEmptyPacketCount",
    "reverseDataByteCount",
    "reverseAverageInterarrivalTime",
    "reverseFirstNonEmptyPacketSize",
    "reverseLargePacketCount",
    "reverseMaxPacketSize",
    "reverseStandardDeviationPayloadLength",
    "reverseStandardDeviationInterarrivalTime",
]

# Optional binary packet direction features (firstEightNonEmptyPacketDirections)
JSON_PKT_DIR_FEATURES = [f"pkt_dir_{i}" for i in range(8)]

# Update dimensions based on filtered features
JSON_N_CONTINUOUS = len(JSON_FEATURES_TO_KEEP)
JSON_N_BINARY = len(JSON_PKT_DIR_FEATURES)
JSON_INPUT_SIZE = JSON_N_CONTINUOUS + JSON_N_BINARY

# 26 device classes from IPFIX Records testbed (Issue 2: MAC-mapped)
JSON_DEVICE_CLASSES = [
    "Nokia body",
    "Panasonic doorphone",
    "Qrio hub",
    "Philips Hue lightbulb",
    "Xiaomi LED",
    "Planex UCA01A camera",
    "Planex pantilt camera",
    "JVC Kenwood camera",
    "Nature remote control",
    "Bitfinder aware sensor",
    "Google Home",
    "Apple Homepod",
    "Sony Bravia TV",
    "iRobot roomba",
    "Sesame access point",
    "JVC Kenwood hub",
    "Wansview camera",
    "Qwatch camera",
    "Fredi camera",
    "Planex outdoor camera",
    "PowerElec WIFI plug",
    "Line Clova speaker",
    "Sony smart speaker",
    "Amazon Echo",
    "Amazon Echo Show",
    "MCJ room hub",
]

# Pipeline selection flag: "csv" or "json"
PIPELINE_MODE = "csv"

# Hybrid Adversarial Training Configuration
# Default: 60% clean + 20% feature-level + 20% sequence-level
HYBRID_SPLIT_CONFIG = {
    "clean": 0.6,
    "feature": 0.2,
    "sequence": 0.2,
}

# 4.4 Classes de Dispositifs IoT - IoT IPFIX Home Dataset (18 classes)
# Ces classes correspondent aux appareils specifiques demandes
IOT_IPFIX_HOME_CLASSES = [
    "Eclear",
    "Sleep",
    "Esensor",
    "Hub Plus",
    "Humidifier",
    "Home Unit",
    "Ink Jet Printer",
    "Smart Wi-Fi Plug Mini",
    "Smart Power Strip",
    "Echo Dot",
    "Fire 7 Tablet",
    "Google Nest Mini",
    "Google Chromecast",
    "Atom Cam",
    "Kasa Camera Pro",
    "Kasa Smart LED Lamp",
    "Fire TV Stick 4K",
    "Qrio Hub",
]

# Alias pour compatibilite (kebab-case pour CSV)
IOT_DEVICE_CLASSES = [
    "eclear",
    "sleep",
    "esensor",
    "hub-plus",
    "humidifier",
    "home-unit",
    "inkjet-printer",
    "smart-wifi-plug-mini",
    "smart-power-strip",
    "echo-dot",
    "fire7-tablet",
    "google-nest-mini",
    "google-chromecast",
    "atom-cam",
    "kasa-camera-pro",
    "kasa-smart-led-lamp",
    "fire-tv-stick-4k",
    "qrio-hub",
]

# IPFIX Records Dataset - 17 classes specifiques demandees
IPFIX_RECORDS_CLASSES = [
    "Qrio Hub",
    "Philips Hue Light Bulb",
    "Planex Pan–Tilt Camera 1",
    "JVC Kenwood Camera",
    "Planex Pan–Tilt Camera 2",
    "Google Home",
    "Apple HomePod",
    "Sony Bravia TV",
    "Wansview Camera",
    "Qwatch Camera",
    "Fredi Camera",
    "Planex Outdoor Camera",
    "Powerlec Wi-Fi Plug",
    "LINE Clova Speaker",
    "Sony Smart Speaker",
    "Amazon Echo",
    "Amazon Echo Show",
]

# ─── CNN-BiLSTM-Transformer Configuration ─────────────────────────────────────
# Architecture hybride (docs/architectures §5)
CNN_BILSTM_TRANSFORMER_CONFIG = {
    # CNN branches (deux branches parallèles, noyaux 3 et 5)
    "cnn_channels": 64,
    "cnn_kernel_small": 3,
    "cnn_kernel_large": 5,
    "cnn_pool_size": 2,
    # BiLSTM
    "bilstm_hidden": 128,
    "bilstm_layers": 2,
    "bilstm_dropout": 0.3,
    # Transformer encoder
    "transformer_d_model": 256,
    "transformer_nhead": 4,
    "transformer_layers": 2,
    "transformer_ff_dim": 512,
    "transformer_dropout": 0.2,
    # FC head
    "fc_dropout": 0.4,
}
