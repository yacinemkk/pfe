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
    "duration",
    "ipProto",
    "outPacketCount",
    "outByteCount",
    "inPacketCount",
    "inByteCount",
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
    "http",
    "https",
    "smb",
    "dns",
    "ntp",
    "tcp",
    "udp",
    "ssdp",
    "lan",
    "wan",
    "deviceInitiated",
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
    "name",
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

LSTM_CONFIG = {
    "hidden_size": 128,
    "num_layers": 2,
    "bidirectional": True,
    "dropout": 0.3,
}

TRANSFORMER_CONFIG = {
    "d_model": 64,
    "nhead": 4,
    "num_encoder_layers": 3,
    "dim_feedforward": 256,
    "dropout": 0.2,
}

MIN_SAMPLES_PER_CLASS = 500

# ─── JSON Pipeline Configuration (IPFIX Records dataset) ────────────────────
# Issue 1: CSV and JSON are completely different datasets — use independently

JSON_DATA_DIR = DATA_DIR / "pcap" / "IPFIX Records (UNSW IoT Analytics)"
JSON_PROCESSED_DATA_DIR = DATA_DIR / "processed" / "json_native"

# JSON-specific behavioral features (Issue 4: identifying columns excluded)
JSON_FEATURES_TO_KEEP = [
    "flowDurationMilliseconds",
    "reverseFlowDeltaMilliseconds",
    "protocolIdentifier",
    "packetTotalCount",
    "octetTotalCount",
    "reversePacketTotalCount",
    "reverseOctetTotalCount",
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

# 8 binary packet direction features (Issue 5: decoded from hex)
JSON_PKT_DIR_FEATURES = [f"pkt_dir_{i}" for i in range(8)]

# Total features: 28 continuous + 8 binary = 36
JSON_N_CONTINUOUS = len(JSON_FEATURES_TO_KEEP)
JSON_N_BINARY = 8
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
PIPELINE_MODE = "json"

# Hybrid Adversarial Training Configuration
# Default: 60% clean + 20% feature-level + 20% sequence-level
HYBRID_SPLIT_CONFIG = {
    "clean": 0.6,
    "feature": 0.2,
    "sequence": 0.2,
}

# 4.4 Classes de Dispositifs IoT (18 classes)
# NOTE: Utiliser kebab-case pour correspondre aux noms dans les CSV
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
