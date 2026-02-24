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
]

LABEL_COLUMN = "device"

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
