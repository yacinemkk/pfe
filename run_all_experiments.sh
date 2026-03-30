#!/bin/bash
# run_all_experiments.sh
# 
# Master script to evaluate all 7 models systematically on both dual datasets:
# - IPFIX ML Instances (CSV)
# - IPFIX Records (JSON)
#
# Generates complete adversarial and standard evaluation reports.

# Exit on any failure
set -e

# Sequence length optimized for general models
SEQ_LENGTH=10
# Epochs per phase (total 20-30 max)
EPOCHS=30

MODELS=(
    "lstm"
    "bilstm"
    "cnn_lstm"
    "xgboost_lstm"
    "transformer"
    "cnn_bilstm"
    "cnn_bilstm_transformer"
)

echo "=========================================================="
echo " Starting Full Dual-Dataset Adversarial Protocol "
echo "=========================================================="

for model in "${MODELS[@]}"; do
    echo "----------------------------------------------------------"
    echo " Evaluating Model: $model"
    echo "----------------------------------------------------------"
    
    # Run train_adversarial.py with --dual_dataset which processes both CSV and JSON automatically
    python3 train_adversarial.py \
        --model "$model" \
        --seq_length "$SEQ_LENGTH" \
        --adv_method "hybrid" \
        --adv_ratio 0.2 \
        --epochs "$EPOCHS" \
        --batch_size 128 \
        --dual_dataset \
        --phase_checkpoints
        
    echo "[OK] Model $model completed."
done

echo "=========================================================="
echo " All experiments completed! Results saved in results_dual/"
echo "=========================================================="
