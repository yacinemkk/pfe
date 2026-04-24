# ─── Cell: Setup ────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import os
if os.path.exists('/content/pfe'):
    !cd /content/pfe && git pull
else:
    !git clone https://github.com/yacinemkk/pfe.git /content/pfe

%cd /content/pfe

!pip install -q torch torchvision tqdm numpy pandas scikit-learn matplotlib xgboost psutil
# ─── Cell: Configuration ─────────────────────────────────────────────────
import os

JSON_DATA_DIR = '/content/drive/MyDrive/PFE/IPFIX_Records'
CSV_DATA_DIR = '/content/drive/MyDrive/PFE/IPFIX_ML_Instances'
DRIVE_RESULTS_DIR = '/content/drive/MyDrive/PFE/results'
DATASETS = 'both'

SEQ_LENGTH = 10
STRIDE = 10
BATCH_SIZE = 32          # reduced from 64 to avoid OOM on L4 GPU
LEARNING_RATE = 5e-4
USE_AMP = True           # mixed-precision (fp16) — cuts VRAM ~50%

# Lighter CNN-BiLSTM-Transformer to fit within 22 GB VRAM
CNN_BILSTM_TRANSFORMER_OVERRIDE = {
    'cnn_channels': 32,          # was 64
    'bilstm_hidden': 64,         # was 128  → bilstm output = 128
    'bilstm_layers': 2,
    'bilstm_dropout': 0.3,
    'transformer_d_model': 128,  # was 256
    'transformer_nhead': 4,
    'transformer_layers': 2,
    'transformer_ff_dim': 512,   # reverted to 512
    'transformer_dropout': 0.2,
    'fc_dropout': 0.4,
}

# Greedy adversarial training phases
PHASE_A_EPOCHS = 15
PHASE_B_EPOCHS = 30
PHASE_C_EPOCHS = 50
PHASE_A_MIX_RATIO = 0.0
PHASE_B_MIX_RATIO = 0.30
PHASE_C_MIX_RATIO = 0.70
PHASE_B_K_MAX = 2
PHASE_C_K_MAX = 4
PHASE_D_EPOCHS = 80           # +15 epochs vs avant (65→80)
PHASE_D_MIX_RATIO = 0.85     # 85% adv (au lieu de 100%) — évite l'effondrement clean
PHASE_D_K_MAX = 4             # k=4 comme Phase C — k=5 était contre-productif

GREEDY_STRATEGIES = ['Zero', 'Mimic_Mean', 'Mimic_95th', 'Padding_x10']

MAX_FILES = None
MAX_RECORDS = None
EVAL_SUBSAMPLE = 1000
EVAL_BATCH_SIZE = 32

os.makedirs(DRIVE_RESULTS_DIR, exist_ok=True)

print(f'CSV data:     {CSV_DATA_DIR}')
print(f'JSON data:    {JSON_DATA_DIR}')
print(f'Results dir:  {DRIVE_RESULTS_DIR}')
print(f'Datasets:     {DATASETS}')
print(f'Seq length:   {SEQ_LENGTH}')
print(f'Phase A: epochs 1-15   | mix=0%   | k_max=0 (clean only)')
print(f'Phase B: epochs 16-30  | mix=30%  | k_max=2')
print(f'Phase C: epochs 31-50 | mix=70%  | k_max=4')
print(f'Batch size:   {BATCH_SIZE}')
print(f'LR:           {LEARNING_RATE}')

import glob
csv_files = glob.glob(f'{CSV_DATA_DIR}/home*_labeled.csv')
print(f'\nFound {len(csv_files)} CSV file(s)')
for f in sorted(csv_files)[:5]:
    size_mb = os.path.getsize(f) / (1024**2)
    print(f'  {os.path.basename(f)} ({size_mb:.1f} MB)')
if len(csv_files) > 5:
    print(f'  ... and {len(csv_files) - 5} more')

json_files = glob.glob(f'{JSON_DATA_DIR}/**/*.json', recursive=True)
print(f'\nFound {len(json_files)} JSON file(s)')
for f in json_files:
    size_gb = os.path.getsize(f) / (1024**3)
    print(f'  {os.path.basename(f)} ({size_gb:.1f} GB)')
# ─── Cell: RAM Monitoring & Data Loading ─────────────────────────────────
import gc
import psutil
import torch
import os
import numpy as np
import pickle
import glob

def get_memory_usage():
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / (1024**3)
    gpu_gb = 0
    if torch.cuda.is_available():
        gpu_gb = torch.cuda.memory_allocated() / (1024**3)
    return ram_gb, gpu_gb

def log_memory(label=''):
    ram_gb, gpu_gb = get_memory_usage()
    print(f'  [RAM {label}] {ram_gb:.2f} GB | [GPU {label}] {gpu_gb:.2f} GB')

def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    ram_gb, gpu_gb = get_memory_usage()
    print(f'  [Cleanup] RAM: {ram_gb:.2f} GB | GPU: {gpu_gb:.2f} GB')

print('RAM monitoring utilities loaded.')
log_memory('startup')


def load_and_display_csv_dataset(csv_data_dir, seq_length=10, stride=10, save_dir=None):
    import sys
    sys.path.insert(0, '/content/pfe')
    from src.data.preprocessor import IoTDataProcessor

    print('\n' + '=' * 70)
    print('  LOADING CSV DATASET')
    print('=' * 70)

    processor = IoTDataProcessor()
    result = processor.process_all(
        max_files=None,
        data_dir=csv_data_dir,
        seq_length=seq_length,
        stride=stride,
        apply_balancing=False,
    )

    X_train, X_val, X_test, y_train, y_val, y_test, features, scaler, label_encoder = result
    n_continuous = len(features)

    print(f'  Features ({n_continuous}): {features[:5]}...')
    print(f'  Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}')
    print(f'  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f'  Saving preprocessed CSV to Drive...')
        np.save(f'{save_dir}/X_train.npy', X_train)
        np.save(f'{save_dir}/X_val.npy', X_val)
        np.save(f'{save_dir}/X_test.npy', X_test)
        np.save(f'{save_dir}/y_train.npy', y_train)
        np.save(f'{save_dir}/y_val.npy', y_val)
        np.save(f'{save_dir}/y_test.npy', y_test)
        with open(f'{save_dir}/csv_metadata.pkl', 'wb') as f:
            pickle.dump({
                'features': features, 'scaler': scaler,
                'label_encoder': label_encoder,
                'n_continuous': n_continuous,
                'seq_length': seq_length, 'stride': stride,
            }, f)
        with open(f'{save_dir}/csv_ready', 'w') as f:
            f.write('ready')
        print(f'  CSV dataset saved to Drive.')

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'features': features, 'scaler': scaler,
        'label_encoder': label_encoder, 'n_continuous': n_continuous
    }


def load_and_display_json_dataset(json_data_dir, seq_length=10, stride=10, max_records=None, save_dir=None):
    import sys
    sys.path.insert(0, '/content/pfe')
    from src.data.json_preprocessor import JsonIoTDataProcessor

    print('\n' + '=' * 70)
    print('  LOADING JSON DATASET')
    print('=' * 70)

    processor = JsonIoTDataProcessor()
    result = processor.process_all(
        data_dir=json_data_dir,
        seq_length=seq_length,
        stride=stride,
        max_records=max_records,
        apply_balancing=False,
    )

    X_train, X_val, X_test, y_train, y_val, y_test, features, scaler, label_encoder = result
    n_continuous = 36

    print(f'  Features ({len(features)}): {features[:5]}...')
    print(f'  Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}')
    print(f'  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f'  Saving preprocessed JSON to Drive...')
        np.save(f'{save_dir}/X_train.npy', X_train)
        np.save(f'{save_dir}/X_val.npy', X_val)
        np.save(f'{save_dir}/X_test.npy', X_test)
        np.save(f'{save_dir}/y_train.npy', y_train)
        np.save(f'{save_dir}/y_val.npy', y_val)
        np.save(f'{save_dir}/y_test.npy', y_test)
        with open(f'{save_dir}/json_metadata.pkl', 'wb') as f:
            pickle.dump({
                'features': features, 'scaler': scaler,
                'label_encoder': label_encoder,
                'n_continuous': n_continuous,
                'seq_length': seq_length, 'stride': stride,
            }, f)
        with open(f'{save_dir}/json_ready', 'w') as f:
            f.write('ready')
        print(f'  JSON dataset saved to Drive.')

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'features': features, 'scaler': scaler,
        'label_encoder': label_encoder, 'n_continuous': n_continuous
    }


def load_dataset_from_drive(dataset_type):
    preprocessed_dir = f'{DRIVE_RESULTS_DIR}/preprocessed/{dataset_type}'
    ready_file = f'{preprocessed_dir}/{dataset_type}_ready'

    if os.path.exists(ready_file) and os.path.exists(f'{preprocessed_dir}/X_train.npy'):
        print(f'  Loading preprocessed {dataset_type.upper()} from Drive...')
        X_train = np.load(f'{preprocessed_dir}/X_train.npy')
        X_val = np.load(f'{preprocessed_dir}/X_val.npy')
        X_test = np.load(f'{preprocessed_dir}/X_test.npy')
        y_train = np.load(f'{preprocessed_dir}/y_train.npy')
        y_val = np.load(f'{preprocessed_dir}/y_val.npy')
        y_test = np.load(f'{preprocessed_dir}/y_test.npy')

        meta_file = f'{preprocessed_dir}/{dataset_type}_metadata.pkl'
        with open(meta_file, 'rb') as f:
            metadata = pickle.load(f)

        data = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'features': metadata['features'], 'scaler': metadata['scaler'],
            'label_encoder': metadata['label_encoder'], 'n_continuous': metadata['n_continuous']
        }
        print(f'  Preprocessed {dataset_type.upper()} loaded: {len(X_train):,} train samples')
        return data
    else:
        print(f'  No preprocessed data found. Loading {dataset_type.upper()} fresh...')
        if dataset_type == 'csv':
            return load_and_display_csv_dataset(
                CSV_DATA_DIR, seq_length=SEQ_LENGTH, stride=STRIDE,
                save_dir=CSV_PREPROCESSED_DIR
            )
        else:
            return load_and_display_json_dataset(
                JSON_DATA_DIR, seq_length=SEQ_LENGTH, stride=STRIDE,
                max_records=MAX_RECORDS, save_dir=JSON_PREPROCESSED_DIR
            )

CSV_PREPROCESSED_DIR = f'{DRIVE_RESULTS_DIR}/preprocessed/csv'
JSON_PREPROCESSED_DIR = f'{DRIVE_RESULTS_DIR}/preprocessed/json'

print('Data loading functions ready.')
# ─── Cell: Load CSV Dataset ──────────────────────────────────────────────
if DATASETS in ['csv', 'both']:
    csv_data = load_dataset_from_drive('csv')
else:
    csv_data = None
    print('Skipping CSV dataset')
# ─── Cell: GreedyAttackSimulator + Training Functions ─────────────────────
import sys
sys.path.insert(0, '/content/pfe')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import json
import os

from src.models.lstm import LSTMClassifier
from src.models.bilstm import BiLSTMClassifier
from src.models.cnn_lstm import CNNLSTMClassifier
from src.models.xgboost_lstm import XGBoostLSTMClassifier
from src.models.transformer import TransformerClassifier
from src.models.cnn_bilstm_transformer import CNNBiLSTMTransformerClassifier
from src.models.transformer import NLPTransformerClassifier
from src.data.tokenizer import create_tokenizer
from src.models.transformer import NLPTransformerClassifier
from src.data.tokenizer import create_tokenizer
from src.models.transformer import NLPTransformerClassifier
from src.data.tokenizer import create_tokenizer
from src.training.trainer import IoTSequenceDataset
from src.adversarial.robust_losses import AFDLoss


# =====================================================================
# FIXED: nlp_cnn_bilstm_transformer properly uses embedded layer logic
# GreedyAttackSimulator — replique exactement adversarial_search_seq.py
# =====================================================================
# FIXED: nlp_cnn_bilstm_transformer properly uses embedded layer logic

class GreedyAttackSimulator:
    def __init__(self, sensitivity_results, feature_stats, verbose=True):
        self.results = sensitivity_results
        self.stats = feature_stats
        
        self.feature_pool = {}
        self.feature_weights = {}
        epsilon = 0.05  # Exploratory minimum probability
        
        for fi, st, drop in sensitivity_results:
            if fi not in self.feature_pool:
                self.feature_pool[fi] = []
                self.feature_weights[fi] = max(0.0, drop) + epsilon
            
            # Maintain a pool of valid strategies even if drop <= 0, we keep them for exploratory testing
            self.feature_pool[fi].append(st)
                
        self.available_features = list(self.feature_pool.keys())
        if len(self.available_features) > 0:
            weights = np.array([self.feature_weights[f] for f in self.available_features])
            if weights.sum() > 0:
                self.sampling_probs = weights / weights.sum()
            else:
                self.sampling_probs = np.ones(len(weights)) / len(weights)
        else:
            self.sampling_probs = np.array([])
            
        if verbose:
            print(f"  [Simulator] Vulnerability Dictionary created with {len(self.available_features)} distinct features.")
            print(f"  [Simulator] Top 3 features logic overview:")
            for idx, feat in enumerate(self.available_features[:3]):
                print(f"     -> Feature {feat} mapped to {len(self.feature_pool[feat])} strategies (prob={self.sampling_probs[idx]:.3f})")

    @classmethod
    def compute_feature_stats(cls, X_train):
        X_flat = X_train.reshape(-1, X_train.shape[-1])
        stats = {}
        for i in range(X_flat.shape[1]):
            col = X_flat[:, i]
            stats[i] = {
                'mean': float(col.mean()),
                'p95': float(np.percentile(col, 95)),
                'std': float(col.std()),
            }
        return stats

    def apply_strategy(self, X, feat_idx, strategy):
        X = X.copy()
        if strategy == 'Zero':
            X[:, :, feat_idx] = 0.0
        elif strategy == 'Mimic_Mean':
            X[:, :, feat_idx] = self.stats[feat_idx]['mean']
        elif strategy == 'Mimic_95th':
            X[:, :, feat_idx] = self.stats[feat_idx]['p95']
        elif strategy == 'Padding_x10':
            X[:, :, feat_idx] = np.clip(X[:, :, feat_idx] * 10.0, -5.0, 5.0)
        return X

    def generate_greedy(self, X, k):
        X_adv = X.copy()
        n_avail = len(self.available_features)
        if n_avail == 0:
            return X_adv
            
        k_actual = min(k, n_avail)
        chosen_features = np.random.choice(self.available_features, size=k_actual, replace=False, p=self.sampling_probs)
        
        for feat_idx in chosen_features:
            strategy = np.random.choice(self.feature_pool[feat_idx])
            X_adv = self.apply_strategy(X_adv, feat_idx, strategy)
            
        return X_adv

    def generate_training_batch(self, X, k_max=4, mix_ratio=0.5):
        n = len(X)
        n_adv = int(n * mix_ratio)
        n_cln = n - n_adv

        idx_adv = np.random.choice(n, n_adv, replace=False)
        idx_cln = np.setdiff1d(np.arange(n), idx_adv)[:n_cln]

        X_out = X.copy()
        flags = np.zeros(n, dtype=np.float32)

        for i in idx_adv:
            k = np.random.randint(1, k_max + 1)
            X_out[[i]] = self.generate_greedy(X[[i]], k)
            flags[i] = 1.0

        return X_out, flags


def load_sensitivity_results(csv_path, feature_names):
    df = pd.read_csv(csv_path).sort_values('drop', ascending=False)
    idx = {name: i for i, name in enumerate(feature_names)}
    result = []
    for _, row in df.iterrows():
        feat = row['feature']
        if feat in idx:
            result.append((idx[feat], row['strategy'], float(row['drop'])))
    print(f"  -> {len(result)} (feature, strategy) pairs loaded from sensitivity analysis")
    print(f"  Top 5 most vulnerable:")
    for i, (fi, st, dr) in enumerate(result[:5], 1):
        print(f"     {i}. {feature_names[fi]:<25} | {st:<14} | drop={dr*100:.1f}%")
    return result


def create_model(model_type, input_size, num_classes):
    if model_type == 'lstm':
        return LSTMClassifier(input_size, num_classes)
    elif model_type == 'bilstm':
        return BiLSTMClassifier(input_size, num_classes)
    elif model_type == 'cnn_lstm':
        return CNNLSTMClassifier(input_size, num_classes)
    elif model_type == 'xgboost_lstm':
        return XGBoostLSTMClassifier(input_size, num_classes)
    elif model_type == 'transformer':
        return TransformerClassifier(input_size, num_classes)
    elif model_type == 'cnn_bilstm_transformer':
        return CNNBiLSTMTransformerClassifier(input_size, num_classes, seq_length=SEQ_LENGTH,
                                              config=CNN_BILSTM_TRANSFORMER_OVERRIDE)
    elif model_type == 'nlp_cnn_bilstm_transformer':
        return CNNBiLSTMTransformerClassifier(input_size=128, num_classes=num_classes, seq_length=576, vocab_size=52000, config=CNN_BILSTM_TRANSFORMER_OVERRIDE)
    elif model_type == 'nlp_transformer':
        return NLPTransformerClassifier(vocab_size=52000, num_classes=num_classes, max_seq_length=576, pad_token_id=2)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =====================================================================
# FIXED: nlp_cnn_bilstm_transformer properly uses embedded layer logic
# train_greedy_phase — train model for one phase (A/B/C)
# =====================================================================
# FIXED: nlp_cnn_bilstm_transformer properly uses embedded layer logic

def train_greedy_phase(
    model, X_train, y_train, X_val, y_val,
    phase, start_epoch, end_epoch,
    mix_ratio, k_max,
    p_drop=0.0, sigma_noise=0.0, afd_lambda=0.0,
    simulator=None, device=None,
    lr=5e-4, batch_size=64, save_path=None,
    is_nlp=False, tokenizer=None, features=None,
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[15, 30], gamma=0.5
    )
    use_amp = USE_AMP and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if start_epoch > 1:
        for _ in range(start_epoch - 1):
            optimizer.step()
            scheduler.step()

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)

    phase_names = {'A': 'Fondation (clean only)', 'B': 'Introduction (30% adv, k_max=2)', 'C': 'Principal (70% adv, k_max=4)', 'D': 'Consolidation (85% adv, k_max=4, epochs 51-80)'}
    print(f"\n{'='*60}")
    print(f"  PHASE {phase} — epochs {start_epoch}-{end_epoch}")
    print(f"  {phase_names.get(phase, '')}")
    print(f"  mix_ratio={mix_ratio} | k_max={k_max}")
    print(f"{'='*60}")

    best_val_acc = 0.0
    best_combined = 0.0   # score = 0.4*clean + 0.6*adv (phases adv seulement)
    best_epoch = start_epoch

    label_sm_map = {'A': 0.05, 'B': 0.08, 'C': 0.10}
    label_sm = label_sm_map.get(phase, 0.05)

    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        criterion = nn.CrossEntropyLoss(label_smoothing=label_sm)

        total_loss, total_correct, total_n = 0.0, 0, 0

        if afd_lambda > 0:
            num_classes = len(np.unique(y_train))
            afd_criterion = AFDLoss(num_classes, num_classes, lambda_intra=1.0, lambda_inter=0.5).to(device)

        for batch_idx, (X_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)):
            X_np = X_batch.numpy()
            y_input = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                if mix_ratio > 0 and simulator is not None:
                    if afd_lambda > 0:
                        X_clean_t = X_batch.to(device)
                        X_adv_mixed, _ = simulator.generate_training_batch(X_np, k_max=k_max, mix_ratio=mix_ratio)
                        X_adv_t = torch.FloatTensor(X_adv_mixed).to(device)
                        
                        if is_nlp:
                            X_clean_t = torch.LongTensor(tokenizer.transform(X_batch.numpy(), features)).to(device)
                            X_adv_t = torch.LongTensor(tokenizer.transform(X_adv_mixed, features)).to(device)
                            
                        if sigma_noise > 0:
                            X_clean_t = X_clean_t + torch.randn_like(X_clean_t) * sigma_noise
                            X_adv_t = X_adv_t + torch.randn_like(X_adv_t) * sigma_noise
                            
                        if batch_idx == 0:
                            import sys
                            print(f"
  [VERBOSE] X_clean_t shape: {X_clean_t.shape}, dtype: {X_clean_t.dtype}", file=sys.stderr)
                            print(f"  [VERBOSE] Model type: {type(model)}", file=sys.stderr)
                            sys.stderr.flush()
                        
                        logits_clean = model(X_clean_t)
                        logits_adv = model(X_adv_t)
                        
                        loss_ce = criterion(logits_adv, y_input)
                        loss_afd = afd_criterion(logits_clean, logits_adv, y_input)
                        loss = loss_ce + afd_lambda * loss_afd
                        logits = logits_adv
                    else:
                        X_mixed, _ = simulator.generate_training_batch(X_np, k_max=k_max, mix_ratio=mix_ratio)
                        if is_nlp:
                            X_input = torch.LongTensor(tokenizer.transform(X_mixed, features)).to(device)
                        else:
                            X_input = torch.FloatTensor(X_mixed).to(device)
                        if p_drop > 0 and not is_nlp:
                            mask = (torch.rand(X_input.shape[0], 1, X_input.shape[2], device=device) > p_drop).float()
                            X_input = X_input * mask / (1.0 - p_drop)
                        if sigma_noise > 0:
                            X_input = X_input + torch.randn_like(X_input) * sigma_noise
                        
                        if batch_idx == 0:
                            import sys
                            print(f"
  [VERBOSE] X_input shape: {X_input.shape}, dtype: {X_input.dtype}", file=sys.stderr)
                            sys.stderr.flush()
                        
                        logits = model(X_input)
                        loss = criterion(logits, y_input)
                else:
                    if is_nlp:
                        X_input = torch.LongTensor(tokenizer.transform(X_np, features)).to(device)
                    else:
                        X_input = X_batch.to(device)
                    if p_drop > 0 and not is_nlp:
                        mask = (torch.rand(X_input.shape[0], 1, X_input.shape[2], device=device) > p_drop).float()
                        X_input = X_input * mask / (1.0 - p_drop)
                    if sigma_noise > 0:
                        X_input = X_input + torch.randn_like(X_input) * sigma_noise
                    
                    if batch_idx == 0:
                        import sys
                        print(f"
  [VERBOSE] X_batch shape: {X_batch.shape}, X_input shape: {X_input.shape}, dtype: {X_input.dtype}", file=sys.stderr)
                        try:
                            # In case it's CNNBiLSTMTransformerClassifier
                            if hasattr(model, 'cnn_branch1'):
                                print(f"  [VERBOSE] cnn_ch: {model.cnn_branch1[0].out_channels}", file=sys.stderr)
                                xt = X_input.permute(0, 2, 1)
                                b1 = model.cnn_branch1(xt)
                                b2 = model.cnn_branch2(xt)
                                fused = torch.cat([b1, b2], dim=1).permute(0, 2, 1).contiguous()
                                print(f"  [VERBOSE] fused shape: {fused.shape}", file=sys.stderr)
                                print(f"  [VERBOSE] bilstm expects input_size: {model.bilstm.input_size}", file=sys.stderr)
                        except Exception as e:
                            print("  [VERBOSE] Debug print exception:", e, file=sys.stderr)
                        sys.stderr.flush()
                    
                    logits = model(X_input)
                    loss = criterion(logits, y_input)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(y_batch)
            total_correct += (logits.argmax(1) == y_input).sum().item()
            total_n += len(y_batch)

        scheduler.step()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        train_loss = total_loss / total_n
        train_acc = total_correct / total_n

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                end = min(i + batch_size, len(X_val))
                if is_nlp:
                    X_val_t = torch.LongTensor(tokenizer.transform(X_val[i:end], features)).to(device)
                else:
                    X_val_t = torch.FloatTensor(X_val[i:end]).to(device)
                y_val_t = torch.LongTensor(y_val[i:end]).to(device)
                val_correct += (model(X_val_t).argmax(1) == y_val_t).sum().item()
                val_total += len(y_val_t)
            val_clean_acc = val_correct / val_total

            val_adv_acc = 0.0
            if simulator is not None and k_max > 0:
                val_adv_correct = 0
                n_eval = min(EVAL_SUBSAMPLE, len(X_val))
                for i in range(0, n_eval, batch_size):
                    end = min(i + batch_size, n_eval)
                    X_adv_np = simulator.generate_greedy(X_val[i:end], k=k_max)
                    if is_nlp:
                        X_adv_t = torch.LongTensor(tokenizer.transform(X_adv_np, features)).to(device)
                    else:
                        X_adv_t = torch.FloatTensor(X_adv_np).to(device)
                    y_sub = torch.LongTensor(y_val[i:end]).to(device)
                    val_adv_correct += (model(X_adv_t).argmax(1) == y_sub).sum().item()
                val_adv_acc = val_adv_correct / n_eval

        print(f"  Epoch {epoch:3d}/{end_epoch} [Ph{phase}] "
              f"Loss={train_loss:.4f} TrainAcc={train_acc:.4f} "
              f"CleanAcc={val_clean_acc:.4f} AdvAcc={val_adv_acc:.4f}")

        # Phase A : sélection sur clean seulement (pas encore d'attaques)
        # Phases B/C/D : score combiné → favorise la robustesse adversariale
        if phase == 'A':
            selection_score = val_clean_acc
            is_better = val_clean_acc > best_val_acc
        else:
            # 0.4 clean + 0.6 adv : on cherche ~85% adv sans effondrer le clean
            selection_score = 0.4 * val_clean_acc + 0.6 * val_adv_acc
            is_better = selection_score > best_combined

        if is_better:
            best_val_acc = val_clean_acc
            best_combined = selection_score
            best_epoch = epoch
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_clean_acc': val_clean_acc,
                    'val_adv_acc': val_adv_acc,
                    'phase': phase,
                    'combined_score': selection_score,
                }, save_path)

    if phase == 'A':
        print(f"  Best epoch: {best_epoch} | Best val clean acc: {best_val_acc:.4f}")
    else:
        print(f"  Best epoch: {best_epoch} | Best combined score: {best_combined:.4f} "
              f"(clean={best_val_acc:.4f}, adv tracked per-epoch)")
    return model


# =====================================================================
# FIXED: nlp_cnn_bilstm_transformer properly uses embedded layer logic
# crash_test_greedy — evaluate clean + adversarial (k=1..4)
# =====================================================================
# FIXED: nlp_cnn_bilstm_transformer properly uses embedded layer logic

def crash_test_greedy(model, X_val, y_val, simulator, device=None,
                      k_values=None, label='', is_nlp=False, tokenizer=None, features=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if k_values is None:
        k_values = [1, 2, 3, 4]

    model.eval()
    model = model.to(device)

    n_eval = min(EVAL_SUBSAMPLE, len(X_val))
    X_eval = X_val[:n_eval]
    y_eval = y_val[:n_eval]

    clean_correct = 0
    with torch.no_grad():
        for i in range(0, n_eval, 1024):
            end = min(i + 1024, n_eval)
            X_b = torch.LongTensor(tokenizer.transform(X_eval[i:end], features)).to(device) if is_nlp else torch.FloatTensor(X_eval[i:end]).to(device)
            y_b = torch.LongTensor(y_eval[i:end]).to(device)
            clean_correct += (model(X_b).argmax(1) == y_b).sum().item()
    clean_acc = clean_correct / n_eval

    results = {'clean': clean_acc}

    print(f"  [Crash Test {label}] Clean={clean_acc:.4f}", end='')

    if simulator is not None:
        for k in k_values:
            adv_correct = 0
            with torch.no_grad():
                for i in range(0, n_eval, 1024):
                    end = min(i + 1024, n_eval)
                    X_adv = simulator.generate_greedy(X_eval[i:end], k=k)
                    X_adv_t = torch.LongTensor(tokenizer.transform(X_adv, features)).to(device) if is_nlp else torch.FloatTensor(X_adv).to(device)
                    y_b = torch.LongTensor(y_eval[i:end]).to(device)
                    adv_correct += (model(X_adv_t).argmax(1) == y_b).sum().item()
            
            adv_acc = adv_correct / n_eval
            results[f'adv_k{k}'] = adv_acc
            rr = adv_acc / max(clean_acc, 1e-8)
            print(f" | k={k}: {adv_acc:.4f}(RR={rr:.3f})", end='')

    print()
    return results


# =====================================================================
# FIXED: nlp_cnn_bilstm_transformer properly uses embedded layer logic
# run_sensitivity_analysis — run sensitivity_analysis_seq.py logic
# =====================================================================
# FIXED: nlp_cnn_bilstm_transformer properly uses embedded layer logic

def run_sensitivity_analysis(model, X_val, y_val, feature_names, num_classes,
                             n_continuous, save_csv_path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n  Running sensitivity analysis...")
    model.eval()
    model = model.to(device)

    from src.adversarial.attacks import SensitivityAnalysis

    n_sens = min(5000, len(X_val))
    sens_indices = np.random.choice(len(X_val), n_sens, replace=False)
    X_sens = X_val[sens_indices].copy()
    y_sens = y_val[sens_indices].copy()

    sa = SensitivityAnalysis(
        X_sens, y_sens,
        feature_names if feature_names else [f'f{i}' for i in range(X_val.shape[2])],
        num_classes,
        n_continuous_features=n_continuous,
    )

    results = sa.analyze(model, X_sens, y_sens, device=device)

    rows = []
    for entry in results:
        rows.append({
            'feature': entry['feature'],
            'strategy': entry['strategy'],
            'drop': entry['drop'],
            'original_acc': entry.get('original_acc', 0) if 'original_acc' in entry else (entry['accuracy'] + entry['drop']),
            'perturbed_acc': entry['accuracy'],
        })

    df = pd.DataFrame(rows).sort_values('drop', ascending=False)
    os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
    df.to_csv(save_csv_path, index=False)
    print(f"  Sensitivity results saved to {save_csv_path}")
    print(f"  Top 5 vulnerable features:")
    for _, row in df.head(5).iterrows():
        print(f"    {row['feature']:<25} | {row['strategy']:<14} | drop={row['drop']*100:.1f}%")

    return df


# =====================================================================
# FIXED: nlp_cnn_bilstm_transformer properly uses embedded layer logic
# Discriminator and Router
# =====================================================================
# FIXED: nlp_cnn_bilstm_transformer properly uses embedded layer logic

class Discriminator(nn.Module):
    def __init__(self, input_size, seq_length, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h_cat = torch.cat([h[0], h[1]], dim=1)
        return self.head(h_cat).squeeze(1)
    
    def predict_proba(self, x):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

def train_discriminator(discriminator, X_train, simulator, device=None, epochs=25, batch_size=64, lr=1e-3, save_path='discriminator.pt'):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*65}\n  ENTRAÎNEMENT DU DISCRIMINATEUR\n{'='*65}")
    discriminator = discriminator.to(device)
    optimizer = torch.optim.AdamW(discriminator.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    n = len(X_train)
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        discriminator.train()
        n_half = n // 2
        idx = np.random.permutation(n)
        idx_clean = idx[:n_half]
        idx_adv = idx[n_half:n_half*2]
        k_values = np.random.randint(1, 4, size=n_half)
        X_adv_list = []
        for orig_i, k in zip(idx_adv, k_values):
            X_adv_list.append(simulator.generate_greedy(X_train[[orig_i]], k=k))
        X_adv_ep = np.concatenate(X_adv_list, axis=0)
        X_combined = np.concatenate([X_train[idx_clean], X_adv_ep], axis=0)
        labels_bin = np.array([0.0]*n_half + [1.0]*n_half, dtype=np.float32)
        perm = np.random.permutation(len(X_combined))
        X_combined = X_combined[perm]
        labels_bin = labels_bin[perm]
        loader = DataLoader(TensorDataset(torch.FloatTensor(X_combined), torch.FloatTensor(labels_bin)), batch_size=batch_size, shuffle=True)
        total_loss, total_correct, total_n = 0.0, 0, 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = discriminator(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            optimizer.step()
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == yb).sum().item()
            total_loss += loss.item() * len(yb)
            total_n += len(yb)
        acc = total_correct / total_n
        print(f"  Epoch {epoch:3d}/{epochs}  Loss={total_loss/total_n:.4f}  Acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({'model_state_dict': discriminator.state_dict(), 'accuracy': acc}, save_path)
    print(f"\n  Discriminateur — meilleure accuracy : {best_acc:.4f}\n  Sauvegardé → {save_path}")
    ckpt = torch.load(save_path, map_location=device)
    discriminator.load_state_dict(ckpt['model_state_dict'])
    return discriminator, best_acc

class IoTRouter(nn.Module):
    def __init__(self, normal_model, adversarial_model, discriminator, threshold=0.5, is_nlp=False, tokenizer=None, features=None):
        self.is_nlp = is_nlp
        self.tokenizer = tokenizer
        self.features = features
        super().__init__()
        self.normal = normal_model
        self.adversarial = adversarial_model
        self.discriminator = discriminator
        self.threshold = threshold
    
    @torch.no_grad()
    def predict(self, X):
        self.normal.eval()
        self.adversarial.eval()
        self.discriminator.eval()
        attack_scores = self.discriminator.predict_proba(X)
        is_attacked = (attack_scores >= self.threshold)
        logits_normal = self.normal(X)
        if hasattr(self, 'is_nlp') and self.is_nlp and self.tokenizer is not None:
            # Need X as numpy for tokenizer
            X_np = X.cpu().numpy()
            X_ids = self.tokenizer.transform(X_np, self.features)
            X_adj = torch.LongTensor(X_ids).to(X.device)
            logits_adv = self.adversarial(X_adj)
        else:
            logits_adv = self.adversarial(X)
        pred_normal = logits_normal.argmax(1)
        pred_adv = logits_adv.argmax(1)
        predictions = torch.where(is_attacked, pred_adv, pred_normal)
        routes = is_attacked.long()
        return predictions, routes, attack_scores
    
    def calibrate_threshold(self, X_clean, X_attacked, target_recall=0.95):
        with torch.no_grad():
            scores_clean = self.discriminator.predict_proba(X_clean).cpu().numpy()
            scores_attacked = self.discriminator.predict_proba(X_attacked).cpu().numpy()
        all_scores = np.concatenate([scores_clean, scores_attacked])
        all_labels = np.array([0]*len(scores_clean) + [1]*len(scores_attacked))
        thresholds = np.linspace(0.0, 1.0, 200)
        best_t, best_f1 = 0.5, 0.0
        for t in thresholds:
            preds = (all_scores >= t).astype(int)
            tp = ((preds == 1) & (all_labels == 1)).sum()
            fp = ((preds == 1) & (all_labels == 0)).sum()
            fn = ((preds == 0) & (all_labels == 1)).sum()
            recall = tp / (tp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            if recall >= target_recall and f1 > best_f1:
                best_f1 = f1
                best_t = t
        self.threshold = best_t
        print(f"  Seuil calibré : {best_t:.3f}  (recall attaques ≥ {target_recall:.0%})")
        return best_t

# =====================================================================
# FIXED: nlp_cnn_bilstm_transformer properly uses embedded layer logic
# train_model_greedy — orchestrate full 3-phase training for one model
# =====================================================================
# FIXED: nlp_cnn_bilstm_transformer properly uses embedded layer logic

def train_model_greedy(
    model_type, dataset_type, data_dict,
    batch_size=64, lr=5e-4,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = f'{DRIVE_RESULTS_DIR}/models/{model_type}_greedy_{dataset_type}'
    os.makedirs(save_dir, exist_ok=True)

    X_train = data_dict['X_train']
    X_val = data_dict['X_val']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_val = data_dict['y_val']
    y_test = data_dict['y_test']
    features = data_dict.get('features', [])
    n_continuous = data_dict.get('n_continuous', X_train.shape[2])
    label_encoder = data_dict.get('label_encoder', None)

    input_size = X_train.shape[2]
    num_classes = len(np.unique(y_train))
    is_nlp = ('nlp' in model_type)
    tokenizer = None
    if is_nlp:
        tokenizer = create_tokenizer()
        print(f"\n  [TOKENIZER] Fitting BPE tokenizer on training data...")
        tokenizer.fit(X_train, features, verbose=False)

    print(f"\n{'#'*80}")
    print(f"  GREEDY ADVERSARIAL TRAINING — {model_type.upper()} on {dataset_type.upper()}")
    print(f"{'#'*80}")
    print(f"  Input size: {input_size} | Classes: {num_classes}")
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"  Save dir: {save_dir}")

    all_crash_results = {}

    # ─── PHASE A (epochs 1-15): Clean data only ──────────────────────────
    phase_a_path = f'{save_dir}/phase_a_model.pt'
    sens_csv_path = f'{save_dir}/sensitivity_results.csv'

    model = create_model(model_type, input_size, num_classes)

    if os.path.exists(phase_a_path):
        print(f"\n  Phase A model found in Drive. Loading...")
        ckpt = torch.load(phase_a_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        print(f"  Loaded Phase A model (epoch {ckpt.get('epoch', '?')}, "
              f"clean_acc={ckpt.get('val_clean_acc', 0):.4f})")
    else:
        model = train_greedy_phase(
            model, X_train, y_train, X_val, y_val,
            phase='A', start_epoch=1, end_epoch=PHASE_A_EPOCHS,
            mix_ratio=PHASE_A_MIX_RATIO, k_max=0,
            p_drop=0.0, sigma_noise=0.0, afd_lambda=0.0,
            simulator=None, device=device, lr=lr,
            batch_size=batch_size, save_path=phase_a_path, is_nlp=is_nlp, tokenizer=tokenizer, features=features
        )

    ct_a = crash_test_greedy(model, X_val, y_val, simulator=None, device=device, label='Phase A', is_nlp=is_nlp, tokenizer=tokenizer, features=features)
    all_crash_results['phase_a'] = ct_a

    # ─── Sensitivity Analysis (after Phase A) ─────────────────────────────
    feature_names = features if features else [f'f{i}' for i in range(input_size)]

    if os.path.exists(sens_csv_path):
        print(f"\n  Sensitivity results found in Drive. Loading...")
        sensitivity = load_sensitivity_results(sens_csv_path, feature_names)
    else:
        run_sensitivity_analysis(
            model, X_val, y_val, feature_names, num_classes,
            n_continuous, sens_csv_path, device=device,
        )
        sensitivity = load_sensitivity_results(sens_csv_path, feature_names)

    feature_stats = GreedyAttackSimulator.compute_feature_stats(X_train)
    simulator = GreedyAttackSimulator(sensitivity, feature_stats)

    # ─── PHASE B (epochs 16-30): 30% adversarial, k_max=2 ─────────────────
    phase_b_path = f'{save_dir}/phase_b_model.pt'

    if os.path.exists(phase_b_path):
        print(f"\n  Phase B model found in Drive. Loading...")
        ckpt = torch.load(phase_b_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        print(f"  Loaded Phase B model (epoch {ckpt.get('epoch', '?')}, "
              f"clean_acc={ckpt.get('val_clean_acc', 0):.4f}, "
              f"adv_acc={ckpt.get('val_adv_acc', 0):.4f})")
    else:
        model = train_greedy_phase(
            model, X_train, y_train, X_val, y_val,
            phase='B', start_epoch=PHASE_A_EPOCHS + 1, end_epoch=PHASE_B_EPOCHS,
            mix_ratio=PHASE_B_MIX_RATIO, k_max=PHASE_B_K_MAX,
            p_drop=0.1, sigma_noise=0.01, afd_lambda=0.5,
            simulator=simulator, device=device, lr=lr,
            batch_size=batch_size, save_path=phase_b_path, is_nlp=is_nlp, tokenizer=tokenizer, features=features
        )

    ct_b = crash_test_greedy(model, X_val, y_val, simulator=simulator, device=device, label='Phase B', is_nlp=is_nlp, tokenizer=tokenizer, features=features)
    all_crash_results['phase_b'] = ct_b

    # ─── PHASE C (epochs 31-50): 70% adversarial, k_max=4 ─────────────────
    phase_c_path = f'{save_dir}/phase_c_model.pt'

    if os.path.exists(phase_c_path):
        print(f"\n  Phase C model found in Drive. Loading...")
        ckpt = torch.load(phase_c_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        print(f"  Loaded Phase C model (epoch {ckpt.get('epoch', '?')}, "
              f"clean_acc={ckpt.get('val_clean_acc', 0):.4f}, "
              f"adv_acc={ckpt.get('val_adv_acc', 0):.4f})")
    else:
        model = train_greedy_phase(
            model, X_train, y_train, X_val, y_val,
            phase='C', start_epoch=PHASE_B_EPOCHS + 1, end_epoch=PHASE_C_EPOCHS,
            mix_ratio=PHASE_C_MIX_RATIO, k_max=PHASE_C_K_MAX,
            p_drop=0.2, sigma_noise=0.01, afd_lambda=1.0,
            simulator=simulator, device=device, lr=lr,
            batch_size=batch_size, save_path=phase_c_path, is_nlp=is_nlp, tokenizer=tokenizer, features=features
        )

    ct_c = crash_test_greedy(model, X_val, y_val, simulator=simulator, device=device, label='Phase C', is_nlp=is_nlp, tokenizer=tokenizer, features=features)
    all_crash_results['phase_c'] = ct_c

    # ─── PHASE D (epochs 51-65): 95% adversarial, k_max=4 ─────────────────
    phase_d_path = f'{save_dir}/phase_d_model.pt'

    if os.path.exists(phase_d_path):
        print(f"\n  Phase D model found in Drive. Loading...")
        ckpt = torch.load(phase_d_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        print(f"  Loaded Phase D model (epoch {ckpt.get('epoch', '?')}, "
              f"clean_acc={ckpt.get('val_clean_acc', 0):.4f}, "
              f"adv_acc={ckpt.get('val_adv_acc', 0):.4f})")
    else:
        model = train_greedy_phase(
            model, X_train, y_train, X_val, y_val,
            phase='D', start_epoch=PHASE_C_EPOCHS + 1, end_epoch=PHASE_D_EPOCHS,
            mix_ratio=PHASE_D_MIX_RATIO, k_max=PHASE_D_K_MAX,
            p_drop=0.2, sigma_noise=0.01, afd_lambda=0.0,
            simulator=simulator, device=device, lr=lr,
            batch_size=batch_size, save_path=phase_d_path, is_nlp=is_nlp, tokenizer=tokenizer, features=features
        )

    ct_d = crash_test_greedy(model, X_val, y_val, simulator=simulator, device=device, label='Phase D', is_nlp=is_nlp, tokenizer=tokenizer, features=features)
    all_crash_results['phase_d'] = ct_d

    # ─── PHASE E: Discriminator ──────────────────────────────────────────
    disc_path = f'{save_dir}/discriminator.pt'
    disc = Discriminator(input_size=input_size, seq_length=10, hidden_size=64)
    if os.path.exists(disc_path):
        print(f"\n  Discriminator model found in Drive. Loading...")
        ckpt = torch.load(disc_path, map_location=device)
        disc.load_state_dict(ckpt['model_state_dict'])
        disc_acc = ckpt.get('accuracy', 0.95)
        disc = disc.to(device)
        print(f"  Loaded Discriminator (acc={disc_acc:.4f})")
    else:
        disc, disc_acc = train_discriminator(
            discriminator=disc,
            X_train=X_train,
            simulator=simulator,
            device=device,
            epochs=25,
            batch_size=batch_size,
            save_path=disc_path
        )
    
    # ─── Final evaluation on test set with Router ────────────────────────
    print(f"\n{'='*80}")
    print(f"  RÉSULTATS ATTENDUS APRÈS ENTRAÎNEMENT")
    print(f"{'='*80}")
    model.eval() # Adversarial model
    disc.eval()

    # Create & load normal model
    normal_model_path = f'{DRIVE_RESULTS_DIR}/models/{model_type}_{dataset_type}/best_val_model.pt'
    if not os.path.exists(normal_model_path):
        normal_model_path = phase_a_path
    normal_model = create_model(model_type, input_size, num_classes)
    if os.path.exists(normal_model_path):
        try:
            ckpt_nor = torch.load(normal_model_path, map_location=device)
            if 'model_state_dict' in ckpt_nor:
                normal_model.load_state_dict(ckpt_nor['model_state_dict'])
            else:
                normal_model.load_state_dict(ckpt_nor)
            print(f"  Normal model loaded from {normal_model_path}")
        except Exception as e:
            print(f"  Could not load normal model: {e}. Using random weights.")
    else:
        print(f"  WARN: Normal model not found at {normal_model_path}. Using random weights.")
    normal_model = normal_model.to(device)
    normal_model.eval()

    router = IoTRouter(normal_model, model, disc, threshold=0.5, is_nlp=is_nlp, tokenizer=tokenizer, features=features)
    
    # Calibrate router limit
    X_val_clean_sub = torch.FloatTensor(X_val[:min(len(X_val), 1000)]).to(device)
    X_val_adv_sub = torch.FloatTensor(simulator.generate_greedy(X_val[:min(len(X_val), 1000)], k=4)).to(device)
    router.calibrate_threshold(X_val_clean_sub, X_val_adv_sub, target_recall=0.95)
    
    # Evaluate Clean
    test_dataset = IoTSequenceDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE)
    correct_clean = 0
    total = 0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            pred, _, _ = router.predict(X_b)
            total += y_b.size(0)
            correct_clean += pred.eq(y_b).sum().item()
    clean_acc = correct_clean / total

    # Evaluate pure models
    correct_normal_clean = 0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            _, pred = normal_model(X_b).max(1)
            correct_normal_clean += pred.eq(y_b).sum().item()
    normal_clean_acc = correct_normal_clean / max(total, 1)

    # Evaluate Adv k=4 on Adv Model
    adv_results = {}
    correct_adv_model_k4 = 0
    correct_router_adv_k4 = 0
    total_adv = 0
    
    for k in [1, 2, 3, 4]:
        c_adv_k = 0
        t_adv_k = 0
        for i in range(0, min(len(X_test), EVAL_SUBSAMPLE), EVAL_BATCH_SIZE):
            batch_end = min(i + EVAL_BATCH_SIZE, len(X_test), EVAL_SUBSAMPLE)
            X_sub = X_test[i:batch_end]
            y_sub = y_test[i:batch_end]
            
            X_adv = simulator.generate_greedy(X_sub, k=k)
            X_adv_t = torch.LongTensor(tokenizer.transform(X_adv, features)).to(device) if is_nlp else torch.FloatTensor(X_adv).to(device)
            y_sub_t = torch.LongTensor(y_sub).to(device)
            
            with torch.no_grad():
                _, pred_adv = model(X_adv_t).max(1)
                
            c_adv_k += pred_adv.eq(y_sub_t).sum().item()
            t_adv_k += len(y_sub)
        adv_results[f'k{k}'] = c_adv_k / max(t_adv_k, 1)
        if k == 4:
            adv_model_k4_acc = adv_results[f'k{k}']
            # router test on k4
            for i in range(0, min(len(X_test), EVAL_SUBSAMPLE), EVAL_BATCH_SIZE):
                batch_end = min(i + EVAL_BATCH_SIZE, len(X_test), EVAL_SUBSAMPLE)
                X_sub = X_test[i:batch_end]
                y_sub = y_test[i:batch_end]
                X_adv = simulator.generate_greedy(X_sub, k=4)
                X_adv_t = torch.LongTensor(tokenizer.transform(X_adv, features)).to(device) if is_nlp else torch.FloatTensor(X_adv).to(device)
                y_sub_t = torch.LongTensor(y_sub).to(device)
                with torch.no_grad():
                    pred_router, _, _ = router.predict(X_adv_t)
                correct_router_adv_k4 += pred_router.eq(y_sub_t).sum().item()
            router_k4_acc = correct_router_adv_k4 / max(t_adv_k, 1)

    global_acc = (clean_acc + router_k4_acc) / 2.0

    print(f"──────────────────────────────────────")
    print(f"    Modèle normal     → Clean accuracy     : {normal_clean_acc*100:.1f}%")
    print(f"    Modèle antagoniste→ Adversarial acc k4 : {adv_model_k4_acc*100:.1f}%")
    print(f"    Discriminateur    → Détection attaque  : {disc_acc*100:.1f}%")
    print(f"    Système complet   → Accuracy globale   : {global_acc*100:.1f}%")
    print(f"================================================================================")
    
    results = {
        'model_type': model_type,
        'dataset_type': dataset_type,
        'clean_accuracy': clean_acc,
        'adversarial_accuracies': adv_results,
        'crash_tests': all_crash_results,
        'input_size': input_size,
        'num_classes': num_classes,
        'phases': {
            'A': {'epochs': f'1-{PHASE_A_EPOCHS}', 'mix_ratio': PHASE_A_MIX_RATIO, 'k_max': 0},
            'B': {'epochs': f'{PHASE_A_EPOCHS+1}-{PHASE_B_EPOCHS}', 'mix_ratio': PHASE_B_MIX_RATIO, 'k_max': PHASE_B_K_MAX},
            'C': {'epochs': f'{PHASE_B_EPOCHS+1}-{PHASE_C_EPOCHS}', 'mix_ratio': PHASE_C_MIX_RATIO, 'k_max': PHASE_C_K_MAX},
            'D': {'epochs': f'{PHASE_C_EPOCHS+1}-{PHASE_D_EPOCHS}', 'mix_ratio': PHASE_D_MIX_RATIO, 'k_max': PHASE_D_K_MAX},
        }
    }

    with open(f'{save_dir}/greedy_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    aggressive_cleanup()
    return results


print('Greedy adversarial training functions loaded.')
# ─── MODEL: LSTM on CSV (Greedy Adversarial) ────────────────────────
MODEL = 'lstm'
print(f'\n{"#"*80}')
print(f'  GREEDY ADVERSARIAL — LSTM on CSV')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_csv')

if DATASETS in ['csv', 'both']:
    data = load_dataset_from_drive('csv')
    if data is not None:
        results = train_model_greedy(
            model_type=MODEL,
            dataset_type='csv',
            data_dict=data,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
        )
        log_memory(f'after_{MODEL}_csv')
        del data  # release numpy arrays from RAM
        try:
            del results
        except Exception:
            pass
        aggressive_cleanup()
    else:
        print('Failed to load CSV data')
else:
    print('Skipping CSV dataset')

print(f'\n LSTM on CSV DONE')
# ─── MODEL: BiLSTM on CSV (Greedy Adversarial) ────────────────────────
MODEL = 'bilstm'
print(f'\n{"#"*80}')
print(f'  GREEDY ADVERSARIAL — BILSTM on CSV')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_csv')

if DATASETS in ['csv', 'both']:
    data = load_dataset_from_drive('csv')
    if data is not None:
        results = train_model_greedy(
            model_type=MODEL,
            dataset_type='csv',
            data_dict=data,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
        )
        log_memory(f'after_{MODEL}_csv')
        del data  # release numpy arrays from RAM
        try:
            del results
        except Exception:
            pass
        aggressive_cleanup()
    else:
        print('Failed to load CSV data')
else:
    print('Skipping CSV dataset')

print(f'\n BILSTM on CSV DONE')
# ─── MODEL: CNN-LSTM on CSV (Greedy Adversarial) ────────────────────────
MODEL = 'cnn_lstm'
print(f'\n{"#"*80}')
print(f'  GREEDY ADVERSARIAL — CNN-LSTM on CSV')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_csv')

if DATASETS in ['csv', 'both']:
    data = load_dataset_from_drive('csv')
    if data is not None:
        results = train_model_greedy(
            model_type=MODEL,
            dataset_type='csv',
            data_dict=data,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
        )
        log_memory(f'after_{MODEL}_csv')
        del data  # release numpy arrays from RAM
        try:
            del results
        except Exception:
            pass
        aggressive_cleanup()
    else:
        print('Failed to load CSV data')
else:
    print('Skipping CSV dataset')

print(f'\n CNN-LSTM on CSV DONE')
# ─── MODEL: XGBoost-LSTM on CSV (Greedy Adversarial) ────────────────────────
MODEL = 'xgboost_lstm'
print(f'\n{"#"*80}')
print(f'  GREEDY ADVERSARIAL — XGBOOST-LSTM on CSV')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_csv')

if DATASETS in ['csv', 'both']:
    data = load_dataset_from_drive('csv')
    if data is not None:
        results = train_model_greedy(
            model_type=MODEL,
            dataset_type='csv',
            data_dict=data,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
        )
        log_memory(f'after_{MODEL}_csv')
        del data  # release numpy arrays from RAM
        try:
            del results
        except Exception:
            pass
        aggressive_cleanup()
    else:
        print('Failed to load CSV data')
else:
    print('Skipping CSV dataset')

print(f'\n XGBOOST-LSTM on CSV DONE')
# ─── MODEL: Transformer on CSV (Greedy Adversarial) ────────────────────────
MODEL = 'transformer'
print(f'\n{"#"*80}')
print(f'  GREEDY ADVERSARIAL — TRANSFORMER on CSV')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_csv')

if DATASETS in ['csv', 'both']:
    data = load_dataset_from_drive('csv')
    if data is not None:
        results = train_model_greedy(
            model_type=MODEL,
            dataset_type='csv',
            data_dict=data,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
        )
        log_memory(f'after_{MODEL}_csv')
        del data  # release numpy arrays from RAM
        try:
            del results
        except Exception:
            pass
        aggressive_cleanup()
    else:
        print('Failed to load CSV data')
else:
    print('Skipping CSV dataset')

print(f'\n TRANSFORMER on CSV DONE')
# ─── MODEL: CNN-BiLSTM-Transformer on CSV (Greedy Adversarial) ────────────────────────
MODEL = 'cnn_bilstm_transformer'
print(f'\n{"#"*80}')
print(f'  GREEDY ADVERSARIAL — CNN-BILSTM-TRANSFORMER on CSV')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_csv')

if DATASETS in ['csv', 'both']:
    data = load_dataset_from_drive('csv')
    if data is not None:
        results = train_model_greedy(
            model_type=MODEL,
            dataset_type='csv',
            data_dict=data,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
        )
        log_memory(f'after_{MODEL}_csv')
        del data  # release numpy arrays from RAM
        try:
            del results
        except Exception:
            pass
        aggressive_cleanup()
    else:
        print('Failed to load CSV data')
else:
    print('Skipping CSV dataset')

print(f'\n CNN-BILSTM-TRANSFORMER on CSV DONE')
# ─── MODEL: NLP-CNN-BiLSTM-Transformer on CSV (Greedy Adversarial) ────────────────────────
MODEL = 'nlp_cnn_bilstm_transformer'
print(f'\n{"#"*80}')
print(f'  GREEDY ADVERSARIAL — NLP-CNN-BILSTM-TRANSFORMER on CSV')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_csv')

if DATASETS in ['csv', 'both']:
    data = load_dataset_from_drive('csv')
    if data is not None:
        results = train_model_greedy(
            model_type=MODEL,
            dataset_type='csv',
            data_dict=data,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
        )
        log_memory(f'after_{MODEL}_csv')
        del data  # release numpy arrays from RAM
        try:
            del results
        except Exception:
            pass
        aggressive_cleanup()
    else:
        print('Failed to load CSV data')
else:
    print('Skipping CSV dataset')

print(f'\n NLP-CNN-BILSTM-TRANSFORMER on CSV DONE')
# ─── MODEL: NLP-Transformer on CSV (Greedy Adversarial) ────────────────────────
MODEL = 'nlp_transformer'
print(f'\n{"#"*80}')
print(f'  GREEDY ADVERSARIAL — NLP-TRANSFORMER on CSV')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_csv')

if DATASETS in ['csv', 'both']:
    data = load_dataset_from_drive('csv')
    if data is not None:
        results = train_model_greedy(
            model_type=MODEL,
            dataset_type='csv',
            data_dict=data,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
        )
        log_memory(f'after_{MODEL}_csv')
        del data  # release numpy arrays from RAM
        try:
            del results
        except Exception:
            pass
        aggressive_cleanup()
    else:
        print('Failed to load CSV data')
else:
    print('Skipping CSV dataset')

print(f'\n NLP-TRANSFORMER on CSV DONE')
# ─── CLEANUP RAM BEFORE JSON PHASE ──────────────────────────────────────
print('\n' + '='*80)
print('  CLEANING RAM BEFORE JSON PHASE...')
print('='*80 + '\n')

aggressive_cleanup()
print('\n RAM cleaned. Ready for JSON phase.')
# ─── Cell: Load JSON Dataset ─────────────────────────────────────────────
if DATASETS in ['json', 'both']:
    json_data = load_dataset_from_drive('json')
else:
    json_data = None
    print('Skipping JSON dataset')
# ─── MODEL: LSTM on JSON (Greedy Adversarial) ────────────────────────
MODEL = 'lstm'
print(f'\n{"#"*80}')
print(f'  GREEDY ADVERSARIAL — LSTM on JSON')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_json')

if DATASETS in ['json', 'both']:
    data = load_dataset_from_drive('json')
    if data is not None:
        results = train_model_greedy(
            model_type=MODEL,
            dataset_type='json',
            data_dict=data,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
        )
        log_memory(f'after_{MODEL}_json')
        del data  # release numpy arrays from RAM
        try:
            del results
        except Exception:
            pass
        aggressive_cleanup()
    else:
        print('Failed to load JSON data')
else:
    print('Skipping JSON dataset')

print(f'\n LSTM on JSON DONE')
# ─── MODEL: BiLSTM on JSON (Greedy Adversarial) ────────────────────────
MODEL = 'bilstm'
print(f'\n{"#"*80}')
print(f'  GREEDY ADVERSARIAL — BILSTM on JSON')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_json')

if DATASETS in ['json', 'both']:
    data = load_dataset_from_drive('json')
    if data is not None:
        results = train_model_greedy(
            model_type=MODEL,
            dataset_type='json',
            data_dict=data,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
        )
        log_memory(f'after_{MODEL}_json')
        del data  # release numpy arrays from RAM
        try:
            del results
        except Exception:
            pass
        aggressive_cleanup()
    else:
        print('Failed to load JSON data')
else:
    print('Skipping JSON dataset')

print(f'\n BILSTM on JSON DONE')
# ─── MODEL: CNN-LSTM on JSON (Greedy Adversarial) ────────────────────────
MODEL = 'cnn_lstm'
print(f'\n{"#"*80}')
print(f'  GREEDY ADVERSARIAL — CNN-LSTM on JSON')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_json')

if DATASETS in ['json', 'both']:
    data = load_dataset_from_drive('json')
    if data is not None:
        results = train_model_greedy(
            model_type=MODEL,
            dataset_type='json',
            data_dict=data,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
        )
        log_memory(f'after_{MODEL}_json')
        del data  # release numpy arrays from RAM
        try:
            del results
        except Exception:
            pass
        aggressive_cleanup()
    else:
        print('Failed to load JSON data')
else:
    print('Skipping JSON dataset')

print(f'\n CNN-LSTM on JSON DONE')
# ─── MODEL: XGBoost-LSTM on JSON (Greedy Adversarial) ────────────────────────
MODEL = 'xgboost_lstm'
print(f'\n{"#"*80}')
print(f'  GREEDY ADVERSARIAL — XGBOOST-LSTM on JSON')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_json')

if DATASETS in ['json', 'both']:
    data = load_dataset_from_drive('json')
    if data is not None:
        results = train_model_greedy(
            model_type=MODEL,
            dataset_type='json',
            data_dict=data,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
        )
        log_memory(f'after_{MODEL}_json')
        del data  # release numpy arrays from RAM
        try:
            del results
        except Exception:
            pass
        aggressive_cleanup()
    else:
        print('Failed to load JSON data')
else:
    print('Skipping JSON dataset')

print(f'\n XGBOOST-LSTM on JSON DONE')
# ─── MODEL: Transformer on JSON (Greedy Adversarial) ────────────────────────
MODEL = 'transformer'
print(f'\n{"#"*80}')
print(f'  GREEDY ADVERSARIAL — TRANSFORMER on JSON')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_json')

if DATASETS in ['json', 'both']:
    data = load_dataset_from_drive('json')
    if data is not None:
        results = train_model_greedy(
            model_type=MODEL,
            dataset_type='json',
            data_dict=data,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
        )
        log_memory(f'after_{MODEL}_json')
        del data  # release numpy arrays from RAM
        try:
            del results
        except Exception:
            pass
        aggressive_cleanup()
    else:
        print('Failed to load JSON data')
else:
    print('Skipping JSON dataset')

print(f'\n TRANSFORMER on JSON DONE')
# ─── MODEL: CNN-BiLSTM-Transformer on JSON (Greedy Adversarial) ────────────────────────
MODEL = 'cnn_bilstm_transformer'
print(f'\n{"#"*80}')
print(f'  GREEDY ADVERSARIAL — CNN-BILSTM-TRANSFORMER on JSON')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_json')

if DATASETS in ['json', 'both']:
    data = load_dataset_from_drive('json')
    if data is not None:
        results = train_model_greedy(
            model_type=MODEL,
            dataset_type='json',
            data_dict=data,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
        )
        log_memory(f'after_{MODEL}_json')
        del data  # release numpy arrays from RAM
        try:
            del results
        except Exception:
            pass
        aggressive_cleanup()
    else:
        print('Failed to load JSON data')
else:
    print('Skipping JSON dataset')

print(f'\n CNN-BILSTM-TRANSFORMER on JSON DONE')
# ─── MODEL: NLP-CNN-BiLSTM-Transformer on JSON (Greedy Adversarial) ────────────────────────
MODEL = 'nlp_cnn_bilstm_transformer'
print(f'\n{"#"*80}')
print(f'  GREEDY ADVERSARIAL — NLP-CNN-BILSTM-TRANSFORMER on JSON')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_json')

if DATASETS in ['json', 'both']:
    data = load_dataset_from_drive('json')
    if data is not None:
        results = train_model_greedy(
            model_type=MODEL,
            dataset_type='json',
            data_dict=data,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
        )
        log_memory(f'after_{MODEL}_json')
        del data  # release numpy arrays from RAM
        try:
            del results
        except Exception:
            pass
        aggressive_cleanup()
    else:
        print('Failed to load JSON data')
else:
    print('Skipping JSON dataset')

print(f'\n NLP-CNN-BILSTM-TRANSFORMER on JSON DONE')
# ─── MODEL: NLP-Transformer on JSON (Greedy Adversarial) ────────────────────────
MODEL = 'nlp_transformer'
print(f'\n{"#"*80}')
print(f'  GREEDY ADVERSARIAL — NLP-TRANSFORMER on JSON')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_json')

if DATASETS in ['json', 'both']:
    data = load_dataset_from_drive('json')
    if data is not None:
        results = train_model_greedy(
            model_type=MODEL,
            dataset_type='json',
            data_dict=data,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
        )
        log_memory(f'after_{MODEL}_json')
        del data  # release numpy arrays from RAM
        try:
            del results
        except Exception:
            pass
        aggressive_cleanup()
    else:
        print('Failed to load JSON data')
else:
    print('Skipping JSON dataset')

print(f'\n NLP-TRANSFORMER on JSON DONE')
# ─── Cell: Visualize Results ─────────────────────────────────────────────
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path(DRIVE_RESULTS_DIR) / 'models'
if not results_dir.exists():
    print('No results found yet — run the training cells first.')
else:
    greedy_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and 'greedy' in d.name])

    if not greedy_dirs:
        print('No greedy results found yet.')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        for idx, dataset in enumerate(['csv', 'json']):
            ax = axes[idx]
            ds_dirs = [d for d in greedy_dirs if dataset in d.name]
            models = []
            clean_accs = []
            adv_k1 = []
            adv_k2 = []
            adv_k3 = []
            adv_k4 = []

            for d in sorted(ds_dirs):
                rf = d / 'greedy_results.json'
                if not rf.exists():
                    continue
                with open(rf) as f:
                    res = json.load(f)

                model_name = d.name.replace(f'_greedy_{dataset}', '').upper()
                models.append(model_name)
                clean_accs.append(res.get('clean_accuracy', 0))
                adv_k1.append(res.get('adversarial_accuracies', {}).get('k1', 0))
                adv_k2.append(res.get('adversarial_accuracies', {}).get('k2', 0))
                adv_k3.append(res.get('adversarial_accuracies', {}).get('k3', 0))
                adv_k4.append(res.get('adversarial_accuracies', {}).get('k4', 0))

            if not models:
                ax.set_title(f'{dataset.upper()} — No results yet')
                continue

            x = np.arange(len(models))
            width = 0.15

            ax.bar(x - 2*width, clean_accs, width, label='Clean', color='#2ecc71')
            ax.bar(x - width, adv_k1, width, label='Adv k=1', color='#3498db')
            ax.bar(x, adv_k2, width, label='Adv k=2', color='#e67e22')
            ax.bar(x + width, adv_k3, width, label='Adv k=3', color='#e74c3c')
            ax.bar(x + 2*width, adv_k4, width, label='Adv k=4', color='#9b59b6')

            ax.set_ylabel('Accuracy')
            ax.set_title(f'{dataset.upper()} — Greedy Adversarial Results')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend()
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        save_path = f'{DRIVE_RESULTS_DIR}/greedy_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f'Plot saved to {save_path}')

        # Phase progression plot per model
        for d in greedy_dirs:
            rf = d / 'greedy_results.json'
            if not rf.exists():
                continue
            with open(rf) as f:
                res = json.load(f)

            ct = res.get('crash_tests', {})
            if not ct:
                continue

            fig, ax = plt.subplots(figsize=(8, 5))
            phases = sorted(ct.keys())
            clean_vals = [ct[p].get('clean', 0) for p in phases]
            ax.plot(phases, clean_vals, 'o-', label='Clean', color='#2ecc71', linewidth=2)
            for k in [1, 2, 3, 4]:
                k_key = f'adv_k{k}'
                vals = [ct[p].get(k_key, None) for p in phases]
                if any(v is not None for v in vals):
                    vals = [v if v is not None else 0 for v in vals]
                    ax.plot(phases, vals, 'o--', label=f'Adv k={k}', alpha=0.7)

            ax.set_title(f"{d.name} — Accuracy by Phase")
            ax.set_xlabel('Phase')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 1.05)
            plt.tight_layout()
            plt.show()
# ─── Cell: Comparative Summary Table ─────────────────────────────────────
import json
from pathlib import Path

results_dir = Path(DRIVE_RESULTS_DIR) / 'models'
if not results_dir.exists():
    print('No results found.')
else:
    greedy_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and 'greedy' in d.name])

    print(f"{'Model + Dataset':<40} {'Clean':>8} {'k=1':>8} {'k=2':>8} {'k=3':>8} {'k=4':>8}")
    print('-' * 80)

    for d in greedy_dirs:
        rf = d / 'greedy_results.json'
        if not rf.exists():
            continue
        with open(rf) as f:
            res = json.load(f)

        name = d.name
        clean = res.get('clean_accuracy', 0)
        adv = res.get('adversarial_accuracies', {})
        k1 = adv.get('k1', 0)
        k2 = adv.get('k2', 0)
        k3 = adv.get('k3', 0)
        k4 = adv.get('k4', 0)

        print(f"{name:<40} {clean:>8.4f} {k1:>8.4f} {k2:>8.4f} {k3:>8.4f} {k4:>8.4f}")

    print('-' * 80)

    # Crash test summary
    print(f"\n{'Model + Dataset':<40} {'Phase':>8} {'Clean':>8} {'k=1':>8} {'k=2':>8} {'k=3':>8} {'k=4':>8}")
    print('-' * 100)
    for d in greedy_dirs:
        rf = d / 'greedy_results.json'
        if not rf.exists():
            continue
        with open(rf) as f:
            res = json.load(f)

        name = d.name
        ct = res.get('crash_tests', {})
        for phase_key in sorted(ct.keys()):
            phase_data = ct[phase_key]
            clean = phase_data.get('clean', 0)
            k1 = phase_data.get('adv_k1', 0)
            k2 = phase_data.get('adv_k2', 0)
            k3 = phase_data.get('adv_k3', 0)
            k4 = phase_data.get('adv_k4', 0)
            print(f"{name:<40} {phase_key:>8} {clean:>8.4f} {k1:>8.4f} {k2:>8.4f} {k3:>8.4f} {k4:>8.4f}")
    print('-' * 100)
    print('\n Comparison complete.')
# ─── Git Push ──────────────────────────────────────────────────────────────
import subprocess

print('Pushing to GitHub...')
result = subprocess.run(['git', 'add', '-A'], capture_output=True, text=True, cwd='/content/pfe')
print(f'  git add: {result.returncode}')

result = subprocess.run(['git', 'commit', '-m', 'Update greedy adversarial training results'], capture_output=True, text=True, cwd='/content/pfe')
if result.returncode == 0:
    print(f'  git commit: OK')
else:
    print(f'  git commit: {result.stdout.strip()} {result.stderr.strip()}')

result = subprocess.run(['git', 'push'], capture_output=True, text=True, cwd='/content/pfe')
if result.returncode == 0:
    print('  Push successful!')
else:
    print(f'  git push stderr: {result.stderr.strip()[:500]}')