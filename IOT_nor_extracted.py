# ─── Cell 1: Setup ───────────────────────────────────────────────────────────
# Mount Google Drive FIRST so the paths in Cell 2 resolve correctly
from google.colab import drive
drive.mount('/content/drive')

# Clone the repository (pull latest if already cloned)
import os
if os.path.exists('/content/pfe'):
    !cd /content/pfe && git pull
else:
    !git clone https://github.com/yacinemkk/pfe.git /content/pfe

%cd /content/pfe

# Install dependencies
!pip install -q torch torchvision tqdm numpy pandas scikit-learn matplotlib xgboost psutil
# ─── Cell 2: Configure ───────────────────────────────────────────────────────
# ⚠️  UPDATE THESE PATHS before pressing Run All.

# Path to your JSON IPFIX files on Google Drive.
# Expected structure:
#   JSON_DATA_DIR/
#     20-01-31(1)/ipfix_202001_fixed.json
#     20-03-31/ipfix_202003.json
#     20-04-30/ipfix_202004.json
JSON_DATA_DIR = '/content/drive/MyDrive/PFE/IPFIX_Records'

# Path to your CSV IPFIX ML Instances on Google Drive.
# Expected structure:
#   CSV_DATA_DIR/
#     home1_labeled.csv
#     home2_labeled.csv
#     ...
CSV_DATA_DIR = '/content/drive/MyDrive/PFE/IPFIX_ML_Instances'

# Where to save trained models, results.json, history.json, and plots.
# This folder is on Drive — it persists after the runtime disconnects.
DRIVE_RESULTS_DIR = '/content/drive/MyDrive/PFE/results'

# Datasets to train on: 'csv', 'json', or 'both'
DATASETS = 'both'  # Options: 'csv', 'json', 'both'

# ─── Training hyperparameters — Curriculum 4 Phases (65 epochs total) ─────────
SEQ_LENGTH   = 10      # Sequence length (timesteps per sample)
# Curriculum phases (epoch boundaries, 1-indexed):
#   Phase 0 — Fondation       : epochs  1-15
#   Phase 1 — Robustesse Douce: epochs 16-35
#   Phase 2 — Robustesse Forte: epochs 36-55
#   Phase 3 — Consolidation   : epochs 56-65
TOTAL_EPOCHS = 65
BATCH_SIZE   = 64      # Batch size
LEARNING_RATE = 1e-3   # Learning rate
MAX_FILES    = None    # Max CSV files to load (None = all). Use small int for testing.
MAX_RECORDS  = None    # Max JSON records to load (None = all). Use small int for testing.
ADV_METHOD   = 'hybrid'
ADV_RATIO    = 0.4

# ─── Couches défensives permanentes ────────────────────────────────────────────
USE_INPUT_DEFENSE = True   # InputDefenseLayer actif depuis Phase 0 (clip ±3.5σ + lissage)
USE_AFD           = True   # Feature Dropout (p variable par phase)
USE_RS            = True   # Randomized Smoothing (σ variable par phase)
USE_CUTMIX        = True   # Adversarial CutMix (prob variable par phase)
USE_MULTI_ATTACK  = True   # Worst-of-K (k variable par phase)
USE_IBP           = True   # IBP — certifié (actif dès Phase 1)
MULTI_ATTACK_STRATEGIES = ['Zero', 'Mimic_Mean', 'Mimic_95th', 'Padding_x10']

# ─── IBP (Interval Bound Propagation) ──────────────────────────────────────────
IBP_EPSILON       = 0.10    # IBP ε final
LAMBDA_IBP        = 1.0     # Poids loss IBP
IBP_METHOD        = 'crown' # Méthode IBP: 'crown' (précis) ou 'ibp' (rapide)
IBP_EPSILON_START = 0.05    # ε de démarrage du warmup IBP
IBP_WARMUP_EPOCHS = 5       # Epochs de warmup IBP (Phase 1)

# ─── TRADES ────────────────────────────────────────────────────────────────────
# β et ε sont gérés automatiquement par get_phase_config(epoch)
TRADES_PGD_STEPS = 10      # Steps PGD TRADES (gardé fixe)

# ─── RS (évaluation certifiée) ─────────────────────────────────────────────────
RS_SIGMA     = 0.25
RS_N_SAMPLES = 50



# ─── RAM Optimization ─────────────────────────────────────────────────────────
STRIDE = 10             # Larger stride = fewer sequences = less RAM
EVAL_SUBSAMPLE = 1000  # Max samples for adversarial eval
EVAL_BATCH_SIZE = 32   # Smaller batch for evaluation (saves RAM)
LEARNING_RATE = 1e-3   # Learning rate

# Create results directory if needed
os.makedirs(DRIVE_RESULTS_DIR, exist_ok=True)

print(f'CSV data:     {CSV_DATA_DIR}')
print(f'JSON data:    {JSON_DATA_DIR}')
print(f'Results dir:  {DRIVE_RESULTS_DIR}')
print(f'Datasets:     {DATASETS}')
print(f'Seq length:   {SEQ_LENGTH}')
print(f'Epochs:       {TOTAL_EPOCHS} (curriculum 4 phases: 0→15→35→55→65)')
print(f'Batch size:   {BATCH_SIZE}')
print(f'Adv method:   {ADV_METHOD}')
print(f'Adv ratio:    {ADV_RATIO}')

# Verify CSV data exists
import glob
csv_files = glob.glob(f'{CSV_DATA_DIR}/home*_labeled.csv')
print(f'\nFound {len(csv_files)} CSV file(s):')
for f in sorted(csv_files)[:5]:
    size_mb = os.path.getsize(f) / (1024**2)
    print(f'  {os.path.basename(f)} ({size_mb:.1f} MB)')
if len(csv_files) > 5:
    print(f'  ... and {len(csv_files) - 5} more')

# Verify JSON data exists
json_files = glob.glob(f'{JSON_DATA_DIR}/**/*.json', recursive=True)
print(f'\nFound {len(json_files)} JSON file(s):')
for f in json_files:
    size_gb = os.path.getsize(f) / (1024**3)
    print(f'  {os.path.basename(f)} ({size_gb:.1f} GB)')
# ─── RAM Monitoring & Cleanup Utilities ───────────────────────────────────────
import gc
import psutil
import torch
import os

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
    """Charge COMPLÈTEMENT le dataset CSV, affiche les infos, et sauvegarde sur Drive."""
    import sys
    import gc
    import numpy as np
    import pickle

    sys.path.insert(0, '/content/pfe')

    print("\n" + "=" * 70)
    print("  CHARGEMENT COMPLET DU DATASET CSV")
    print("=" * 70)

    print(f"\n  Répertoire : {csv_data_dir}")
    print(f"  Seq length : {seq_length} | Stride : {stride}")

    csv_files = sorted(glob.glob(f'{csv_data_dir}/home*_labeled.csv'))
    print(f"\n  Fichiers CSV trouvés : {len(csv_files)}")

    for f in csv_files:
        size_mb = os.path.getsize(f) / (1024**2)
        print(f"    {os.path.basename(f):<30s} : {size_mb:>10.1f} MB")

    total_gb = sum(os.path.getsize(f) for f in csv_files) / (1024**3)
    print(f"\n  Taille totale : {total_gb:.2f} GB")

    # Charger le dataset complet via le vrai pipeline CSV
    print("\n  Chargement via le pipeline CSV (IoTDataProcessor)...")
    from src.data.preprocessor import IoTDataProcessor

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

    print(f"\n  {'='*70}")
    print(f"  RÉSULTAT DU CHARGEMENT")
    print(f"  {'='*70}")
    print(f"    Features ({n_continuous}) : {features[:5]}...")
    print(f"    Classes ({len(label_encoder.classes_)}) : {list(label_encoder.classes_)}")
    print(f"\n  Shapes des séquences (seq_length={seq_length}, stride={stride}) :")
    print(f"    Train : {X_train.shape}  →  {len(X_train):,} séquences")
    print(f"    Val   : {X_val.shape}  →  {len(X_val):,} séquences")
    print(f"    Test  : {X_test.shape}  →  {len(X_test):,} séquences")
    print(f"    Total : {len(X_train) + len(X_val) + len(X_test):,} séquences")

    print(f"\n  Distribution des classes (train) :")
    for cls in label_encoder.classes_:
        cls_id = label_encoder.transform([cls])[0]
        count = int(np.sum(y_train == cls_id))
        bar = '█' * max(1, count // 50)
        print(f"    {cls:<30s} : {count:>6,}  {bar}")

    # ─── Sauvegarder sur Drive ────────────────────────────────────────────
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n  💾 Sauvegarde du dataset CSV pré-traité sur Drive...")
        print(f"     Répertoire : {save_dir}")

        np.save(f'{save_dir}/X_train.npy', X_train)
        np.save(f'{save_dir}/X_val.npy', X_val)
        np.save(f'{save_dir}/X_test.npy', X_test)
        np.save(f'{save_dir}/y_train.npy', y_train)
        np.save(f'{save_dir}/y_val.npy', y_val)
        np.save(f'{save_dir}/y_test.npy', y_test)

        with open(f'{save_dir}/csv_metadata.pkl', 'wb') as f:
            pickle.dump({
                'features': features,
                'scaler': scaler,
                'label_encoder': label_encoder,
                'n_continuous': n_continuous,
                'seq_length': seq_length,
                'stride': stride,
            }, f)

        # Marker file to indicate preprocessing is complete
        with open(f'{save_dir}/csv_ready', 'w') as f:
            f.write('ready')

        saved_gb = (X_train.nbytes + X_val.nbytes + X_test.nbytes +
                    y_train.nbytes + y_val.nbytes + y_test.nbytes) / (1024**3)
        print(f"  ✅ Dataset CSV sauvegardé ({saved_gb:.2f} GB)")
        print(f"     Fichiers : X_train, X_val, X_test, y_train, y_val, y_test, csv_metadata.pkl")

    print(f"\n  {'='*70}")
    print(f"  ✅ Dataset CSV chargé complètement en RAM")
    print(f"  {'='*70}\n")

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'features': features, 'scaler': scaler,
        'label_encoder': label_encoder, 'n_continuous': n_continuous
    }

# ─── CSV preprocessing directory on Drive ─────────────────────────────
CSV_PREPROCESSED_DIR = f'{DRIVE_RESULTS_DIR}/preprocessed/csv'

if DATASETS in ['csv', 'both']:
    csv_data = load_and_display_csv_dataset(
        CSV_DATA_DIR, seq_length=SEQ_LENGTH, stride=STRIDE,
        save_dir=CSV_PREPROCESSED_DIR
    )
else:
    csv_data = None
    print('Skipping CSV dataset loading — DATASETS is not csv or both')

# ─── Training with Countermeasures & Data Loading Helpers ────────────────────────
import sys
sys.path.insert(0, '/content/pfe')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import gc
import os
import json
import pickle

from src.adversarial.trades import TRADESAttack, FeatureAttackGenerator, MultiAttackTRADES
from src.adversarial.input_transform import create_input_transform
from src.adversarial.cutmix import create_adversarial_cutmix
from src.adversarial.feature_dropout import AdversarialFeatureDropout, create_adversarial_feature_dropout
from src.adversarial.attacks import SensitivityAnalysis, AdversarialSearch
from src.adversarial.ibp import IntervalBoundPropagation, IBPTrainer
from src.adversarial.robust_losses import (
    InputDefenseLayer, AFDLoss, feature_diversity_loss,
    get_phase_config, combined_loss_normalized,
    AdversarialEarlyStopping, compute_feature_stats,
)
from src.models.lstm import LSTMClassifier
from src.models.bilstm import BiLSTMClassifier
from src.models.cnn_lstm import CNNLSTMClassifier
from src.models.xgboost_lstm import XGBoostLSTMClassifier
from src.models.transformer import TransformerClassifier
from src.models.cnn_bilstm_transformer import CNNBiLSTMTransformerClassifier
from src.training.trainer import IoTSequenceDataset
from config.config import LSTM_CONFIG, TRANSFORMER_CONFIG, CNN_BILSTM_TRANSFORMER_CONFIG, LEARNING_RATE, WEIGHT_DECAY


def load_dataset_from_drive(dataset_type):
    """Load preprocessed data from Drive or fresh loading."""
    import numpy as np
    import pickle
    import os

    preprocessed_dir = f'{DRIVE_RESULTS_DIR}/preprocessed/{dataset_type}'
    ready_file = f'{preprocessed_dir}/{dataset_type}_ready'

    if os.path.exists(ready_file) and os.path.exists(f'{preprocessed_dir}/X_train.npy'):
        print(f'  📂 Loading preprocessed {dataset_type.upper()} data from Drive...')
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
        print(f'  ✅ Preprocessed {dataset_type.upper()} data loaded: {len(X_train):,} train samples')
        return data
    else:
        print(f'  📂 No preprocessed data found. Loading {dataset_type.upper()} fresh...')
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


def create_model(model_type, input_size, num_classes):
    """Create model instance based on model_type."""
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
        return CNNBiLSTMTransformerClassifier(input_size, num_classes, seq_length=SEQ_LENGTH)
    else:
        raise ValueError(f"Model type {model_type} not supported")


# =============================================================================
# BOUCLE UNIFIÉE PAR EPOCH — Curriculum 4 Phases
# =============================================================================

def _crash_test(model, val_loader, device, epoch, pgd_attack,
                active_attack, is_multi_attack, label='', input_defense=None):
    """Crash test rapide : clean + PGD + WoK sur ~10 batches."""
    model.eval()
    # 1. Clean accuracy
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            if input_defense is not None:
                X = input_defense(X)
            out = model(X)
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    clean_acc = correct / max(total, 1)

    # 2. PGD accuracy (10 batches)
    pgd_correct, pgd_total, n = 0, 0, 0
    for X, y in val_loader:
        if n >= 10:
            break
        X, y = X.to(device), y.to(device)
        if input_defense is not None:
            X = input_defense(X)
        with torch.no_grad():
            X_adv = pgd_attack.generate(model, X, y, device)
            out = model(X_adv)
        _, pred = out.max(1)
        pgd_total += y.size(0)
        pgd_correct += pred.eq(y).sum().item()
        n += 1
    pgd_acc = pgd_correct / max(pgd_total, 1)

    # 3. WoK accuracy (10 batches)
    wok_acc = pgd_acc
    if active_attack is not None and is_multi_attack:
        wok_correct, wok_total, n = 0, 0, 0
        for X, y in val_loader:
            if n >= 10:
                break
            X, y = X.to(device), y.to(device)
            if input_defense is not None:
                X = input_defense(X)
            with torch.no_grad():
                result = active_attack.generate(model, X, y, device)
                X_adv = result[0] if isinstance(result, tuple) else result
                out = model(X_adv)
            _, pred = out.max(1)
            wok_total += y.size(0)
            wok_correct += pred.eq(y).sum().item()
            n += 1
        wok_acc = wok_correct / max(wok_total, 1)

    rr_pgd = pgd_acc / max(clean_acc, 1e-8)
    rr_wok = wok_acc / max(clean_acc, 1e-8)
    print(f"  [Crash Test Epoch {epoch} {label}] "
          f"Clean={clean_acc:.4f} | PGD={pgd_acc:.4f}(RR={rr_pgd:.3f}) "
          f"| WoK={wok_acc:.4f}(RR={rr_wok:.3f})")
    return {'clean': clean_acc, 'pgd': pgd_acc, 'wok': wok_acc,
            'rr_pgd': rr_pgd, 'rr_wok': rr_wok}


def train_epoch_robust(
    epoch, model, train_loader, optimizer, device,
    # Components
    input_defense, afd_loss, feature_stats,
    ibp_trainer, active_attack, pgd_attack,
    # Phase config (from get_phase_config)
    phase_cfg,
    # Flags
    use_cutmix=True, use_ibp=True,
):
    """
    Boucle d'entraînement unifiée — pilotée par phase_cfg.
    Respecte le curriculum 4-phases du plan d'implémentation.
    """
    model.train()
    phase = phase_cfg['phase']
    epsilon = phase_cfg['epsilon']
    trades_beta = phase_cfg['trades_beta']
    trades_steps = phase_cfg['trades_steps']
    afd_lambda = phase_cfg['afd_lambda']
    div_lambda = phase_cfg['diversity_lambda']
    label_smoothing = phase_cfg['label_smoothing']
    worst_k = phase_cfg['worst_k']
    cutmix_prob = phase_cfg['cutmix_prob']
    # Randomized Smoothing sigma par phase
    rs_sigma_map = {0: 0.0, 1: 0.10, 2: 0.25, 3: 0.25}
    rs_sigma = rs_sigma_map.get(phase, 0.0)
    # Feature Dropout prob par phase
    fd_prob_map = {0: 0.10, 1: 0.20, 2: 0.30, 3: 0.20}
    fd_prob = fd_prob_map.get(phase, 0.0)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    is_multi = isinstance(active_attack, MultiAttackTRADES)
    adv_cutmix = create_adversarial_cutmix(alpha=1.0, prob=cutmix_prob) if use_cutmix and cutmix_prob > 0 else None
    cutmix_available = create_adversarial_cutmix is not None

    # IBP warmup
    if ibp_trainer is not None:
        ibp_trainer.set_epoch(epoch)

    # Metric accumulators
    total_loss = ce_loss_acc = trades_loss_acc = afd_loss_acc = ibp_loss_acc = div_loss_acc = 0.0
    correct = total_samples = 0
    n_batches = 0

    from tqdm.auto import tqdm
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # ── Couche 1: InputDefenseLayer (permanent) ──────────────────────────
        if input_defense is not None:
            X_batch = input_defense(X_batch)

        # ── Couche 2: Randomized Smoothing (Phase 1+) ────────────────────────
        if rs_sigma > 0:
            X_batch = X_batch + rs_sigma * torch.randn_like(X_batch)

        # ── Couche 3: Feature Dropout (toutes phases, p variable) ───────────
        if fd_prob > 0:
            mask = (torch.rand(X_batch.shape[0], 1, X_batch.shape[2],
                               device=device) > fd_prob).float()
            X_batch = X_batch * mask

        # ── Phase 0 : entraînement propre (pas d'attaques) ──────────────────
        if phase == 0:
            optimizer.zero_grad()
            # Forward clean
            clean_logits = model(X_batch)
            # CE loss
            loss_ce = criterion(clean_logits, y_batch)

            # AFD loss (latent space)
            with torch.no_grad():
                feat_clean = X_batch.detach()  # use input as proxy if no encoder
                feat_adv = feat_clean  # no attack in phase 0
            loss_afd = torch.tensor(0.0, device=device)
            if afd_loss is not None:
                try:
                    loss_afd = afd_loss(feat_clean, feat_adv, y_batch)
                except Exception:
                    pass

            # Feature Diversity loss
            loss_div = torch.tensor(0.0, device=device)
            try:
                loss_div = feature_diversity_loss(model, X_batch, y_batch, top_k=5)
            except Exception:
                pass

            losses = {'ce': loss_ce, 'afd': loss_afd, 'div': loss_div}
            weights = {'ce': 1.0, 'afd': afd_lambda, 'div': div_lambda}
            loss = combined_loss_normalized(losses, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track
            ce_loss_acc += loss_ce.item()
            afd_loss_acc += loss_afd.item() if hasattr(loss_afd, 'item') else float(loss_afd)
            div_loss_acc += loss_div.item() if hasattr(loss_div, 'item') else float(loss_div)

        # ── Phases 1-3 : entraînement adversarial ──────────────────────────
        else:
            # Générer k aléatoire selon phase
            if phase == 1:
                k = np.random.choice([1, 2])
            elif phase == 2:
                k = np.random.randint(1, 6)  # 1..5
            else:  # phase 3
                k = np.random.choice([3, 4, 5])

            # Générer l'exemple adversarial
            with torch.no_grad():
                result = active_attack.generate(model, X_batch, y_batch, device)
                if is_multi:
                    X_adv, _ = result
                else:
                    X_adv = result

            # CutMix adversarial/propre
            if adv_cutmix is not None and np.random.rand() < cutmix_prob:
                X_adv, y_batch = adv_cutmix(X_batch, X_adv, y_batch)

            optimizer.zero_grad()

            # Forward clean
            clean_logits = model(X_batch)
            loss_ce = criterion(clean_logits, y_batch)

            # Forward adversarial
            adv_logits = model(X_adv)

            # TRADES: KL(clean || adv)
            clean_probs = F.softmax(clean_logits, dim=1).detach()
            adv_log_probs = F.log_softmax(adv_logits, dim=1)
            loss_trades = F.kl_div(adv_log_probs, clean_probs, reduction='batchmean')

            # CE sur adversarial (Worst-of-K)
            loss_adv_ce = criterion(adv_logits, y_batch)

            # AFD loss
            with torch.no_grad():
                feat_clean = X_batch
                feat_adv_proxy = X_adv
            loss_afd = torch.tensor(0.0, device=device)
            if afd_loss is not None:
                try:
                    loss_afd = afd_loss(feat_clean, feat_adv_proxy, y_batch)
                except Exception:
                    pass

            # Feature Diversity loss
            loss_div = torch.tensor(0.0, device=device)
            try:
                loss_div = feature_diversity_loss(model, X_batch, y_batch, top_k=5)
            except Exception:
                pass

            # IBP certified loss (Phase 1+)
            loss_ibp = torch.tensor(0.0, device=device)
            if use_ibp and ibp_trainer is not None:
                try:
                    loss_ibp, _ = ibp_trainer.compute_loss(X_batch, y_batch)
                except Exception:
                    pass

            # Combined normalized loss
            losses  = {'ce': loss_ce, 'trades': loss_trades,
                       'adv_ce': loss_adv_ce, 'afd': loss_afd,
                       'ibp': loss_ibp, 'div': loss_div}
            weights = {'ce': 1.0, 'trades': trades_beta,
                       'adv_ce': 1.5, 'afd': afd_lambda,
                       'ibp': 1.0 if phase >= 1 else 0.0,
                       'div': div_lambda}
            loss = combined_loss_normalized(losses, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track
            ce_loss_acc += loss_ce.item()
            trades_loss_acc += loss_trades.item()
            afd_loss_acc += loss_afd.item() if hasattr(loss_afd, 'item') else float(loss_afd)
            ibp_loss_acc  += loss_ibp.item()  if hasattr(loss_ibp, 'item')  else float(loss_ibp)
            div_loss_acc  += loss_div.item()  if hasattr(loss_div, 'item')  else float(loss_div)

            del X_adv

        total_loss += loss.item()
        with torch.no_grad():
            _, predicted = model(X_batch).max(1)
        total_samples += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()
        n_batches += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    nb = max(n_batches, 1)
    return {
        'loss': total_loss / nb,
        'acc': correct / max(total_samples, 1),
        'ce': ce_loss_acc / nb,
        'trades': trades_loss_acc / nb,
        'afd': afd_loss_acc / nb,
        'ibp': ibp_loss_acc / nb,
        'div': div_loss_acc / nb,
    }


def train_model_with_countermeasures(
    model_type,
    dataset_type,
    data_dict=None,
    pgd_steps=10,
    batch_size=64,
    lr=1e-3,
    use_cutmix=True,
    use_afd=True,
    use_randomized_smoothing=True,
    rs_sigma=0.25,
    rs_n_samples=50,
    use_multi_attack=True,
    multi_attack_strategies=None,
    save_dir=None,
    use_class_weights=True,
    use_ibp=True,
    ibp_epsilon=0.1,
    lambda_ibp=1.0,
    ibp_method='crown',
    ibp_warmup_epochs=5,
    ibp_epsilon_start=0.05,
    total_epochs=65,
    use_input_defense=True,
):
    """
    Entraîne un modèle avec le curriculum 4-phases (docs/Implementation Plan).

    Curriculum :
      Phase 0 — Fondation (epochs  1-15) : CE + AFD + Diversity, sans attaques
      Phase 1 — Robustesse Douce  (16-35): CE + TRADES(β=1, ε↗) + WoK(k=1-2) + IBP warmup
      Phase 2 — Robustesse Forte  (36-55): CE + TRADES(β=2, ε↗) + WoK(k=1-5) + IBP fixe
      Phase 3 — Consolidation     (56-65): ε gelé + WoK(k=3-5) + affinage

    8 couches défensives actives selon la phase :
      1. InputDefenseLayer (permanent)        5. TRADES
      2. Feature Dropout (p variable)         6. IBP
      3. Randomized Smoothing (σ variable)    7. CutMix
      4. AFD Loss (λ décroissant)             8. Feature Diversity Loss
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"  CURRICULUM 4 PHASES — ENTRAÎNEMENT ROBUSTE")
    print(f"{'='*80}")
    print(f"  Model: {model_type.upper()}")
    print(f"  Dataset: {dataset_type.upper()}")
    print(f"  Device: {device}")
    print(f"  Total epochs: {total_epochs} (phases 0→15→35→55→{total_epochs})")
    print(f"  Couche 1 - InputDefenseLayer: {use_input_defense}")
    print(f"  Couche 3,4 - AFD + Feature Diversity: {use_afd}")
    print(f"  Couche 5 - TRADES Worst-of-K: {use_multi_attack}")
    if use_multi_attack and multi_attack_strategies:
        print(f"    Strategies: {multi_attack_strategies}")
    print(f"  Couche 6 - IBP: {use_ibp} (ε {ibp_epsilon_start}→{ibp_epsilon}, méthode={ibp_method})")
    print(f"  Couche 7 - CutMix: {use_cutmix}")
    print(f"  Couche 2 - Randomized Smoothing: {use_randomized_smoothing} (σ={rs_sigma})")
    print(f"{'='*80}\n")

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

    print(f"  Input size: {input_size}")
    print(f"  Num classes: {num_classes}")
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Val samples: {len(X_val):,}")
    print(f"  Test samples: {len(X_test):,}")

    train_dataset = IoTSequenceDataset(X_train, y_train)
    val_dataset = IoTSequenceDataset(X_val, y_val)
    test_dataset = IoTSequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = create_model(model_type, input_size, num_classes)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # ── Sensitivity Analysis (for AFD + feature attacks) ──
    print(f"\n  Computing sensitivity analysis...")
    n_sens = min(5000, len(X_train))
    sens_indices = np.random.choice(len(X_train), n_sens, replace=False)
    sensitivity_analysis = SensitivityAnalysis(
        X_train.reshape(-1, X_train.shape[-1])[:n_sens*10],
        np.repeat(y_train[sens_indices], X_train.shape[1])[:n_sens*10],
        features if features else [f'f{i}' for i in range(input_size)],
        num_classes,
        n_continuous_features=n_continuous,
    )

    # ── Initialisation des composants défensifs ──────────────────────────────
    print(f"\n{'─'*80}")
    print(f"  INITIALISATION — Composants Défensifs (8 couches)")
    print(f"{'─'*80}\n")

    # Couche 1: InputDefenseLayer (permanent)
    input_defense = InputDefenseLayer(clip_min=-3.5, clip_max=3.5, smooth_alpha=0.25).to(device) \
        if use_input_defense else None
    if input_defense:
        print(f"  ✅ Couche 1: InputDefenseLayer actif (clip ±3.5, lissage α=0.25)")

    # AFD Loss — compacité intra-classe + séparation inter-classe
    afd_loss_module = None
    if use_afd:
        try:
            afd_loss_module = AFDLoss(
                num_classes=num_classes,
                feature_dim=input_size,
                lambda_intra=1.0, lambda_inter=0.5
            ).to(device)
            print(f"  ✅ AFD Loss: num_classes={num_classes}, feature_dim={input_size}")
        except Exception as e:
            print(f"  ⚠️ AFD Loss init error: {e}")

    # Worst-of-K Multi-Attack TRADES
    pgd_attack_base = TRADESAttack(epsilon=0.05, alpha=0.005, num_steps=pgd_steps)
    active_attack = pgd_attack_base
    if use_multi_attack and sensitivity_analysis is not None:
        feature_attack = FeatureAttackGenerator(
            benign_stats=sensitivity_analysis.benign_stats,
            feature_names=sensitivity_analysis.feature_names,
            num_classes=num_classes,
            n_continuous_features=n_continuous,
            strategies=multi_attack_strategies,
        )
        active_attack = MultiAttackTRADES(
            trades_attack=pgd_attack_base,
            feature_attack=feature_attack,
            strategies=multi_attack_strategies or FeatureAttackGenerator.VALID_STRATEGIES,
        )
        print(f"  ✅ Couche 5: Worst-of-K actif: PGD + {active_attack.strategies}")
    elif use_multi_attack:
        print(f"  ⚠️ Worst-of-K: pas de sensitivity_analysis, PGD seulement.")
    is_multi_attack_flag = isinstance(active_attack, MultiAttackTRADES)

    # IBP
    ibp_trainer = None
    if use_ibp:
        _ibp_bstats = sensitivity_analysis.benign_stats if sensitivity_analysis else None
        _ibp_non_mod = []
        if sensitivity_analysis is not None:
            try:
                _ibp_non_mod = [sensitivity_analysis.feature_names.index(n)
                                 for n in sensitivity_analysis.non_modifiable
                                 if n in sensitivity_analysis.feature_names]
            except (ValueError, AttributeError):
                pass
        ibp_trainer = IBPTrainer(
            model, device,
            epsilon=ibp_epsilon, lambda_ibp=lambda_ibp,
            n_continuous_features=n_continuous, method=ibp_method,
            warmup_epochs=ibp_warmup_epochs, benign_stats=_ibp_bstats,
            perturbation_types=['zero', 'mean', 'p95'],
            non_modifiable_indices=_ibp_non_mod,
            epsilon_start=ibp_epsilon_start,
        )
        print(f"  ✅ Couche 7: IBP (ε {ibp_epsilon_start}→{ibp_epsilon}, méthode={ibp_method})")

    # Optimiseur et scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    # AdversarialEarlyStopping
    adv_early_stop = AdversarialEarlyStopping(max_gap=0.60, max_adv_loss=10.0, patience=5)
    current_epsilon = 0.01  # démarrage Phase 1

    # Crash test attack (fixed, pour évaluation)
    crash_pgd = TRADESAttack(epsilon=0.05, alpha=0.005, num_steps=pgd_steps)

    # Suivi
    best_val_loss = float('inf')
    best_state = None
    crash_test_results = {}
    criterion_eval = nn.CrossEntropyLoss()

    print(f"\n{'─'*80}")
    print(f"  CURRICULUM 4 PHASES — {total_epochs} epochs")
    print(f"  Phase 0: epochs  1-15  | Phase 1: epochs 16-35")
    print(f"  Phase 2: epochs 36-55  | Phase 3: epochs 56-{total_epochs}")
    print(f"{'─'*80}\n")

    # ── BOUCLE PRINCIPALE — 4 Phases ────────────────────────────────────────
    for epoch in range(1, total_epochs + 1):
        phase_cfg = get_phase_config(epoch)
        phase = phase_cfg['phase']

        # Mise à jour epsilon dans l'attaque PGD si phase >= 1
        if phase >= 1:
            current_epsilon = max(current_epsilon, phase_cfg['epsilon'])
            pgd_attack_base.epsilon = current_epsilon
            pgd_attack_base.alpha   = current_epsilon / 4

        # Annonce de changement de phase
        if epoch in (1, 16, 36, 56):
            print(f"\n{'═'*60}")
            print(f"  ➤ PHASE {phase} — {phase_cfg['name']}  (epoch {epoch})")
            print(f"{'═'*60}")

        # ── Epoch training ──────────────────────────────────────────────────
        train_stats = train_epoch_robust(
            epoch=epoch, model=model,
            train_loader=train_loader, optimizer=optimizer, device=device,
            input_defense=input_defense,
            afd_loss=afd_loss_module,
            feature_stats=None,
            ibp_trainer=ibp_trainer,
            active_attack=active_attack,
            pgd_attack=pgd_attack_base,
            phase_cfg=phase_cfg,
            use_cutmix=use_cutmix,
            use_ibp=use_ibp,
        )

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                if input_defense is not None:
                    X_b = input_defense(X_b)
                val_loss += criterion_eval(model(X_b), y_b).item()
        val_loss /= max(len(val_loader), 1)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # ── Early stopping adaptatif ────────────────────────────────────────
        if phase >= 1:
            stop, current_epsilon = adv_early_stop.check(
                benign_acc=train_stats['acc'],
                adv_acc=train_stats['acc'],  # proxy
                adv_loss=train_stats['trades'],
                current_eps=current_epsilon,
            )
            if stop:
                print(f"  ⛔ AdversarialEarlyStopping déclenché à epoch {epoch}. Sauvegarde + arrêt.")
                break

        # Log
        print(f"  Epoch {epoch:3d}/{total_epochs} [P{phase}|{phase_cfg['name'][:8]}] "
              f"Acc={train_stats['acc']:.4f} Loss={train_stats['loss']:.4f} "
              f"CE={train_stats['ce']:.4f} TRADES={train_stats['trades']:.4f} "
              f"IBP={train_stats['ibp']:.4f} Div={train_stats['div']:.4f} "
              f"Val={val_loss:.4f} ε={current_epsilon:.4f}")

        # ── Crash tests automatiques ────────────────────────────────────────
        if epoch in (15, 35, 55, total_epochs):
            print(f"\n{'─'*60}")
            ct = _crash_test(model, val_loader, device, epoch,
                             crash_pgd, active_attack, is_multi_attack_flag,
                             label=f'Phase{phase}', input_defense=input_defense)
            crash_test_results[f'epoch_{epoch}'] = ct
            model.train()
            print(f"{'─'*60}\n")

    # Restaurer meilleur état
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
    print(f"\n  ✓ Curriculum complet. Best val loss: {best_val_loss:.4f}")

    # ── Final Evaluation ──
    print(f"\n{'='*80}")
    print(f"  ÉVALUATION FINALE")
    print(f"{'='*80}\n")

    model.eval()
    correct_clean = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            if input_defense is not None:
                X_batch = input_defense(X_batch)
            outputs = model(X_batch)
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct_clean += predicted.eq(y_batch).sum().item()
    clean_acc = correct_clean / total
    print(f"  Clean Accuracy: {clean_acc:.4f}")

    correct_adv = 0
    total = 0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        if input_defense is not None:
            X_batch = input_defense(X_batch)
        with torch.no_grad():
            result = active_attack.generate(model, X_batch, y_batch, device)
            if is_multi_attack:
                X_adv, _ = result
            else:
                X_adv = result
        with torch.no_grad():
            outputs = model(X_adv)
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct_adv += predicted.eq(y_batch).sum().item()
        del X_adv
    adv_acc = correct_adv / total
    robustness_ratio = adv_acc / max(clean_acc, 1e-8)
    print(f"  Adversarial Accuracy: {adv_acc:.4f}")
    print(f"  Robustness Ratio: {robustness_ratio:.4f}")

    # Couche 3: Randomized Smoothing evaluation
    rs_results = None
    if use_randomized_smoothing:
        from src.adversarial.randomized_smoothing import RandomizedSmoothing
        print(f"\n  🛡️  Couche 3: Randomized Smoothing evaluation (σ={rs_sigma}, N={rs_n_samples})...")
        smoother = RandomizedSmoothing(model, device, sigma=rs_sigma, n_samples=rs_n_samples)
        rs_results = smoother.evaluate_robust(X_test, y_test)
        print(f"    RS Clean Accuracy: {rs_results['clean_accuracy']:.4f}")
        print(f"    Mean Certified Radius: {rs_results['mean_certified_radius']:.4f}")
        print(f"    % Certified: {rs_results['pct_certified']:.4f}")

    # Couche 4: IBP Certified Robustness evaluation
    ibp_results = None
    if use_ibp and ibp_trainer is not None:
        print(f"\n  🛡️  Couche 4: IBP Certified Robustness evaluation...")
        try:
            ibp_results = ibp_trainer.certify(test_loader, max_samples=2000)
            print(f"    Certified Accuracy:  {ibp_results['certified_accuracy']:.4f}")
            print(f"    Clean Accuracy:      {ibp_results['clean_accuracy']:.4f}")
            print(f"    Avg Certified Radius: {ibp_results['avg_certified_radius']:.4f}")
            print(f"    % Certified:         {ibp_results['certified_ratio']:.2%}")
            if 'perturbation_types' in ibp_results:
                print(f"    Perturbation types:   {ibp_results['perturbation_types']}")
        except Exception as e:
            print(f"    ⚠️  IBP certification error: {e}")
            ibp_results = {'error': str(e)}

    results = {
        'model_type': model_type,
        'dataset_type': dataset_type,
        'clean_accuracy': clean_acc,
        'adversarial_accuracy': adv_acc,
        'robustness_ratio': robustness_ratio,
        'use_afd': use_afd,
        'use_multi_attack': use_multi_attack,
        'use_randomized_smoothing': use_randomized_smoothing,
        'use_ibp': use_ibp,
        'ibp_epsilon': ibp_epsilon,
        'lambda_ibp': lambda_ibp,
        'ibp_method': ibp_method,
        'total_epochs': total_epochs,
        'use_input_defense': use_input_defense,
        'use_cutmix': use_cutmix,
    }
    if rs_results is not None:
        results['rs_clean_accuracy'] = rs_results['clean_accuracy']
        results['rs_mean_certified_radius'] = rs_results['mean_certified_radius']
        results['rs_pct_certified'] = rs_results['pct_certified']

    if ibp_results is not None:
        results['ibp_certified_accuracy'] = ibp_results.get('certified_accuracy', 0)
        results['ibp_avg_certified_radius'] = ibp_results.get('avg_certified_radius', 0)
        results['ibp_certified_ratio'] = ibp_results.get('certified_ratio', 0)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), f'{save_dir}/model.pt')
        with open(f'{save_dir}/countermeasures_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  💾 Model and results saved to: {save_dir}")

    del train_loader, val_loader, test_loader
    del train_dataset, val_dataset, test_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results

print('✅ Training functions loaded: train_model_with_countermeasures, load_dataset_from_drive, create_model')

# ─── MODEL: LSTM on CSV (with Countermeasures) ────────────────────────────
MODEL = 'lstm'
print(f'\n{"#"*80}')
print(f'  PHASE 3a — LSTM with Countermeasures on CSV')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_csv')

if DATASETS in ['csv', 'both']:
    data = load_dataset_from_drive('csv')

    if data is not None:
        results = train_model_with_countermeasures(
            model_type=MODEL,
            dataset_type='csv',
            data_dict=data,
            pgd_steps=TRADES_PGD_STEPS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            use_cutmix=USE_CUTMIX,
            use_afd=USE_AFD,
            use_randomized_smoothing=USE_RS,
            rs_sigma=RS_SIGMA,
            rs_n_samples=RS_N_SAMPLES,
            use_multi_attack=USE_MULTI_ATTACK,
            multi_attack_strategies=MULTI_ATTACK_STRATEGIES,
            save_dir=f'{DRIVE_RESULTS_DIR}/models/{MODEL}_countermeasures_csv',
            use_class_weights=True,
            use_ibp=USE_IBP,
            ibp_epsilon=IBP_EPSILON,
            lambda_ibp=LAMBDA_IBP,
            ibp_method=IBP_METHOD,
            ibp_warmup_epochs=IBP_WARMUP_EPOCHS,
            ibp_epsilon_start=IBP_EPSILON_START,
            total_epochs=TOTAL_EPOCHS,
            use_input_defense=USE_INPUT_DEFENSE,
            use_randomized_smoothing=USE_RS,
            rs_sigma=RS_SIGMA,
            rs_n_samples=RS_N_SAMPLES,
        )

        print(f"\n✅ {MODEL.upper()} with Countermeasures on CSV complete!")
        print(f"  Clean Accuracy: {results['clean_accuracy']:.4f}")
        print(f"  Adversarial Accuracy: {results['adversarial_accuracy']:.4f}")
        print(f"  Robustness Ratio: {results['robustness_ratio']:.4f}")

        log_memory(f'after_{MODEL}_csv')
        aggressive_cleanup()
    else:
        print('❌ Failed to load CSV data')
else:
    print('Skipping CSV dataset')

print(f'\n✅ {MODEL.upper()} on CSV DONE')

# ─── MODEL: BILSTM on CSV (with Countermeasures) ────────────────────────────
MODEL = 'bilstm'
print(f'\n{"#"*80}')
print(f'  PHASE 3a — BILSTM with Countermeasures on CSV')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_csv')

if DATASETS in ['csv', 'both']:
    data = load_dataset_from_drive('csv')

    if data is not None:
        results = train_model_with_countermeasures(
            model_type=MODEL,
            dataset_type='csv',
            data_dict=data,
            pgd_steps=TRADES_PGD_STEPS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            use_cutmix=USE_CUTMIX,
            use_afd=USE_AFD,
            use_randomized_smoothing=USE_RS,
            rs_sigma=RS_SIGMA,
            rs_n_samples=RS_N_SAMPLES,
            use_multi_attack=USE_MULTI_ATTACK,
            multi_attack_strategies=MULTI_ATTACK_STRATEGIES,
            save_dir=f'{DRIVE_RESULTS_DIR}/models/{MODEL}_countermeasures_csv',
            use_class_weights=True,
            use_ibp=USE_IBP,
            ibp_epsilon=IBP_EPSILON,
            lambda_ibp=LAMBDA_IBP,
            ibp_method=IBP_METHOD,
            ibp_warmup_epochs=IBP_WARMUP_EPOCHS,
            ibp_epsilon_start=IBP_EPSILON_START,
            total_epochs=TOTAL_EPOCHS,
            use_input_defense=USE_INPUT_DEFENSE,
            use_randomized_smoothing=USE_RS,
            rs_sigma=RS_SIGMA,
            rs_n_samples=RS_N_SAMPLES,
        )

        print(f"\n✅ {MODEL.upper()} with Countermeasures on CSV complete!")
        print(f"  Clean Accuracy: {results['clean_accuracy']:.4f}")
        print(f"  Adversarial Accuracy: {results['adversarial_accuracy']:.4f}")
        print(f"  Robustness Ratio: {results['robustness_ratio']:.4f}")

        log_memory(f'after_{MODEL}_csv')
        aggressive_cleanup()
    else:
        print('❌ Failed to load CSV data')
else:
    print('Skipping CSV dataset')

print(f'\n✅ {MODEL.upper()} on CSV DONE')

# ─── MODEL: CNN-LSTM on CSV (with Countermeasures) ────────────────────────────
MODEL = 'cnn_lstm'
print(f'\n{"#"*80}')
print(f'  PHASE 3a — CNN-LSTM with Countermeasures on CSV')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_csv')

if DATASETS in ['csv', 'both']:
    data = load_dataset_from_drive('csv')

    if data is not None:
        results = train_model_with_countermeasures(
            model_type=MODEL,
            dataset_type='csv',
            data_dict=data,
            pgd_steps=TRADES_PGD_STEPS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            use_cutmix=USE_CUTMIX,
            use_afd=USE_AFD,
            use_randomized_smoothing=USE_RS,
            rs_sigma=RS_SIGMA,
            rs_n_samples=RS_N_SAMPLES,
            use_multi_attack=USE_MULTI_ATTACK,
            multi_attack_strategies=MULTI_ATTACK_STRATEGIES,
            save_dir=f'{DRIVE_RESULTS_DIR}/models/{MODEL}_countermeasures_csv',
            use_class_weights=True,
            use_ibp=USE_IBP,
            ibp_epsilon=IBP_EPSILON,
            lambda_ibp=LAMBDA_IBP,
            ibp_method=IBP_METHOD,
            ibp_warmup_epochs=IBP_WARMUP_EPOCHS,
            ibp_epsilon_start=IBP_EPSILON_START,
            total_epochs=TOTAL_EPOCHS,
            use_input_defense=USE_INPUT_DEFENSE,
            use_randomized_smoothing=USE_RS,
            rs_sigma=RS_SIGMA,
            rs_n_samples=RS_N_SAMPLES,
        )

        print(f"\n✅ {MODEL.upper()} with Countermeasures on CSV complete!")
        print(f"  Clean Accuracy: {results['clean_accuracy']:.4f}")
        print(f"  Adversarial Accuracy: {results['adversarial_accuracy']:.4f}")
        print(f"  Robustness Ratio: {results['robustness_ratio']:.4f}")

        log_memory(f'after_{MODEL}_csv')
        aggressive_cleanup()
    else:
        print('❌ Failed to load CSV data')
else:
    print('Skipping CSV dataset')

print(f'\n✅ {MODEL.upper()} on CSV DONE')

# ─── MODEL: XGBOOST-LSTM on CSV (with Countermeasures) ────────────────────────────
MODEL = 'xgboost_lstm'
print(f'\n{"#"*80}')
print(f'  PHASE 3a — XGBOOST-LSTM with Countermeasures on CSV')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_csv')

if DATASETS in ['csv', 'both']:
    data = load_dataset_from_drive('csv')

    if data is not None:
        results = train_model_with_countermeasures(
            model_type=MODEL,
            dataset_type='csv',
            data_dict=data,
            pgd_steps=TRADES_PGD_STEPS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            use_cutmix=USE_CUTMIX,
            use_afd=USE_AFD,
            use_randomized_smoothing=USE_RS,
            rs_sigma=RS_SIGMA,
            rs_n_samples=RS_N_SAMPLES,
            use_multi_attack=USE_MULTI_ATTACK,
            multi_attack_strategies=MULTI_ATTACK_STRATEGIES,
            save_dir=f'{DRIVE_RESULTS_DIR}/models/{MODEL}_countermeasures_csv',
            use_class_weights=True,
            use_ibp=USE_IBP,
            ibp_epsilon=IBP_EPSILON,
            lambda_ibp=LAMBDA_IBP,
            ibp_method=IBP_METHOD,
            ibp_warmup_epochs=IBP_WARMUP_EPOCHS,
            ibp_epsilon_start=IBP_EPSILON_START,
            total_epochs=TOTAL_EPOCHS,
            use_input_defense=USE_INPUT_DEFENSE,
            use_randomized_smoothing=USE_RS,
            rs_sigma=RS_SIGMA,
            rs_n_samples=RS_N_SAMPLES,
        )

        print(f"\n✅ {MODEL.upper()} with Countermeasures on CSV complete!")
        print(f"  Clean Accuracy: {results['clean_accuracy']:.4f}")
        print(f"  Adversarial Accuracy: {results['adversarial_accuracy']:.4f}")
        print(f"  Robustness Ratio: {results['robustness_ratio']:.4f}")

        log_memory(f'after_{MODEL}_csv')
        aggressive_cleanup()
    else:
        print('❌ Failed to load CSV data')
else:
    print('Skipping CSV dataset')

print(f'\n✅ {MODEL.upper()} on CSV DONE')

# ─── MODEL: TRANSFORMER on CSV (with Countermeasures) ────────────────────────────
MODEL = 'transformer'
print(f'\n{"#"*80}')
print(f'  PHASE 3a — TRANSFORMER with Countermeasures on CSV')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_csv')

if DATASETS in ['csv', 'both']:
    data = load_dataset_from_drive('csv')

    if data is not None:
        results = train_model_with_countermeasures(
            model_type=MODEL,
            dataset_type='csv',
            data_dict=data,
            pgd_steps=TRADES_PGD_STEPS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            use_cutmix=USE_CUTMIX,
            use_afd=USE_AFD,
            use_randomized_smoothing=USE_RS,
            rs_sigma=RS_SIGMA,
            rs_n_samples=RS_N_SAMPLES,
            use_multi_attack=USE_MULTI_ATTACK,
            multi_attack_strategies=MULTI_ATTACK_STRATEGIES,
            save_dir=f'{DRIVE_RESULTS_DIR}/models/{MODEL}_countermeasures_csv',
            use_class_weights=True,
            use_ibp=USE_IBP,
            ibp_epsilon=IBP_EPSILON,
            lambda_ibp=LAMBDA_IBP,
            ibp_method=IBP_METHOD,
            ibp_warmup_epochs=IBP_WARMUP_EPOCHS,
            ibp_epsilon_start=IBP_EPSILON_START,
            total_epochs=TOTAL_EPOCHS,
            use_input_defense=USE_INPUT_DEFENSE,
            use_randomized_smoothing=USE_RS,
            rs_sigma=RS_SIGMA,
            rs_n_samples=RS_N_SAMPLES,
        )

        print(f"\n✅ {MODEL.upper()} with Countermeasures on CSV complete!")
        print(f"  Clean Accuracy: {results['clean_accuracy']:.4f}")
        print(f"  Adversarial Accuracy: {results['adversarial_accuracy']:.4f}")
        print(f"  Robustness Ratio: {results['robustness_ratio']:.4f}")

        log_memory(f'after_{MODEL}_csv')
        aggressive_cleanup()
    else:
        print('❌ Failed to load CSV data')
else:
    print('Skipping CSV dataset')

print(f'\n✅ {MODEL.upper()} on CSV DONE')

# ─── MODEL: CNN-BILSTM-TRANSFORMER on CSV (with Countermeasures) ────────────────────────────
MODEL = 'cnn_bilstm_transformer'
print(f'\n{"#"*80}')
print(f'  PHASE 3a — CNN-BILSTM-TRANSFORMER with Countermeasures on CSV')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_csv')

if DATASETS in ['csv', 'both']:
    data = load_dataset_from_drive('csv')

    if data is not None:
        results = train_model_with_countermeasures(
            model_type=MODEL,
            dataset_type='csv',
            data_dict=data,
            pgd_steps=TRADES_PGD_STEPS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            use_cutmix=USE_CUTMIX,
            use_afd=USE_AFD,
            use_randomized_smoothing=USE_RS,
            rs_sigma=RS_SIGMA,
            rs_n_samples=RS_N_SAMPLES,
            use_multi_attack=USE_MULTI_ATTACK,
            multi_attack_strategies=MULTI_ATTACK_STRATEGIES,
            save_dir=f'{DRIVE_RESULTS_DIR}/models/{MODEL}_countermeasures_csv',
            use_class_weights=True,
            use_ibp=USE_IBP,
            ibp_epsilon=IBP_EPSILON,
            lambda_ibp=LAMBDA_IBP,
            ibp_method=IBP_METHOD,
            ibp_warmup_epochs=IBP_WARMUP_EPOCHS,
            ibp_epsilon_start=IBP_EPSILON_START,
            total_epochs=TOTAL_EPOCHS,
            use_input_defense=USE_INPUT_DEFENSE,
            use_randomized_smoothing=USE_RS,
            rs_sigma=RS_SIGMA,
            rs_n_samples=RS_N_SAMPLES,
        )

        print(f"\n✅ {MODEL.upper()} with Countermeasures on CSV complete!")
        print(f"  Clean Accuracy: {results['clean_accuracy']:.4f}")
        print(f"  Adversarial Accuracy: {results['adversarial_accuracy']:.4f}")
        print(f"  Robustness Ratio: {results['robustness_ratio']:.4f}")

        log_memory(f'after_{MODEL}_csv')
        aggressive_cleanup()
    else:
        print('❌ Failed to load CSV data')
else:
    print('Skipping CSV dataset')

print(f'\n✅ {MODEL.upper()} on CSV DONE')

# ─── CLEANUP RAM BEFORE JSON PHASE ──────────────────────────────────────
print('\n' + '='*80)
print('  🧹 CLEANING RAM BEFORE JSON PHASE...')
print('='*80 + '\n')

aggressive_cleanup()

print('\n✅ RAM cleaned. Ready for JSON phase.')

def load_and_display_json_dataset(json_data_dir, seq_length=10, stride=10, max_records=None, save_dir=None):
    """Charge COMPLÈTEMENT le dataset JSON, affiche les infos, et sauvegarde sur Drive."""
    import sys
    import gc
    import numpy as np
    import pickle

    sys.path.insert(0, '/content/pfe')

    print("\n" + "=" * 70)
    print("  CHARGEMENT COMPLET DU DATASET JSON")
    print("=" * 70)

    print(f"\n  Répertoire : {json_data_dir}")
    print(f"  Seq length : {seq_length} | Stride : {stride}")
    print(f"  Max records: {max_records if max_records else 'Tous'}")

    json_files = sorted(glob.glob(f'{json_data_dir}/**/*.json', recursive=True))
    print(f"\n  Fichiers JSON trouvés : {len(json_files)}")

    for f in json_files:
        size_gb = os.path.getsize(f) / (1024**3)
        print(f"    {os.path.basename(f):<30s} : {size_gb:>10.2f} GB")

    total_gb = sum(os.path.getsize(f) for f in json_files) / (1024**3)
    print(f"\n  Taille totale : {total_gb:.2f} GB")

    # Charger le dataset complet via le vrai pipeline JSON
    print("\n  Chargement via le pipeline JSON (JsonIoTDataProcessor)...")
    from src.data.json_preprocessor import JsonIoTDataProcessor

    processor = JsonIoTDataProcessor()
    result = processor.process_all(
        data_dir=json_data_dir,
        seq_length=seq_length,
        stride=stride,
        max_records=max_records,
        apply_balancing=False,
    )

    X_train, X_val, X_test, y_train, y_val, y_test, features, scaler, label_encoder = result
    n_continuous = 36  # JSON pipeline features

    print(f"\n  {'='*70}")
    print(f"  RÉSULTAT DU CHARGEMENT")
    print(f"  {'='*70}")
    print(f"    Features ({len(features)}) : {features[:5]}...")
    print(f"    Classes ({len(label_encoder.classes_)}) : {list(label_encoder.classes_)}")
    print(f"\n  Shapes des séquences (seq_length={seq_length}, stride={stride}) :")
    print(f"    Train : {X_train.shape}  →  {len(X_train):,} séquences")
    print(f"    Val   : {X_val.shape}  →  {len(X_val):,} séquences")
    print(f"    Test  : {X_test.shape}  →  {len(X_test):,} séquences")
    print(f"    Total : {len(X_train) + len(X_val) + len(X_test):,} séquences")

    print(f"\n  Distribution des classes (train) :")
    for cls in label_encoder.classes_:
        cls_id = label_encoder.transform([cls])[0]
        count = int(np.sum(y_train == cls_id))
        bar = '█' * max(1, count // 50)
        print(f"    {cls:<30s} : {count:>6,}  {bar}")

    # ─── Sauvegarder sur Drive ────────────────────────────────────────────
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n  💾 Sauvegarde du dataset JSON pré-traité sur Drive...")
        print(f"     Répertoire : {save_dir}")

        np.save(f'{save_dir}/X_train.npy', X_train)
        np.save(f'{save_dir}/X_val.npy', X_val)
        np.save(f'{save_dir}/X_test.npy', X_test)
        np.save(f'{save_dir}/y_train.npy', y_train)
        np.save(f'{save_dir}/y_val.npy', y_val)
        np.save(f'{save_dir}/y_test.npy', y_test)

        with open(f'{save_dir}/json_metadata.pkl', 'wb') as f:
            pickle.dump({
                'features': features,
                'scaler': scaler,
                'label_encoder': label_encoder,
                'n_continuous': n_continuous,
                'seq_length': seq_length,
                'stride': stride,
            }, f)

        # Marker file to indicate preprocessing is complete
        with open(f'{save_dir}/json_ready', 'w') as f:
            f.write('ready')

        saved_gb = (X_train.nbytes + X_val.nbytes + X_test.nbytes +
                    y_train.nbytes + y_val.nbytes + y_test.nbytes) / (1024**3)
        print(f"  ✅ Dataset JSON sauvegardé ({saved_gb:.2f} GB)")
        print(f"     Fichiers : X_train, X_val, X_test, y_train, y_val, y_test, json_metadata.pkl")

    print(f"\n  {'='*70}")
    print(f"  ✅ Dataset JSON chargé complètement en RAM")
    print(f"  {'='*70}\n")

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'features': features, 'scaler': scaler,
        'label_encoder': label_encoder, 'n_continuous': n_continuous
    }

# ─── JSON preprocessing directory on Drive ─────────────────────────────
JSON_PREPROCESSED_DIR = f'{DRIVE_RESULTS_DIR}/preprocessed/json'

if DATASETS in ['json', 'both']:
    json_data = load_and_display_json_dataset(
        JSON_DATA_DIR, seq_length=SEQ_LENGTH, stride=STRIDE,
        max_records=MAX_RECORDS, save_dir=JSON_PREPROCESSED_DIR
    )
else:
    json_data = None
    print('Skipping JSON dataset loading — DATASETS is not json or both')

# ─── MODEL: LSTM on JSON (with Countermeasures) ────────────────────────────
MODEL = 'lstm'
print(f'\n{"#"*80}')
print(f'  PHASE 3b — LSTM with Countermeasures on JSON')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_json')

if DATASETS in ['json', 'both']:
    data = load_dataset_from_drive('json')

    if data is not None:
        results = train_model_with_countermeasures(
            model_type=MODEL,
            dataset_type='json',
            data_dict=data,
            pgd_steps=TRADES_PGD_STEPS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            use_cutmix=USE_CUTMIX,
            use_afd=USE_AFD,
            use_randomized_smoothing=USE_RS,
            rs_sigma=RS_SIGMA,
            rs_n_samples=RS_N_SAMPLES,
            use_multi_attack=USE_MULTI_ATTACK,
            multi_attack_strategies=MULTI_ATTACK_STRATEGIES,
            save_dir=f'{DRIVE_RESULTS_DIR}/models/{MODEL}_countermeasures_json',
            use_class_weights=True,
            use_ibp=USE_IBP,
            ibp_epsilon=IBP_EPSILON,
            lambda_ibp=LAMBDA_IBP,
            ibp_method=IBP_METHOD,
            ibp_warmup_epochs=IBP_WARMUP_EPOCHS,
            ibp_epsilon_start=IBP_EPSILON_START,
            total_epochs=TOTAL_EPOCHS,
            use_input_defense=USE_INPUT_DEFENSE,
            use_randomized_smoothing=USE_RS,
            rs_sigma=RS_SIGMA,
            rs_n_samples=RS_N_SAMPLES,
        )

        print(f"\n✅ {MODEL.upper()} with Countermeasures on JSON complete!")
        print(f"  Clean Accuracy: {results['clean_accuracy']:.4f}")
        print(f"  Adversarial Accuracy: {results['adversarial_accuracy']:.4f}")
        print(f"  Robustness Ratio: {results['robustness_ratio']:.4f}")

        log_memory(f'after_{MODEL}_json')
        aggressive_cleanup()
    else:
        print('❌ Failed to load JSON data')
else:
    print('Skipping JSON dataset')

print(f'\n✅ {MODEL.upper()} on JSON DONE')

# ─── MODEL: BILSTM on JSON (with Countermeasures) ────────────────────────────
MODEL = 'bilstm'
print(f'\n{"#"*80}')
print(f'  PHASE 3b — BILSTM with Countermeasures on JSON')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_json')

if DATASETS in ['json', 'both']:
    data = load_dataset_from_drive('json')

    if data is not None:
        results = train_model_with_countermeasures(
            model_type=MODEL,
            dataset_type='json',
            data_dict=data,
            pgd_steps=TRADES_PGD_STEPS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            use_cutmix=USE_CUTMIX,
            use_afd=USE_AFD,
            use_randomized_smoothing=USE_RS,
            rs_sigma=RS_SIGMA,
            rs_n_samples=RS_N_SAMPLES,
            use_multi_attack=USE_MULTI_ATTACK,
            multi_attack_strategies=MULTI_ATTACK_STRATEGIES,
            save_dir=f'{DRIVE_RESULTS_DIR}/models/{MODEL}_countermeasures_json',
            use_class_weights=True,
            use_ibp=USE_IBP,
            ibp_epsilon=IBP_EPSILON,
            lambda_ibp=LAMBDA_IBP,
            ibp_method=IBP_METHOD,
            ibp_warmup_epochs=IBP_WARMUP_EPOCHS,
            ibp_epsilon_start=IBP_EPSILON_START,
            total_epochs=TOTAL_EPOCHS,
            use_input_defense=USE_INPUT_DEFENSE,
            use_randomized_smoothing=USE_RS,
            rs_sigma=RS_SIGMA,
            rs_n_samples=RS_N_SAMPLES,
        )

        print(f"\n✅ {MODEL.upper()} with Countermeasures on JSON complete!")
        print(f"  Clean Accuracy: {results['clean_accuracy']:.4f}")
        print(f"  Adversarial Accuracy: {results['adversarial_accuracy']:.4f}")
        print(f"  Robustness Ratio: {results['robustness_ratio']:.4f}")

        log_memory(f'after_{MODEL}_json')
        aggressive_cleanup()
    else:
        print('❌ Failed to load JSON data')
else:
    print('Skipping JSON dataset')

print(f'\n✅ {MODEL.upper()} on JSON DONE')

# ─── MODEL: CNN-LSTM on JSON (with Countermeasures) ────────────────────────────
MODEL = 'cnn_lstm'
print(f'\n{"#"*80}')
print(f'  PHASE 3b — CNN-LSTM with Countermeasures on JSON')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_json')

if DATASETS in ['json', 'both']:
    data = load_dataset_from_drive('json')

    if data is not None:
        results = train_model_with_countermeasures(
            model_type=MODEL,
            dataset_type='json',
            data_dict=data,
            pgd_steps=TRADES_PGD_STEPS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            use_cutmix=USE_CUTMIX,
            use_afd=USE_AFD,
            use_randomized_smoothing=USE_RS,
            rs_sigma=RS_SIGMA,
            rs_n_samples=RS_N_SAMPLES,
            use_multi_attack=USE_MULTI_ATTACK,
            multi_attack_strategies=MULTI_ATTACK_STRATEGIES,
            save_dir=f'{DRIVE_RESULTS_DIR}/models/{MODEL}_countermeasures_json',
            use_class_weights=True,
            use_ibp=USE_IBP,
            ibp_epsilon=IBP_EPSILON,
            lambda_ibp=LAMBDA_IBP,
            ibp_method=IBP_METHOD,
            ibp_warmup_epochs=IBP_WARMUP_EPOCHS,
            ibp_epsilon_start=IBP_EPSILON_START,
            total_epochs=TOTAL_EPOCHS,
            use_input_defense=USE_INPUT_DEFENSE,
            use_randomized_smoothing=USE_RS,
            rs_sigma=RS_SIGMA,
            rs_n_samples=RS_N_SAMPLES,
        )

        print(f"\n✅ {MODEL.upper()} with Countermeasures on JSON complete!")
        print(f"  Clean Accuracy: {results['clean_accuracy']:.4f}")
        print(f"  Adversarial Accuracy: {results['adversarial_accuracy']:.4f}")
        print(f"  Robustness Ratio: {results['robustness_ratio']:.4f}")

        log_memory(f'after_{MODEL}_json')
        aggressive_cleanup()
    else:
        print('❌ Failed to load JSON data')
else:
    print('Skipping JSON dataset')

print(f'\n✅ {MODEL.upper()} on JSON DONE')

# ─── MODEL: XGBOOST-LSTM on JSON (with Countermeasures) ────────────────────────────
MODEL = 'xgboost_lstm'
print(f'\n{"#"*80}')
print(f'  PHASE 3b — XGBOOST-LSTM with Countermeasures on JSON')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_json')

if DATASETS in ['json', 'both']:
    data = load_dataset_from_drive('json')

    if data is not None:
        results = train_model_with_countermeasures(
            model_type=MODEL,
            dataset_type='json',
            data_dict=data,
            pgd_steps=TRADES_PGD_STEPS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            use_cutmix=USE_CUTMIX,
            use_afd=USE_AFD,
            use_randomized_smoothing=USE_RS,
            rs_sigma=RS_SIGMA,
            rs_n_samples=RS_N_SAMPLES,
            use_multi_attack=USE_MULTI_ATTACK,
            multi_attack_strategies=MULTI_ATTACK_STRATEGIES,
            save_dir=f'{DRIVE_RESULTS_DIR}/models/{MODEL}_countermeasures_json',
            use_class_weights=True,
            use_ibp=USE_IBP,
            ibp_epsilon=IBP_EPSILON,
            lambda_ibp=LAMBDA_IBP,
            ibp_method=IBP_METHOD,
            ibp_warmup_epochs=IBP_WARMUP_EPOCHS,
            ibp_epsilon_start=IBP_EPSILON_START,
            total_epochs=TOTAL_EPOCHS,
            use_input_defense=USE_INPUT_DEFENSE,
            use_randomized_smoothing=USE_RS,
            rs_sigma=RS_SIGMA,
            rs_n_samples=RS_N_SAMPLES,
        )

        print(f"\n✅ {MODEL.upper()} with Countermeasures on JSON complete!")
        print(f"  Clean Accuracy: {results['clean_accuracy']:.4f}")
        print(f"  Adversarial Accuracy: {results['adversarial_accuracy']:.4f}")
        print(f"  Robustness Ratio: {results['robustness_ratio']:.4f}")

        log_memory(f'after_{MODEL}_json')
        aggressive_cleanup()
    else:
        print('❌ Failed to load JSON data')
else:
    print('Skipping JSON dataset')

print(f'\n✅ {MODEL.upper()} on JSON DONE')

# ─── MODEL: TRANSFORMER on JSON (with Countermeasures) ────────────────────────────
MODEL = 'transformer'
print(f'\n{"#"*80}')
print(f'  PHASE 3b — TRANSFORMER with Countermeasures on JSON')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_json')

if DATASETS in ['json', 'both']:
    data = load_dataset_from_drive('json')

    if data is not None:
        results = train_model_with_countermeasures(
            model_type=MODEL,
            dataset_type='json',
            data_dict=data,
            pgd_steps=TRADES_PGD_STEPS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            use_cutmix=USE_CUTMIX,
            use_afd=USE_AFD,
            use_randomized_smoothing=USE_RS,
            rs_sigma=RS_SIGMA,
            rs_n_samples=RS_N_SAMPLES,
            use_multi_attack=USE_MULTI_ATTACK,
            multi_attack_strategies=MULTI_ATTACK_STRATEGIES,
            save_dir=f'{DRIVE_RESULTS_DIR}/models/{MODEL}_countermeasures_json',
            use_class_weights=True,
            use_ibp=USE_IBP,
            ibp_epsilon=IBP_EPSILON,
            lambda_ibp=LAMBDA_IBP,
            ibp_method=IBP_METHOD,
            ibp_warmup_epochs=IBP_WARMUP_EPOCHS,
            ibp_epsilon_start=IBP_EPSILON_START,
            total_epochs=TOTAL_EPOCHS,
            use_input_defense=USE_INPUT_DEFENSE,
            use_randomized_smoothing=USE_RS,
            rs_sigma=RS_SIGMA,
            rs_n_samples=RS_N_SAMPLES,
        )

        print(f"\n✅ {MODEL.upper()} with Countermeasures on JSON complete!")
        print(f"  Clean Accuracy: {results['clean_accuracy']:.4f}")
        print(f"  Adversarial Accuracy: {results['adversarial_accuracy']:.4f}")
        print(f"  Robustness Ratio: {results['robustness_ratio']:.4f}")

        log_memory(f'after_{MODEL}_json')
        aggressive_cleanup()
    else:
        print('❌ Failed to load JSON data')
else:
    print('Skipping JSON dataset')

print(f'\n✅ {MODEL.upper()} on JSON DONE')

# ─── MODEL: CNN-BILSTM-TRANSFORMER on JSON (with Countermeasures) ────────────────────────────
MODEL = 'cnn_bilstm_transformer'
print(f'\n{"#"*80}')
print(f'  PHASE 3b — CNN-BILSTM-TRANSFORMER with Countermeasures on JSON')
print(f'{"#"*80}\n')

log_memory(f'before_{MODEL}_json')

if DATASETS in ['json', 'both']:
    data = load_dataset_from_drive('json')

    if data is not None:
        results = train_model_with_countermeasures(
            model_type=MODEL,
            dataset_type='json',
            data_dict=data,
            pgd_steps=TRADES_PGD_STEPS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            use_cutmix=USE_CUTMIX,
            use_afd=USE_AFD,
            use_randomized_smoothing=USE_RS,
            rs_sigma=RS_SIGMA,
            rs_n_samples=RS_N_SAMPLES,
            use_multi_attack=USE_MULTI_ATTACK,
            multi_attack_strategies=MULTI_ATTACK_STRATEGIES,
            save_dir=f'{DRIVE_RESULTS_DIR}/models/{MODEL}_countermeasures_json',
            use_class_weights=True,
            use_ibp=USE_IBP,
            ibp_epsilon=IBP_EPSILON,
            lambda_ibp=LAMBDA_IBP,
            ibp_method=IBP_METHOD,
            ibp_warmup_epochs=IBP_WARMUP_EPOCHS,
            ibp_epsilon_start=IBP_EPSILON_START,
            total_epochs=TOTAL_EPOCHS,
            use_input_defense=USE_INPUT_DEFENSE,
            use_randomized_smoothing=USE_RS,
            rs_sigma=RS_SIGMA,
            rs_n_samples=RS_N_SAMPLES,
        )

        print(f"\n✅ {MODEL.upper()} with Countermeasures on JSON complete!")
        print(f"  Clean Accuracy: {results['clean_accuracy']:.4f}")
        print(f"  Adversarial Accuracy: {results['adversarial_accuracy']:.4f}")
        print(f"  Robustness Ratio: {results['robustness_ratio']:.4f}")

        log_memory(f'after_{MODEL}_json')
        aggressive_cleanup()
    else:
        print('❌ Failed to load JSON data')
else:
    print('Skipping JSON dataset')



print(f'\n{"="*80}')
print('  ✅ ALL 6 MODELS TRAINED ON BOTH DATASETS — COMPLETE!')
print(f'{"="*80}')

# ─── Cell: Display ALL Plots ─────────────────────────────────────────────────
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from IPython.display import display, HTML

results_dir = Path(DRIVE_RESULTS_DIR) / 'models'
if not results_dir.exists():
    print('No results found yet — run the training cells first.')
else:
    result_dirs = sorted(results_dir.iterdir())

    for rd in result_dirs:
        if not rd.is_dir():
            continue

        model_name = rd.name
        print(f"\n{'='*80}")
        print(f"  📁 {model_name}")
        print(f"{'='*80}")

        # Show results summary
        results_file = rd / 'results.json'
        if results_file.exists():
            with open(results_file) as f:
                res = json.load(f)
            print(f"  Model: {res.get('model_type', '?')} | Input: {res.get('input_size', '?')} | Classes: {res.get('num_classes', '?')}")
            print(f"  Clean Accuracy: {res.get('test_accuracy_clean', 0):.4f}")
            if 'clean_metrics' in res:
                cm = res['clean_metrics']
                print(f"  Macro F1: {cm.get('macro_f1', 0):.4f} | Macro P: {cm.get('macro_precision', 0):.4f} | Macro R: {cm.get('macro_recall', 0):.4f}")
            if 'robustness_ratios' in res:
                print(f"  Robustness Ratios: {res['robustness_ratios']}")

        # Display all plots
        plot_files = sorted(rd.glob('fig_*.png'))
        if plot_files:
            for pf in plot_files:
                print(f"\n  📊 {pf.name}")
                fig, ax = plt.subplots(figsize=(16, 8))
                img = mpimg.imread(str(pf))
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(pf.stem.replace('_', ' ').title(), fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.show()
        else:
            print('  No plots found.')

        # Show curriculum report if available
        report_file = rd / 'curriculum_report.json'
        if report_file.exists():
            with open(report_file) as f:
                report = json.load(f)
            print(f"\n  📋 Curriculum Report:")
            for phase_key, phase_data in report.get('phases', {}).items():
                phase_num = phase_data.get('phase', '?')
                clean = phase_data.get('clean', {})
                feat = phase_data.get('feature_attack', {})
                pgd = phase_data.get('sequence_pgd', {})
                fgsm = phase_data.get('sequence_fgsm', {})
                print(f"    Phase {phase_num}:")
                print(f"      Clean  — Acc: {clean.get('accuracy', 0):.4f}  F1: {clean.get('f1_score', 0):.4f}  Macro-F1: {clean.get('macro_f1', 0):.4f}")
                if feat:
                    print(f"      FeatAdv — Acc: {feat.get('accuracy', 0):.4f}  F1: {feat.get('f1_score', 0):.4f}  RR: {feat.get('robustness_ratio', 0):.4f}")
                if pgd:
                    print(f"      SeqPGD — Acc: {pgd.get('accuracy', 0):.4f}  F1: {pgd.get('f1_score', 0):.4f}  RR: {pgd.get('robustness_ratio', 0):.4f}")
                if fgsm:
                    print(f"      SeqFGSM — Acc: {fgsm.get('accuracy', 0):.4f}  F1: {fgsm.get('f1_score', 0):.4f}  RR: {fgsm.get('robustness_ratio', 0):.4f}")

print('\n\n✅ All results displayed.')
# ─── Cell: Comparative Summary Table ─────────────────────────────────────────
import json
from pathlib import Path

results_dir = Path(DRIVE_RESULTS_DIR) / 'models'
if not results_dir.exists():
    print('No results found.')
else:
    print(f"{'Model':<30} {'Clean Acc':>10} {'Macro F1':>10} {'Feat Adv':>10} {'Seq PGD':>10} {'Seq FGSM':>10} {'Hybrid':>10}")
    print(f"{'-'*100}")

    for rd in sorted(results_dir.iterdir()):
        rf = rd / 'results.json'
        if not rf.exists():
            continue
        with open(rf) as f:
            res = json.load(f)

        model = rd.name
        clean_acc = res.get('test_accuracy_clean', 0)
        macro_f1 = res.get('clean_metrics', {}).get('macro_f1', 0)

        adv = res.get('adversarial_results', {})
        feat_acc = adv.get('feature_level', {}).get('accuracy', 0)
        pgd_acc = adv.get('sequence_pgd', {}).get('accuracy', 0)
        fgsm_acc = adv.get('sequence_fgsm', {}).get('accuracy', 0)
        hybrid_acc = adv.get('hybrid', {}).get('accuracy', 0)

        print(f"{model:<30} {clean_acc:>10.4f} {macro_f1:>10.4f} {feat_acc:>10.4f} {pgd_acc:>10.4f} {fgsm_acc:>10.4f} {hybrid_acc:>10.4f}")

    print(f"{'-'*100}")
    print('\n✅ Comparison complete.')
# ─── Git Push ──────────────────────────────────────────────────────────────────
import subprocess

print('Pushing to GitHub...')
result = subprocess.run(['git', 'add', '-A'], capture_output=True, text=True, cwd='/content/pfe')
print(f'  git add: {result.returncode}')

result = subprocess.run(['git', 'commit', '-m', 'Update models with countermeasures training'], capture_output=True, text=True, cwd='/content/pfe')
if result.returncode == 0:
    print(f'  git commit: OK')
else:
    print(f'  git commit: {result.stdout.strip()} {result.stderr.strip()}')

result = subprocess.run(['git', 'push'], capture_output=True, text=True, cwd='/content/pfe')
if result.returncode == 0:
    print('  ✅ Push successful!')
else:
    print(f'  git push stderr: {result.stderr.strip()[:500]}')

