import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ajouter le repo à sys.path
PROJECT_ROOT = Path('/home/pc/Desktop/pfe')
sys.path.insert(0, str(PROJECT_ROOT))
print(f"Projet PFE ajouté à sys.path: {PROJECT_ROOT}")

from src.models.lstm import IoTSequenceDataset
from src.models import LSTMClassifier, TransformerClassifier, CNNLSTMClassifier, CNNClassifier, XGBoostLSTMClassifier
from src.adversarial.attacks import FeatureLevelAttack, SequenceLevelAttack, HybridAdversarialAttack
from train_adversarial import load_and_preprocess_data

# Dossiers locaux
MODELS_DIR = PROJECT_ROOT / 'results(2)' / 'results' / 'models'
# Le dossier "20-01-31(1)" où sont vos 3 fichiers json
DATA_DIR = PROJECT_ROOT / 'data' / 'pcap' / 'IPFIX Records (UNSW IoT Analytics)' / '20-01-31(1)'
OUTPUT_DIR = PROJECT_ROOT / 'results(2)' / 'crash_tests_plots'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

def create_model(model_type, input_size, num_classes, seq_length):
    from config.config import LSTM_CONFIG, TRANSFORMER_CONFIG
    if model_type == 'lstm':
        model = LSTMClassifier(input_size, num_classes, LSTM_CONFIG)
    elif model_type == 'transformer':
        model = TransformerClassifier(input_size, num_classes, seq_length, TRANSFORMER_CONFIG)
    elif model_type == 'cnn_lstm':
        model = CNNLSTMClassifier(input_size, num_classes)
    elif model_type == 'xgboost_lstm':
        model = XGBoostLSTMClassifier(input_size, num_classes, LSTM_CONFIG)
    elif model_type == 'cnn':
        model = CNNClassifier(input_size, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model.to(device)

def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    return get_metrics(all_labels, all_preds)

def plot_phase_metrics(metrics_dict, model_name, save_path):
    phases = sorted(metrics_dict.keys(), key=lambda x: int(x.split(' ')[1]))
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f"{model_name} - Métriques par Phase", fontsize=18, fontweight='bold')
    axes = axes.flatten()
    
    x = np.arange(len(phases))
    width = 0.2
    
    colors = ['royalblue', 'orange', 'mediumseagreen', 'crimson']
    
    for i, m_name in enumerate(metric_names):
        ax = axes[i]
        
        vals_norm = [metrics_dict[p]['Normal'][m_name] for p in phases]
        vals_feat = [metrics_dict[p]['Feature'][m_name] for p in phases]
        vals_fgsm = [metrics_dict[p]['Seq FGSM'][m_name] for p in phases]
        vals_pgd = [metrics_dict[p]['Seq PGD'][m_name] for p in phases]
        
        rects1 = ax.bar(x - 1.5*width, vals_norm, width, label='Normal', color=colors[0])
        rects2 = ax.bar(x - 0.5*width, vals_feat, width, label='Feature Adv', color=colors[1])
        rects3 = ax.bar(x + 0.5*width, vals_fgsm, width, label='Seq FGSM', color=colors[2])
        rects4 = ax.bar(x + 1.5*width, vals_pgd, width, label='Seq PGD', color=colors[3])
        
        ax.set_ylabel(m_name.capitalize(), fontsize=12)
        ax.set_title(f"{m_name.capitalize()} par Phase", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(phases, fontsize=12)
        ax.set_ylim(0, 1.1) 
        ax.grid(axis='y', alpha=0.3)
        if i == 0:
            ax.legend(fontsize=11)
            
        for rects in [rects1, rects2, rects3, rects4]:
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f"{height:.2f}",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, rotation=90)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path / f"metrics_évolution_{model_name}.png", dpi=300)
    plt.close()
    print(f"Plot enregistré: {save_path / f'metrics_évolution_{model_name}.png'}")

def run_local_crash_test():
    if not MODELS_DIR.exists():
        print(f"Modèles introuvables au chemin : {MODELS_DIR}")
        return
        
    for model_folder in sorted(os.listdir(MODELS_DIR)):
        model_path = MODELS_DIR / model_folder
        if not model_path.is_dir():
            continue
            
        prep_file = model_path / "preprocessor.pkl"
        res_file = model_path / "results.json"
        
        if not prep_file.exists() or not res_file.exists():
            continue
            
        print(f"\\n{'='*80}")
        print(f"Analyse du modèle: {model_folder}")
        print(f"{'='*80}")
        
        with open(prep_file, "rb") as f:
            prep_data = pickle.load(f)
            seq_length = prep_data["seq_length"]
            features = prep_data["features"]
            
        with open(res_file, "r") as f:
            res_data = json.load(f)
            model_type = res_data["model_type"]
            input_size = res_data["input_size"]
            num_classes = res_data["num_classes"]
            
        phase_files = [f for f in os.listdir(model_path) if f.startswith("best_model_phase") and f.endswith(".pt")]
        if not phase_files:
            print(f"Aucun checkpoint de phase trouvé pour {model_folder}")
            continue
            
        phase_files.sort(key=lambda x: int(x.split("phase")[1].split(".pt")[0]))
        
        # Add the best_model.pt as the final phase
        if (model_path / "best_model.pt").exists():
            phase_files.append("best_model.pt")
            
        print(f"Chargement des données pour le test...")
        try:
            if input_size == 37:
                print(" -> Modèle (CSV) détecté. Chargement des données via l'ancien Pipeline strict...")
                import pandas as pd
                from src.data.preprocessor import IoTDataProcessor
                from config.config import LABEL_COLUMN
                
                scaler = prep_data["scaler"]
                label_encoder = prep_data["label_encoder"]
                if "feature_names" in prep_data:
                    old_features = prep_data["feature_names"]
                elif "features" in prep_data:
                    old_features = prep_data["features"]
                else: raise ValueError("Failed to retrieve feature list.")
                
                proc = IoTDataProcessor()
                csv_dir = PROJECT_ROOT / 'data' / 'pcap' / 'IPFIX ML Instances'
                csv_file = sorted(csv_dir.glob("home*_labeled.csv"))[0]
                df = pd.read_csv(csv_file, nrows=200000)
                
                df = df.drop_duplicates().dropna(subset=[LABEL_COLUMN])
                df = df[df[LABEL_COLUMN].isin(label_encoder.classes_)]
                
                X_mat = df[old_features].values
                y_raw = df[LABEL_COLUMN].values
                
                X_norm = scaler.transform(X_mat)
                y_enc = label_encoder.transform(y_raw)
                
                X_test, y_test = proc.create_sequences(X_norm, y_enc)
                n_continuous_features = 37
                features = old_features
                print(f"  Load OK: shape={X_test.shape}")
                del df, proc, X_mat, y_raw, X_norm, y_enc; import gc; gc.collect()
            else:
                print(f" -> Modèle (JSON) détecté. Chargement des données via nouveau Pipeline (JSON)...")
                data = load_and_preprocess_data(seq_length, max_records=20000, pipeline_mode="json", data_dir=DATA_DIR)
                X_test, y_test = data[2], data[5]
                label_encoder, n_continuous_features = data[8], data[9]
                features = data[6]
        except Exception as e:
            print(f"Erreur data: {e}")
            import traceback; traceback.print_exc()
            continue
            
        n_eval = min(5000, len(X_test))
        eval_indices = np.random.choice(len(X_test), n_eval, replace=False)
        X_eval = X_test[eval_indices]
        y_eval = y_test[eval_indices]
        
        # Backward compatibility for old models trained with 37 features instead of 36
        if X_eval.shape[-1] < input_size:
            pad_size = input_size - X_eval.shape[-1]
            pad = np.zeros((*X_eval.shape[:-1], pad_size))
            X_eval = np.concatenate([X_eval, pad], axis=-1)
            n_continuous_features += pad_size
            features = list(features) + ["padded"] * pad_size
        
        eval_dataset = IoTSequenceDataset(X_eval, y_eval)
        eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
        
        print("Préparation des attaques...")
        tmp_model = create_model(model_type, input_size, num_classes, seq_length)
        final_model_path = model_path / "best_model.pt"
        if final_model_path.exists():
            chk = torch.load(final_model_path, map_location=device)
            if 'model_state_dict' in chk: tmp_model.load_state_dict(chk['model_state_dict'])
            else: tmp_model.load_state_dict(chk)
        else:
            chk = torch.load(model_path / phase_files[-1], map_location=device)
            if 'model_state_dict' in chk: tmp_model.load_state_dict(chk['model_state_dict'])
            else: tmp_model.load_state_dict(chk)
            
        sequence_attack = SequenceLevelAttack(
            tmp_model, device, epsilon=0.1, alpha=0.01, num_steps=10, 
            n_continuous_features=n_continuous_features
        )
        
        X_eval_flat = X_eval.reshape(-1, X_eval.shape[-1])
        y_eval_expanded = np.repeat(y_eval, X_eval.shape[1])
        feature_attack = FeatureLevelAttack(
            X_eval_flat, y_eval_expanded, features[:X_eval.shape[-1]], num_classes,
            n_continuous_features=n_continuous_features
        )
        
        print(" -> Génération Feature-Level...")
        X_adv_feature_flat = feature_attack.generate_batch(X_eval_flat, y_eval_expanded, verbose=False)
        X_adv_feature = X_adv_feature_flat.reshape(X_eval.shape)
        
        print(" -> Génération Sequence-Level (FGSM)...")
        X_adv_fgsm = sequence_attack.generate_batch(X_eval, y_eval, method="fgsm", verbose=False)
        
        print(" -> Génération Sequence-Level (PGD)...")
        X_adv_pgd = sequence_attack.generate_batch(X_eval, y_eval, method="pgd", verbose=False)
        
        adv_feat_dataset = IoTSequenceDataset(X_adv_feature, y_eval)
        adv_feat_loader = DataLoader(adv_feat_dataset, batch_size=128, shuffle=False)
        
        adv_fgsm_dataset = IoTSequenceDataset(X_adv_fgsm, y_eval)
        adv_fgsm_loader = DataLoader(adv_fgsm_dataset, batch_size=128, shuffle=False)
        
        adv_pgd_dataset = IoTSequenceDataset(X_adv_pgd, y_eval)
        adv_pgd_loader = DataLoader(adv_pgd_dataset, batch_size=128, shuffle=False)
        
        phase_metrics = {}
        
        for p_file in phase_files:
            if p_file == "best_model.pt":
                phase_name = f"Phase {len(phase_files)}"
            else:
                phase_num = p_file.split("phase")[1].split(".pt")[0]
                phase_name = f"Phase {phase_num}"
            print(f"\\n--- Évaluation des métriques pour {phase_name} ---")
            
            model = create_model(model_type, input_size, num_classes, seq_length)
            chk = torch.load(model_path / p_file, map_location=device)
            if 'model_state_dict' in chk: model.load_state_dict(chk['model_state_dict'])
            else: model.load_state_dict(chk)
            model.eval()
            
            norm_metrics = evaluate_model(model, eval_loader)
            feat_metrics = evaluate_model(model, adv_feat_loader)
            fgsm_metrics = evaluate_model(model, adv_fgsm_loader)
            pgd_metrics = evaluate_model(model, adv_pgd_loader)
            
            print(f"  -> Normal:   Acc: {norm_metrics['accuracy']:.4f}, F1: {norm_metrics['f1']:.4f}")
            print(f"  -> Feature:  Acc: {feat_metrics['accuracy']:.4f}, F1: {feat_metrics['f1']:.4f}")
            print(f"  -> Seq FGSM: Acc: {fgsm_metrics['accuracy']:.4f}, F1: {fgsm_metrics['f1']:.4f}")
            print(f"  -> Seq PGD:  Acc: {pgd_metrics['accuracy']:.4f}, F1: {pgd_metrics['f1']:.4f}")
            
            phase_metrics[phase_name] = {
                "Normal": norm_metrics,
                "Feature": feat_metrics,
                "Seq FGSM": fgsm_metrics,
                "Seq PGD": pgd_metrics
            }
            
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        plot_phase_metrics(phase_metrics, model_folder, OUTPUT_DIR)

if __name__ == "__main__":
    run_local_crash_test()
