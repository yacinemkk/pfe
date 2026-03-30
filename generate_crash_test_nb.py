import json

notebook = {
  "cells": [],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}

def add_markdown(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")]
    })

md_intro = """# Crash Test exhaustif par Phase pour les modèles IoT Adversarial
Ce notebook permet d'évaluer les performances (Accuracy, Recall, Precision, F1-Score) sur:
1. Données **Normales**
2. Données **Adversariales Feature-Level**
3. Données **Adversariales Sequence-Level (FGSM)**
4. Données **Adversariales Sequence-Level (PGD)**

L'évaluation se fait pour **chaque phase d'entraînement** des modèles enregistrés, permettant d'observer l'acquisition de la robustesse.
Assurez-vous d'avoir téléchargé votre dossier de projet dans votre Google Drive."""
add_markdown(md_intro)

code_mount = """from google.colab import drive
drive.mount('/content/drive')

import os
# Clonage ou mise à jour du repository GitHub
if os.path.exists('/content/pfe'):
    !cd /content/pfe && git pull
else:
    !git clone https://github.com/yacinemkk/pfe.git /content/pfe

%cd /content/pfe"""
add_code(code_mount)

code_path = """import sys
from pathlib import Path

# Ajouter le repo à sys.path pour les imports 'src'
sys.path.insert(0, '/content/pfe')
print("Projet PFE ajouté à sys.path")

# Remplacez ce chemin par le chemin réel vers le dossier PFE dans votre Drive
DRIVE_PFE_DIR = Path('/content/drive/MyDrive/PFE')

# Models directory path
MODELS_DIR = DRIVE_PFE_DIR / 'results(2)'
DATA_DIR = DRIVE_PFE_DIR / 'IPFIX_Records'"""
add_code(code_path)

code_imports = """import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm.notebook import tqdm
import gc

# Import modules from your project
from src.models.lstm import IoTSequenceDataset
from src.models import LSTMClassifier, TransformerClassifier, CNNLSTMClassifier, CNNClassifier, XGBoostLSTMClassifier, CNNBiLSTMTransformerClassifier
from src.adversarial.attacks import FeatureLevelAttack, SequenceLevelAttack, HybridAdversarialAttack
from train_adversarial import load_and_preprocess_data"""
add_code(code_imports)

md_utils = """## Fonctions Utilitaires
*   `create_model`: Instancier le modèle
*   `evaluate_model`: Calcule Accuracy, Précision, Rappel et F1 macro
*   `plot_phase_metrics`: Dessine les graphiques demandés par phase pour comparer les 4 types de données"""
add_markdown(md_utils)

code_utils = """device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

def create_model(model_type, input_size, num_classes, seq_length):
    from config.config import LSTM_CONFIG, TRANSFORMER_CONFIG, CNN_BILSTM_TRANSFORMER_CONFIG
    if model_type == 'lstm':
        model = LSTMClassifier(input_size, num_classes, LSTM_CONFIG)
    elif model_type == 'bilstm':
        from src.models.bilstm import BiLSTMClassifier
        model = BiLSTMClassifier(input_size, num_classes)
    elif model_type == 'cnn_bilstm':
        from src.models.cnn_bilstm import CNNBiLSTMClassifier
        model = CNNBiLSTMClassifier(input_size, num_classes)
    elif model_type == 'nlp_transformer':
        from src.models.transformer import NLPTransformerClassifier
        model = NLPTransformerClassifier(52000, num_classes, 512)
    elif model_type == 'cnn_bilstm_transformer':
        model = CNNBiLSTMTransformerClassifier(input_size, num_classes, seq_length, CNN_BILSTM_TRANSFORMER_CONFIG)
    elif model_type == 'transformer':
        model = TransformerClassifier(
            input_size, num_classes, seq_length, TRANSFORMER_CONFIG
        )
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

def plot_phase_metrics(metrics_dict, model_name):
    phases = sorted(metrics_dict.keys(), key=lambda x: int(x.split(' ')[1]))
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f"{model_name} - Métriques par Phase", fontsize=18, fontweight='bold')
    axes = axes.flatten()
    
    x = np.arange(len(phases))
    width = 0.2  # On a 4 barres, on réduit la largeur
    
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
            
        # Affichage des valeurs sur les barres (rotation pour gain de place)
        for rects in [rects1, rects2, rects3, rects4]:
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f"{height:.2f}",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, rotation=90)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()"""
add_code(code_utils)

md_eval = """## Evaluation Exhaustive des Modèles
Ce block va scanner automatiquement le modèle sélectionné, créer les attaques Feature-Level, FGSM et PGD sur les données de test, puis évaluer les metrics pour chaque phase de ce modèle."""
add_markdown(md_eval)

code_eval = """def run_crash_test_for_all_models():
    if not isinstance(MODELS_DIR, Path) or not MODELS_DIR.exists():
        print(f"Modèles introuvables au chemin : {MODELS_DIR}")
        return
        
    # On va iterer par dossier modele trouvé
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
            
        # Identifier les phases
        phase_files = [f for f in os.listdir(model_path) if f.startswith("best_model_phase") and f.endswith(".pt")]
        if not phase_files:
            print(f"Aucun checkpoint de phase trouvé pour {model_folder}")
            continue
            
        # Tri correct : Phase 1, Phase 2, ...
        phase_files.sort(key=lambda x: int(x.split("phase")[1].split(".pt")[0]))
        
        print(f"Chargement des données pour le test...")
        try:
            if input_size == 37:
                print(" -> Modèle (CSV) détecté. Chargement des données via l'ancien Pipeline strict...")
                
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
                
                X_test, y_test = proc.create_sequences(X_norm, y_enc, seq_length=seq_length) # Added seq_length here
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
            
        # Add the best_model.pt as the final phase
        if (model_path / "best_model.pt").exists():
            phase_files.append("best_model.pt")
            
        # Selection de donnees de test aleatoires pour le crash test (~2000 examples max ou ce qui existe)
        n_eval = min(2000, len(X_test))
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
        
        print("Préparation des attaques adversariales (Feature, FGSM, PGD)...")
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
            
        # Setup Sequence-Level Attack
        sequence_attack = SequenceLevelAttack(
            tmp_model, device, epsilon=0.1, alpha=0.01, num_steps=10, 
            n_continuous_features=n_continuous_features
        )
        
        # Setup Feature-Level Attack
        X_eval_flat = X_eval.reshape(-1, X_eval.shape[-1])
        y_eval_expanded = np.repeat(y_eval, X_eval.shape[1])
        feature_attack = FeatureLevelAttack(
            X_eval_flat, y_eval_expanded, features[:X_eval.shape[-1]], num_classes,
            n_continuous_features=n_continuous_features
        )
        
        # Generation
        print("  -> Génération Feature-Level...")
        X_adv_feature_flat = feature_attack.generate_batch(X_eval_flat, y_eval_expanded, verbose=False)
        X_adv_feature = X_adv_feature_flat.reshape(X_eval.shape)
        
        print("  -> Génération Sequence-Level (FGSM)...")
        X_adv_fgsm = sequence_attack.generate_batch(X_eval, y_eval, method="fgsm", verbose=False)
        
        print("  -> Génération Sequence-Level (PGD)...")
        X_adv_pgd = sequence_attack.generate_batch(X_eval, y_eval, method="pgd", verbose=False)
        
        # Loaders
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
            
            # Normal Evaluation
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
            torch.cuda.empty_cache()
            
        # Draw Plot for the model showing metrics per phase
        plot_phase_metrics(phase_metrics, model_folder)

# Lancer la boucle de test pour tous les modèles
run_crash_test_for_all_models()"""
add_code(code_eval)

save_path = '/home/pc/Desktop/pfe/CrashTest_Phases_Colab.ipynb'
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)
print(f"Notebook Created at {save_path}!")
