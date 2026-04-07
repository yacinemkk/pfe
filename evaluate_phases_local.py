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
from tqdm import tqdm

# Ajouter le repo à sys.path
PROJECT_ROOT = Path("/home/pc/Desktop/pfe")
sys.path.insert(0, str(PROJECT_ROOT))
print(f"Projet PFE ajouté à sys.path: {PROJECT_ROOT}")

from src.models.lstm import IoTSequenceDataset
from src.models import (
    LSTMClassifier,
    TransformerClassifier,
    CNNLSTMClassifier,
    CNNClassifier,
    XGBoostLSTMClassifier,
)
from src.adversarial.attacks import (
    SensitivityAnalysis,
    AdversarialSearch,
)
from train_adversarial import load_and_preprocess_data

# Dossiers locaux
MODELS_DIR = PROJECT_ROOT / "results(2)" / "results" / "models"
DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "pcap"
    / "IPFIX Records (UNSW IoT Analytics)"
    / "20-01-31(1)"
)
OUTPUT_DIR = PROJECT_ROOT / "results(2)" / "crash_tests_plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Taille de séquence fixe pour les modèles transformer et cnn-bilstm-transformer
SEQ_LENGTH_TRANSFORMER = 25


def create_model(model_type, input_size, num_classes, seq_length):
    from config.config import (
        LSTM_CONFIG,
        TRANSFORMER_CONFIG,
        CNN_BILSTM_TRANSFORMER_CONFIG,
    )

    if model_type == "lstm":
        model = LSTMClassifier(input_size, num_classes, LSTM_CONFIG)
    elif model_type == "bilstm":
        from src.models.bilstm import BiLSTMClassifier

        model = BiLSTMClassifier(input_size, num_classes)
    elif model_type == "cnn_bilstm":
        from src.models.cnn_bilstm import CNNBiLSTMClassifier

        model = CNNBiLSTMClassifier(input_size, num_classes)
    elif model_type == "nlp_transformer":
        from src.models.transformer import NLPTransformerClassifier

        model = NLPTransformerClassifier(
            52000, num_classes, max_seq_length=576, pad_token_id=2
        )
    elif model_type == "cnn_bilstm_transformer":
        from src.models.cnn_bilstm_transformer import CNNBiLSTMTransformerClassifier

        model = CNNBiLSTMTransformerClassifier(
            input_size,
            num_classes,
            SEQ_LENGTH_TRANSFORMER,
            CNN_BILSTM_TRANSFORMER_CONFIG,
        )
    elif model_type == "transformer":
        model = TransformerClassifier(
            input_size, num_classes, SEQ_LENGTH_TRANSFORMER, TRANSFORMER_CONFIG
        )
    elif model_type == "cnn_lstm":
        model = CNNLSTMClassifier(input_size, num_classes)
    elif model_type == "xgboost_lstm":
        model = XGBoostLSTMClassifier(input_size, num_classes, LSTM_CONFIG)
    elif model_type == "cnn":
        model = CNNClassifier(input_size, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model.to(device)


def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in tqdm(dataloader, desc="Evaluating", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    return get_metrics(all_labels, all_preds)


def plot_phase_metrics(metrics_dict, model_name, save_path):
    phases = sorted(metrics_dict.keys(), key=lambda x: int(x.split(" ")[1]))
    metric_names = ["accuracy", "precision", "recall", "f1"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f"{model_name} - Métriques par Phase", fontsize=18, fontweight="bold")
    axes = axes.flatten()

    x = np.arange(len(phases))
    width = 0.2

    colors = ["royalblue", "orange", "mediumseagreen", "crimson"]

    for i, m_name in enumerate(metric_names):
        ax = axes[i]

        vals_norm = [metrics_dict[p]["Normal"][m_name] for p in phases]
        vals_feat = [metrics_dict[p]["Feature"][m_name] for p in phases]
        vals_search = [metrics_dict[p]["Seq Search"][m_name] for p in phases]

        rects1 = ax.bar(x - width, vals_norm, width, label="Normal", color=colors[0])
        rects2 = ax.bar(x, vals_feat, width, label="Feature Adv", color=colors[1])
        rects3 = ax.bar(
            x + width, vals_search, width, label="Seq Search", color=colors[2]
        )

        ax.set_ylabel(m_name.capitalize(), fontsize=12)
        ax.set_title(f"{m_name.capitalize()} par Phase", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(phases, fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)
        if i == 0:
            ax.legend(fontsize=11)

        for rects in [rects1, rects2, rects3]:
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path / f"metrics_évolution_{model_name}.png", dpi=300)
    plt.close()
    print(f"Plot enregistré: {save_path / f'metrics_évolution_{model_name}.png'}")


def display_dataset_info(data_dir, pipeline_mode="json", max_records=20000):
    """Affiche les informations détaillées du dataset chargé."""
    print("\n" + "=" * 70)
    print("  CHARGEMENT DU DATASET")
    print("=" * 70)

    print(f"\n  Mode pipeline : {pipeline_mode.upper()}")
    print(f"  Répertoire des données : {data_dir}")
    print(f"  Nombre max d'enregistrements : {max_records:,}")

    data = load_and_preprocess_data(
        seq_length=SEQ_LENGTH_TRANSFORMER,
        max_records=max_records,
        pipeline_mode=pipeline_mode,
        data_dir=data_dir,
    )

    X_train, X_val, X_test = data[0], data[1], data[2]
    y_train, y_val, y_test = data[3], data[4], data[5]
    features = data[6]
    scaler = data[7]
    label_encoder = data[8]
    n_continuous_features = data[9]

    print(f"\n  Récapitulatif des données :")
    print(f"    Features ({len(features)}) : {features[:5]}...")
    print(f"    Features continues : {n_continuous_features}")
    print(
        f"    Classes ({len(label_encoder.classes_)}) : {list(label_encoder.classes_)}"
    )
    print(f"\n  Shapes des séquences (seq_length={SEQ_LENGTH_TRANSFORMER}) :")
    print(f"    Train : {X_train.shape}  →  {len(X_train):,} séquences")
    print(f"    Val   : {X_val.shape}  →  {len(X_val):,} séquences")
    print(f"    Test  : {X_test.shape}  →  {len(X_test):,} séquences")
    print(f"\n  Distribution des classes (train) :")
    for cls in label_encoder.classes_:
        cls_id = label_encoder.transform([cls])[0]
        count = np.sum(y_train == cls_id)
        bar = "█" * max(1, count // 50)
        print(f"    {cls:<30} : {count:>6,}  {bar}")

    print("\n" + "=" * 70)

    return data


def verify_tokenization_for_transformers(
    X_sample, features, num_classes, seq_length=25
):
    """Vérifie que la tokenization fonctionne avec les 2 modèles transformer."""
    from src.data.tokenizer import IoTTokenizer, SimpleTokenizer, create_tokenizer
    from src.models.transformer import TransformerClassifier, NLPTransformerClassifier
    from src.models.cnn_bilstm_transformer import CNNBiLSTMTransformerClassifier
    from config.config import CNN_BILSTM_TRANSFORMER_CONFIG

    print("\n" + "=" * 70)
    print("  VÉRIFICATION DE LA TOKENIZATION")
    print("=" * 70)

    print("\n  [1] Création du tokenizer...")
    tokenizer = create_tokenizer()

    print("  [2] Entraînement du tokenizer sur un échantillon...")
    n_sample = min(500, len(X_sample))
    sample_indices = np.random.choice(len(X_sample), n_sample, replace=False)
    X_tok_sample = X_sample[sample_indices]

    tokenizer.fit(X_tok_sample, features, verbose=True)

    print("\n  [3] Transformation des séquences en tokens...")
    X_tokenized = tokenizer.transform(X_sample, features)
    print(f"      Shape des tokens : {X_tokenized.shape}")
    print(f"      Type : {X_tokenized.dtype}")
    print(f"      Token IDs uniques : {len(np.unique(X_tokenized))}")

    input_size = X_sample.shape[-1]

    print("\n  [4] Vérification avec TransformerClassifier...")
    transformer_model = TransformerClassifier(
        input_size=input_size,
        num_classes=num_classes,
        seq_length=seq_length,
    )

    x_tensor = torch.FloatTensor(X_sample[:2])
    try:
        out = transformer_model(x_tensor)
        print(
            f"      ✅ TransformerClassifier : input {x_tensor.shape} → output {out.shape}"
        )
        transformer_ok = True
    except Exception as e:
        print(f"      ❌ TransformerClassifier échec : {e}")
        transformer_ok = False

    print("\n  [5] Vérification avec NLPTransformerClassifier...")
    vocab_size = (
        tokenizer.tokenizer.get_vocab_size()
        if hasattr(tokenizer, "tokenizer") and tokenizer.tokenizer
        else 52000
    )
    nlp_transformer = NLPTransformerClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        max_seq_length=X_tokenized.shape[1],
    )

    x_tok_tensor = torch.LongTensor(X_tokenized[:2])
    try:
        out = nlp_transformer(x_tok_tensor)
        print(
            f"      ✅ NLPTransformerClassifier : input {x_tok_tensor.shape} → output {out.shape}"
        )
        nlp_ok = True
    except Exception as e:
        print(f"      ❌ NLPTransformerClassifier échec : {e}")
        nlp_ok = False

    print("\n  [6] Vérification avec CNNBiLSTMTransformerClassifier...")
    hybrid_model = CNNBiLSTMTransformerClassifier(
        input_size=input_size,
        num_classes=num_classes,
        seq_length=seq_length,
        config=CNN_BILSTM_TRANSFORMER_CONFIG,
    )

    try:
        out = hybrid_model(x_tensor)
        print(
            f"      ✅ CNNBiLSTMTransformerClassifier : input {x_tensor.shape} → output {out.shape}"
        )
        hybrid_ok = True
    except Exception as e:
        print(f"      ❌ CNNBiLSTMTransformerClassifier échec : {e}")
        hybrid_ok = False

    print("\n" + "=" * 70)
    print("  RÉSUMÉ DE LA VÉRIFICATION")
    print("=" * 70)
    print(
        f"    Tokenizer BPE           : {'✅ OK' if tokenizer.tokenizer else '❌ Échec'}"
    )
    print(f"    Vocabulaire size        : {vocab_size:,} tokens")
    print(
        f"    TransformerClassifier   : {'✅ Compatible (features brutes)' if transformer_ok else '❌ Incompatible'}"
    )
    print(
        f"    NLPTransformerClassifier: {'✅ Compatible (tokens)' if nlp_ok else '❌ Incompatible'}"
    )
    print(
        f"    CNNBiLSTMTransformer    : {'✅ Compatible (features brutes)' if hybrid_ok else '❌ Incompatible'}"
    )
    print("=" * 70 + "\n")

    return {
        "tokenizer": tokenizer,
        "transformer_ok": transformer_ok,
        "nlp_transformer_ok": nlp_ok,
        "hybrid_ok": hybrid_ok,
        "vocab_size": vocab_size,
    }


def run_local_crash_test():
    # Charger et afficher les infos du dataset
    print("\n" + "=" * 80)
    print("  CHARGEMENT DU DATASET AVANT ÉVALUATION")
    print("=" * 80)
    dataset_info = display_dataset_info(
        DATA_DIR, pipeline_mode="json", max_records=20000
    )

    # Vérifier la tokenization pour les modèles transformer
    X_test_raw = dataset_info[2]
    features_raw = dataset_info[6]
    label_encoder_raw = dataset_info[8]
    num_classes_raw = len(label_encoder_raw.classes_)

    tokenization_results = verify_tokenization_for_transformers(
        X_test_raw, features_raw, num_classes_raw, seq_length=SEQ_LENGTH_TRANSFORMER
    )

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

        print(f"\\n{'=' * 80}")
        print(f"Analyse du modèle: {model_folder}")
        print(f"{'=' * 80}")

        with open(prep_file, "rb") as f:
            prep_data = pickle.load(f)
            seq_length = prep_data["seq_length"]
            features = prep_data["features"]

        with open(res_file, "r") as f:
            res_data = json.load(f)
            model_type = res_data["model_type"]
            input_size = res_data["input_size"]
            num_classes = res_data["num_classes"]

        # Pour les modèles transformer et cnn-bilstm-transformer, utiliser seq_length=25
        if model_type in ["transformer", "cnn_bilstm_transformer", "nlp_transformer"]:
            seq_length = SEQ_LENGTH_TRANSFORMER
            print(
                f"  -> Modèle Transformer détecté. seq_length forcé à {SEQ_LENGTH_TRANSFORMER}"
            )

        phase_files = [
            f
            for f in os.listdir(model_path)
            if f.startswith("best_model_phase") and f.endswith(".pt")
        ]
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
                print(
                    " -> Modèle (CSV) détecté. Chargement des données via l'ancien Pipeline strict..."
                )
                import pandas as pd
                from src.data.preprocessor import IoTDataProcessor
                from config.config import LABEL_COLUMN

                scaler = prep_data["scaler"]
                label_encoder = prep_data["label_encoder"]
                if "feature_names" in prep_data:
                    old_features = prep_data["feature_names"]
                elif "features" in prep_data:
                    old_features = prep_data["features"]
                else:
                    raise ValueError("Failed to retrieve feature list.")

                proc = IoTDataProcessor()
                csv_dir = PROJECT_ROOT / "data" / "pcap" / "IPFIX ML Instances"
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
                del df, proc, X_mat, y_raw, X_norm, y_enc
                import gc

                gc.collect()
            else:
                print(
                    f" -> Modèle (JSON) détecté. Chargement des données via nouveau Pipeline (JSON)..."
                )
                data = load_and_preprocess_data(
                    seq_length,
                    max_records=20000,
                    pipeline_mode="json",
                    data_dir=DATA_DIR,
                )
                X_test, y_test = data[2], data[5]
                label_encoder, n_continuous_features = data[8], data[9]
                features = data[6]
        except Exception as e:
            print(f"Erreur data: {e}")
            import traceback

            traceback.print_exc()
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
            if "model_state_dict" in chk:
                tmp_model.load_state_dict(chk["model_state_dict"])
            else:
                tmp_model.load_state_dict(chk)
        else:
            chk = torch.load(model_path / phase_files[-1], map_location=device)
            if "model_state_dict" in chk:
                tmp_model.load_state_dict(chk["model_state_dict"])
            else:
                tmp_model.load_state_dict(chk)

        X_eval_flat = X_eval.reshape(-1, X_eval.shape[-1])
        y_eval_expanded = np.repeat(y_eval, X_eval.shape[1])
        sensitivity_analysis = SensitivityAnalysis(
            X_eval_flat,
            y_eval_expanded,
            features[: X_eval.shape[-1]],
            num_classes,
            n_continuous_features=n_continuous_features,
        )

        adversarial_search = AdversarialSearch(
            tmp_model,
            device,
            sensitivity_analysis=sensitivity_analysis,
            target_accuracy=0.5,
            batch_size=128,
        )

        print(" -> Génération adversariale (Mimicry par défaut)...")
        X_adv_default = sensitivity_analysis.generate_adversarial(
            X_eval, y_eval, verbose=False
        )

        print(" -> Analyse de sensibilité...")
        sensitivity_results = sensitivity_analysis.analyze(
            tmp_model, X_eval, y_eval, device, batch_size=128, verbose=False
        )

        print(" -> Génération Adversarial (Greedy Search)...")
        X_adv_search = adversarial_search.generate_adversarial(
            X_eval, y_eval, sensitivity_results=sensitivity_results, verbose=False
        )

        adv_dataset = IoTSequenceDataset(X_adv_default, y_eval)
        adv_loader = DataLoader(adv_dataset, batch_size=128, shuffle=False)

        adv_search_dataset = IoTSequenceDataset(X_adv_search, y_eval)
        adv_search_loader = DataLoader(
            adv_search_dataset, batch_size=128, shuffle=False
        )

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
            if "model_state_dict" in chk:
                model.load_state_dict(chk["model_state_dict"])
            else:
                model.load_state_dict(chk)
            model.eval()

            norm_metrics = evaluate_model(model, eval_loader)
            adv_metrics = evaluate_model(model, adv_loader)
            search_metrics = evaluate_model(model, adv_search_loader)

            print(
                f"  -> Normal:   Acc: {norm_metrics['accuracy']:.4f}, F1: {norm_metrics['f1']:.4f}"
            )
            print(
                f"  -> Adversarial:  Acc: {adv_metrics['accuracy']:.4f}, F1: {adv_metrics['f1']:.4f}"
            )
            print(
                f"  -> Greedy Search: Acc: {search_metrics['accuracy']:.4f}, F1: {search_metrics['f1']:.4f}"
            )

            phase_metrics[phase_name] = {
                "Normal": norm_metrics,
                "Adversarial": adv_metrics,
                "Greedy Search": search_metrics,
            }

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        plot_phase_metrics(phase_metrics, model_folder, OUTPUT_DIR)


if __name__ == "__main__":
    run_local_crash_test()
