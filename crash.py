# ═══════════════════════════════════════════════════════════════
# CRASH TEST — Charger le modèle LSTM depuis Drive + Évaluation
# Avec 5 Couches de Contremesures
# ═══════════════════════════════════════════════════════════════

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import gc
import json
import glob
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader

# ── 1. Monter Drive + Cloner le repo ─────────────────────────
from google.colab import drive

drive.mount("/content/drive")

if not os.path.exists("/content/pfe"):
    !git clone https://github.com/yacinemkk/pfe.git /content/pfe
else:
    !cd /content/pfe && git pull

DRIVE_ROOT = "/content/drive/MyDrive/PFE"
MODEL_DIR = os.path.join(DRIVE_ROOT, "results", "models", "lstm_countermeasures_csv")
DATA_DIR = os.path.join(DRIVE_ROOT, "results", "preprocessed", "csv")
PFE_ROOT = "/content/pfe"

# ── 2. Définir le modèle LSTM (même architecture que l'entraînement) ──


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes,
        hidden_size=64,
        num_layers=2,
        embedding_dim=128,
        dropout=0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )
        self.embedding_proj = nn.Linear(hidden_size, embedding_dim)
        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        hidden_out = hidden[-1]
        embedding = self.relu(self.embedding_proj(hidden_out))
        return self.classifier(embedding)


# ── 3. Dataset utilitaire ────────────────────────────────────


class IoTSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── 4. Charger le modèle sauvegardé ──────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_path = os.path.join(MODEL_DIR, "best_model.pt")
if not os.path.exists(ckpt_path):
    ckpt_path = os.path.join(MODEL_DIR, "model.pt")
state_dict = torch.load(ckpt_path, map_location=device)

meta_path = os.path.join(MODEL_DIR, "model_meta.json")
if os.path.exists(meta_path):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    input_size = meta["input_size"]
    num_classes = meta["num_classes"]
    seq_length = meta.get("seq_length", 10)
    hidden_size = meta.get("hidden_size", 64)
    num_layers = meta.get("num_layers", 2)
    embedding_dim = meta.get("embedding_dim", 128)
    dropout = meta.get("dropout", 0.3)
    feature_names = meta.get("feature_names", None)
    label_encoder_classes = meta.get("label_encoder_classes", None)
else:
    lstm_w = state_dict["lstm.weight_ih_l0"]
    hidden_size = lstm_w.shape[0] // 4
    input_size = lstm_w.shape[1]
    embedding_dim = state_dict["embedding_proj.weight"].shape[0]
    num_layers = 1
    for key in state_dict:
        if key.startswith("lstm.weight_ih_l"):
            layer_idx = int(key.split("_l")[1])
            num_layers = max(num_layers, layer_idx + 1)
    num_classes = (
        state_dict["classifier.4.weight"].shape[0]
        if "classifier.4.weight" in state_dict
        else state_dict[
            [
                k
                for k in state_dict
                if k.startswith("classifier.") and k.endswith(".weight")
            ][-1]
        ].shape[0]
    )
    seq_length = 10
    dropout = 0.3
    feature_names = None
    label_encoder_classes = None
    print(
        f"⚠️ model_meta.json non trouvé, dimensions auto-détectées depuis le checkpoint:"
    )
    print(
        f"   input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, "
        f"embedding_dim={embedding_dim}, num_classes={num_classes}"
    )

model = LSTMClassifier(
    input_size=input_size,
    num_classes=num_classes,
    hidden_size=hidden_size,
    num_layers=num_layers,
    embedding_dim=embedding_dim,
    dropout=dropout,
).to(device)

model.load_state_dict(state_dict)
model.eval()
print(f"✅ Modèle chargé depuis {ckpt_path}")
print(f"   input_size={input_size}, num_classes={num_classes}, seq_length={seq_length}")

# ── 5. Charger les données de test (depuis .npy pré-traités) ───
import pickle

npy_test_path = os.path.join(DATA_DIR, "X_test.npy")
npy_label_path = os.path.join(DATA_DIR, "y_test.npy")
meta_pkl_path = os.path.join(DATA_DIR, "csv_metadata.pkl")

if os.path.exists(npy_test_path) and os.path.exists(npy_label_path):
    X_test = np.load(npy_test_path)
    y_test = np.load(npy_label_path)
    print(f"✅ Données test chargées: X_test={X_test.shape}, y_test={y_test.shape}")

    if os.path.exists(meta_pkl_path):
        with open(meta_pkl_path, "rb") as f:
            csv_meta = pickle.load(f)
        feature_names = list(csv_meta["features"])
        seq_length = csv_meta.get("seq_length", seq_length)
        label_encoder = csv_meta.get("label_encoder", None)
        print(f"   Features ({len(feature_names)}): {feature_names[:5]}...")
        if label_encoder is not None:
            print(
                f"   Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}"
            )
    else:
        print("⚠️ csv_metadata.pkl non trouvé, feature_names par défaut")
else:
    raise FileNotFoundError(
        f"Fichiers .npy non trouvés dans {DATA_DIR}\nAttendu: X_test.npy, y_test.npy"
    )

# ── 6. SensitivityAnalysis + AdversarialSearch ───────────────
sys.path.insert(0, PFE_ROOT)
from src.adversarial.attacks import SensitivityAnalysis, AdversarialSearch

n_test = len(X_test)
n_half = n_test // 2
X_clean_reserved = X_test[:n_half]
y_clean_reserved = y_test[:n_half]
X_adv_src = X_test[n_half:]
y_adv_src = y_test[n_half:]

print(f"\n📊 Split test set:")
print(f"   Clean réservé : {len(X_clean_reserved)} échantillons")
print(f"   Source attaque : {len(X_adv_src)} échantillons")

sensitivity_analysis = SensitivityAnalysis(
    X_adv_src.reshape(-1, X_adv_src.shape[-1]),
    np.repeat(y_adv_src, X_adv_src.shape[1]),
    feature_names,
    num_classes,
)

sensitivity_results = sensitivity_analysis.analyze(
    model,
    X_adv_src,
    y_adv_src,
    device,
    batch_size=64,
    verbose=True,
)

adversarial_search = AdversarialSearch(
    model=model,
    device=device,
    sensitivity_analysis=sensitivity_analysis,
    target_accuracy=0.5,
    batch_size=64,
)

X_adv = adversarial_search.generate_adversarial(
    X_adv_src,
    y_adv_src,
    sensitivity_results=sensitivity_results,
    verbose=True,
)

del sensitivity_results
gc.collect()

print(f"\n⚔️  Exemples adverses générés: {X_adv.shape}")

# ═══════════════════════════════════════════════════════════════
# ── 7. CRASH TEST — Standard (sans contremesures) ─────────────
# ═══════════════════════════════════════════════════════════════
from src.training.evaluator import CrashTestEvaluator

print(f"\n{'=' * 70}")
print("  💥 CRASH TEST — STANDARD (sans contremesures)")
print(f"{'=' * 70}")

crash_evaluator = CrashTestEvaluator(model, device)
crash_results_standard = crash_evaluator.run_crash_test(
    X_clean_reserved,
    y_clean_reserved,
    X_adv,
    y_adv_src,
    batch_size=64,
    verbose=True,
)

print("\n" + "=" * 70)
print("  💥 CRASH TEST STANDARD — RÉSULTATS")
print("=" * 70)
for test_name, metrics in crash_results_standard.items():
    acc = metrics["accuracy"]
    f1 = metrics["macro_f1"]
    print(f"  {test_name:<25}  Acc={acc:.4f}  Macro-F1={f1:.4f}")

f1_clean_std = crash_results_standard["test1_clean"]["macro_f1"]
f1_adv_std = crash_results_standard["test2_adversarial"]["macro_f1"]
f1_drop_std = f1_clean_std - f1_adv_std
rr_std = f1_adv_std / max(f1_clean_std, 1e-8)
print(f"\n  📉 Chute F1: {f1_drop_std:.4f} ({f1_drop_std * 100:.1f}%) | RR: {rr_std:.4f}")

# ═══════════════════════════════════════════════════════════════
# ── 8. CRASH TEST — Avec Couche 3: DAE (Denoising Autoencoder) ──
# ═══════════════════════════════════════════════════════════════
from src.adversarial.denoising_autoencoder import (
    DenoisingAutoencoder,
    PerturbationGenerator,
)

print(f"\n{'=' * 70}")
print("  🛡️  CRASH TEST — Couche 3: DAE (Denoising Autoencoder)")
print(f"{'=' * 70}")

dae_path = os.path.join(MODEL_DIR, "dae_model.pt")
if os.path.exists(dae_path):
    dae = DenoisingAutoencoder(
        input_size=input_size,
        latent_dim=16,
        hidden_dim=32,
    ).to(device)
    dae.load_state_dict(torch.load(dae_path, map_location=device))
    dae.eval()
    print(f"  ✅ DAE chargé depuis {dae_path}")
else:
    print(f"  ⚠️  DAE non trouvé à {dae_path}, entraînement rapide...")
    pert_gen = PerturbationGenerator(
        n_features=input_size,
        p_zero=0.3,
        p_noise=0.3,
        p_scale=0.2,
        p_mimic=0.2,
        noise_std=0.3,
        benign_stats=sensitivity_analysis.benign_stats,
    )
    dae = DenoisingAutoencoder(
        input_size=input_size,
        latent_dim=16,
        hidden_dim=32,
    ).to(device)
    dae.train()
    optimizer_dae = torch.optim.Adam(dae.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    X_train_dae = torch.FloatTensor(X_adv_src).to(device)
    n_dae_epochs = 10
    for epoch in range(n_dae_epochs):
        total_dae_loss = 0
        for i in range(0, len(X_train_dae), 64):
            batch = X_train_dae[i : i + 64]
            X_pert = pert_gen.generate(batch)
            X_recon = dae(X_pert)
            loss = mse_loss(X_recon, batch)
            optimizer_dae.zero_grad()
            loss.backward()
            optimizer_dae.step()
            total_dae_loss += loss.item()
        print(f"    DAE Epoch {epoch + 1}/{n_dae_epochs} | MSE: {total_dae_loss / (len(X_train_dae) // 64):.4f}")
    dae.eval()
    print(f"  ✅ DAE entraîné")

    torch.save(dae.state_dict(), dae_path)
    print(f"  💾 DAE sauvegardé: {dae_path}")


def evaluate_with_dae(model, dae, X, y, device, batch_size=64):
    model.eval()
    dae.eval()
    correct, total = 0, 0
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_batch = torch.FloatTensor(X[i : i + batch_size]).to(device)
            X_denoised = dae(X_batch)
            outputs = model(X_denoised)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            total += len(predicted)
            y_batch = y[i : i + batch_size]
            correct += (predicted.cpu().numpy() == y_batch).sum()
    acc = correct / total if total > 0 else 0.0
    f1 = f1_score(y, all_preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": f1}


results_dae = {}
r1 = evaluate_with_dae(model, dae, X_clean_reserved, y_clean_reserved, device)
results_dae["test1_clean"] = r1
r2 = evaluate_with_dae(model, dae, X_adv, y_adv_src, device)
results_dae["test2_adversarial"] = r2
X_mix = np.concatenate([X_clean_reserved, X_adv])
y_mix = np.concatenate([y_clean_reserved, y_adv_src])
r3 = evaluate_with_dae(model, dae, X_mix, y_mix, device)
results_dae["test3_mixed"] = r3

print(f"\n  📊 DAE Crash Test Results:")
for test_name, metrics in results_dae.items():
    print(f"    {test_name:<25}  Acc={metrics['accuracy']:.4f}  F1={metrics['macro_f1']:.4f}")
f1_drop_dae = results_dae["test1_clean"]["macro_f1"] - results_dae["test2_adversarial"]["macro_f1"]
rr_dae = results_dae["test2_adversarial"]["macro_f1"] / max(results_dae["test1_clean"]["macro_f1"], 1e-8)
print(f"    📉 Chute F1: {f1_drop_dae:.4f} | RR: {rr_dae:.4f}")

# ═══════════════════════════════════════════════════════════════
# ── 9. CRASH TEST — Avec Couche 4: Randomized Smoothing ────────
# ═══════════════════════════════════════════════════════════════
from src.adversarial.randomized_smoothing import RandomizedSmoothing

print(f"\n{'=' * 70}")
print("  🛡️  CRASH TEST — Couche 4: Randomized Smoothing (σ=0.25, N=50)")
print(f"{'=' * 70}")

smoother = RandomizedSmoothing(model, device, sigma=0.25, n_samples=50)

rs_results_clean = smoother.evaluate_robust(X_clean_reserved, y_clean_reserved)
rs_results_adv = smoother.evaluate_robust(X_adv, y_adv_src)

results_rs = {
    "test1_clean": {"accuracy": rs_results_clean["clean_accuracy"], "macro_f1": rs_results_clean.get("pct_certified", 0)},
    "test2_adversarial": {"accuracy": rs_results_adv["clean_accuracy"], "macro_f1": rs_results_adv.get("adversarial_accuracy", 0)},
}
if "adversarial_accuracy" in rs_results_adv:
    results_rs["test2_adversarial"]["accuracy"] = rs_results_adv["adversarial_accuracy"]

print(f"\n  📊 Randomized Smoothing Results:")
print(f"    Clean Accuracy:  {rs_results_clean['clean_accuracy']:.4f}")
print(f"    Adv Accuracy:    {rs_results_adv.get('adversarial_accuracy', 'N/A')}")
print(f"    Mean Cert Radius: {rs_results_clean['mean_certified_radius']:.4f}")
print(f"    % Certified:     {rs_results_clean['pct_certified']:.4f}")

# ═══════════════════════════════════════════════════════════════
# ── 10. CRASH TEST — Toutes contremesures combinées ─────────────
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  🛡️  CRASH TEST — Toutes Contremesures Combinées (DAE + RS)")
print(f"{'=' * 70}")


def evaluate_dae_plus_rs(model, dae, smoother, X, y, device, batch_size=64):
    model.eval()
    dae.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_batch = torch.FloatTensor(X[i : i + batch_size])
            X_denoised = dae(X_batch.to(device)).cpu()
            preds = smoother.predict(X_denoised, batch_size=batch_size)
            all_preds.extend(preds.numpy() if hasattr(preds, 'numpy') else preds)
    acc = np.mean(np.array(all_preds) == y)
    f1 = f1_score(y, all_preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": f1}


results_combined = {}
r1 = evaluate_dae_plus_rs(model, dae, smoother, X_clean_reserved, y_clean_reserved, device)
results_combined["test1_clean"] = r1
r2 = evaluate_dae_plus_rs(model, dae, smoother, X_adv, y_adv_src, device)
results_combined["test2_adversarial"] = r2
r3 = evaluate_dae_plus_rs(model, dae, smoother, X_mix, y_mix, device)
results_combined["test3_mixed"] = r3

print(f"\n  📊 Combined (DAE + RS) Results:")
for test_name, metrics in results_combined.items():
    print(f"    {test_name:<25}  Acc={metrics['accuracy']:.4f}  F1={metrics['macro_f1']:.4f}")
f1_drop_combo = results_combined["test1_clean"]["macro_f1"] - results_combined["test2_adversarial"]["macro_f1"]
rr_combo = results_combined["test2_adversarial"]["macro_f1"] / max(results_combined["test1_clean"]["macro_f1"], 1e-8)
print(f"    📉 Chute F1: {f1_drop_combo:.4f} | RR: {rr_combo:.4f}")

# ═══════════════════════════════════════════════════════════════
# ── 11. RÉSUMÉ COMPARATIF ──────────────────────────────────────
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  📋 RÉSUMÉ COMPARATIF — Toutes les Contremesures")
print(f"{'=' * 70}")

print(f"\n  {'Méthode':<30} {'F1 Clean':>10} {'F1 Adv':>10} {'F1 Drop':>10} {'RR':>8}")
print(f"  {'─' * 30} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 8}")
print(f"  {'Standard (sans défense)':<30} {f1_clean_std:>10.4f} {f1_adv_std:>10.4f} {f1_drop_std:>10.4f} {rr_std:>8.4f}")

f1_clean_dae = results_dae["test1_clean"]["macro_f1"]
f1_adv_dae = results_dae["test2_adversarial"]["macro_f1"]
print(f"  {'Couche 3: DAE':<30} {f1_clean_dae:>10.4f} {f1_adv_dae:>10.4f} {f1_drop_dae:>10.4f} {rr_dae:>8.4f}")

f1_clean_combo = results_combined["test1_clean"]["macro_f1"]
f1_adv_combo = results_combined["test2_adversarial"]["macro_f1"]
print(f"  {'DAE + RS (Combiné)':<30} {f1_clean_combo:>10.4f} {f1_adv_combo:>10.4f} {f1_drop_combo:>10.4f} {rr_combo:>8.4f}")

print(f"\n  Remarque: Les Couches 1 (AFD) et 2 (TRADES corrigé) sont actives")
print(f"  pendant l'entraînement. Pour les évaluer, ré-entraînez le modèle")
print(f"  avec train_adversarial.py (les nouveaux paramètres sont appliqués).")
print(f"  La Couche 5 (Ensemble) se construit avec plusieurs modèles entraînés.")
print(f"{'=' * 70}")

# ── 12. Sauvegarder les résultats ─────────────────────────────
all_results = {
    "standard": crash_results_standard,
    "dae": results_dae,
    "randomized_smoothing": {
        "clean": rs_results_clean,
        "adversarial": rs_results_adv,
    },
    "combined_dae_rs": results_combined,
}

results_path = os.path.join(MODEL_DIR, "crash_test_results.json")
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n💾 Résultats sauvegardés: {results_path}")
