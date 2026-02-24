# =====================================================
# ENTRAÎNEMENT TRANSFORMER - Identification IoT
# Dataset: IoT IPFIX Home (18 classes)
# Features: SDN-compatible (sans IP/ports/MAC)
# Attaques: Adversariales IoT-SDN spécifiques
# =====================================================

# 1. Monter Google Drive
from google.colab import drive

drive.mount("/content/drive")

# 2. Imports
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# =====================================================
# CONFIGURATION
# =====================================================
DATA_DIR = "/content/drive/MyDrive/IPFIX ML Instances"
SEQUENCE_LENGTH = 10
BATCH_SIZE = 64
EPOCHS_PHASE1 = 50
EPOCHS_PHASE2 = 30
ADVERSARIAL_RATIO = 0.2

# Hyperparamètres Transformer
NUM_HEADS = 8
FF_DIM = 128
NUM_TRANSFORMER_BLOCKS = 4
EMBEDDING_DIM = 64

# 18 Classes IoT IPFIX Home
VALID_CLASSES = [
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

# Features non-modifiables (selon documentation)
NON_MODIFIABLE_FEATURES = [
    "ipProto",
    "http",
    "https",
    "dns",
    "ntp",
    "tcp",
    "udp",
    "ssdp",
]
DEPENDENT_FEATURES = {
    "inPacketCount": "outPacketCount",
    "inByteCount": "outByteCount",
    "inAvgIAT": "outAvgIAT",
    "inAvgPacketSize": "outAvgPacketSize",
}

print("TensorFlow version:", tf.__version__)
print("GPU disponible:", len(tf.config.list_physical_devices("GPU")) > 0)

# =====================================================
# 3. COLONNES À SUPPRIMER
# =====================================================
COLS_TO_DROP = [
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

# =====================================================
# 4. FONCTIONS DE PRÉTRAITEMENT
# =====================================================


def load_all_data(data_dir):
    all_data = []
    for i in range(1, 13):
        csv_file = os.path.join(data_dir, f"home{i}_labeled.csv")
        if os.path.exists(csv_file):
            print(f"Chargement {csv_file}...")
            df = pd.read_csv(csv_file)
            all_data.append(df)

    df = pd.concat(all_data, ignore_index=True)
    print(f"Dataset total: {len(df):,} lignes")
    return df


def filter_classes(df, valid_classes):
    df_filtered = df[df["name"].isin(valid_classes)].copy()
    print(f"Après filtrage: {len(df_filtered):,} lignes")

    print("\nDistribution:")
    for cls in valid_classes:
        count = len(df_filtered[df_filtered["name"] == cls])
        print(f"  {cls}: {count:,}")

    return df_filtered


def preprocess_for_sdn(df):
    df = df.drop_duplicates()
    print(f"Après doublons: {len(df):,}")

    df = df.dropna()
    print(f"Après NaN: {len(df):,}")

    feature_cols = [col for col in df.columns if col not in COLS_TO_DROP]
    print(f"\nFeatures SDN ({len(feature_cols)}): {feature_cols}")

    X = df[feature_cols].values.astype(np.float32)
    y = df["name"].values

    return X, y, feature_cols


def normalize_and_standardize(X_train, X_test):
    scaler_norm = MinMaxScaler()
    X_train_norm = scaler_norm.fit_transform(X_train)
    X_test_norm = scaler_norm.transform(X_test)

    scaler_std = StandardScaler()
    X_train_scaled = scaler_std.fit_transform(X_train_norm)
    X_test_scaled = scaler_std.transform(X_test_norm)

    return X_train_scaled, X_test_scaled, scaler_norm, scaler_std


def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []

    for i in range(len(X) - seq_length):
        X_seq.append(X[i : i + seq_length])
        y_seq.append(y[i + seq_length])

    return np.array(X_seq), np.array(y_seq)


# =====================================================
# 5. ATTAQUE ADVERSARIALE IoT-SDN
# =====================================================


class IoTAdversarialAttack:
    """
    Attaque adversariale spécifique IoT-SDN:
    x_adv = Projection[x0 + c·t·mask·sign(μ_target - x0)·|Difference(μ_target, x0)|]
    """

    def __init__(self, X_train, y_train, feature_cols, label_encoder):
        self.X_train = X_train
        self.y_train = y_train
        self.feature_cols = feature_cols
        self.label_encoder = label_encoder
        self.num_classes = len(label_encoder.classes_)

        self.modifiable_indices = self._get_modifiable_indices()
        self.dependent_pairs = self._get_dependent_pairs()
        self.class_centroids = self._compute_class_centroids()
        self.nearest_classes = self._find_nearest_classes()
        self.masks = self._generate_masks()

    def _get_modifiable_indices(self):
        modifiable = []
        for i, col in enumerate(self.feature_cols):
            if col not in NON_MODIFIABLE_FEATURES:
                modifiable.append(i)
        return np.array(modifiable)

    def _get_dependent_pairs(self):
        pairs = []
        for dep, indep in DEPENDENT_FEATURES.items():
            if dep in self.feature_cols and indep in self.feature_cols:
                dep_idx = self.feature_cols.index(dep)
                indep_idx = self.feature_cols.index(indep)
                pairs.append((indep_idx, dep_idx))
        return pairs

    def _compute_class_centroids(self):
        centroids = {}
        for class_idx in range(self.num_classes):
            mask = self.y_train == class_idx
            if np.sum(mask) > 0:
                centroids[class_idx] = np.mean(self.X_train[mask], axis=0)
        return centroids

    def _find_nearest_classes(self, k=3):
        """L2 minimization: trouve les k classes les plus proches"""
        nearest = {}
        class_indices = list(self.class_centroids.keys())
        centroids_matrix = np.array([self.class_centroids[i] for i in class_indices])

        for class_idx in class_indices:
            centroid = self.class_centroids[class_idx]
            distances = np.sqrt(np.sum((centroids_matrix - centroid) ** 2, axis=1))
            distances[class_indices.index(class_idx)] = np.inf
            nearest_indices = np.argsort(distances)[:k]
            nearest[class_idx] = [class_indices[i] for i in nearest_indices]

        return nearest

    def _generate_masks(self, n_masks=20):
        """L0 minimization: génère les masques les plus impactants"""
        n_features = len(self.modifiable_indices)
        masks = []

        # Masque complet
        masks.append(np.ones(len(self.feature_cols)))

        # Masques avec différents pourcentages
        for pct in [0.25, 0.5, 0.75]:
            n_active = max(1, int(n_features * pct))
            for _ in range(3):
                mask = np.zeros(len(self.feature_cols))
                active_indices = np.random.choice(
                    self.modifiable_indices, n_active, replace=False
                )
                mask[active_indices] = 1
                masks.append(mask)

        # Masques basés sur la variance
        variances = np.var(self.X_train, axis=0)
        for top_k in [5, 10, 15]:
            mask = np.zeros(len(self.feature_cols))
            top_indices = np.argsort(variances)[-top_k:]
            mask[top_indices] = 1
            masks.append(mask)

        return np.array(masks[:n_masks])

    def projection(self, X):
        """Maintient les contraintes sémantiques"""
        X_proj = X.copy()
        X_proj = np.clip(X_proj, -3.0, 3.0)

        # Cohérence features dépendantes
        for indep_idx, dep_idx in self.dependent_pairs:
            ratio = np.abs(X_proj[:, dep_idx]) / (np.abs(X_proj[:, indep_idx]) + 1e-8)
            X_proj[:, dep_idx] = X_proj[:, indep_idx] * np.mean(ratio)

        return X_proj

    def difference_function(self, mu_target, x0):
        return np.abs(mu_target - x0)

    def generate_adversarial(self, model, X, y, max_iter=20, c=0.1):
        """Génère des exemples adversariaux IoT-SDN"""
        X_adv = X.copy()
        n_samples = len(X)

        for i in range(n_samples):
            x0 = X[i]
            true_class = np.argmax(y[i])

            # Choisir classe cible parmi les 3 plus proches
            if true_class in self.nearest_classes:
                target_candidates = self.nearest_classes[true_class]
                target_class = np.random.choice(target_candidates)
            else:
                other_classes = [c for c in range(self.num_classes) if c != true_class]
                target_class = np.random.choice(other_classes)

            if target_class not in self.class_centroids:
                continue

            mu_target = self.class_centroids[target_class]
            mask = self.masks[np.random.randint(len(self.masks))]

            x_adv = x0.copy()
            for t in range(1, max_iter + 1):
                diff = self.difference_function(mu_target, x0)
                direction = np.sign(mu_target - x0)
                perturbation = c * t * mask * direction * diff

                x_adv = x0 + perturbation
                x_adv = self.projection(x_adv.reshape(1, -1)).flatten()

                pred = model.predict(x_adv.reshape(1, SEQUENCE_LENGTH, -1), verbose=0)
                pred_class = np.argmax(pred)

                if pred_class == target_class:
                    X_adv[i] = x_adv
                    break

            X_adv[i] = x_adv

        return X_adv


# =====================================================
# 6. POSITIONAL ENCODING
# =====================================================


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_length, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.seq_length = seq_length
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            shape=(self.seq_length, self.embed_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="positional_embedding",
        )

    def call(self, inputs):
        return inputs + self.pos_emb

    def get_config(self):
        config = super().get_config()
        config.update({"seq_length": self.seq_length, "embed_dim": self.embed_dim})
        return config


# =====================================================
# 7. BLOC TRANSFORMER
# =====================================================


def transformer_block(inputs, num_heads, ff_dim, dropout=0.1):
    """Bloc encodeur Transformer avec multi-head attention"""

    attn_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=inputs.shape[-1] // num_heads, dropout=dropout
    )(inputs, inputs)
    attn_output = Dropout(dropout)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ff_output = Dense(ff_dim, activation="relu")(out1)
    ff_output = Dropout(dropout)(ff_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ff_output)

    return out2


def create_transformer_model(
    input_shape,
    num_classes,
    embed_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    num_blocks=NUM_TRANSFORMER_BLOCKS,
    dropout=0.3,
):
    inputs = Input(shape=input_shape)

    # Projection vers embedding
    x = Dense(embed_dim)(inputs)

    # Positional Encoding
    x = PositionalEncoding(input_shape[0], embed_dim)(x)

    # Blocs Transformer
    for _ in range(num_blocks):
        x = transformer_block(x, num_heads, ff_dim, dropout)

    # Global Pooling
    x = GlobalAveragePooling1D()(x)

    # Classification head
    x = Dense(256, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(dropout)(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


# =====================================================
# 8. CHARGEMENT DONNÉES
# =====================================================
print("\n" + "=" * 60)
print("CHARGEMENT ET PRÉTRAITEMENT")
print("=" * 60)

df = load_all_data(DATA_DIR)
df = filter_classes(df, VALID_CLASSES)
X, y, feature_cols = preprocess_for_sdn(df)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
print(f"\nClasses: {num_classes}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

X_train_scaled, X_test_scaled, scaler_norm, scaler_std = normalize_and_standardize(
    X_train, X_test
)

print(f"\nCréation séquences (longueur={SEQUENCE_LENGTH})...")
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, SEQUENCE_LENGTH)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, SEQUENCE_LENGTH)

print(f"X_train_seq: {X_train_seq.shape}")
print(f"X_test_seq: {X_test_seq.shape}")

y_train_cat = to_categorical(y_train_seq, num_classes)
y_test_cat = to_categorical(y_test_seq, num_classes)

# =====================================================
# 9. PHASE 1: ENTRAÎNEMENT TRAFIC BÉNIN
# =====================================================
print("\n" + "=" * 60)
print("PHASE 1: ENTRAÎNEMENT TRAFIC BÉNIN (100%)")
print("=" * 60)

input_shape = (SEQUENCE_LENGTH, len(feature_cols))
model_transformer = create_transformer_model(input_shape, num_classes)
model_transformer.summary()

callbacks_phase1 = [
    EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    ),
    ModelCheckpoint(
        "/content/drive/MyDrive/transformer_phase1_benin.h5",
        save_best_only=True,
        monitor="val_loss",
        verbose=1,
    ),
]

history_phase1 = model_transformer.fit(
    X_train_seq,
    y_train_cat,
    validation_split=0.2,
    epochs=EPOCHS_PHASE1,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_phase1,
    verbose=1,
)

loss_p1, acc_p1 = model_transformer.evaluate(X_test_seq, y_test_cat, verbose=0)
print(f"\nPhase 1 - Loss: {loss_p1:.4f} | Accuracy: {acc_p1:.4f}")

# =====================================================
# 10. PHASE 2: ADVERSARIAL TRAINING IoT-SDN
# =====================================================
print("\n" + "=" * 60)
print("PHASE 2: ADVERSARIAL TRAINING IoT-SDN")
print("=" * 60)

attack_generator = IoTAdversarialAttack(
    X_train_scaled, y_train, feature_cols, label_encoder
)

print("Génération d'exemples adversariaux IoT-SDN...")
n_adv = int(len(X_train_seq) * ADVERSARIAL_RATIO)
adv_indices = np.random.choice(len(X_train_seq), n_adv, replace=False)

X_adv_iot = attack_generator.generate_adversarial(
    model_transformer,
    X_train_seq[adv_indices],
    y_train_cat[adv_indices],
    max_iter=15,
    c=0.1,
)

print(f"Exemples adversariaux générés: {len(X_adv_iot)}")

X_train_mix = np.vstack([X_train_seq, X_adv_iot])
y_train_mix = np.vstack([y_train_cat, y_train_cat[adv_indices]])

shuffle_idx = np.random.permutation(len(X_train_mix))
X_train_mix = X_train_mix[shuffle_idx]
y_train_mix = y_train_mix[shuffle_idx]

print(f"\nDataset mixte: {len(X_train_mix):,}")
print(
    f"  Bénin: {len(X_train_seq):,} ({len(X_train_seq) / len(X_train_mix) * 100:.1f}%)"
)
print(f"  Adversarial IoT-SDN: {n_adv:,} ({n_adv / len(X_train_mix) * 100:.1f}%)")

callbacks_phase2 = [
    EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    ),
    ModelCheckpoint(
        "/content/drive/MyDrive/transformer_phase2_adversarial.h5",
        save_best_only=True,
        monitor="val_loss",
        verbose=1,
    ),
]

history_phase2 = model_transformer.fit(
    X_train_mix,
    y_train_mix,
    validation_split=0.2,
    epochs=EPOCHS_PHASE2,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_phase2,
    verbose=1,
)

# =====================================================
# 11. ÉVALUATION FINALE
# =====================================================
print("\n" + "=" * 60)
print("ÉVALUATION FINALE")
print("=" * 60)

loss_clean, acc_clean = model_transformer.evaluate(X_test_seq, y_test_cat, verbose=0)
print(f"\n[Propres] Loss: {loss_clean:.4f} | Accuracy: {acc_clean:.4f}")

n_test = min(2000, len(X_test_seq))
test_indices = np.random.choice(len(X_test_seq), n_test, replace=False)

X_test_adv = attack_generator.generate_adversarial(
    model_transformer,
    X_test_seq[test_indices],
    y_test_cat[test_indices],
    max_iter=20,
    c=0.15,
)

loss_adv, acc_adv = model_transformer.evaluate(
    X_test_adv, y_test_cat[test_indices], verbose=0
)
print(f"[IoT-SDN Attack] Loss: {loss_adv:.4f} | Accuracy: {acc_adv:.4f}")

y_pred = model_transformer.predict(X_test_seq, verbose=0)
y_pred_cls = np.argmax(y_pred, axis=1)
y_true_cls = np.argmax(y_test_cat, axis=1)

print("\n--- Classification Report ---")
print(
    classification_report(y_true_cls, y_pred_cls, target_names=label_encoder.classes_)
)

plt.figure(figsize=(14, 12))
cm = confusion_matrix(y_true_cls, y_pred_cls)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
)
plt.title("Matrice de Confusion - Transformer (IoT IPFIX Home)")
plt.xlabel("Prédictions")
plt.ylabel("Vraies valeurs")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/transformer_confusion_matrix.png", dpi=150)
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(history_phase1.history["loss"], label="Train")
axes[0, 0].plot(history_phase1.history["val_loss"], label="Val")
axes[0, 0].set_title("Phase 1 - Loss")
axes[0, 0].legend()

axes[0, 1].plot(history_phase1.history["accuracy"], label="Train")
axes[0, 1].plot(history_phase1.history["val_accuracy"], label="Val")
axes[0, 1].set_title("Phase 1 - Accuracy")
axes[0, 1].legend()

axes[1, 0].plot(history_phase2.history["loss"], label="Train")
axes[1, 0].plot(history_phase2.history["val_loss"], label="Val")
axes[1, 0].set_title("Phase 2 - Loss (Adversarial)")
axes[1, 0].legend()

axes[1, 1].plot(history_phase2.history["accuracy"], label="Train")
axes[1, 1].plot(history_phase2.history["val_accuracy"], label="Val")
axes[1, 1].set_title("Phase 2 - Accuracy (Adversarial)")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("/content/drive/MyDrive/transformer_training_curves.png", dpi=150)
plt.show()

# =====================================================
# 12. RÉSUMÉ
# =====================================================
print("\n" + "=" * 60)
print("RÉSUMÉ DES RÉSULTATS")
print("=" * 60)

print("\n| Test | Loss | Accuracy | Robustesse |")
print("|------|------|----------|------------|")
print(f"| Propres | {loss_clean:.4f} | {acc_clean:.4f} | 100.0% |")
print(
    f"| IoT-SDN Attack | {loss_adv:.4f} | {acc_adv:.4f} | {(acc_adv / acc_clean) * 100:.1f}% |"
)

# =====================================================
# 13. SAUVEGARDE
# =====================================================
print("\n" + "=" * 60)
print("SAUVEGARDE")
print("=" * 60)

model_transformer.save("/content/drive/MyDrive/transformer_iot_sdn_final.h5")
print("Modèle: /content/drive/MyDrive/transformer_iot_sdn_final.h5")

preprocessing = {
    "label_encoder": label_encoder,
    "scaler_norm": scaler_norm,
    "scaler_std": scaler_std,
    "feature_cols": feature_cols,
    "sequence_length": SEQUENCE_LENGTH,
    "num_classes": num_classes,
    "valid_classes": VALID_CLASSES,
    "non_modifiable_features": NON_MODIFIABLE_FEATURES,
    "dependent_features": DEPENDENT_FEATURES,
    "num_heads": NUM_HEADS,
    "ff_dim": FF_DIM,
    "embedding_dim": EMBEDDING_DIM,
    "num_blocks": NUM_TRANSFORMER_BLOCKS,
}

with open("/content/drive/MyDrive/transformer_preprocessing.pkl", "wb") as f:
    pickle.dump(preprocessing, f)
print("Preprocessing: /content/drive/MyDrive/transformer_preprocessing.pkl")

print("\n✅ ENTRAÎNEMENT TRANSFORMER TERMINÉ!")
print(f"   - Classes: {num_classes}")
print(f"   - Features SDN: {len(feature_cols)}")
print(f"   - Accuracy (propre): {acc_clean:.4f}")
print(f"   - Accuracy (IoT-SDN Attack): {acc_adv:.4f}")
print(f"   - Robustesse: {(acc_adv / acc_clean) * 100:.1f}%")
