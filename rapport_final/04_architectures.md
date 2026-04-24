# Chapitre 4 — Architectures des Modèles de Deep Learning

## 4.1 Vue d'Ensemble des Six Architectures

Ce projet implémente et compare **six architectures de Deep Learning** pour la classification de séquences de flux IoT. Ces architectures forment une progression logique en complexité, illustrant comment chaque ajout architectural améliore les performances.

| Architecture | Modules | Paramètres approx. | Points forts |
|--------------|---------|---------------------|--------------|
| LSTM | LSTM × 2 | ~120K | Baseline temporel |
| BiLSTM | BiLSTM × 2 | ~240K | Contexte bidirectionnel |
| CNN-LSTM | Conv1D + LSTM | ~180K | Extraction locale + temporel |
| XGBoost-LSTM | LSTM + XGBoost | ~120K + gradient boosting | Hybride ML/DL |
| Transformer | TransformerEncoder | ~3.5M | Attention globale |
| **CNN-BiLSTM-Transformer** | CNN + BiLSTM + Transformer | **~2.1M** | **Architecture ultime** |

---

## 4.2 Architecture 1 : LSTM — Le Socle Temporel

**Fichier :** `src/models/lstm.py` — classe `LSTMClassifier`

### 4.2.1 Description

Le **Long Short-Term Memory (LSTM)** constitue l'architecture de référence pour les séquences temporelles. Il est conçu spécifiquement pour capturer les **dépendances à longue portée** dans les séries temporelles, un problème que les RNN simples ne peuvent pas résoudre à cause du gradient évanescent.

### 4.2.2 Architecture Détaillée

```
Input: (batch, 10, F)          # F = nombre de features
   ↓
LSTM Layer 1: input=F, hidden=64, batch_first=True
   → Output: (batch, 10, 64)
   ↓
Dropout(0.3)
   ↓
LSTM Layer 2: input=64, hidden=128    # Embedding de 128 dimensions
   → Output: (batch, 10, 128)
   ↓
last_hidden = output[:, -1, :]        # Dernier état caché uniquement
   → Shape: (batch, 128)
   ↓
ReLU()
   ↓
Linear(128 → num_classes)
   → Logits: (batch, num_classes)
```

### 4.2.3 Mécanisme Interne de la Cellule LSTM

À chaque pas de temps `t`, la cellule LSTM calcule :
- **Porte d'oubli :** `f_t = σ(W_f · [h_{t-1}, x_t] + b_f)` — quoi oublier de la mémoire
- **Porte d'entrée :** `i_t = σ(W_i · [h_{t-1}, x_t] + b_i)` — quoi mémoriser en nouveau
- **Mise à jour mémoire :** `C_t = f_t ⊙ C_{t-1} + i_t ⊙ tanh(W_C · [h_{t-1}, x_t] + b_C)`
- **Porte de sortie :** `h_t = o_t ⊙ tanh(C_t)`

Ce mécanisme permet d'apprendre à mémoriser sélectivement les caractéristiques pertinentes d'un flux et d'oublier les informations non pertinentes.

### 4.2.4 Justification dans le Contexte IoT

La nature temporelle des comportements IoT — un appareil envoie des rafales suivies de périodes de silence, avec des patterns caractéristiques de connexion — est parfaitement adaptée au LSTM. Les deux couches empilées permettent d'apprendre des représentations hiérarchiques : la première couche capturant les dépendances immédiates, la seconde les patterns à plus longue portée.

---

## 4.3 Architecture 2 : BiLSTM — La Vision Bidirectionnelle

**Fichier :** `src/models/bilstm.py` — classe `BiLSTMClassifier`

### 4.3.1 Description

Le **BiLSTM** étend le LSTM en traitant la séquence **dans les deux sens** simultanément — de gauche à droite (passé → futur) et de droite à gauche (futur → passé). La représentation finale est la **concaténation** des états cachés des deux directions.

### 4.3.2 Architecture Détaillée

```
Input: (batch, 10, F)
   ↓
BiLSTM Layer 1: hidden=64, bidirectional=True
   → Output: (batch, 10, 128)   # 64 forward + 64 backward
   ↓
Dropout(0.3)
   ↓
BiLSTM Layer 2: hidden=64
   → Output: (batch, 10, 128)
   ↓
Attention pooling ou last state
   ↓
Linear(128 → num_classes)
```

### 4.3.3 Justification de la Bidirectionnalité

Dans une séquence de 10 flux consécutifs d'un appareil IoT, le flux numéro 6 peut être mieux interprété si on connaît à la fois les flux 1-5 (contexte passé) ET les flux 7-10 (contexte futur dans la séquence capturée). Par exemple, un burst de trafic fort suivi d'une période calme a une signature différente d'un pic isolé en milieu de séquence.

---

## 4.4 Architecture 3 : CNN-LSTM — L'Extracteur Spatio-Temporel

**Fichier :** `src/models/cnn_lstm.py` — classe `CNNLSTMClassifier`

### 4.4.1 Description

Cette architecture combine un **CNN 1D** (extraction de motifs locaux) avec un **LSTM** (modélisation temporelle). Le CNN agit comme un extracteur de features automatique sur la séquence, et le LSTM modélise les dépendances entre ces features extraites.

### 4.4.2 Architecture Détaillée

```
Input: (batch, 10, F)
   ↓ permute → (batch, F, 10)   # Format CNN: (batch, channels, length)
Conv1d(F, 64, kernel_size=3, padding=1) → ReLU()
   → (batch, 64, 10)
MaxPool1d(kernel_size=2)
   → (batch, 64, 5)
   ↓ permute → (batch, 5, 64)   # Retour format séquence
LSTM(64, hidden=100, batch_first=True)
   → (batch, 5, 100)
last_hidden = output[:, -1, :]
   → (batch, 100)
Linear(100 → num_classes)
```

### 4.4.3 Rôle du CNN 1D

Le filtre convolutif de taille 3 (kernel_size=3) apprend des motifs sur 3 flux consécutifs. Par exemple, si les flux i, i+1, i+2 présentent systématiquement un pattern de tailles croissantes pour un certain type d'appareil (ex. : initialisation de session), le CNN l'apprend comme un filtre spécialisé. Le MaxPool compresse cette information en ne retenant que le signal le plus fort de chaque fenêtre.

---

## 4.5 Architecture 4 : XGBoost-LSTM — L'Hybride ML/DL

**Fichier :** `src/models/xgboost_lstm.py` — classe `XGBoostLSTMClassifier`

### 4.5.1 Description

Cette architecture utilise le **LSTM comme extracteur de features** (sans couche de classification finale) et un **XGBoost** comme classifieur sur ces features extraites.

### 4.5.2 Fonctionnement

```
Phase 1 — Feature Extraction (LSTM)
   Input: (batch, 10, F)
   LSTM(hidden=64) → output[:, -1, :]  # Vecteur latent de 64 dimensions
   
Phase 2 — Classification (XGBoost)
   Input: vecteur latent 64-dim
   XGBClassifier(n_estimators=100, max_depth=6)
   Output: classe IoT
```

### 4.5.3 Justification

XGBoost est reconnu pour ses performances exceptionnelles sur des features tabulaires. En combinant la capacité du LSTM à créer des représentations temporelles riches avec la puissance de classification non-linéaire de XGBoost, cette architecture tente d'obtenir le meilleur des deux mondes. **Limitation :** cette architecture est difficile à rendre robuste aux attaques adversariales car le XGBoost n'est pas différentiable et ne peut pas être entraîné de manière antagoniste par backpropagation.

---

## 4.6 Architecture 5 : Transformer — L'Attention Globale

**Fichier :** `src/models/transformer.py` — classe `TransformerClassifier`

### 4.6.1 TransformerClassifier (Données Numériques)

```
Input: (batch, 10, F)
   ↓
Linear(F → d_model=768)           # Projection vers l'espace d'attention
   ↓
PositionalEncoding(d_model=768)   # Encodage de la position dans la séquence
   ↓
TransformerEncoder:
   × 6 couches de TransformerEncoderLayer:
      - MultiHeadSelfAttention(nhead=12, d_model=768)
      - FeedForward(d_model=768 → 3072 → 768) avec activation GELU
      - LayerNorm
   → Output: (batch, 10, 768)
   ↓
MeanPooling → (batch, 768)
   ↓
Linear(768 → 256) → ReLU() → Linear(256 → num_classes)
```

### 4.6.2 Mécanisme d'Attention Multi-Têtes

Le cœur du Transformer est le mécanisme d'**attention multi-têtes** :

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

Avec 12 têtes d'attention et d_model=768, chaque tête opère sur des sous-espaces de dimension 64 (768/12). Cette multi-tête permet au modèle d'apprendre différents types de relations en parallèle — certaines têtes peuvent apprendre à relier les pics de débit, d'autres les patterns d'inter-arrivée.



---

## 4.7 Architecture 6 : CNN-BiLSTM-Transformer — Le Modèle Hybride Principal

**Fichier :** `src/models/cnn_bilstm_transformer.py` — classe `CNNBiLSTMTransformerClassifier`

Cette architecture est le **modèle principal** du projet, combinant les forces des trois approches précédentes en un pipeline séquentiel.

### 4.7.1 Vue d'Ensemble de l'Architecture

```
Input: (batch, seq_length=10, input_size=F)
   │
   ├─ permute → (batch, F, 10)
   │
   ├─[CNN Branch 1]─ Conv1d(F, 32, k=3) → ReLU → MaxPool1d → BatchNorm1d
   │                 → (batch, 32, 10)
   │
   └─[CNN Branch 2]─ Conv1d(F, 32, k=5) → ReLU → MaxPool1d → BatchNorm1d
                     → (batch, 32, 10)
                     
   concat([B1, B2], dim=1) → (batch, 64, 10)
   permute → (batch, 10, 64)
   │
   BiLSTM(input=64, hidden=64, layers=2, bidirectional=True)
   → (batch, 10, 128)            # 64 forward + 64 backward
   │
   Linear(128 → 128) [Projection si bilstm_out ≠ d_model]
   ↓
   PositionalEncoding(d_model=128)
   ↓
   TransformerEncoder:
      × 2 couches:
         - MultiHeadAttention(nhead=4, d_model=128)
         - FeedForward(128 → 512 → 128, activation=GELU)
         - LayerNorm
   → (batch, 10, 128)
   ↓
   MeanPooling → (batch, 128)
   ↓
   Dropout(0.4) → Linear(128 → 64) → ReLU → Dropout(0.2) → Linear(64 → num_classes)
   → Logits: (batch, num_classes)
```

### 4.7.2 Module CNN : Extraction Multi-Échelle

**Pourquoi deux branches parallèles avec kernels différents ?**

La clé de cette conception est l'utilisation de deux noyaux convolutifs de tailles différentes (k=3 et k=5), qui capturent des patterns à des **granularités différentes** :

- **Branche 1 (k=3)** : analyse des patterns sur 3 flux consécutifs — détecte des motifs à court terme comme des changements brusques de débit, ou des triplets de paquets caractéristiques d'une connexion IoT.
- **Branche 2 (k=5)** : analyse des patterns sur 5 flux consécutifs — détecte des structures rhythmiques plus larges, comme un cycle de 5 flux caractéristique d'un protocole de heartbeat.

La **concaténation** des deux sorties (64 canaux total = 32+32) permet au modèle de conserver les informations des deux échelles simultanément, avant de les envoyer dans le BiLSTM.

**BatchNorm1d** après le pooling stabilise l'entraînement en normalisant les activations, permettant des valeurs de learning rate plus élevées.

### 4.7.3 Module BiLSTM : Dépendances Temporelles Bidirectionnelles

Le BiLSTM reçoit la sortie fusionnée du CNN `(batch, 10, 64)` et modélise les **dépendances temporelles bidirectionnelles** entre les 10 positions de la séquence.

```python
self.bilstm = nn.LSTM(
    input_size=64,          # Sortie du CNN (cnn_ch * 2)
    hidden_size=64,         # Unités cachées par direction
    num_layers=2,           # LSTM empilé sur 2 couches
    batch_first=True,
    bidirectional=True,     # Traitement dans les deux sens
    dropout=0.0,            # Pas de dropout (géré par Feature Dropout en entrée)
)
# Sortie: (batch, 10, 128)  # 64 forward + 64 backward
```

**Fixation du bug CuDNN LSTM :** une workaround a été nécessaire pour contourner un bug de mémoire de CuDNN sur GPU L4 (28 GB de VRAM alloués par le workspace CuDNN pour le backward du LSTM en AMP) :

```python
with torch.backends.cudnn.flags(enabled=False):
    lstm_out, _ = self.bilstm(fused)
```

Cette désactivation force PyTorch à utiliser son propre backend LSTM (compatible FP32) au lieu de cuDNN, évitant l'allocation massive de workspace.

### 4.7.4 Module Transformer : Attention Globale

Le Transformer reçoit la sortie du BiLSTM projetée dans l'espace `d_model=128` et applique le mécanisme d'attention multi-têtes sur la séquence complète.

```python
enc_layer = nn.TransformerEncoderLayer(
    d_model=128,
    nhead=4,            # 4 têtes → chaque tête opère sur 32 dimensions
    dim_feedforward=512,
    dropout=0.2,
    activation="gelu",  # GELU = Gaussian Error Linear Unit (meilleur que ReLU pour NLP)
    batch_first=True,
)
self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
self.layer_norm = nn.LayerNorm(128)
```

**Pourquoi ajouter le Transformer après le BiLSTM ?** Le BiLSTM capture les dépendances temporelles locales de manière séquentielle, mais il a du mal à relier directement des positions éloignées (position 1 et position 9 dans une séquence de 10). Le Transformer, via son attention directe entre toutes les paires de positions, comble ce déficit. Le résultat est un modèle qui combine compréhension locale (CNN), contexte séquentiel (BiLSTM) et relations globales (Transformer).

### 4.7.5 Tête de Classification

```python
self.classifier = nn.Sequential(
    nn.Dropout(0.4),        # Régularisation forte (40%)
    nn.Linear(128, 64),     # Réduction 128 → 64
    nn.ReLU(),
    nn.Dropout(0.2),        # Régularisation douce (20%)
    nn.Linear(64, num_classes),  # Classification finale
)
```

Le **MeanPooling** avant la tête de classification agrège toutes les positions de la séquence (plutôt que de prendre uniquement le dernier hidden state comme le ferait un LSTM pur), ce qui donne une représentation plus robuste de l'ensemble de la séquence.

### 4.7.6 Hyperparamètres Finaux (Version Optimisée pour GPU L4)

```python
CNN_BILSTM_TRANSFORMER_OVERRIDE = {
    'cnn_channels': 32,          # Réduit de 64 à 32 pour économiser la VRAM
    'bilstm_hidden': 64,         # Réduit de 128 à 64 → sortie BiLSTM = 128
    'bilstm_layers': 2,
    'bilstm_dropout': 0.3,
    'transformer_d_model': 128,  # Réduit de 256 à 128
    'transformer_nhead': 4,
    'transformer_layers': 2,
    'transformer_ff_dim': 512,
    'transformer_dropout': 0.2,
    'fc_dropout': 0.4,
}
```

Ces hyperparamètres ont été calibrés pour tenir dans les 22 GB de VRAM d'un GPU NVIDIA L4 avec un batch size de 32 et la précision mixte (AMP).

---

## 4.8 Entraînement : Paramètres Communs

Tous les modèles partagent la même configuration d'entraînement :

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| **Optimiseur** | AdamW | Intègre la décroissance de poids (weight decay=1e-4) |
| **Learning Rate** | 5e-4 | Conservateur pour la stabilité de l'AMP |
| **Scheduler** | MultiStepLR (milestones=[15,30], γ=0.5) | Diminution du LR aux epochs clés |
| **Gradient Clipping** | max_norm=1.0 | Prévient les explosions de gradient (important pour BiLSTM) |
| **Mixed Precision** | AMP (FP16) | Réduit la VRAM d'environ 50% sur GPU compatible |
| **Batch Size** | 32 | Réduit de 64 pour éviter les OOM sur L4 GPU |
| **Label Smoothing** | Phase A: 0.05, Phase B: 0.08, Phase C: 0.10 | Régularisation progressive |
