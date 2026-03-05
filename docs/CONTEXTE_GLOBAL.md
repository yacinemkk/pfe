# CONTEXTE GLOBAL DU PROJET PFE

## Identification des Dispositifs IoT avec Apprentissage Adversarial pour SDN-NAC

---

## 1. Resume Executif

Ce projet de fin d'etudes (PFE) vise a developper un systeme robuste d'identification des dispositifs IoT dans un environnement SDN-NAC (Software Defined Networking - Network Access Control). L'innovation principale reside dans l'utilisation de modeles d'apprentissage profond (LSTM et Transformers) formules avec un entrainement adversarial pour resister aux attaques antagonistes.

---

## 2. Problematique

### 2.1 Contexte IoT + SDN-NAC

Les reseaux IoT contiennent des milliers d'objets heterogenes (cameras, capteurs, badges, etc.)

Le SDN-NAC decide :
- Qui a le droit de se connecter au reseau ?
- A quel niveau d'acceder ?
- Pour cela, on utilise l'identification des dispositifs IoT a partir du trafic reseau.

### 2.2 Le Probleme des Methodes Classiques

Les approches classiques :
- Utilisent du Machine Learning classique
- Basees sur des features statiques : taille moyenne des paquets, ports utilises, protocoles, statistiques globales

**Faiblesse majeure** : Ces modeles sont vulnerables aux attaques antagonistes :
- Un attaquant modifie legerement le trafic
- Le modele se trompe d'identite
- Resultat : acces non autorise ou contournement du NAC

### 2.3 Qu'est-ce qu'une Attaque Antagoniste ?

Une attaque antagoniste (adversarial attack) :
- N'altegre pas la fonctionnalite de l'objet IoT
- Modifie subtilement les caracteristiques du trafic
- Objectif : tromper le modele ML

**Exemple concret** : Une camera IP modifie legerement le timing ou la taille de ses paquets -> le modele croit que c'est un capteur de temperature -> mauvaise politique de securite appliquee.

---

## 3. Solution Proposee

### 3.1 Architecture Globale

```
IoT Device
    |
    | 1. Generate network traffic
    v
SDN Switch
    |
    | 2. Mirror traffic (SDN monitoring)
    v
Flow Collector
    |
    | 3. Extract flows & build time sequences
    v
AAD Validator
    |
    | 4. Validate traffic integrity
    |-----------------------------|
    |  traffic normal             |
    |  traffic suspicious          |
    |-----------------------------|
    v
Adversarial Discriminator
    |
    | 5. Detect adversarial patterns
    |  -> score_adv
    v
Identification Model (LSTM / Transformer)
    |
    | 6. Identify IoT device type
    |  -> device_label
    |  -> confidence_score
    v
Adaptive Defense Engine
    |
    | 7. Correlate: device_label, confidence_score, score_adv
    | 8. Decide security policy
    v
SDN Controller (NAC)
    |
    | 9. Enforce access control
    |----------------------------|
    |  Full access               |
    |  Restricted access         |
    |  Quarantine                |
    |  Block                      |
    |----------------------------|
    v
Network Enforcement
```

### 3.2 Composants du Systeme

#### 3.2.1 Validator (AAD) - Pre-filtrage

**Role** :
- Detecter des perturbations evidentes (bruit adversarial excessif)
- Eliminer les flux fortement suspects avant l'identification
- Verifier : coherence des distributions, valeurs hors domaine, anomalies grossieres, violations de contraintes reseau

**Avantage** : Reduit le bruit, ameliore la stabilite des LSTM / Transformers

#### 3.2.2 Adversarial Discriminator (AAD)

**Role** :
- Detecter attaque vs trafic normal
- Produire un score de confiance adversarial : `score_adv ∈ [0,1]`

Ce score alimente le moteur adaptatif :
- Si score_adv eleve : activer Transformer, seuil strict
- Sinon : LSTM standard

#### 3.2.3 LSTM / Transformer (Identification IoT)

**LSTM** :
- Specialise dans les donnees temporelles
- Capte les dependances long terme et les comportements recurrents
- Tres adapte aux flux reseau IoT
- Meme si l'attaquant modifie quelques paquets, le comportement global reste reconnaissable

**Transformers** :
- Utilisent le mecanisme d'attention
- Peuvent analyser toute la sequence en parallele
- Reparent : incoherences, motifs anormaux, changements de comportement subtils
- Tres efficaces contre les attaques adaptatives et evolutives

#### 3.2.4 Adaptive Defense Engine

**Innovation du projet** : Le moteur decide dynamiquement :
- Quels seuils appliquer
- Quelle action reseau declencher

Actions possibles :
- Autorisation complete
- Acces restreint
- Quarantaine
- Blocage temporaire

---

## 4. Jeu de Donnees

### 4.1 Source

**Dataset** : IPFIX ML Instances (UNSW IoT Analytics)

**Localisation** : `data/pcap/IPFIX ML Instances/`

### 4.2 Description

| Fichier | Lignes |
|---------|--------|
| home1_labeled.csv | 1,152,574 |
| home2_labeled.csv | 1,401,332 |
| home3_labeled.csv | 1,707,786 |
| home4_labeled.csv | 8,071,109 |
| home5_labeled.csv | 1,791,773 |
| home6_labeled.csv | 4,643,642 |
| home7_labeled.csv | 2,652,648 |
| home8_labeled.csv | 1,518,574 |
| home9_labeled.csv | 1,137,143 |
| home10_labeled.csv | 5,373,493 |
| home11_labeled.csv | 1,037,094 |
| home12_labeled.csv | 1,630,018 |

**Total** : 32+ millions de lignes

### 4.3 Features (44 colonnes)

**Features de flux** :
- `duration`, `ipProto`
- `outPacketCount`, `outByteCount`, `inPacketCount`, `inByteCount`
- `outAvgIAT`, `outMaxPktSize`, `outStdevPayloadSize`, `outAvgPacketSize`
- Features entrantes similaires (in*)

**Flags de protocole** :
- `http`, `https`, `smb`, `dns`, `ntp`, `tcp`, `udp`, `ssdp`

**Flags reseau** :
- `lan`, `wan`, `deviceInitiated`

**Labels** :
- `device` (ID), `name` (device type)

### 4.4 Classes de Dispositifs IoT (18 classes)

1. Eclear
2. Sleep
3. Esensor
4. Hub Plus
5. Humidifier
6. Home Unit
7. Ink Jet Printer
8. Smart Wi-Fi Plug Mini
9. Smart Power Strip
10. Echo Dot
11. Fire 7 Tablet
12. Google Nest Mini
13. Google Chromecast
14. Atom Cam
15. Kasa Camera Pro
16. Kasa Smart LED Lamp
17. Fire TV Stick 4K
18. Qrio Hub

### 4.5 Features SDN-Compatibles (37 features)

Features conservees (excluant IP, MAC, ports) :
- `duration`, `ipProto`
- `outPacketCount`, `outByteCount`, `inPacketCount`, `inByteCount`
- `outSmallPktCount`, `outLargePktCount`, `outNonEmptyPktCount`, `outDataByteCount`
- `outAvgIAT`, `outFirstNonEmptyPktSize`, `outMaxPktSize`
- `outStdevPayloadSize`, `outStdevIAT`, `outAvgPacketSize`
- Features entrantes similaires
- Flags de protocole : `http`, `https`, `smb`, `dns`, `ntp`, `tcp`, `udp`, `ssdp`
- Flags reseau : `lan`, `wan`, `deviceInitiated`

**Features exclues** : `start`, `srcMac`, `destMac`, `srcIP`, `destIP`, `srcPort`, `destPort`

---

## 5. Pretraitement des Donnees

### 5.1 Etapes

1. **Normalisation** : Donnees remises a l'echelle dans [0, 1]
2. **Standardisation** : Moyenne = 0, ecart-type = 1
3. **Suppression des doublons** : Via Pandas
4. **Gestion des valeurs manquantes** : Exclusion des instances incompletes
5. **Selection des features SDN** : Conservation uniquement des caracteristiques accessibles via API SDN

### 5.2 Construction des Sequences

- **Sequence length** : 10 (configurable)
- **Stride** : 5
- Les donnees sont transformees en sequences temporelles pour LSTM/Transformer

### 5.3 Split des Donnees

- Train : 70%
- Validation : 10%
- Test : 20%

---

## 6. Modeles

### 6.1 Configuration LSTM

```python
LSTM_CONFIG = {
    "hidden_size": 128,
    "num_layers": 2,
    "bidirectional": True,
    "dropout": 0.3,
}
```

- Bidirectional LSTM
- Optimiseur : Adam
- Loss : CrossEntropyLoss
- Learning rate scheduling : ReduceLROnPlateau

### 6.2 Configuration Transformer

```python
TRANSFORMER_CONFIG = {
    "d_model": 64,
    "nhead": 4,
    "num_encoder_layers": 3,
    "dim_feedforward": 256,
    "dropout": 0.2,
}
```

- Encoder-only Transformer
- Positional Encoding
- Self-attention mechanism

### 6.3 Hyperparametres d'Entrainement

```python
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MIN_SAMPLES_PER_CLASS = 500
```

---

## 7. Apprentissage Adversarial

### 7.1 Fondements de l'Apprentissage Antagoniste

L'apprentissage antagoniste (adversarial learning) est un paradigme d'apprentissage automatique dans lequel deux modèles ou plus sont entraînés simultanément dans un cadre compétitif. Ce concept repose sur la théorie des jeux et l'optimisation min-max.

#### 7.1.1 Réseaux Antagonistes Génératifs (GAN)

Les GAN (Generative Adversarial Networks), introduits par Ian Goodfellow en 2014, constituent une architecture fondamentale de l'apprentissage antagoniste.

**Architecture** :
- **Générateur (G)** : Réseau neuronal qui prend en entrée un vecteur de bruit aléatoire (issu d'une distribution gaussienne) et génère des données synthétiques similaires aux données réelles.
- **Discriminateur (D)** : Réseau neuronal qui distingue les données réelles des données générées.

**Objectif min-max** :
```
min_G max_D V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]
```

Le générateur cherche à minimiser cette fonction tandis que le discriminateur cherche à la maximiser.

#### 7.1.2 Application à la Sécurité IoT

Dans le contexte de ce projet :
- Le **générateur** peut être utilisé pour créer des exemples adversariaux réalistes (trafic IoT modifié)
- Le **discriminateur** (Adversarial Discriminator dans l'architecture) détecte les patterns adversariaux

### 7.2 Strategie d'Entrainement Adversarial

**Phase 1 - Entrainement initial** :
- Donnees 100% benignes
- Apprentissage du comportement reel naturel des appareils

**Phase 2 - Entrainement adversarial** :
- 80% trafic benin
- 20% trafic adversarial
- Le modele apprend des frontieres de decision plus robustes

### 7.3 Approches Adversariales pour Sequences

#### Approche 1 : PGD/FGSM pour Sequences

```
Pour chaque sequence x = [x1, x2, ..., xT]:
    x_adv = Project(x + epsilon * sign(grad_x L(theta, x, y)))

Gradient calcule via BPTT (Backpropagation Through Time)
```

**Avantages** : Perturbations optimales (guidees par gradient)
**Inconvenients** : Peut violer les contraintes semantiques

#### Approche 2 : Framework PANTS (AML + SMT Solvers)

1. Generer perturbation initiale (PGD/ZOO)
2. Encoder contraintes trafic comme formules SMT
3. Utiliser solver SMT (Z3) pour trouver assignation valide
4. Si unsat, relaxer contraintes et reessayer

**Avantage** : Garantit que les echantillons sont realisables
**Inconvenient** : Plus lent (~10x)

#### Approche 3 : Data Augmentation

Methodes d'augmentation pour time series :
1. Jittering : Ajouter bruit gaussien
2. Scaling : Mise a l'echelle aleatoire
3. Permutation : Melanger segments
4. Magnitude Warping : Courbe de scaling lisse
5. Time Warping : Distorsion temporelle lisse

**Avantage** : Seulement 14% overhead computationnel
**Inconvenient** : Pas de garantie contre des attaques specifiques

#### Approche 4 : Attaque TSFool/TANTRA

**TSFool** :
- Attaque gray-box pour RNN-based time series classification
- Optimisation multi-objectif : erreur classification + L2 + coefficient de camouflage

**TANTRA** :
- Attaque basee sur le timing
- Utilise LSTM pour apprendre distribution des inter-arrival times
- 99.99% succes en evasion NIDS

#### Approche 5 : Approche Hybride (Recommandee)

Combine methode actuelle + attaque gradient-based :

```
Phase 1: Attaque feature-level (IoT-SDN attack)
         - Cible features statistiques
         - Respecte dependances features

Phase 2: Attaque sequence-level (gradient-based)
         - Cible ordonnancement temporel
         - Utilise BPTT pour gradients
         - Contrainte de preservation semantique

Phase 3: Entrainement Adversarial
         - 60% clean + 20% feature-level + 20% sequence-level
```

---

## 8. Defense Adaptive

### 8.1 Principe

Le systeme observe le trafic en continu, detecte les derivees, et s'adapte en temps reel.

### 8.2 Mecanismes

- Changement de seuils
- Ajustement des poids
- Selection dynamique du modele (LSTM vs Transformer)
- Activation automatique de mecanisme de defense

### 8.3 Integration SDN-NAC

Le controleur SDN recoit la decision du modele et applique dynamiquement :
- Autorisation
- Limitation
- Isolement
- Quarantaine

---



## 12. Technologies et Frameworks

### 12.1 Principal : PyTorch

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
```

### 12.2 Secondaire : TensorFlow/Keras (pour Colab)

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, MultiHeadAttention
```

### 12.3 Bibliotheques Support

- scikit-learn : StandardScaler, LabelEncoder, train_test_split
- pandas/numpy : Manipulation de donnees
- matplotlib/seaborn : Visualisation
- tqdm : Progress bars
- pickle : Serialisation

---

## 13. Prochaines Etapes

1. Implementer le preprocesseur de donnees
2. Implementer les modeles LSTM et Transformer
3. Implementer les attaques adversariales (IoT-SDN + gradient-based)
4. Implementer l'entrainement adversarial
5. Evaluer sur differentes longueurs de sequences (10, 50, 100)
6. Comparer LSTM, Transformer
7. Tester contre multiples types d'attaques
8. Implementer le moteur de defense adaptive
9. Integrer avec simulateur SDN

---

## 14. References Cles

### Repositories

1. `GoktugOcal/time-series-adversarial-attacks` : FGSM, PGD 

