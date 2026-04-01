# Rapport d'Analyse et Documentation du Projet PFE

**Date de génération** : 2026-04-01  
**Projet** : IoT Device Identification — Adversarial Robustness Framework  
**Type** : Projet de Fin d'Études (PFE)

---

# PARTIE 1 : Documentation du Projet

## 1. Vue d'ensemble

### Objectif
Identification de dispositifs IoT à partir de flux réseau IPFIX, avec évaluation systématique de la robustesse face aux attaques adversariales. Le projet compare plusieurs architectures de deep learning et démontre que l'entraînement antagoniste (adversarial training) restaure la résilience des modèles.

### Contexte
Deux datasets sources sont utilisés indépendamment :

| Dataset | Format | Classes | Source |
|---------|--------|---------|--------|
| **IPFIX ML Instances** | CSV | 18 appareils | Eclear, Echo Dot, Atom Cam, Fire TV Stick 4K, etc. |
| **IPFIX Records** | JSON | 17 appareils | Qrio Hub, Philips Hue, Amazon Echo, Wansview Camera, etc. |

### Contrainte Majeure
**Zéro data leakage** — split temporel strict par appareil (80% passé = train, 20% futur = test), normalisation fit sur train uniquement, séquences générées indépendamment sur train et test.

---

## 2. Architecture Technique

### Structure du Projet

```
pfe/
├── config/
│   ├── config.yaml              # Configuration centrale (hyperparamètres, features, modèles)
│   └── config.py                # Configuration Python (chemins, constantes, device classes)
├── src/
│   ├── data/                    # Pipeline de données
│   │   ├── loader.py                # Chargement des fichiers CSV
│   │   ├── preprocessor.py          # Préprocesseur CSV (4 étapes + anti-leakage)
│   │   ├── json_preprocessor.py     # Préprocesseur JSON (4 étapes + anti-leakage)
│   │   ├── sequence.py              # Génération de séquences (sliding window)
│   │   ├── split.py                 # Split temporel anti-leakage
│   │   ├── tokenizer.py             # Tokenisation BPE (IoT-Tokenize)
│   │   ├── categorical_encoder.py   # Encodage catégoriel par type de modèle
│   │   └── preprocessing.py         # Préprocesseur alternatif
│   ├── models/                  # Architectures de modèles
│   │   ├── lstm.py                  # LSTM 2 couches, 64 unités, 128-dim embedding
│   │   ├── bilstm.py                # BiLSTM bidirectionnel
│   │   ├── cnn_lstm.py              # CNN 1D → LSTM
│   │   ├── xgboost_lstm.py          # LSTM extracteur + XGBoost classifieur
│   │   ├── transformer.py           # Transformer (768d, 12 heads, 6 layers) + NLP variant
│   │   ├── hybrid.py                # CNN(3,5) → BiLSTM → Transformer
│   │   ├── cnn_bilstm.py            # CNN → BiLSTM
│   │   └── cnn_bilstm_transformer.py  # Variante hybride
│   ├── adversarial/             # Attaques adversariales
│   │   └── attacks.py               # Feature-level, Sequence-level (FGSM/PGD), Hybrid
│   └── training/                # Entraînement et évaluation
│       ├── trainer.py               # Boucle d'entraînement + AdversarialTrainer
│       └── evaluator.py             # Evaluator, ModelComparator, CrashTestEvaluator
├── main.py                      # Entry point pipeline complet (legacy)
├── train_adversarial.py         # Entraînement antagoniste 2 phases (script principal)
├── analyze_leakage.py           # Analyse de fuite de données
├── evaluate_phases_local.py     # Évaluation locale des phases
├── requirements.txt             # Dépendances Python
├── data/                        # Données brutes et transformées
│   ├── pcap/                        # Données IPFIX brutes (CSV + JSON)
│   ├── processed/                   # Données prétraitées (.npy, .pkl)
│   └── flow_csv/                    # CSV additionnels
├── results/                     # Résultats (modèles, métriques, graphiques)
└── docs/                        # Documentation technique
    ├── general                    # Pipeline général anti-leakage
    ├── architectures              # Spécifications des 6 architectures
    ├── train                      # Protocole d'entraînement 2 phases
    ├── pretraitement              # 4 étapes de prétraitement
    ├── featureselection           # Sélection de features
    ├── structure                  # Structure du projet
    └── rapport_analyse_specifications.md  # Analyse d'écart code vs specs
```

### Composants Clés

#### Pipeline de Données (2 pipelines parallèles)

| Étape | CSV Pipeline | JSON Pipeline |
|-------|-------------|---------------|
| **Étape 1** : Filtrage SDN | Suppression IP/MAC/ports, 30 features | Suppression IP/MAC/ports, 28+8 features |
| **Étape 2** : Équilibrage | Borderline-SMOTE + Isolation Forest + LOF | Identique |
| **Étape 3** : Sélection features | XGBoost + Chi2 + Mutual Info (elbow method) | Identique |
| **Étape 4** : Normalisation | Min-Max uniquement | Min-Max + StandardScaler |
| **Anti-leakage** | Split temporel 80/20 par device | Split temporel 80/20 par device |
| **Séquences** | Sliding window (length=10, stride=5) | Identique |

---

## 3. Fonctionnalités Implémentées

### Modèles (8 architectures)

| Modèle | Description | Statut |
|--------|-------------|--------|
| **LSTM** | 2 couches, 64 unités, 128-dim embedding, ReLU | ✅ Aligné spec |
| **BiLSTM** | Bidirectionnel, 2 couches, 64 unités | ✅ Aligné spec |
| **CNN-LSTM** | Conv1D → MaxPool → LSTM(64) → Dense(100) | ✅ Aligné spec |
| **XGBoost-LSTM** | LSTM comme extracteur de features + XGBoost classifieur | ✅ Aligné spec |
| **Transformer** | 6 layers, 768d, 12 heads, 576 max_seq, GELU | ✅ Aligné spec |
| **NLPTransformer** | BPE tokenization (52k vocab) + 3 embeddings (word + position + token-type) | ✅ Aligné spec |
| **Hybrid** | CNN(k=3,5) → BiLSTM → Transformer → Mean Pool → FC | ✅ Aligné spec |
| **CNN-BiLSTM** | CNN → BiLSTM | ✅ Présent |

### Attaques Adversariales

| Attaque | Type | Description |
|---------|------|-------------|
| **Feature-level** | IoT-SDN style | Perturbation ciblée vers centroides de classes voisines, avec masques et contraintes sémantiques |
| **Sequence-level FGSM** | Gradient-based (1 étape) | Perturbation sur séquences complètes via signe du gradient |
| **Sequence-level PGD** | Gradient-based itératif (BPTT) | Perturbation itérative avec projection, split continu/binaire |
| **Hybrid** | Combinaison | Mélange 50/50 feature-level + sequence-level |

### Protocole d'Entraînement (2 phases)

- **Phase 1** : Entraînement standard (données bénignes uniquement, max 20 epochs) → **Crash Test 1** (3 tests : bénin, adversaire, mixte)
- **Phase 2** : Entraînement antagoniste (corpus conjoint bénin + adversaire, max 10 epochs) → **Crash Test 2** (mêmes 3 tests)
- **Early stopping** par phase avec patience réinitialisée
- **Checkpointing** des meilleurs poids par phase

### Tokenisation IoT-Tokenize (5 étapes)

1. Transformation structurée : `nom_feature valeur; nom_feature valeur; ...`
2. Création du vocabulaire : features + valeurs catégorielles + tokens spéciaux
3. Entraînement BPE sur données d'entraînement
4. Encodage en token IDs avec padding/truncation
5. Passage au modèle NLPTransformerClassifier (3 embeddings : word + position + token-type)

---

## 4. Stack Technique

| Catégorie | Technologie | Version |
|-----------|-------------|---------|
| Langage | Python | 3.10+ |
| Deep Learning | PyTorch | ≥1.12 |
| Machine Learning | scikit-learn, XGBoost, imbalanced-learn | ≥1.0, ≥1.6, ≥0.9 |
| Tokenisation | HuggingFace `tokenizers` (BPE) | — |
| Data | NumPy, Pandas, SciPy | ≥1.21, ≥1.3, ≥1.7 |
| Visualisation | Matplotlib, Seaborn | ≥3.5, ≥0.12 |
| Configuration | PyYAML | ≥6.0 |
| Progression | tqdm | ≥4.64 |

---

## 5. Flux de Données

```
Données brutes (CSV/JSON IPFIX)
  │
  ├── Pipeline CSV ──→ SDN Filter → Temporal Split (80/20 par device) →
  │                     SMOTE → Feature Selection → Min-Max →
  │                     Sequences (length=10, stride=5) → Val Split (10%)
  │
  └── Pipeline JSON ──→ MAC Mapping → SDN Filter → Temporal Split (80/20) →
                        SMOTE → Feature Selection → Min-Max + StandardScaler →
                        Sequences → Val Split (10%)
  │
  ├── Modèles DL ──→ DataLoader → Training (2 phases) → Évaluation → Comparaison
  │
  └── Tokenizer (NLP) ──→ BPE → Token IDs → NLPTransformerClassifier
```

---

## 6. Décisions Clés

### 6.1 Double Pipeline CSV/JSON
Les deux datasets sont traités indépendamment car leurs schémas de features diffèrent fondamentalement (noms de colonnes, nombre de features, format binaire des directions de paquets dans JSON).

### 6.2 Pas de One-Hot pour les Features Catégorielles dans Transformer
`ipProto`/`protocolIdentifier` restent en entiers bruts, convertis en labels textuels (`6 → tcp`, `17 → udp`) par le tokenizer BPE. Le mécanisme d'attention évalue dynamiquement leur importance.

### 6.3 Normalisation Différente CSV vs JSON
- **CSV** : Min-Max uniquement (0-1)
- **JSON** : Min-Max puis StandardScaler (redondance mathématique identifiée comme gap)

### 6.4 Split par Label (pas par Timestamp)
La colonne `start` est supprimée par le filtre SDN. Le split temporel se fait donc par regroupement de label (appareil), en supposant que les données sont déjà triées chronologiquement dans les fichiers sources.

### 6.5 Features Non Modifiables dans les Attaques
- CSV : `ipProto` (champ SDN match field)
- JSON : `protocolIdentifier` + 8 bits `pkt_dir_*` (direction binaire des paquets)

---

# PARTIE 2 : Rapport des Gaps

## Gaps Critiques (P0)

---

### G1 : `main.py` utilise un split non chronologique (fuite de données potentielle)

- **Description** : Le split dans `main.py` (lignes 368-392) regroupe par label et prend les 80% premiers sans tri temporel. Le code contient un avertissement explicite (lignes 356-366) signalant que ce n'est PAS un vrai split chronologique.
- **Impact** : Critique — peut introduire un data leakage significatif si les données ne sont pas déjà triées.
- **Solution suggérée** : `main.py` est déjà marqué comme legacy. Le script `train_adversarial.py` délègue correctement aux préprocesseurs (`IoTDataProcessor.process_all()` / `JsonIoTDataProcessor.process_all()`) qui implémentent le vrai split temporel. Supprimer `main.py` ou le réécrire pour utiliser les preprocessors.
- **Priorité** : P0

---

### G2 : Aucune suite de tests automatisés

- **Description** : Aucun test unitaire, d'intégration ou end-to-end n'existe dans le projet. Aucun dossier `tests/`, aucun fichier `test_*.py`, aucune configuration pytest.
- **Impact** : Critique — impossible de valider que les modifications ne cassent pas le pipeline, ni de garantir la non-régression.
- **Solution suggérée** : Ajouter un dossier `tests/` avec pytest. Commencer par :
  - Tests unitaires sur `SequenceGenerator` (vérifier shapes, pas de chevauchement train/test)
  - Tests sur `TemporalSplitter` (vérifier pas de leakage temporel)
  - Tests sur `FeatureLevelAttack` et `SequenceLevelAttack` (vérifier contraintes sémantiques)
  - Tests sur les modèles (vérifier shapes d'entrée/sortie)
- **Priorité** : P0

---

### G3 : Absence totale de README

- **Description** : Aucun fichier `README.md` à la racine du projet. Un nouveau développeur ne sait pas comment installer, configurer ou exécuter le projet.
- **Impact** : Critique — aucune documentation d'entrée, aucune instruction d'installation, aucun exemple d'utilisation.
- **Solution suggérée** : Créer un `README.md` avec :
  - Objectif du projet et contexte
  - Prérequis (Python 3.10+, CUDA optionnel)
  - Installation (`pip install -r requirements.txt`)
  - Usage (`python train_adversarial.py --model lstm --seq_length 10`)
  - Structure du projet
  - Référence aux documents dans `docs/`
- **Priorité** : P0

---

## Gaps Majeurs (P1)

---

### G4 : Normalisation redondante dans le pipeline JSON

- **Description** : `json_preprocessor.py` (lignes 832-846) applique Min-Max **puis** StandardScaler. Appliquer StandardScaler après Min-Max (qui produit déjà des valeurs dans [0,1]) est mathématiquement redondant. Le CSV pipeline applique uniquement Min-Max, créant une incohérence entre les deux pipelines.
- **Impact** : Majeur — incohérence entre pipelines, potentiellement dégradation des performances, confusion documentaire.
- **Solution suggérée** : Standardiser le choix : soit Min-Max seul (comme CSV), soit StandardScaler seul. Documenter le choix dans `docs/pretraitement`.
- **Priorité** : P1

---

### G5 : Gestion des classes inconnues au test non spécifiée

- **Description** : Si un appareil apparaît dans le test mais pas dans le train (possible avec le split temporel), le `LabelEncoder` lui attribue la classe 0 par défaut (`preprocessor.py` lignes 510-514). Cela fausse les métriques d'évaluation sans avertissement.
- **Impact** : Majeur — métriques faussées, pas de détection de dérive de distribution.
- **Solution suggérée** : Ajouter une détection explicite des classes inconnues et les logger. Optionnellement, implémenter un mécanisme de "unknown class" rejection avec seuil de confiance.
- **Priorité** : P1

---

### G6 : Pas de métriques de performance cibles

- **Description** : Aucune métrique cible n'est définie. Le spec mentionne ">90%" comme référence informelle mais sans seuil formel de validation. Impossible de déterminer automatiquement si un modèle est "bon".
- **Impact** : Majeur — pas de critère objectif de succès, pas de validation automatique.
- **Solution suggérée** : Définir des seuils dans `config.yaml` (ex: `min_macro_f1: 0.85`, `min_robustness_ratio: 0.7`) et les vérifier automatiquement post-entraînement avec alerte si non atteints.
- **Priorité** : P1

---

### G7 : Feature-level attack lente (boucle Python sur chaque échantillon)

- **Description** : `FeatureLevelAttack.generate_batch()` (lignes 222-239 de `attacks.py`) itère sur chaque échantillon individuellement dans une boucle Python pure. Pour des datasets de 100k+ samples, c'est extrêmement lent.
- **Impact** : Majeur — temps d'entraînement et d'évaluation multiplié, rend les expérimentations à grande échelle impraticables.
- **Solution suggérée** : Vectoriser `generate_batch()` avec NumPy pour traiter tous les échantillons en parallèle. Les opérations (perturbation, projection, contraintes dépendantes) sont toutes vectorisables.
- **Priorité** : P1

---

### G8 : `train_adversarial.py` mélange deux protocoles d'entraînement

- **Description** : `fit()` implémente un curriculum learning en 3 phases (60-20-20, non documenté), tandis que `fit_with_phase_checkpoints()` implémente le protocole officiel en 2 phases. La méthode `fit()` est marquée DEPRECATED dans son docstring mais reste présente et potentiellement appelable.
- **Impact** : Majeur — confusion sur le protocole à utiliser, risque d'utiliser la mauvaise méthode.
- **Solution suggérée** : Supprimer `fit()` ou la renommer explicitement en `fit_curriculum_deprecated()`. S'assurer que seul `fit_with_phase_checkpoints()` est utilisé dans les scripts d'entraînement.
- **Priorité** : P1

---

### G9 : Dépendance `tokenizers` optionnelle mais critique pour le Transformer NLP

- **Description** : `IoTTokenizer` nécessite la librairie `tokenizers` (HuggingFace), qui n'est pas listée dans `requirements.txt`. Si absente, le fallback `SimpleTokenizer` est utilisé mais produit des résultats qualitativement différents.
- **Impact** : Majeur — le modèle `nlp_transformer` ne fonctionnera pas correctement sans la dépendance, et l'installation échouera silencieusement.
- **Solution suggérée** : Ajouter `tokenizers` à `requirements.txt`. Lever une erreur explicite si le modèle `nlp_transformer` est demandé sans la dépendance disponible.
- **Priorité** : P1

---

## Gaps Mineurs (P2)

---

### G10 : Redondance documentaire massive

- **Description** : 6 documents dans `docs/` (`general`, `architectures`, `train`, `pretraitement`, `featureselection`, `structure`) se chevauchent fortement. Le `rapport_analyse_specifications.md` est déjà une synthèse partielle.
- **Impact** : Mineur — maintenance difficile, risque de divergence entre documents.
- **Solution suggérée** : Fusionner en un seul document de référence avec des sections claires. Supprimer les redondances.
- **Priorité** : P2

---

### G11 : Pas de versioning des modèles sauvegardés

- **Description** : Les checkpoints sont sauvegardés dans `results/models/<model>/` sans horodatage ni version ni hash de configuration. Difficile de retrouver un modèle spécifique après plusieurs runs.
- **Impact** : Mineur — perte de traçabilité des expériences.
- **Solution suggérée** : Ajouter un timestamp ou un hash de config dans le chemin de sauvegarde (ex: `results/models/lstm/20260401_143022/checkpoint_phase1.pt`).
- **Priorité** : P2

---

### G12 : `docs/important.md` référencé mais inexistant

- **Description** : Plusieurs fichiers (`main.py` ligne 5, `sequence.py` ligne 3, `split.py` ligne 3, `evaluator.py` ligne 3) référencent `docs/important.md` qui n'existe pas dans le dossier `docs/`.
- **Impact** : Mineur — références cassées, confusion pour les développeurs.
- **Solution suggérée** : Créer le fichier ou supprimer les références dans les docstrings.
- **Priorité** : P2

---

### G13 : Pas de logging structuré

- **Description** : Toute la sortie se fait via `print()`. Impossible de rediriger vers un fichier, de filtrer par niveau, ou de parser automatiquement les résultats.
- **Impact** : Mineur — difficulté de débogage et d'analyse post-run.
- **Solution suggérée** : Remplacer les `print()` par le module `logging` Python avec niveaux (INFO, WARNING, ERROR). Ajouter un handler fichier optionnel.
- **Priorité** : P2

---

### G14 : Pas de gestion des appareils inconnus au déploiement

- **Description** : Le modèle ne peut classifier que les classes vues à l'entraînement. En production, un nouvel appareil IoT serait mal classifié sans avertissement.
- **Impact** : Mineur — risque de fausses classifications silencieuses en déploiement réel.
- **Solution suggérée** : Ajouter un seuil de confiance sur la sortie softmax (ex: max(softmax) < 0.5 → "unknown device").
- **Priorité** : P2

---

### G15 : Scripts utilitaires orphelins non documentés

- **Description** : Les fichiers `fix_presentation.py`, `fix_titles_final.py`, `generate_crash_test_nb.py`, `update_nb.py`, `extract_pdf.py`, `dump_vis.py` sont des scripts utilitaires sans documentation ni contexte d'utilisation.
- **Impact** : Mineur — confusion, fichiers potentiellement obsolètes.
- **Solution suggérée** : Documenter chaque script dans un docstring ou les déplacer dans un dossier `scripts/` avec un README explicatif.
- **Priorité** : P2

---

### G16 : `.gitignore` incomplet

- **Description** : Les fichiers `__pycache__/`, `.venv/`, `.ruff_cache/`, les fichiers `.pptx` volumineux, et les artefacts de `results/` et `data/processed/` ne sont pas exclus du versionnement.
- **Impact** : Mineur — repo pollué par des artefacts, taille du repo gonflée.
- **Solution suggérée** : Mettre à jour `.gitignore` pour exclure :
  ```
  __pycache__/
  *.pyc
  .venv/
  .ruff_cache/
  results/
  data/processed/
  *.pptx
  *.pkl
  *.npy
  ```
- **Priorité** : P2

---

### G17 : Pas de cross-validation

- **Description** : Les résultats dépendent d'un seul split 80/20. Une variation aléatoire pourrait changer significativement les performances rapportées.
- **Impact** : Mineur — résultats potentiellement non reproductibles, variance non estimée.
- **Solution suggérée** : Implémenter un mode k-fold ou repeated temporal split pour estimer la variance des performances.
- **Priorité** : P2

---

### G18 : Defensive distillation non implémentée

- **Description** : Mentionnée comme optionnelle dans les specs (Phase 5) mais jamais implémentée.
- **Impact** : Mineur — fonctionnalité manquante mais non critique.
- **Solution suggérée** : Implémenter si le temps le permet, ou documenter explicitement comme "hors scope" dans le rapport final.
- **Priorité** : P2

---

## Résumé des Priorités

| Priorité | Nombre | Gaps |
|----------|--------|------|
| **P0 — Critique** | 3 | G1 (split main.py), G2 (pas de tests), G3 (pas de README) |
| **P1 — Majeur** | 6 | G4 (normalisation), G5 (classes inconnues), G6 (métriques cibles), G7 (performance attaque), G8 (double protocole), G9 (dépendance tokenizers) |
| **P2 — Mineur** | 9 | G10-G18 (documentation, logging, versioning, etc.) |

---

*Document généré automatiquement par analyse du codebase — 2026-04-01*
