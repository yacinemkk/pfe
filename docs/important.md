# Plan Très Détaillé pour Implémenter les Requirements

## Vue d'Ensemble

Le projet nécessite l'implémentation d'un pipeline complet d'identification d'appareils IoT avec 6 modèles de Deep Learning, un système anti-data leakage, et une évaluation de robustesse face aux attaques adverses.

---

## PHASE 1: Prétraitement des Données (Feature Engineering)

### 1.1 Chargement et Exploration des Datasets
- [ ] Charger les datasets `IoT IPFIX Home` (47 jours, 12 foyers, 24 types) et `IPFIX Records` (3 mois, 26 appareils, 9M+ enregistrements)
- [ ] Analyser la distribution des classes et identifier le déséquilibre
- [ ] Documenter les 28 attributs de flux disponibles

### 1.2 Filtrage Orienté SDN
- [ ] Supprimer les colonnes interdites: `IP source`, `IP destination`, `MAC source`, `MAC destination`, `port source`, `port destination`
- [ ] Conserver uniquement: `inter-arrival time`, `taille moyenne paquets entrants/sortants`, `type protocole IP`, `durée flux`, `nombre paquets`
- [ ] Filtrer les protocoles de gestion: ARP, DHCP, ICMP, NTP

### 1.3 Nettoyage
- [ ] Supprimer les doublons avec Pandas (`drop_duplicates()`)
- [ ] Gérer les valeurs manquantes (suppression ou imputation)
- [ ] Valider l'intégrité des données

### 1.4 Gestion du Déséquilibre des Classes
- [ ] Identifier les classes sous-représentées (capteurs sommeil, etc.)
- [ ] Option A: Réduire à 18 classes (IPFIX Home) / 17 classes (IPFIX Records)
- [ ] Option B: Appliquer Borderline-SMOTE ou GAN pour augmentation
- [ ] Documenter la distribution finale

### 1.5 Normalisation et Standardisation
- [ ] Appliquer Min-Max Scaling (plage 0-1) sur variables continues
- [ ] Appliquer standardisation (moyenne=0, écart-type=1)
- [ ] **IMPORTANT**: Calculer les paramètres UNIQUEMENT sur train, appliquer sur train ET test
- [ ] Ne PAS utiliser one-hot encoding (garder catégorielles pour attention)

---

## PHASE 2: Pipeline Anti-Data Leakage

### 2.1 Regroupement par Appareil
- [ ] Grouper les flux par identifiant appareil (`label` ou `MAC` avant suppression)
- [ ] Valider qu'aucun groupe n'est vide

### 2.2 Tri Chronologique
- [ ] Pour chaque groupe: trier par `flow_start` (ordre croissant)
- [ ] Vérifier l'intégrité temporelle

### 2.3 Split Temporel 80/20
- [ ] Pour chaque appareil: premiers 80% → Train, derniers 20% → Test
- [ ] Garantir: Train = Passé, Test = Futur
- [ ] Vérifier: aucun chevauchement

### 2.4 Génération des Séquences
- [ ] Définir taille fenêtre (ex: 10 flux consécutifs)
- [ ] Appliquer sliding window SÉPARÉMENT sur Train et Test
- [ ] Garantir: chaque séquence = flux d'un SEUL appareil
- [ ] Aucune séquence ne doit traverser la frontière train/test

---

## PHASE 3: Tokenisation pour Transformer (IoT-Tokenize)

### 3.1 Transformation Structurée
- [ ] Convertir chaque flux en format: `nom_feature1 valeur1; nom_feature2 valeur2; ...`
- [ ] Exemple: `ptcl 6; ipv 4; bi_dur 12.5; bi_pkt 150;`

### 3.2 Création du Vocabulaire
- [ ] Définir tokens prédéfinis: `ptcl`, `ipv`, `bi_dur`, `bi_pkt`, etc.
- [ ] Définir tokens réservés: `<s>`, `</s>`, `<pad>`, `<unk>`, `<mask>`
- [ ] Taille vocabulaire: ~52 000 tokens

### 3.3 Encodage BPE
- [ ] Implémenter Byte-Level BPE tokenizer
- [ ] Limiter risque "out of vocabulary"
- [ ] Encoder les valeurs numériques

### 3.4 Padding et Tenseurs
- [ ] Définir longueur max séquence (512 ou 1024)
- [ ] Right-padding avec `<pad>`
- [ ] Convertir en tenseurs PyTorch/TensorFlow

---

## PHASE 4: Implémentation des 6 Architectures

### 4.1 Modèle LSTM (Priorité ⭐⭐⭐⭐⭐)
- [ ] Architecture: 2 couches LSTM, 64 unités/couche, activation ReLU
- [ ] Dropout pour régularisation
- [ ] Couche Dense de sortie (classification multi-classe)
- [ ] Vecteur embedding: 128 dimensions

### 4.2 Modèle BiLSTM (Priorité ⭐⭐⭐⭐)
- [ ] Remplacer LSTM par BiLSTM bidirectionnel
- [ ] Même configuration: 2 couches, 64 unités
- [ ] Capture contexte avant/arrière

### 4.3 Modèle CNN-LSTM (Priorité ⭐⭐⭐⭐⭐)
- [ ] Couche Conv1D avec MaxPooling (extraction spatiale)
- [ ] Couche LSTM 64 neurones (return_sequences=True)
- [ ] MaxPool1D + Flatten
- [ ] Dense 100 neurones + sortie

### 4.4 Modèle XGBoost-LSTM (Priorité ⭐⭐⭐)
- [ ] LSTM pré-entraîné comme extracteur de features
- [ ] Sortie: vecteur latent de taille fixe
- [ ] XGBoost avec hyperparamètres: `lr=[0.01-0.3]`, `max_depth=[3-15]`, `min_child_weight=[1-7]`, `gamma=[0-0.5]`

### 4.5 Modèle Transformer (Priorité ⭐⭐⭐⭐⭐)
- [ ] 6 couches encodeur
- [ ] 768 dimensions, 12 têtes d'attention
- [ ] FFN 3072 dimensions, activation GELU
- [ ] Connexions résiduelles + LayerNorm

### 4.6 Architecture Hybride CNN-BiLSTM-Transformer (Priorité ⭐⭐⭐⭐⭐)
- [ ] **Branche CNN 1**: Conv1D kernel=3, ReLU, MaxPool1D, BatchNorm, Flatten
- [ ] **Branche CNN 2**: Conv1D kernel=5, ReLU, MaxPool1D, BatchNorm, Flatten
- [ ] **Couche Fusion**: Concaténation des 2 branches
- [ ] **Module BiLSTM**: Modélisation temporelle bidirectionnelle
- [ ] **Module Transformer**: Encodeurs avec Multi-Head Attention
- [ ] **Sortie**: Mean Pooling + FC + Sigmoid/Softmax

---

## PHASE 5: Entraînement - Phase 1 (Standard)

### 5.1 Configuration d'Entraînement
- [ ] Définir loss: CrossEntropyLoss
- [ ] Optimiseur: Adam ou AdamW
- [ ] Learning rate scheduler
- [ ] Early stopping

### 5.2 Entraînement Initial (Données Bénignes)
- [ ] Pour chaque modèle: entraîner sur 80% train (trafic normal uniquement)
- [ ] Validation sur subset de train
- [ ] Sauvegarder checkpoints

### 5.3 Évaluation Standard
- [ ] Métriques globales: Macro F1-Score, Accuracy
- [ ] Métriques par classe: Précision, Rappel, F1-Score
- [ ] Matrice de confusion
- [ ] Attendu: >90% sur données normales

### 5.4 Crash Test 1 (Vulnérabilité)
- [ ] **Générer attaques adverses**: FGSM, BIM, PGD
- [ ] **Test 1**: Données bénignes → scores élevés
- [ ] **Test 2**: Données adverses uniquement → observer chute F1 (ex: 90% → 20-40%)
- [ ] **Test 3**: Mélange bénignes + adverses → évaluation réaliste
- [ ] Documenter les vulnérabilités

---

## PHASE 6: Entraînement - Phase 2 (Antagoniste)

### 6.1 Création du Corpus Conjoint
- [ ] Générer échantillons adverses à partir de train
- [ ] Étiqueter correctement (vraie classe appareil)
- [ ] Mélanger avec données bénignes pures

### 6.2 Ré-entraînement (Adversarial Training)
- [ ] Ré-entraîner chaque modèle sur corpus conjoint
- [ ] Ajuster hyperparamètres si nécessaire
- [ ] Sauvegarder modèles robustes

### 6.3 Crash Test 2 (Résilience)
- [ ] **Test 1**: Données bénignes → vérifier trade-off (perte ~2% acceptable)
- [ ] **Test 2**: Données adverses uniquement → observer remontée F1 (ex: 20% → 80%+)
- [ ] **Test 3**: Mélange → confirmer gestion trafic compromis

---

## PHASE 7: Comparaison et Rapport Final

### 7.1 Comparaison des Modèles
- [ ] Tableau comparatif: Performance normale vs Performance adverse
- [ ] Graphiques: F1-Score par modèle, par phase
- [ ] Identifier le meilleur modèle

### 7.2 Validation Architecture Hybride
- [ ] Démontrer supériorité CNN-BiLSTM-Transformer
- [ ] Plus haute précision (vision spatio-temporelle + globale)
- [ ] Meilleure résilience face aux attaques

### 7.3 Documentation
- [ ] Rapport complet avec toutes les métriques
- [ ] Visualisations et interprétations
- [ ] Recommandations pour déploiement SDN

---

## Structure de Code Proposée

```
src/
├── data/
│   ├── loader.py           # Chargement datasets
│   ├── preprocessing.py    # Filtrage, nettoyage
│   ├── split.py           # Split temporel anti-leakage
│   ├── sequence.py        # Génération séquences
│   └── tokenizer.py       # IoT-Tokenize BPE
├── models/
│   ├── lstm.py
│   ├── bilstm.py
│   ├── cnn_lstm.py
│   ├── xgboost_lstm.py
│   ├── transformer.py
│   └── hybrid.py          # CNN-BiLSTM-Transformer
├── training/
│   ├── trainer.py         # Boucle entraînement
│   ├── adversarial.py     # Génération attaques (FGSM, BIM, PGD)
│   └── evaluator.py       # Métriques et tests
├── config/
│   └── config.yaml        # Hyperparamètres
└── main.py
```

---

## Résumé des Contraintes Critiques

### Anti-Data Leakage (OBLIGATOIRE)
1. Pas de chevauchement entre train et test
2. Pas de mélange temporel (train = passé, test = futur)
3. Pas de mélange entre appareils dans une même séquence
4. Normalisation calculée sur train, appliquée sur test

### Pipeline SDN
1. Supprimer IP, MAC, ports avant entraînement
2. Conserver uniquement caractéristiques de flux globales

### Évaluation Robustesse
1. Phase 1: Évaluer vulnérabilité face aux attaques
2. Phase 2: Prouver résilience après adversarial training
3. Métrique principale: Macro F1-Score
