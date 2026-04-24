# Chapitre 3 — Datasets et Prétraitement des Données

## 3.1 Vue d'Ensemble des Deux Datasets

Ce projet utilise deux datasets IPFIX indépendants, capturés dans des environnements domestiques réels. Le traitement en parallèle de deux sources hétérogènes reflète la réalité des déploiements réseau où différentes infrastructures coexistent.

| Caractéristique | Dataset CSV | Dataset JSON |
|-----------------|-------------|--------------|
| **Format** | CSV (plusieurs fichiers `home*_labeled.csv`) | JSON (un fichier unique > 1 Go) |
| **Classes IoT** | 18 appareils | 17 appareils |
| **Features initiales** | ~30 colonnes IPFIX | 28 features continues + 8 bits direction |
| **Features après sélection** | 15–20 features continues + `ipProto` | 28 features continues + 8 bits |
| **Normalisation** | StandardScaler | StandardScaler |
| **Format des séquences** | `(N, 10, F)` tenseur 3D | `(N, 10, 36)` tenseur 3D |

**Appareils Dataset CSV (18 classes) :** Eclear, Amazon Echo Dot, Microsoft Atom Cam, Amazon Fire TV Stick 4K, Google Nest Hub, Philips Hue, TP-Link Tapo, Wemo Insight Switch, etc.

**Appareils Dataset JSON (17 classes) :** Qrio Hub, Philips Hue Bridge, Amazon Echo Dot, Wansview Camera, Google Nest, Sony Bravia, Netatmo, iRobot Roomba, etc.

---

## 3.2 Philosophie Anti-Data-Leakage

### 3.2.1 Définition du Data Leakage

Le *data leakage* (fuite de données) est un problème critique en apprentissage automatique : il survient lorsque des informations du jeu de test "contaminent" le processus d'entraînement, conduisant à des mesures de performance artificiellement gonflées qui ne se généralisent pas au déploiement réel.

Dans le contexte IoT avec séquences temporelles, trois formes de leakage sont particulièrement dangereuses :

1. **Leakage temporel :** si des données "futures" sont utilisées pour entraîner le modèle sur des données "passées", le modèle bénéficie d'une information qu'il n'aurait pas en production.
2. **Leakage inter-appareils :** si une séquence de flux mélange des flux de deux appareils différents, le modèle apprend à associer des comportements non causalement liés.
3. **Leakage du scaler :** si les paramètres de normalisation (min, max) sont calculés sur l'ensemble de toutes les données (train + test), le modèle bénéficie implicitement de la distribution du test.

### 3.2.2 Protocole Anti-Leakage Implémenté

Le projet implémente un protocole strict en 5 garanties, documenté dans `docs/general` et implémenté dans `src/data/preprocessor.py` (classe `IoTDataProcessor`) et `src/data/json_preprocessor.py` (classe `JsonIoTDataProcessor`) :

1. **Groupement par appareil :** tous les flux sont séparés en sous-ensembles distincts, un par type d'appareil IoT (par label ou par MAC).
2. **Tri chronologique strict :** pour chaque sous-ensemble, les flux sont triés par horodatage croissant (`flow_start` ou colonne `start`).
3. **Split temporel 72/18/10 par appareil :** les 72% premiers flux (dans le temps) constituent le train, les 18% suivants la validation, les 10% finaux le test. Ce split est fait **par appareil**, garantissant que le modèle ne voit jamais de flux "futurs" lors de l'entraînement.
4. **Génération des séquences indépendamment :** les fenêtres glissantes sont appliquées séparément sur les sets train, val et test déjà séparés.
5. **Fit des scalers uniquement sur le train :** les paramètres de normalisation spatiale (StandardScaler) sont calculés exclusivement sur les données d'entraînement, puis appliqués aux données de validation et de test.

---

## 3.3 Pipeline de Prétraitement CSV — 4 Étapes Détaillées

Implémenté dans `src/data/preprocessor.py`, classe `IoTDataProcessor`.

### 3.3.1 Étape 1 : Filtrage et Adaptation au SDN (`sdn_filter`)

**Objectif :** Supprimer toutes les colonnes qui ne sont pas accessibles via l'API d'un contrôleur SDN — notamment les adresses IP et MAC — et conserver uniquement les statistiques de flux accessibles via IPFIX/OpenFlow.

**Colonnes supprimées :**
```python
sdn_excluded = [
    "srcMac",    # Adresse MAC source — facilement usurpable
    "destMac",   # Adresse MAC destination
    "srcIP",     # Adresse IP source — non fiable (DHCP, NAT)
    "destIP",    # Adresse IP destination — non SDN-compatible
    "srcPort",   # Port source — variable par connexion
    "destPort",  # Port destination
    "device",    # Identifiant interne de l'appareil
]
```

**Pourquoi ces colonnes sont supprimées :** Dans un réseau SDN, les match fields disponibles via OpenFlow incluent les statistiques de compteurs (paquets, octets), les durées de flux, et les statistiques d'inter-arrivée — mais pas les adresses IP/MAC *source* qui sont des identifiants trop facilement falsifiables. Supprimer ces colonnes force le modèle à apprendre des patterns comportementaux intrinsèques.

**Note importante :** La colonne `start` (timestamp du début du flux) est **conservée** à cette étape pour permettre le tri chronologique lors du split temporel. Elle est supprimée **après** le split.

**Filtrage des classes :** seuls les appareils listés dans `IOT_IPFIX_HOME_CLASSES` et `IOT_DEVICE_CLASSES` (18 classes officielles) sont conservés. Les appareils avec moins de `MIN_SAMPLES_PER_CLASS` flux sont éliminés pour assurer une représentation statistiquement significative.

**Colonnes conservées après filtrage (environ 30 features) :**
durée du flux, compteurs de paquets entrants/sortants, compteurs d'octets entrants/sortants, taille moyenne/min/max des paquets, temps moyen d'inter-arrivée, variance des tailles de paquets, ratio de paquets, protocole IP (`ipProto`), indicateurs TCP, etc.

---

### 3.3.2 Étape 2 : Équilibrage et Filtrage du Bruit (`balance_and_filter_noise`)

**Contexte :** les datasets IoT sont notoirement déséquilibrés. Un réseau domestique génère beaucoup plus de trafic pour la télévision connectée (Netflix en streaming) que pour le thermostat intelligent (quelques paquets par heure). Sans correction, le modèle apprend à maximiser l'accuracy en se spécialisant sur les classes majoritaires.

Cette étape est composée de trois sous-étapes appliquées **uniquement sur les données d'entraînement** :

#### 3.3.2.1 Gestion du Déséquilibre (Class Imbalance)

Initialement, des techniques d'oversampling comme **Borderline-SMOTE** étaient envisagées pour pallier le déséquilibre inhérent aux datasets IoT (un téléviseur génère plus de flux qu'un thermostat). Cependant, elles ont été **écartées** dans la version finale. En effet, la génération de données synthétiques (interpolation entre points) perturbe la structure *temporelle* naturelle et introduit du bruit indésirable, déstabilisant les modèles séquentiels.
Le déséquilibre est désormais géré **intrinsèquement** au niveau de l'entraînement par les modèles (architectures avancées) et par l'utilisation de fonctions de Loss pondérées ainsi que par la dynamique de la boucle de formation antagoniste (GreedyAttackSimulator), qui obligent le modèle à généraliser.

#### 3.3.2.2 Isolation Forest (Détection d'Anomalies)

Le dataset peut contenir des flux aberrants ou perturbés. **Isolation Forest** (Liu et al., 2008) construit des arbres de décision aléatoires et mesure à quelle profondeur les exemples sont isolés : les exemples faciles à isoler (peu de coupures nécessaires) sont des anomalies.

```python
iso_forest = IsolationForest(
    contamination=0.05,    # On suppose que 5% des données sont des anomalies
    random_state=42,
    n_jobs=-1              # Parallélisation sur tous les cœurs
)
outliers_if = iso_forest.fit_predict(X_resampled)
# Retient uniquement les inliers (label == 1)
X_filtered = X_resampled[outliers_if == 1]
```

**Pourquoi Isolation Forest ?** Contrairement aux méthodes basées sur la distance (LOF, k-NN), Isolation Forest est efficient en O(n log n) et fonctionne bien en haute dimension (caractéristique importante avec 20-30 features).

#### 3.3.2.3 Local Outlier Factor (Filtrage Résiduel)

Après Isolation Forest, **LOF** (Breunig et al., 2000) applique un second filtre basé sur la densité locale. LOF calcule le ratio entre la densité locale d'un point et la densité locale de ses voisins : un point dans une région peu dense par rapport à ses voisins est probablement anormal.

```python
lof = LocalOutlierFactor(
    n_neighbors=20,        # Comparaison avec les 20 plus proches voisins
    contamination=0.05,
    n_jobs=-1
)
outliers_lof = lof.fit_predict(X_filtered)
X_final = X_filtered[outliers_lof == 1]
```

**Complémentarité avec Isolation Forest :** LOF est plus précis pour détecter des anomalies locales que Isolation Forest qui a une perspective globale. Les deux filtres combinés assurent une nettoyage robuste.

---

### 3.3.3 Étape 3 : Sélection Hybride des Features (`hybrid_feature_selection`)

**Motivations de la sélection de features :**
- Réduire la dimensionnalité → moins de risque d'overfitting, modèles plus rapides
- Éliminer les features redondantes ou non informatives
- Respecter la contrainte SDN : conserver uniquement les features sémantiquement pertinentes
- **Cap entre 15 et 20 features continues**, tel que spécifié dans `docs/featureselection`

La sélection est un processus hybride combinant trois méthodes indépendantes, dont les scores sont combinés par une moyenne pondérée.

#### 3.3.3.1 XGBoost Feature Importance

XGBoost (Extreme Gradient Boosting) est entraîné en tant que classifieur multi-classe sur les données d'entraînement. L'importance de chaque feature est calculée par le gain moyen obtenu en utilisant cette feature dans les splits des arbres de gradient boosting.

```python
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="mlogloss"
)
xgb_clf.fit(X, y)
xgb_importance = xgb_clf.feature_importances_
```

**Avantage :** XGBoost capture les interactions non-linéaires entre features et les effets de combinaisons. Une feature peut être peu informative seule mais très puissante combinée à d'autres.

#### 3.3.3.2 Test du Chi-Carré (Chi²)

Le test Chi² mesure la dépendance statistique entre chaque feature et la variable cible. Une valeur Chi² élevée indique que la distribution de la feature varie significativement selon la classe — ce qui est signe d'utilité discriminante.

```python
X_positive = X - X.min() + 1e-6   # Chi² requiert des valeurs positives
chi2_scores, _ = chi2(X_positive, y)
chi2_scores = chi2_scores / (chi2_scores.max() + 1e-10)  # Normalisation
```

**Avantage :** le Chi² est basé sur la statistique et non sur un modèle particulier, ce qui le rend complémentaire à XGBoost.

**Limitation :** le Chi² suppose une indépendance conditionnelle et ne capture pas les interactions entre features.

#### 3.3.3.3 Information Mutuelle (MI)

L'information mutuelle mesure la réduction d'incertitude sur la variable cible `Y` apportée par la connaissance d'une feature `X_i` :
```
I(X_i; Y) = H(Y) - H(Y | X_i)
```

```python
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_scores = mi_scores / (mi_scores.max() + 1e-10)
```

**Avantage :** l'information mutuelle capture les dépendances non-linéaires entre features et cible, là où le test Chi² est limité aux dépendances linéaires.

#### 3.3.3.4 Combinaison et Méthode du Coude

Les trois scores sont combinés par une moyenne pondérée :
```python
# Normalisation de XGBoost
xgb_norm = xgb_importance / (xgb_importance.max() + 1e-10)

# Score combiné (0.4 XGBoost + 0.3 Chi² + 0.3 MI)
combined_scores = 0.4 * xgb_norm + 0.3 * chi2_scores + 0.3 * mi_scores
```

**Justification des poids :** XGBoost reçoit un poids légèrement supérieur (0.4) car il modélise directement le problème de classification en capturant les interactions entre features. Chi² et MI reçoivent des poids égaux (0.3) comme critères statistiques complémentaires.

**Méthode du Coude (Elbow Method)** pour déterminer le nombre optimal de features `k` :

```python
@staticmethod
def find_elbow_k(scores: np.ndarray) -> int:
    """
    Calcule le k optimal en maximisant la distance perpendiculaire
    de chaque point à la droite reliant le premier et le dernier point
    de la courbe des scores triés en ordre décroissant.
    """
    sorted_scores = np.sort(scores)[::-1]
    # Construire la droite p1-p2
    p1 = np.array([0, sorted_scores[0]])
    p2 = np.array([len(scores)-1, sorted_scores[-1]])
    line_vec = p2 - p1
    # Distance perpendiculaire de chaque point
    distances = [abs(np.cross(line_vec, [i, s] - p1)) / np.linalg.norm(line_vec)
                 for i, s in enumerate(sorted_scores)]
    # Le coude est le point le plus éloigné
    elbow_idx = np.argmax(distances)
    return max(1, elbow_idx + 1)
```

La méthode du coude identifie automatiquement le point d'inflexion de la courbe d'importance, au-delà duquel les features supplémentaires n'apportent que peu d'information. Le résultat est ensuite borné dans l'intervalle [15, 20] features conformément aux spécifications.

---

### 3.3.4 Étape 4 : Normalisation (`standard_scaler.fit_transform`)

**Pourquoi normaliser ?** Les réseaux de neurones profonds ont des difficultés à converger lorsque les features ont des échelles très différentes (une feature en millisecondes, une autre en octets). La normalisation garantit que toutes les features contribuent équitablement au signal de gradient.

**Standardisation :**
La Normalisation Min-Max a été supprimée suite à la constatation d'une redondance avec le StandardScaler et de sa forte sensibilité aux valeurs aberrantes (outliers) réseau. **StandardScaler** est utilisé en substitution, assurant une moyenne de 0 et un écart-type de 1 pour toutes les features, facilitant grandement la convergence de la descente de gradient dans l'architecture Deep Learning.

**Règle critique anti-leakage :**
```python
# FIT uniquement sur train
X_train_scaled = standard_scaler.fit_transform(X_train_continuous)

# TRANSFORM seulement sur val et test (pas de fit)
X_val_scaled = standard_scaler.transform(X_val_continuous)
X_test_scaled = standard_scaler.transform(X_test_continuous)
```

**Features catégorielles :** la feature `ipProto` (valeurs entières : 6=TCP, 17=UDP, 1=ICMP) est intégrée tel quel (ou via un one-hot ou embedding dédié).

---

## 3.4 Pipeline de Prétraitement JSON — Spécificités

Le pipeline JSON (`src/data/json_preprocessor.py`, classe `JsonIoTDataProcessor`) suit les mêmes 4 étapes que le pipeline CSV mais avec les adaptations suivantes :

### 3.4.1 Lecture et Parsing du JSON

Le fichier JSON unique est parsé de manière **streaming** (par chunks) pour éviter de charger les > 1 Go en mémoire RAM. Chaque enregistrement IPFIX est converti en ligne DataFrame.

**Mapping MAC → Label :** le dataset JSON ne contient pas directement les labels d'appareils mais les adresses MAC. Un dictionnaire de correspondance MAC→type est appliqué pour créer la colonne label.

### 3.4.2 Features Spéciales : Les 8 Bits de Direction

Le dataset JSON inclut 8 features binaires `pkt_dir_0` à `pkt_dir_7` encodant la direction (entrant=1, sortant=0) des 8 premiers paquets du flux. Ces features sont :
- **Binaires par nature** (valeurs 0 ou 1 uniquement)
- **Non modifiables** par un attaquant (imposées par le protocole TCP/UDP)
- **Très informatives** : le pattern d'initiation d'une connexion est caractéristique du type d'appareil

Ces 8 bits sont conservés tels quels (sans normalisation) et concaténés après les 28 features continues normalisées, donnant `input_size = 36` features pour le modèle JSON.

### 3.4.3 Normalisation Simple : StandardScaler

```python
X_train_standard = standard_scaler.fit_transform(X_train_continuous)
```

Un simple **StandardScaler** est appliqué pour ramener les données à une moyenne de 0 et une variance de 1, retirant l'ancienne redondance mathématique (MinMax suivi de StandardScaler) qui altérait défavorablement les distributions pour l'apprentissage.

---

## 3.5 Génération des Séquences par Fenêtre Glissante

### 3.5.1 Paramètres

```python
SEQ_LENGTH = 10    # Longueur de chaque séquence (10 flux consécutifs)
STRIDE = 10        # Pas entre deux séquences (pas de chevauchement)
```

### 3.5.2 Mécanisme

Pour chaque appareil `d` dans l'ensemble d'entraînement :
1. Extraire tous les flux de l'appareil `d` : `X_d = X[y == d]`
2. Créer des fenêtres : séquence `i` = flux `[i*stride : i*stride + seq_length]`
3. Label de la séquence = label du dernier flux de la fenêtre

```python
for label in unique_labels:
    mask = (y == label)
    X_group = X[mask]    # Flux de cet appareil uniquement
    for i in range(0, len(X_group) - seq_length + 1, stride):
        X_seq.append(X_group[i : i + seq_length])  # Fenêtre de 10 flux
        y_seq.append(y_group[i + seq_length - 1])   # Label = dernier flux
```

**Garantie :** aucune séquence ne mélange des flux de deux appareils différents.

### 3.5.3 Format de Sortie

Les séquences sont des tenseurs 3D : `(N, 10, F)` où :
- `N` = nombre total de séquences
- `10` = longueur de séquence (seq_length)
- `F` = nombre de features (15–21 pour CSV, 36 pour JSON)

Ce format est directement compatible avec l'entrée attendue par tous les modèles Deep Learning implémentés.


