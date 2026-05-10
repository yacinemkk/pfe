# Pipeline de Chargement et Prétraitement du Dataset JSON IPFIX Records

**Fichier source** : `src/data/json_preprocessor.py`
**Fichier notebook** : `greedy_new_optimized.ipynb`

---

## Vue d'ensemble

Le pipeline de prétraitement s'exécute en **4 étapes** séquentielles. Il prend en entrée les fichiers JSON bruts du dataset IPFIX Records (IoT Analytics UNSW) et produit des séquences prétraitées prêtes pour l'entraînement des modèles.

---

## Étape 1 : Filtrage et Adaptation au SDN

### 1.1 Chargement des fichiers JSON

```python
def load_json_files(self, data_dir, chunk_size=100_000, max_records=None):
```

- **Lecture streaming** : les fichiers JSON sont lus ligne par ligne (`json.loads(line)`) pour éviter de charger tout le fichier en mémoire
- **Décodage hexadécimal** : `decode_packet_directions(hex_str)` décode la chaîne hexadécimale `firstEightNonEmptyPacketDirections` en 8 bits binaires représentant la direction de chaque paquet (1=outbound, 0=inbound)
- **Chunking** : les enregistrements sont accumulés par blocs de 100 000 pour limiter l'empreinte mémoire

### 1.2 Labeling par Adresse MAC

```python
def label_flow(flow, mac_to_device=None):
```

- Chaque flux est labelisé en regardant les adresse MAC source et destination
- **Double lookup** : on vérifie les deux champs MAC
- **Préférence IoT** : on privilégie l'appareil non-passerelle (non-DEFAULT GATEWAY)
- Retourne le nom de l'appareil parmi les 17 classes cibles

### 1.3 Filtrage SDN (Suppression des colonnes interdites)

```python
COLUMNS_TO_DROP = [
    "sourceMacAddress", "destinationMacAddress",
    "sourceIPv4Address", "destinationIPv4Address",
    "sourceTransportPort", "destinationTransportPort",
    "tcpSequenceNumber", "collectorName", "vlanId",
    "flowStartMilliseconds", "flowEndMilliseconds", ...
]
```

- **Suppression des IP et ports** : ces informations ne sont pas accessibles via les API SDN (OpenFlow)
- **Suppression des identifiants de flux temporels** : pour éviter le data leakage
- **Conservation uniquement des caractéristiques statistiques** accessibles via OpenFlow

### 1.4 Features Extraites

| Catégorie | Features (28 continues + 8 binaires) |
|-----------|--------------------------------------|
| **Temps** | `flowDurationMilliseconds`, `averageInterarrivalTime`, `standardDeviationInterarrivalTime` |
| **Protocole** | `protocolIdentifier` (catégorielle) |
| **Métriques globales** | `packetTotalCount`, `octetTotalCount`, `reversePacketTotalCount`, `reverseOctetTotalCount` |
| **Drapeaux TCP** | `initialTCPFlags`, `unionTCPFlags`, `reverseInitialTCPFlags`, `reverseUnionTCPFlags` |
| **Détail forward** | `tcpUrgTotalCount`, `smallPacketCount`, `nonEmptyPacketCount`, `dataByteCount`, `firstNonEmptyPacketSize`, `largePacketCount`, `maxPacketSize`, `standardDeviationPayloadLength`, `bytesPerPacket` |
| **Détail reverse** | `reverseTcpUrgTotalCount`, `reverseSmallPacketCount`, `reverseNonEmptyPacketCount`, `reverseDataByteCount`, `reverseAverageInterarrivalTime`, `reverseFirstNonEmptyPacketSize`, `reverseLargePacketCount`, `reverseMaxPacketSize`, `reverseStandardDeviationPayloadLength`, `reverseStandardDeviationInterarrivalTime` |
| **Directions paquets** | `pkt_dir_0` à `pkt_dir_7` (8 bits décodés du hex) |

**Total : 36 features par flux**

### 1.5 Split Temporel Anti-Leakage

- Split **70/10/20** (train/val/test) **par appareil**
- Les données sont triées par `flowStartMilliseconds` avant le split
- Les séquences sont générées **séparément** sur chaque split pour éviter toute contamination temporelle

---

## Étape 2 : Équilibrage et Filtrage du Bruit

```python
def balance_and_filter_noise(self, X, y, contamination=0.05):
```

### 2.1 Borderline-SMOTE
- **Désactivé actuellement** (ligne 467)
- Était utilisé pour suréchantillonner les classes minoritaires via Borderline-SMOTE

### 2.2 Isolation Forest

```python
iso_forest = IsolationForest(
    contamination=0.05,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
outliers_if = iso_forest.fit_predict(X_resampled)
```

- **Objectif** : détecter les anomalies dans l'espace des features
- **Fonctionnement** : les arbres isolent récursivement les points ; les anomalies sont isolées plus rapidement
- **Paramètre** : 5% des données considérées comme contamination maximale
- **Résultat** : suppression des échantillons identifiés comme aberrants

### 2.3 Local Outlier Factor (LOF)

```python
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=contamination,
    n_jobs=-1
)
outliers_lof = lof.fit_predict(X_filtered)
```

- **Objectif** : filtrer le bruit résiduel non détecté par Isolation Forest
- **Fonctionnement** : compare la densité locale d'un point à celle de ses k=20 voisins les plus proches
- **Résultat** : supprime les points dont la densité est significativement plus faible que celle de leurs voisins

### Ordre d'exécution
```
Données brutes → Borderline-SMOTE (désactivé) → Isolation Forest → LOF → Données filtrées
```

---

## Étape 3 : Sélection Hybride des Caractéristiques

```python
def hybrid_feature_selection(self, X, y, feature_names, top_k=None):
```

Cette étape combine **trois méthodes complémentaires** pour évaluer l'importance de chaque feature.

### 3.1 XGBoost Feature Importance (poids 40%)

```python
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
    eval_metric="mlogloss",
)
xgb_clf.fit(X, y)
xgb_importance = xgb_clf.feature_importances_
```

- **Principe** : un classifieur XGBoost est entraîné, puis les importances sont extraites via `feature_importances_`
- **Ce que ça mesure** : la contribution de chaque feature à la réduction de l'entropie (gain d'information) dans les arbres de décision
- **Avantage** : capture les relations non-linéaires et les interactions entre features

### 3.2 Test du Chi-Carré (poids 30%)

```python
X_positive = X - X.min() + 1e-6
chi2_scores, _ = chi2(X_positive, y)
chi2_scores = chi2_scores / (chi2_scores.max() + 1e-10)
```

- **Principe** : teste l'indépendance statistique entre chaque feature et la variable cible
- **Ce que ça mesure** : une feature avec un score Chi² élevé est fortement dépendante de la classe
- **Condition** : les valeurs doivent être positives (d'où le décalage `X - X.min()`)
- **Normalisation** : divise par le max pour mettre à l'échelle 0-1

### 3.3 Information Mutuelle (poids 30%)

```python
mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
mi_scores = mi_scores / (mi_scores.max() + 1e-10)
```

- **Principe** : calcule la dépendance mutuelle entre chaque feature et la classe
- **Ce que ça mesure** : la quantité d'information gained sur Y en connaissant X
- **Avantage** : capture les dépendances **non-linéaires** que Chi² ne détecte pas

### 3.4 Score Combiné

```python
xgb_norm = xgb_importance / (xgb_importance.max() + 1e-10)
combined_scores = 0.4 * xgb_norm + 0.3 * chi2_scores + 0.3 * mi_scores
```

| Méthode | Poids | Ce qu'elle détecte |
|---------|-------|-------------------|
| XGBoost | 40% | Importance prédictive (non-linéaire) |
| Chi² | 30% | Dépendance statistique linéaire |
| Information Mutuelle | 30% | Dépendance non-linéaire |

### 3.5 Sélection par la Méthode du Coude (Elbow)

```python
def find_elbow_k(scores):
    sorted_scores = np.sort(scores)[::-1]
    # Calcule la distance perpendiculaire de chaque point à la ligne droite
    # Le point le plus éloigné = le coude optimal
    distances = np.zeros(n)
    for i in range(n):
        pt = np.array([x[i], sorted_scores[i]])
        distances[i] = abs(np.cross(line_vec, pt - p1)) / np.sqrt(line_len_sq)
    elbow_idx = int(np.argmax(distances))
    return max(1, elbow_idx + 1)
```

- **Principe** : on trace les scores triés par ordre décroissant, puis on trouve le "coude" de la courbe
- **Méthode** : calcule la distance perpendiculaire maximale entre chaque point et la ligne reliant le premier et le dernier point
- **Résultat** : le nombre optimal de features à sélectionner est automatiquement déterminé

---

## Étape 4 : Normalisation StandardScaler

```python
self.standard_scaler = StandardScaler()
X_train_cont_scaled = self.standard_scaler.fit_transform(X_train_selected)
X_val_cont_scaled = self.standard_scaler.transform(X_val_selected)
X_test_cont_scaled = self.standard_scaler.transform(X_test_selected)
```

### Règles importantes

| Type de feature | Normalisée ? | Raison |
|-----------------|--------------|--------|
| **Continues** | ✅ Oui (StandardScaler) | Centrées (moy=0), réduites (std=1) |
| **Catégorielles** (`protocolIdentifier`) | ❌ Non | Conservées comme entiers bruts pour le tokenizer BPE |
| **Binaires** (`pkt_dir_*`) | ❌ Non | Valeurs 0/1 déjà normalisées |

### Anti-leakage
- Le **StandardScaler est fit uniquement sur le training set** puis appliqué sur val/test

---

## Création des Séquences (Sliding Window)

```python
def create_sequences_with_categorical(
    X_continuous, X_categorical, X_binary,
    y, labels_str=None,
    seq_length=10, stride=10
):
```

- **Fenêtre glissante** : chaque séquence contient `seq_length=10` flux consécutifs
- **Stride** : pas de `10` (non-recouvrant par défaut)
- **Regroupement par appareil** : les flux sont d'abord triés par `flowStartMilliseconds` au sein de chaque classe, puis les fenêtres sont générées
- **Label** : la classe de la séquence est celle du dernier flux (`y_group[i + seq_length - 1]`)

### Ordre final des features dans chaque séquence
```
[continuous (28) | categorical (1) | binary (8)] × seq_length=10
```

---

## Flux Complet du Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  ETAPE 1 : Filtrage SDN                                 │
│  - Chargement JSON (streaming ligne par ligne)          │
│  - Labeling par MAC bidirectionnel                       │
│  - Suppression IP/ports/timestamps                      │
│  - Décodage directions paquets (8 bits)                │
│  - Split temporel 70/10/20 par appareil               │
├─────────────────────────────────────────────────────────┤
│  ETAPE 2 : Équilibrage + Filtrage bruit (train only)    │
│  - Isolation Forest (anomalies)                         │
│  - Local Outlier Factor (bruit résiduel)               │
├─────────────────────────────────────────────────────────┤
│  ETAPE 3 : Sélection hybride des features              │
│  - XGBoost (40%) → importance prédictive               │
│  - Chi² (30%) → dépendance statistique                  │
│  - Information Mutuelle (30%) → dépendance non-linéaire│
│  - Score combiné + méthode du coude                    │
├─────────────────────────────────────────────────────────┤
│  ETAPE 4 : Normalisation                                │
│  - StandardScaler sur features continues (fit sur train)│
│  - Features catégorielles et binaires non modifiées    │
├─────────────────────────────────────────────────────────┤
│  Création des séquences                                │
│  - Sliding window (length=10, stride=10)               │
│  - Label = classe du dernier flux                       │
└─────────────────────────────────────────────────────────┘
```

---

## Résumé des Paramètres Clés

| Paramètre | Valeur par défaut | Description |
|-----------|------------------|-------------|
| `seq_length` | 10 | Nombre de flux par séquence |
| `stride` | 10 | Pas entre deux séquences |
| `contamination` | 0.05 | Proportion d'anomalies attendue |
| `n_estimators` (XGBoost) | 100 | Nombre d'arbres |
| `max_depth` (XGBoost) | 6 | Profondeur maximale des arbres |
| `n_neighbors` (LOF) | 20 | Nombre de voisins pour LOF |
| `RANDOM_STATE` | 42 | Graine aléatoire |

---

## Classes Cibles (17 appareils IoT)

1. Qrio Hub
2. Philips Hue Light Bulb
3. Planex Pan–Tilt Camera 1
4. JVC Kenwood Camera
5. iRobot roomba
6. Google Home
7. Apple HomePod
8. Sony Bravia TV
9. Wansview Camera
10. Qwatch Camera
11. Fredi Camera
12. Planex Outdoor Camera
13. Powerlec Wi-Fi Plug
14. LINE Clova Speaker
15. Sony Smart Speaker
16. Amazon Echo
17. Amazon Echo Show
