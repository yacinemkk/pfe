# Chapitre 5 — Attaques Adversariales sur les Flux IoT

## 5.1 Introduction aux Attaques Adversariales dans le Domaine Réseau

Les attaques adversariales dans le domaine de la classification de trafic réseau présentent des caractéristiques distinctes par rapport aux attaques classiques sur des images. Contrairement aux perturbations pixel-level sur des images, les attaques sur les flux réseau doivent respecter des **contraintes sémantiques strictes** : une perturbation n'est valide que si elle correspond à un comportement réseau physiquement réalisable par un équipement réel.

### 5.1.1 Modèle de Menace

L'adversaire modélisé dans ce projet est :
- **Connaissance partielle :** l'attaquant sait que son trafic sera analysé par un système d'identification IoT mais ne connaît pas les paramètres exacts du modèle (attaque en boîte noire).
- **Capacité d'action limitée :** l'attaquant peut modifier les **statistiques agrégées** de ses flux (délais artificiels, amplification de certaines features) mais ne peut pas modifier les features imposées par le protocole réseau.
- **Objectif :** tromper le classificateur pour qu'un appareil malveillant soit identifié comme un appareil légitime connu (=usurpation d'identité, *device spoofing*).

### 5.1.2 Features Non Modifiables (Contraintes Sémantiques)

Les features suivantes sont **protégées** et ne peuvent pas être perturbées par un attaquant réaliste :

**Dataset CSV :** `ipProto` — le protocole réseau (TCP=6, UDP=17, ICMP=1) est imposé par la connexion réseau sous-jacente. Un attaquant ne peut pas changer le protocole sans changer la nature de sa communication.

**Dataset JSON :** 
- `protocolIdentifier` — même raison que `ipProto`
- `pkt_dir_0` à `pkt_dir_7` — 8 bits binaires encodant la direction des 8 premiers paquets (0=sortant, 1=entrant). Ces bits sont imposés par la structure du protocole TCP (SYN envoi, SYN-ACK réception, etc.).

---

## 5.2 Les Quatre Stratégies d'Attaque Implémentées

Le projet implémente 4 stratégies d'attaque au niveau des features, chacune simulant un comportement réseau différent :

### 5.2.1 Stratégie Zero — Suppression du Trafic

```python
def apply_strategy(self, X, feat_idx, strategy):
    if strategy == 'Zero':
        X[:, :, feat_idx] = 0.0   # Mise à zéro de la feature dans tous les flux
```

**Principe :** annuler complètement une statistique réseau (ex. : `avgInterarrivalTime = 0`).

**Scénario réel :** un appareil malveillant qui supprime artificiellement certains patterns caractéristiques (par exemple, en limitant ses inter-arrivées à zéro lors d'un burst simulé).

**Impact sur le modèle :** si le modèle s'appuie fortement sur une feature particulière pour différencier les classes, sa mise à zéro peut dégrader massivement les performances. Cette stratégie est souvent la plus déstabilisante pour les features statistiques très discriminantes.

### 5.2.2 Stratégie Mimic_Mean — Imitation de la Moyenne

```python
elif strategy == 'Mimic_Mean':
    X[:, :, feat_idx] = self.stats[feat_idx]['mean']  # Remplacer par la moyenne globale
```

**Principe :** remplacer la valeur d'une feature par sa **moyenne calculée sur l'ensemble d'entraînement** pour cette feature (calculée dans `GreedyAttackSimulator.compute_feature_stats`).

**Scénario réel :** un appareil malveillant qui "normalise" son comportement réseau pour ressembler à un appareil générique moyen, rendant son identification difficile.

**Impact :** cette stratégie amène une feature vers la valeur "attendue" par le modèle pour une distribution générique, perturbant les modèles qui utilisent des valeurs extrêmes comme signaux discriminants.

### 5.2.3 Stratégie Mimic_95th — Imitation du 95ème Percentile

```python
elif strategy == 'Mimic_95th':
    X[:, :, feat_idx] = self.stats[feat_idx]['p95']  # Remplacer par le 95ème percentile
```

**Principe :** remplacer la valeur par le **95ème percentile** de la distribution de cette feature dans les données d'entraînement.

**Scénario réel :** un attaquant qui non seulement normalise son trafic mais le "sature" vers les valeurs extrêmes de la distribution normale — utile pour imiter des appareils qui génèrent beaucoup de trafic (ex. : caméra de surveillance en streaming).

**Impact :** complémentaire à Mimic_Mean, cette stratégie est particulièrement efficace contre les modèles basés sur des seuils implicites.

### 5.2.4 Stratégie Padding_x10 — Amplification

```python
elif strategy == 'Padding_x10':
    X[:, :, feat_idx] = np.clip(X[:, :, feat_idx] * 10.0, -5.0, 5.0)
```

**Principe :** amplifier la valeur d'une feature par un facteur 10, clippée dans l'intervalle [-5, 5] pour rester dans des limites raisonnables.

**Scénario réel :** un appareil malveillant qui génère des flux réseau avec des statistiques anormalement élevées — par exemple, envoyer 10× plus de paquets que la normale pour un type d'appareil donné. Dans la réalité, cela peut être réalisé en fragmentant intentionnellement les données.

**Impact :** cette stratégie est particulièrement déstabilisante pour les modèles qui ont appris des plages de valeurs normales. Elle est contrecarrée spécifiquement par l'`InputDefenseLayer` (clipping à [-3.5, 3.5]).

---

## 5.3 Le GreedyAttackSimulator — Échantillonnage de Failles (Vulnerability Sampling)

### 5.3.1 Principe Général — La Loterie des Failles Critiques

Le `GreedyAttackSimulator` est le composant central du système d'attaque. Son architecture repose sur un principe original appelé **Vulnerability Sampling** (ou échantillonnage de faiblesses), qui crée un adversaire intelligent en 3 étapes :

---

**Étape 1 — Structure Dictionnaire des Failles**

Plutôt qu'une liste linéaire de features classées, le simulateur construit un **dictionnaire des features critiques**. Pour chaque feature, il répertorie toutes ses **stratégies dangereuses** (celles qui provoquent une chute de précision significative lors de l'analyse de sensibilité) :

```
feature_pool = {
    feat_i : ["Zero", "Mimic_95th"],       # 2 stratégies dangereuses pour i
    feat_j : ["Padding_x10"],              # 1 stratégie dangereuse pour j
    feat_k : ["Mimic_Mean", "Zero"],       # 2 stratégies dangereuses pour k
    ...
}
feature_weights = {feat_i: drop_i + ε, feat_j: drop_j + ε, ...}   # Poids ∝ vulnérabilité
```

**Étape 2 — Vecteurs d'Attaque Uniques (k features distinctes)**

Quand on demande une attaque `k=3`, le simulateur extrait **3 features distinctes** en utilisant un **sampling sans remise** pondéré par les `drop_scores`. Cela garantit que :
- Aucune feature n'est écrasée deux fois (évite the overwriting)
- Les features les plus vulnérables sont choisies plus souvent (probabilité ∝ drop_score)
- Les features moins vulnérables sont quand même explorées (dâce à `ε=0.05`)

**Étape 3 — Aléatoire Stratégique**

Pour chaque feature choisie, le simulateur tire **au hasard** l'une de ses stratégies dangereuses connues. Si la Feature A a les stratégies {Zero, Padding_x10}, l'une est choisie uniformément. Ce double aléatoire (qui perturber ? → comment perturber ?) crée une **adversarité imprévisible** : le modèle ne peut jamais mémoriser un pattern d'attaque fixe.

---

```python
class GreedyAttackSimulator:
    def __init__(self, sensitivity_results, feature_stats):
        self.stats = feature_stats
        self.feature_pool = {}         # Étape 1 : dictionnaire {feat: [stratégies dangereuses]}
        self.feature_weights = {}      # Poids de vulnérabilité par feature
        epsilon = 0.05                 # Probabilité exploratoire minimale (ε)

        for fi, st, drop in sensitivity_results:
            if fi not in self.feature_pool:
                self.feature_pool[fi] = []
                self.feature_weights[fi] = max(0.0, drop) + epsilon
            self.feature_pool[fi].append(st)   # Ajout de la stratégie dangereuse

        self.available_features = list(self.feature_pool.keys())
        # Distribution de sampling (probabilités proportionnelles aux drop_scores)
        weights = np.array([self.feature_weights[f] for f in self.available_features])
        self.sampling_probs = weights / weights.sum()  # Distribution P(choisir feat_i)
```

### 5.3.2 Analyse de Sensibilité et Mise à Jour de la Distribution

L'analyse de sensibilité est réalisée sur un sous-ensemble de 5000 exemples de validation. Pour chaque feature `f_i` et chaque stratégie `s_j ∈ {Zero, Mimic_Mean, Mimic_95th, Padding_x10}` :

1. Créer `X_perturbed` en appliquant `s_j` à la feature `f_i` de `X_val`
2. Évaluer le modèle sur `X_perturbed`
3. Calculer le `drop_score = accuracy_clean - accuracy_perturbed`

Le résultat est un fichier `sensitivity_results.csv` listant tous les couples (feature, stratégie) triés par `drop_score` décroissant.

**Distribution de sampling (résultats typiques) :**

```
feature               | strategy      | drop  | P(sampling)
--------------------------------------------------------------
averageInterarrivalTime | Mimic_95th  | 32.4% | 0.196
packetTotalCount       | Zero         | 28.1% | 0.170
byteCount              | Padding_x10  | 25.7% | 0.155
flowDuration           | Mimic_Mean   | 19.3% | 0.117
reverseOctetTotalCount | Zero         | 16.8% | 0.102
...autres features...  |              |  ...  | 0.260
```

La colonne `P(sampling)` montre la **distribution de probabilité** avec laquelle chaque feature sera choisie lors d'une attaque. Cette distribution est recalculée dynamiquement : la sensibilité est **re-analysée après chaque phase d'entraînement** (sur la version courante du modèle, pas seulement Phase A), ce qui permet de mettre à jour la distribution `P` au fur et à mesure que le modèle devient plus robuste contre certaines features — le simulateur s'adapte dynamiquement aux nouvelles failles exposées par l'entraînement.

```python
# Recalcul de la sensibilité après chaque phase
sens_csv_path = f'{save_dir}/sensitivity_phase_{phase}.csv'
if not os.path.exists(sens_csv_path):
    run_sensitivity_analysis(model, X_val, y_val, feature_names, ..., sens_csv_path)
sensitivity = load_sensitivity_results(sens_csv_path, feature_names)
feature_stats = GreedyAttackSimulator.compute_feature_stats(X_train)
simulator = GreedyAttackSimulator(sensitivity, feature_stats)  # Nouvelle distribution P
```

### 5.3.3 Génération d'une Attaque via Vulnerability Sampling (Étapes 2 et 3)

```python
def generate_greedy(self, X, k):
    X_adv = X.copy()
    k_actual = min(k, len(self.available_features))

    # Étape 2 : sampling SANS remise pondéré (évite d'attaquer 2x la même feature)
    chosen_features = np.random.choice(
        self.available_features, size=k_actual,
        replace=False, p=self.sampling_probs   # Distribution P ∝ drop_scores
    )
    # Étape 3 : stratégie aléatoire parmi les stratégies dangereuses de chaque feature
    for feat_idx in chosen_features:
        strategy = np.random.choice(self.feature_pool[feat_idx])
        X_adv = self.apply_strategy(X_adv, feat_idx, strategy)
    return X_adv
```

**Propriétés garanties par ce mécanisme :**
- ✅ **Pas d'écrasement** : `replace=False` garantit k features strictement distinctes
- ✅ **Priorité aux failles critiques** : probabilité de choisir `feat_i` ∝ `drop_i + ε`
- ✅ **Exploration des failles secondaires** : le terme `ε=0.05` donne une probabilité minimale non nulle à toute feature, même peu vulnérable
- ✅ **Double imprévisibilité** : qui attaquer (sampling pondéré) ET comment (stratégie aléatoire) → le modèle ne peut jamais mémoriser un vecteur d'attaque fixe

#### 5.3.3.1 — Le Cas `drop ≈ 0` et la Solution Epsilon

**Problème identifié :** après plusieurs phases d'entraînement, le modèle peut apprendre à parfaitement résister à certaines features. Son `drop_score` sur ces features tombe proche de zéro :

```python
# Situation typique après Phase C — Feature X presque "guérie"
feature_weights = {
    'Feature X': 0.02,   # Le modèle a appris à résister → drop quasi-nul
    'Feature Y': 0.45,   # Encore vulnérable
    'Feature Z': 0.38,   # Encore vulnérable
}
# Sampling_probs sans epsilon :
# P('Feature X') = 0.02 / 0.85 = 2.3%  → presque jamais testée en Phase D
```

**Danger :** si une **nouvelle vulnérabilité émerge** sur la Feature X (par exemple à l'epoch 45 en Phase D, le modèle se fragilise à nouveau sur cette feature), le simulateur avec `drop≈0` ne la détecterait pratiquement jamais — créant un angle mort dans la défense.

**Correctif implémenté — `epsilon = 0.05` :**

```python
# Dans greedy_new.ipynb — __init__ du GreedyAttackSimulator
epsilon = 0.05  # Exploratory minimum probability

for fi, st, drop in sensitivity_results:
    if fi not in self.feature_pool:
        self.feature_pool[fi] = []
        # weight = drop_score + ε  (toujours > ε, jamais 0)
        self.feature_weights[fi] = max(0.0, drop) + epsilon
    self.feature_pool[fi].append(st)
```

**Effet de l'epsilon :**

```python
# Avec epsilon = 0.05 :
feature_weights = {
    'Feature X': 0.02 + 0.05 = 0.07,  # Probabilité minimale garantie
    'Feature Y': 0.45 + 0.05 = 0.50,
    'Feature Z': 0.38 + 0.05 = 0.43,
}
# P('Feature X') = 0.07 / 1.00 = 7.0%  → encore testée régulièrement
```

**Tableau de validation du design :**

| Composant | Statut | Code |
|-----------|--------|------|
| Recalibration après chaque phase | ✅ Dynamique | `GreedyAttackSimulator(sensitivity, feature_stats)` |
| `max(drops)` par feature | ✅ Vrai pire cas | `max(0.0, drop)` |
| Normalisation en probabilités | ✅ Mathématiquement propre | `weights / weights.sum()` |
| Epsilon pour probabilité minimale | ✅ **Présent** (`ε=0.05`) | `max(0.0, drop) + epsilon` |

Le design est **complet et défendable** : aucune feature n'est jamais complètement abandonnée, même si le modèle y résiste parfaitement lors de l'analyse précédente.


### 5.3.4 Génération d'un Batch d'Entraînement

```python
def generate_training_batch(self, X, k_max=4, mix_ratio=0.5):
    """
    Génère un batch mixte pour l'entraînement antagoniste.
    
    mix_ratio=0.5 → 50% d'exemples adversariaux + 50% propres
    k aléatoire ∈ [1, k_max] pour chaque exemple adversarial
    → diversité des niveaux d'attaque
    """
    n = len(X)
    n_adv = int(n * mix_ratio)
    
    # Sélection aléatoire des exemples à perturber
    idx_adv = np.random.choice(n, n_adv, replace=False)
    
    X_out = X.copy()
    for i in idx_adv:
        k = np.random.randint(1, k_max + 1)   # k aléatoire ∈ [1, k_max]
        X_out[[i]] = self.generate_greedy(X[[i]], k)
    
    return X_out, flags  # flags[i]=1 si l'exemple i est adversarial
```

**Stratégie de k aléatoire :** utiliser un k fixe conduirait le modèle à apprendre à résister à une force d'attaque précise. Varier k aléatoirement force le modèle à être robuste à **différents niveaux de perturbation**, ce qui est plus réaliste.

---

## 5.4 Le Discriminateur BiLSTM — Détection des Attaques

### 5.4.1 Architecture

Le `Discriminateur` est un modèle séparé du classifieur principal, entraîné spécifiquement pour **détecter si une séquence de flux a été perturbée**.

```python
class Discriminator(nn.Module):
    def __init__(self, input_size, seq_length, hidden_size=64):
        super().__init__()
        # BiLSTM pour capturer les incohérences temporelles dans la séquence
        self.lstm = nn.LSTM(input_size, hidden_size, 
                           num_layers=1, 
                           batch_first=True, 
                           bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),  # 128 → 32
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)   # Sortie scalaire : score d'attaque
        )
    
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h_cat = torch.cat([h[0], h[1]], dim=1)  # Concaténation forward+backward
        return self.head(h_cat).squeeze(1)      # Logit de détection
```

**Pourquoi un BiLSTM ?** Les perturbations greedy créent des **incohérences temporelles** dans la séquence : si la feature `averageInterarrivalTime` est mise à sa valeur moyenne globale (Mimic_Mean), elle devient constante dans le temps, alors qu'un comportement naturel montre une variation temporelle. Le BiLSTM est idéalement positionné pour détecter ces brisures de cohérence temporelle.

### 5.4.2 Entraînement du Discriminateur

```python
def train_discriminator(discriminator, X_train, simulator, ...):
    criterion = nn.BCEWithLogitsLoss()   # Classification binaire
    
    # Génération des labels binaires : 0=propre, 1=adversarial
    # Moitié des exemples perturbés avec k ∈ {1, 2, 3}
    k_values = np.random.randint(1, 4, size=n_half)
    X_adv_list = [simulator.generate_greedy(X[[i]], k=k) for i, k in zip(idx_adv, k_values)]
    
    labels_bin = [0.0]*n_half + [1.0]*n_half   # 0=clean, 1=adversarial
```

Le Discriminateur est entraîné pendant 25 epochs avec un objectif de détection binaire. La **meilleure accuracy** obtenue est sauvegardée.

### 5.4.3 Le Routeur IoT (IoTRouter)

```python
class IoTRouter(nn.Module):
    def predict(self, X):
        # Détection : score ≥ threshold → attaque détectée
        attack_scores = self.discriminator.predict_proba(X)
        is_attacked = (attack_scores >= self.threshold)
        
        # Prédiction du modèle normal (Phase A) pour les flux propres
        logits_normal = self.normal(X)
        
        # Prédiction du modèle robuste (Phase D) pour les flux adversariaux
        logits_adv = self.adversarial(X)
        
        # Routage dynamique
        predictions = torch.where(is_attacked, 
                                  logits_adv.argmax(1), 
                                  logits_normal.argmax(1))
        return predictions, is_attacked, attack_scores
```

**Calibration automatique du seuil :** le seuil de détection est calibré automatiquement pour atteindre un **recall d'attaque de 95%** (95% des attaques détectées) :

```python
router.calibrate_threshold(X_val_clean, X_val_adv, target_recall=0.95)
```

---



---

## 5.6 Crash Test — Démonstration de la Vulnérabilité

Le Crash Test est le protocole de validation de la vulnérabilité des modèles. Il consiste à évaluer un modèle entraîné normalement (Phuse A) face aux attaques greedy avec k = 1, 2, 3, 4 features perturbées.

```python
def crash_test_greedy(model, X_val, y_val, simulator, k_values=[1,2,3,4]):
    # Évaluation propre
    clean_acc = evaluate_clean(model, X_val)
    
    # Évaluation adversariale pour chaque k
    for k in k_values:
        X_adv = simulator.generate_greedy(X_val, k=k)
        adv_acc = evaluate(model, X_adv, y_val)
        RR = adv_acc / clean_acc   # Robustness Ratio
        print(f"k={k}: CleanAcc={clean_acc:.2%} | AdvAcc={adv_acc:.2%} | RR={RR:.3f}")
```

**Résultats typiques pour le modèle CNN-BiLSTM-Transformer (Phase A) :**
| k features perturbées | Accuracy propre | Accuracy adversariale | Taux de Robustesse |
|----------------------|-----------------|----------------------|--------------------|
| 0 (clean) | ~92% | — | 1.000 |
| k=1 | ~92% | ~65% | 0.707 |
| k=2 | ~92% | ~43% | 0.467 |
| k=3 | ~92% | ~28% | 0.304 |
| k=4 | ~92% | ~18% | 0.196 |

Cette chute spectaculaire du Taux de Robustesse (de 1.0 à 0.196) démontre la **fragilité intrinsèque** des modèles non-robustifiés et motive l'entraînement antagoniste des Phases B à D.
