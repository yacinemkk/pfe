# Chapitre 6 — Curriculum d'Entraînement Antagoniste en 4 Phases

## 6.1 Philosophie du Curriculum Learning Adversarial

L'entraînement antagoniste naïf — où on injecte des exemples adversariaux dès le début de l'entraînement — échoue fréquemment dans la pratique pour deux raisons principales :

1. **Effondrement de l'accuracy propre (*clean accuracy collapse*)** : le modèle apprend à être robuste mais oublie les exemples normaux, ce qui est inacceptable en production où la majorité du trafic est légitime.
2. **Instabilité de l'entraînement** : les gradients générés par les exemples adversariaux forts peuvent interférer destructivement avec l'apprentissage des patterns propres.

La solution adoptée dans ce projet est un **curriculum d'entraînement progressif en 4 phases** (A→B→C→D), où la difficulté des exemples adversariaux augmente graduellement. Chaque phase introduit de nouveaux mécanismes de défense adaptés à son niveau de robustesse cible.

---

## 6.2 Phase A — Fondation Propre (Epochs 1–15)

### 6.2.1 Objectif

Établir une base solide d'apprentissage sur des données **entièrement propres** (aucun exemple adversarial). Le modèle apprend à reconnaître les patterns comportementaux normaux de chaque appareil IoT.

### 6.2.2 Configuration

```python
# Paramètres Phase A
PHASE_A_EPOCHS = 15
PHASE_A_MIX_RATIO = 0.0   # 0% d'exemples adversariaux
k_max = 0                  # Aucune attaque
p_drop = 0.0               # Pas de Feature Dropout
sigma_noise = 0.0          # Pas de bruit gaussien
afd_lambda = 0.0           # AFDLoss désactivée
label_smoothing = 0.05     # Légère régularisation (5%)
```

### 6.2.3 Fonction de Perte

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
loss = criterion(model(X_batch), y_batch)
```

La **Label Smoothing** (0.05) remplace les cibles one-hot dures par des distributions douces :
```
y_smooth[c] = y_hard[c] * (1 - ε) + ε/K
```
où `ε=0.05` et `K` = nombre de classes. Elle empêche le modèle de devenir trop confiant (*over-confident*) sur les données d'entraînement.

### 6.2.4 Sélection du Meilleur Checkpoint

En Phase A, la sélection du meilleur modèle se fait uniquement sur l'**accuracy de validation propre** :
```python
if phase == 'A':
    is_better = val_clean_acc > best_val_acc
```

### 6.2.5 Analyse de Sensibilité Post-Phase A

Après la Phase A, une **analyse de sensibilité exhaustive** est réalisée sur le modèle obtenu. C'est un point crítico du pipeline : le GreedyAttackSimulator est construit à partir de ces résultats et utilisera les features les plus vulnérables du modèle Phase A comme cibles pour les phases suivantes.

**Pourquoi après la Phase A et pas avant ?** Les features vulnerables dépendent du modèle appris. On ne peut pas savoir à l'avance quelles features le modèle utilisera pour discriminer. L'analyse post-Phase A révèle les "talons d'Achille" spécifiques à chaque architecture.

```python
# Analyse de 5000 exemples de validation
sa = SensitivityAnalysis(X_val[:5000], y_val[:5000], ...)
results = sa.analyze(model, ...)   # Pour chaque (feature, stratégie) : mesure du drop
# Sauvegarde dans sensitivity_results.csv
# Construction du GreedyAttackSimulator avec les paires (feat, strategy) triées par drop↓
simulator = GreedyAttackSimulator(sensitivity_results, feature_stats)
```

---

## 6.3 Phase B — Introduction à la Robustesse (Epochs 16–25)

### 6.3.1 Objectif

Introduire **progressivement** les exemples adversariaux (30% du batch), en utilisant uniquement des attaques légères (k_max=2 features perturbées). Le modèle commence à développer une résistance aux perturbations simples sans trop dégrader les performances propres.

### 6.3.2 Configuration

```python
# Paramètres Phase B (greedy_new.ipynb)
PHASE_B_EPOCHS = 25          # Epochs 16 à 25 (10 epochs)
PHASE_B_MIX_RATIO = 0.30     # 30% d'exemples adversariaux par batch
PHASE_B_K_MAX = 2            # Au plus 2 features perturbées par exemple
p_drop = 0.1                 # Feature Dropout 10%
sigma_noise = 0.01           # Bruit gaussien σ=0.01
afd_lambda = 0.5             # AFDLoss activée (λ=0.5)
label_smoothing = 0.08       # Légèrement augmenté (8%)
```

### 6.3.3 Mécanisme de la Phase B — Batch Mixte

À chaque batch :
```python
# 1. Générer le batch mixte (30% adversarial, 70% propre)
X_mixed, flags = simulator.generate_training_batch(
    X_np, k_max=2, mix_ratio=0.30
)
# 2. Ajouter Feature Dropout (p=0.1)
mask = (torch.rand(X_input.shape[0], 1, X_input.shape[2]) > 0.1).float()
X_input = X_input * mask / (1.0 - 0.1)   # Mis à l'échelle pour compenser
# 3. Ajouter bruit gaussien (σ=0.01)
X_input = X_input + torch.randn_like(X_input) * 0.01
# 4. Loss composée : CE + AFD
logits_clean = model(X_clean)
logits_adv = model(X_adv_mixed)
loss_ce = criterion(logits_adv, y)
loss_afd = afd_criterion(logits_clean, logits_adv, y)
loss = loss_ce + 0.5 * loss_afd
```

### 6.3.4 Contre-Mesure 1 : Feature Dropout (`p_drop=0.1`)

**Principe :** à chaque forward pass, 10% des features d'entrée sont masquées (mises à zéro) de manière aléatoire. Le masque change à chaque batch.

```python
if p_drop > 0:
    mask = (torch.rand(batch_size, 1, n_features) > p_drop).float()
    X_input = X_input * mask / (1.0 - p_drop)  # Inverted dropout
```

**Justification :** Le Feature Dropout empêche le modèle de trop se focaliser sur un **sous-ensemble restreint de features**. Un modèle qui dépend excessivement de 2-3 features pour classifier est vulnérable aux attaques qui ne ciblent que ces features. En les masquant aléatoirement, le modèle est forcé à distribuer sa dépendance sur l'ensemble des features disponibles. C'est une forme de régularisation spécialisée pour la robustesse adversariale.

**Complémentarité avec les attaques greedy :** les attaques greedy ciblent les features les *plus* vulnérables. Le Feature Dropout force le modèle à ne *pas* s'appuyer excessivement sur ces features, réduisant leur vulnérabilité au fil des epochs.

### 6.3.5 Contre-Mesure 2 : Bruit Gaussien (`sigma_noise=0.01`)

**Principe :** ajouter un bruit gaussien de faible amplitude à toutes les features d'entrée.

```python
X_input = X_input + torch.randn_like(X_input) * sigma_noise
```

**Justification :** le bruit gaussien sert de **régularisateur de robustesse empirique**. En exposant le modèle à de légères perturbations aléatoires pendant l'entraînement, il apprend à construire des représentations internes plus stables qui ne varient pas dramatiquement face à de petites modifications des features d'entrée. C'est une forme de *data augmentation* adverse.

**Différence avec les attaques greedy :** les attaques greedy sont dirigées (elles ciblent les features les plus vulnérables) tandis que le bruit gaussien est isotropique. Les deux sont complémentaires : le bruit améliore la robustesse générale, les attaques ciblées développent la résistance spécifique.

### 6.3.6 Contre-Mesure 3 : AFDLoss — Adversarial Feature Defense

**Implémentation :** `src/adversarial/robust_losses.py` — classe `AFDLoss`

L'`AFDLoss` est une **fonction de perte custom** qui force le modèle à maintenir des représentations distinctes entre les exemples propres et adversariaux, tout en maintenant des centres de classes bien séparés dans l'espace des représentations.

#### 6.3.6.1 Principe Mathématique

L'`AFDLoss` combine deux termes :

**Terme Intra-classe (perte de cohérence) :**
```
L_intra = ||f(x_clean) - center_c||² + ||f(x_adv) - center_c||²
```
Minimiser ce terme force les représentations des exemples propres ET adversariaux à se rapprocher du centre de leur classe. Le modèle apprend ainsi à produire des représentations similaires pour les exemples variés du même appareil.

**Terme Inter-classe (perte de séparabilité) :**
```
L_inter = Σ_{i≠j} max(0, 1 - ||center_i - center_j||₂)
```
Ce terme pénalise les paires de centres de classes qui sont trop proches dans l'espace des représentations, forçant les classes différentes à être bien séparées.

**Loss Totale :**
```
L_AFD = λ_intra × (L_intra_clean + L_intra_adv) - λ_inter × L_inter
```

#### 6.3.6.2 Mise à Jour des Centres par Momentum

Les centres des classes sont maintenus comme un **buffer entraînable** mis à jour par momentum exponential :
```python
@torch.no_grad()
def update_centers(self, features, labels):
    for c in range(self.num_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            feat_mean = features[mask].mean(0)
            if (self.centers[c] == 0).all():
                # Initialisation au premier batch contenant cette classe
                self.centers[c] = feat_mean
            else:
                # Mise à jour par momentum (0.9 = inertie forte)
                self.centers[c] = 0.9 * self.centers[c] + 0.1 * feat_mean
```

**Momentum 0.9 :** ce fort momentum assure que les centres sont stables et ne sautent pas brusquement d'un batch à l'autre, garantissant une estimation fiable de la distribution de chaque classe.

#### 6.3.6.3 Normalisation L2

```python
features = F.normalize(features, p=2, dim=1)
features_adv = F.normalize(features_adv, p=2, dim=1)
```

La normalisation L2 des features avant le calcul AFD assure que la perte est invariante à l'échelle des activations, ce qui rend l'entraînement plus stable.

**Justification globale de l'AFDLoss :** un modèle sans AFDLoss peut apprendre à classifier correctement les exemples adversariaux mais en modifiant ses représentations internes de telle manière que les centres de classes se rapprochent — rendant le modèle fragile aux attaques futures. L'AFDLoss force une structure géométrique robuste dans l'espace des représentations.

### 6.3.7 Sélection du Meilleur Checkpoint (Phases B, C, D)

À partir de la Phase B, la sélection du meilleur checkpoint utilise un **score pondéré** qui favorise la robustesse adversariale :

```python
# Score = 0.4 * accuracy_propre + 0.6 * accuracy_adversariale
selection_score = 0.4 * val_clean_acc + 0.6 * val_adv_acc
is_better = selection_score > best_combined   # Critère de sélection
```

**Justification des poids (0.4/0.6) :** dans un scénario de sécurité réseau, la robustesse adversariale est plus critique que la performance sur données normales. On accepte une légère dégradation (≤5%) de l'accuracy propre en échange d'une robustesse bien supérieure. Le poids 0.6 pour l'accuracy adversariale reflète cette priorité sans aller jusqu'à ignorer complètement l'accuracy propre.

---

## 6.4 Phase C — Robustesse Forte (Epochs 26–35)

### 6.4.1 Objectif

Entraîner le modèle contre des attaques significativement plus fortes : 70% du batch adversarial avec jusqu'à 4 features perturbées simultanément. L'AFDLoss est renforcée (λ=1.0) et le Feature Dropout augmenté (p=0.2).

### 6.4.2 Configuration

```python
# Paramètres Phase C (greedy_new.ipynb)
PHASE_C_EPOCHS = 35          # Epochs 26 à 35 (10 epochs)
PHASE_C_MIX_RATIO = 0.70     # 70% d'exemples adversariaux
PHASE_C_K_MAX = 4            # Jusqu'à 4 features perturbées
p_drop = 0.2                 # Feature Dropout augmenté à 20%
sigma_noise = 0.01           # Bruit gaussien maintenu
afd_lambda = 1.0             # AFDLoss renforcée (λ=1.0)
label_smoothing = 0.10       # Maximum recommandé (10%)
```

### 6.4.3 Rôle de l'InputDefenseLayer (disponible dans `robust_losses.py`)

L'`InputDefenseLayer` est un module de défense **préprocesseur** qui peut être ajouté en entrée du modèle :

```python
class InputDefenseLayer(nn.Module):
    def __init__(self, clip_min=-3.5, clip_max=3.5, smooth_alpha=0.25):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.alpha = smooth_alpha
    
    def forward(self, x):
        # 1. Clipping des valeurs aberrantes (kill Padding_x10)
        x = torch.clamp(x, self.clip_min, self.clip_max)
        
        # 2. Lissage temporel exponentiel (kill Zero/Mimic)
        smoothed = x.clone()
        for t in range(1, x.shape[1]):
            smoothed[:, t] = 0.25 * x[:, t] + 0.75 * smoothed[:, t - 1]
        return smoothed
```

**Pourquoi deux défenses dans l'InputDefenseLayer ?**

**Clipping [-3.5, 3.5] :** la stratégie Padding_x10 multiplie les features par 10, produisant des valeurs pouvant atteindre 10 dans l'espace normalisé [0,1] → espace latent. Le clipping à [-3.5, 3.5] **neutralise complètement** cette stratégie avant même que le signal n'entre dans le modèle. Ce clipping maintient ≥99.9% des valeurs normales (qui sont bien dans [-3.5, 3.5] après standardisation).

**Lissage temporel (EMA, α=0.25) :** les stratégies Zero et Mimic_Mean créent des **discontinuités temporelles** — la feature perturbée prend une valeur constante sur toute la séquence, donnant une série temporelle de variance nulle. Le lissage exponentiel (avec α=0.25 = faible poids sur la valeur courante) atténue ces sauts brusques en les "diffusant" dans le temps, rendant la perturbation moins tranchante.

---

## 6.5 Phase D — Consolidation (Epochs 36–50)

### 6.5.1 Objectif

Consolider la robustesse acquise aux Phases B et C en poussant le ratio d'exemples adversariaux à 80%, tout en maintenant l'AFDLoss comme **ancre géométrique** des représentations. Désactiver l'AFDLoss en Phase D a conduit lors des expérimentations à des résultats catastrophiques.

### 6.5.2 Configuration

```python
# Paramètres Phase D (greedy_new.ipynb)
PHASE_D_EPOCHS = 50          # Epochs 36 à 50 (15 epochs) — 50 epochs total
PHASE_D_MIX_RATIO = 0.80     # 80% adv avec ancrage afd_lambda
PHASE_D_K_MAX = 4            # Maintenu à k=4 (idem Phase C)
p_drop = 0.2                 # Feature Dropout maintenu (20%)
sigma_noise = 0.01           # Bruit gaussien maintenu
afd_lambda = 0.5             # AFDLoss MAINTENUE ACTIVE (désactiver → catastrophique)
label_smoothing = 0.10       # Maintenu (10%)
```

### 6.5.3 Pourquoi Mix_Ratio = 0.80 et Pas 1.0 ?

**Problème identifié :** lors des expérimentations initiales avec `mix_ratio=1.0` (100% adversarial) et `k_max=5`, l'accuracy propre s'effondrait en dessous de 60%, rendant le système inutilisable.

**Solution :** maintenir 20% d'exemples propres dans chaque batch garantit que le modèle ne "oublie" pas les flux normaux. Ce 20% agit comme une **ancre de stabilité** pour l'accuracy clean.

```python
PHASE_D_MIX_RATIO = 0.80     # 80% adv avec ancrage afd_lambda
PHASE_D_K_MAX = 4            # k=4 comme Phase C — k=5 était contre-productif
```

### 6.5.4 Pourquoi l'AFDLoss reste-t-elle Active en Phase D ?

**Leçon critique des expérimentations :** désactiver l'AFDLoss en Phase D a produit des résultats **catastrophiques**. Sans sa contrainte géométrique, le modèle entraîné à 80% adversarial perdait la séparabilité inter-classes dans son espace de représentations. L'AFDLoss est donc maintenue à λ=0.5 en Phase D pour :
1. **Ancrer la structure géométrique** bâtie aux Phases B et C malgré la pression adversariale intense.
2. **Prévenir la convergence des centres de classes** qui causerait des confusions massives entre appareils IoT.

---

## 6.6 Phase E — Entraînement du Discriminateur (Séparé)

Après les 4 phases du classifieur, un **Discriminateur BiLSTM** est entraîné séparément pour détecter les attaques :

```python
disc = Discriminator(input_size=input_size, seq_length=10, hidden_size=64)
disc, disc_acc = train_discriminator(
    discriminator=disc,
    X_train=X_train,
    simulator=simulator,      # Utilise le même simulator Phase D
    device=device,
    epochs=25,                # 25 epochs d'entraînement binaire
    batch_size=batch_size,
    lr=1e-3
)
```

Le Discriminateur est entraîné avec une perte BCEWithLogitsLoss sur des batches équilibrés : 50% de flux propres (label=0) et 50% de flux adversariaux (label=1, générés avec k ∈ {1,2,3}).

---

## 6.7 Récapitulatif du Curriculum (greedy_new.ipynb)

| Phase | Epochs | % Adversarial | k_max | Feature Dropout | AFDLoss | Contre-mesures actives |
|-------|--------|---------------|-------|-----------------|---------|------------------------|
| **A** | 1–15 | 0% | 0 | Non | Non | Label Smoothing |
| **B** | 16–25 | 30% | 2 | 10% | λ=0.5 | LD, FD, Bruit, AFD |
| **C** | 26–35 | 70% | 4 | 20% | λ=1.0 | LD, FD, Bruit, AFD |
| **D** | 36–50 | 80% | 4 | 20% | **λ=0.5** | LD, FD, Bruit, AFD |
| **E** | 25 ep. | — | — | — | — | Discriminateur BiLSTM |

**Légende :** LD = Label Smoothing, FD = Feature Dropout
**Total :** 50 epochs d'entraînement adversarial (A+B+C+D)

La progression graduelle (0% → 30% → 70% → 80% adversarial, k_max: 0 → 2 → 4 → 4) assure une montée en difficulté contrôlée. L'AFDLoss reste active sur toutes les phases adversariales (B, C, D) car la désactiver en Phase D produit un effondrement de la séparabilité des classes.
