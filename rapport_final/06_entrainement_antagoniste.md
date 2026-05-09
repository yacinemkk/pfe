# Chapitre 6 — Curriculum d'Entraînement Antagoniste en 4 Phases

## 6.1 Philosophie du Curriculum Learning Adversarial

L'entraînement antagoniste naïf — où on injecte des exemples adversariaux dès le début de l'entraînement — échoue fréquemment dans la pratique pour deux raisons principales :

1. **Effondrement de l'accuracy propre (*clean accuracy collapse*)** : le modèle apprend à être robuste mais oublie les exemples normaux, ce qui est inacceptable en production où la majorité du trafic est légitime.
2. **Instabilité de l'entraînement** : les gradients générés par les exemples adversariaux forts peuvent interférer destructivement avec l'apprentissage des patterns propres.

La solution adoptée dans ce projet est un **curriculum d'entraînement progressif en 4 phases**, défini formellement dans `src/adversarial/robust_losses.py`. La difficulté des exemples adversariaux (`worst_k` et `epsilon`) augmente graduellement avec l'avancement des époques (epochs), garantissant que le modèle acquiert d'abord une bonne base avant de s'attaquer aux perturbations complexes.

---

## 6.2 Curriculum en 4 Phases (Basé sur les Epochs)

### 6.2.1 Phase 0 — Fondation Propre (Epochs 1 à 15)

**Objectif :** Établir une base solide d'apprentissage sur des données **entièrement propres** (aucun exemple adversarial). Le modèle apprend à reconnaître les patterns comportementaux normaux sans être perturbé.

**Configuration :**
- `epsilon = 0.0`
- `worst_k = 0` (Aucune feature perturbée)
- `cutmix_prob = 0.0`
- `afd_lambda = 0.5`
- `trades_beta = 0.0`

### 6.2.2 Phase 1 — Robustesse Douce (Epochs 16 à 35)

**Objectif :** Introduire les attaques progressivement. L'intensité de la perturbation (`epsilon`) augmente linéairement, et l'on attaque uniquement la feature la plus vulnérable (`worst_k = 1`).

**Configuration :**
- `epsilon` : Croissance linéaire de 0.01 à 0.05.
- `worst_k = 1`
- `cutmix_prob = 0.2` (20% de probabilité d'utiliser CutMix)
- `afd_lambda = 0.2`
- `trades_beta = 1.0`

### 6.2.3 Phase 2 — Robustesse Forte (Epochs 36 à 55)

**Objectif :** Entraîner le modèle contre des attaques fortes en perturbant 3 features simultanément (`worst_k = 3`). L'attention de TRADES est doublée pour forcer la robustesse.

**Configuration :**
- `epsilon` : Croissance linéaire de 0.05 à 0.10.
- `worst_k = 3`
- `cutmix_prob = 0.3` (30% de probabilité d'utiliser CutMix)
- `afd_lambda = 0.1`
- `trades_beta = 2.0`

### 6.2.4 Phase 3 — Consolidation (Epochs > 55)

**Objectif :** Maintenir une pression constante et maximale pour stabiliser les gradients finaux du modèle.

**Configuration :**
- `epsilon = 0.10` (Maximum)
- `worst_k = 3`
- `cutmix_prob = 0.3`
- `afd_lambda = 0.05`
- `trades_beta = 2.0`

---

## 6.3 Early Stopping Antagoniste (Stabilité)

Le passage rigide basé sur les epochs peut conduire à une dégradation si le modèle oublie ses connaissances de base. Un mécanisme de sauvegarde, `AdversarialEarlyStopping`, surveille en continu la santé de l'entraînement :

1. **Surveillance du Gap :** L'écart entre l'accuracy propre et l'accuracy adversariale (`benign_acc - adv_acc`) ne doit pas dépasser `0.60`.
2. **Surveillance de la Loss :** La `adv_loss` ne doit pas exploser au-dessus de `10.0`.
3. **Réduction d'Epsilon :** Si une violation survient, la perturbation `epsilon` est dynamiquement réduite (`new_eps = current_eps * 0.7`). Si les violations persistent pendant 5 epochs (`patience=5`), l'entraînement s'arrête prématurément pour éviter un effondrement catastrophique ("catastrophic forgetting").

---

## 6.4 Phase E — Entraînement du Discriminateur (Séparé)

Après les phases du classifieur, un **Discriminateur BiLSTM** peut être entraîné séparément pour détecter la présence de manipulations adversariales sur le trafic entrant :

```python
disc = Discriminator(input_size=input_size, seq_length=10, hidden_size=64)
disc, disc_acc = train_discriminator(
    discriminator=disc,
    X_train=X_train,
    simulator=simulator,      # Utilise le simulator de la Phase D
    device=device,
    epochs=25,                # Entraînement binaire
    batch_size=batch_size,
    lr=1e-3
)
```

Le Discriminateur est entraîné avec une perte `BCEWithLogitsLoss` sur des batches équilibrés : 50% de flux propres (label=0) et 50% de flux adversariaux (label=1). Il agit en tant que sentinelle pour le système de routage SDN.

---

## 6.5 Récapitulatif du Curriculum

| Phase | Epochs | worst_k | Cutmix % | Epsilon | TRADES β |
|-------|--------|---------|----------|---------|----------|
| **0. Fondation Propre** | 1-15 | 0 | 0% | 0.0 | 0.0 |
| **1. Robustesse Douce** | 16-35 | 1 | 20% | 0.01 → 0.05 | 1.0 |
| **2. Robustesse Forte** | 36-55 | 3 | 30% | 0.05 → 0.10 | 2.0 |
| **3. Consolidation** | > 55 | 3 | 30% | 0.10 | 2.0 |

Cette architecture temporelle assure que les modèles finaux sont véritablement robustes de manière consistante, et non par le biais de fluctuations statistiques aléatoires.
