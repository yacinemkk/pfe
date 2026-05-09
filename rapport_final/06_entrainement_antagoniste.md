# Chapitre 6 — Curriculum d'Entraînement Adversarial Greedy par Mix Ratio

## 6.1 Philosophie du Curriculum Learning Adversarial

L'entraînement antagoniste naïf — où on injecte des exemples adversariaux dès le début de l'entraînement — échoue fréquemment dans la pratique pour deux raisons principales :

1. **Effondrement de l'accuracy propre (*clean accuracy collapse*)** : le modèle apprend à être robuste mais oublie les exemples normaux, ce qui est inacceptable en production où la majorité du trafic est légitime.
2. **Instabilité de l'entraînement** : les gradients générés par les exemples adversariaux forts peuvent interférer destructivement avec l'apprentissage des patterns propres.

La solution adoptée dans ce projet est un **curriculum d'entraînement progressif basé sur un mix ratio croissant d'exemples adversariaux**, défini dans `greedy_new_optimized.ipynb/py`. La stratégie repose sur une augmentation graduelle de la proportion d'exemples adversariaux générés par le **GreedyAttackSimulator** (qui applique les perturbations : Zero, Mimic_Mean, Mimic_95th, Padding_x10), avec contrôle dynamique par seuils de robustesse (`k_max`).

---

## 6.2 Curriculum Adaptatif par Mix Ratio et k_max

### Configuration Actuelle (Curriculum v3 : Threshold-Gated + Replay)

**Paramètres globaux :**
- `MAX_PHASE_EPOCHS = 20` : timeout par phase (même si seuil non atteint)
- `K_THRESHOLD = 0.85` : cible de robustesse par phase
- `K_STABLE = 0.82` : seuil de stabilité des k précédents
- `K_BACKSTEP = 0.80` : déclenche retour D2→D1 si k < 0.80
- `K_RESUME_D2 = 0.83` : hysteresis — reprend D2 si k > 0.83 (après backstep)
- `N_CONSEC_D2 = 3` : nombre d'epochs consécutives à K_THRESHOLD pour arrêt D2

### 6.2.1 Phase 0 — Bootstrap Propre (Epochs 1 à ~15)

**Objectif :** Établir une base solide sur des données **100% propres** (aucun exemple adversarial).

**Configuration :**
- `Mix Ratio = 0.0` → 100% clean data
- `k_max = 0` (Aucune perturbation appliquée)
- **Ratio Clean/Adversarial :** 100% / 0%

### 6.2.2 Phase B1 — Introduction Douce (Epochs ~16 à ~25)

**Objectif :** Introduire graduellement les attaques greedy avec k_max=2 (perturbation sur 2 features max).

**Configuration :**
- `Mix Ratio = 0.4` → 60% clean / 40% adversarial
- `k_max = 2` (Perturbation sur 2 features max)
- **Strategies appliquées :** Zero, Mimic_Mean, Mimic_95th, Padding_x10

### 6.2.3 Phase B2 — Robustesse Progressive (Epochs ~26 à ~30)

**Objectif :** Augmenter la proportion d'adversariaux avec k_max=2.

**Configuration :**
- `Mix Ratio = 0.5` → 50% clean / 50% adversarial
- `k_max = 2`
- **Ratio Clean/Adversarial :** 50% / 50%

### 6.2.4 Phase C — Robustesse Forte (Epochs ~31 à ~50)

**Objectif :** Full adversarial training avec k_max=4 (perturbation sur 4 features max).

**Configuration :**
- `Mix Ratio = 0.7` → 30% clean / 70% adversarial
- `k_max = 4`
- **Ratio Clean/Adversarial :** 30% / 70%

### 6.2.5 Phase D1 — Intensité Maximale (Epochs ~51 à ~60)

**Objectif :** Entraîner le modèle contre des attaques extrêmement fortes.

**Configuration :**
- `Mix Ratio = 0.85` → 15% clean / 85% adversarial
- `k_max = 4`
- **Ratio Clean/Adversarial :** 15% / 85%

### 6.2.6 Phase D2 — Consolidation (Epochs ~61 à ~70+)

**Objectif :** Maintenir la robustesse avec quasi 100% d'exemples adversariaux.

**Configuration :**
- `Mix Ratio = 0.95` → 5% clean / 95% adversarial
- `k_max = 4`
- **Ratio Clean/Adversarial :** 5% / 95%

---

## 6.3 Mécanisme de Contrôle Adaptatif (Threshold-Gated + Replay)

Le curriculum actuel intègre un **contrôle dynamique** basé sur la robustesse mesurée en validation :

1. **Évaluation Continuelle :** Après chaque epoch, on évalue la robustesse du modèle en testant sa performance contre les attaques greedy avec différents `k` (k=1, k=2, k=3, k=4).

2. **Threshold-Gating :** 
   - Si `accuracy_k >= K_THRESHOLD (0.85)`, la phase est considérée "réussie"
   - Si `accuracy_k < K_STABLE (0.82)`, le modèle est jugé instable et on applique un backstep optionnel
   - Si `accuracy_k < K_BACKSTEP (0.80)`, on retourne à la phase précédente (D2→D1)

3. **Hysteresis & Replay :**
   - Après un backstep, la phase peut être reprise si `accuracy_k > K_RESUME_D2 (0.83)`
   - Évite les oscillations excessives et garantit une convergence stable

4. **Nombre d'Epochs :** Chaque phase dure max `MAX_PHASE_EPOCHS = 20` epochs ou jusqu'à atteinte de `N_CONSEC_D2 = 3` epochs consécutives au-dessus du seuil.

---

## 6.4 Stratégies d'Attaque Greedy Applicables

Le **GreedyAttackSimulator** applique 4 stratégies de perturbation sur les features vulnérables identifiées par analyse de sensibilité :

| Stratégie | Description | Effet |
|-----------|-------------|-------|
| **Zero** | Annule la feature (définie à 0) | Suppression totale d'information |
| **Mimic_Mean** | Remplace par la moyenne statistique | Perturbation douce basée sur distribution |
| **Mimic_95th** | Remplace par le percentile 95 | Perturbation forte mais réaliste |
| **Padding_x10** | Multiplie la feature par 10 | Amplification des valeurs |

Ces stratégies sont appliquées simultanément sur les `k_max` features les plus vulnérables, créant ainsi des exemples adversariaux réalistes et furtifs.

---

## 6.5 Early Stopping et Stabilité

Le passage à travers les phases est assuré par les mécanismes suivants :

1. **Surveillance de la Robustesse :** L'accuracy en présence d'attaques greedy est continuellement surveillée.
2. **Prévention de l'Oubli :** En maintenant toujours une proportion minimale de données propres (5% en Phase D2), on évite le "catastrophic forgetting".
3. **Convergence Garantie :** Après max `MAX_PHASE_EPOCHS`, la phase se termine même si les seuils ne sont pas atteints, permettant une progression garantie du curriculum.

---

## 6.6 Résumé Visuel du Curriculum

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

## 6.7 Tableau Récapitulatif du Curriculum Actuel

| Phase | Epochs Approx. | Mix Ratio | Clean % | Adv % | k_max | Strategies | Objectif |
|-------|---|---|---|---|---|---|---|
| **Phase 0** | 1-15 | 0.0 | 100% | 0% | 0 | Aucune | Bootstrap propre |
| **Phase B1** | 16-25 | 0.4 | 60% | 40% | 2 | Zero, Mimic_Mean, Mimic_95th, Padding_x10 | Introduction douce |
| **Phase B2** | 26-30 | 0.5 | 50% | 50% | 2 | Greedy Search | Robustesse progressive |
| **Phase C** | 31-50 | 0.7 | 30% | 70% | 4 | Greedy Search | Full adversarial |
| **Phase D1** | 51-60 | 0.85 | 15% | 85% | 4 | Greedy Search | Intensité maximale |
| **Phase D2** | 61-70+ | 0.95 | 5% | 95% | 4 | Greedy Search | Consolidation |

---

## 6.8 Conclusion du Curriculum

Ce curriculum d'entraînement **adaptatif et multi-étapes** garantit que les modèles finaux atteignent une robustesse certifiée contre les attaques greedy tout en conservant une accuracy propre élevée. En progressant graduellement du clean learning au full adversarial training, puis en appliquant des mécanismes de feedback basés sur seuils (`K_THRESHOLD`, `K_BACKSTEP`), le système évite le "catastrophic forgetting" et converge vers une solution robuste et équilibrée.

L'utilisation du **GreedyAttackSimulator** plutôt que des attaques génériques (FGSM, PGD) garantit que la robustesse développée est **spécifique et certifiée** contre les attaques furtives identifiées dans la phase d'analyse de sensibilité.
