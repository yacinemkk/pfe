# Mise à Jour - Curriculum Greedy Adversarial Multi-Phase

## État Actuel (Mai 2026)

Le curriculum d'entraînement a évolué d'un modèle simple en **2 phases** (clean + adversarial) 
à un **curriculum progressif multi-phase (6 phases)** basé sur un **mix ratio croissant** et 
un **contrôle adaptatif par seuils de robustesse**.

### Fichiers de Référence

- **Documentation complète :** `rapport_final/06_entrainement_antagoniste.md`
- **Plan d'entraînement détaillé :** `docs/train`
- **Implémentation :** `greedy_new_optimized.ipynb` / `greedy_new_optimized.py`

---

## Phases du Curriculum Actuel

### Phase 0 : Bootstrap Propre (Epochs 1-15)
- **Mix Ratio :** 0.0 (100% clean)
- **k_max :** 0 (pas de perturbations)
- **Objectif :** Établir une base de réference sur données pures
- **Résultat attendu :** Clean accuracy > 90%

### Phase B1 : Introduction Douce (Epochs ~16-25)
- **Mix Ratio :** 0.4 (60% clean / 40% adversarial)
- **k_max :** 2 (perturbation sur 2 features max)
- **Objectif :** Commencer l'exposition graduellement aux attaques greedy
- **Strategies :** Zero, Mimic_Mean, Mimic_95th, Padding_x10

### Phase B2 : Robustesse Progressive (Epochs ~26-30)
- **Mix Ratio :** 0.5 (50% clean / 50% adversarial)
- **k_max :** 2
- **Objectif :** Équilibre parfait entre clean et adversarial

### Phase C : Full Adversarial Training (Epochs ~31-50)
- **Mix Ratio :** 0.7 (30% clean / 70% adversarial)
- **k_max :** 4 (perturbation sur 4 features max)
- **Objectif :** Entraîner contre des attaques fortes

### Phase D1 : Intensité Maximale (Epochs ~51-60)
- **Mix Ratio :** 0.85 (15% clean / 85% adversarial)
- **k_max :** 4
- **Objectif :** Robustesse maximale avec maintien critique de données clean

### Phase D2 : Consolidation (Epochs ~61-70+)
- **Mix Ratio :** 0.95 (5% clean / 95% adversarial)
- **k_max :** 4
- **Objectif :** Stabiliser la robustesse avec quasi 100% adversaires

---

## Paramètres de Contrôle Adaptatif

```
MAX_PHASE_EPOCHS  = 20      # timeout par phase
K_THRESHOLD       = 0.85    # cible de robustesse
K_STABLE          = 0.82    # seuil de stabilité
K_BACKSTEP        = 0.80    # déclenche retour D2→D1
K_RESUME_D2       = 0.83    # reprend D2 après backstep
N_CONSEC_D2       = 3       # epochs consécutives au seuil pour progression
```

**Mécanisme :**
- Chaque phase évalue la robustesse : `accuracy_k` pour k=1,2,3,4
- Si `accuracy_k >= K_THRESHOLD` : phase réussie
- Si `accuracy_k < K_BACKSTEP` : retour à la phase précédente (backstep)
- Hysteresis : évite oscillations excessives

---

## Stratégies d'Attaque Greedy

Le **GreedyAttackSimulator** applique 4 stratégies combinées sur les `k_max` features les plus vulnérables :

| Stratégie | Description |
|-----------|-------------|
| **Zero** | Annule la feature (définie à 0) |
| **Mimic_Mean** | Remplace par la moyenne statistique |
| **Mimic_95th** | Remplace par le percentile 95 |
| **Padding_x10** | Multiplie la feature par 10 |

---

## Résultats Attendus

### Crash Test 1 (Phase 0 uniquement)
- Clean accuracy : 90%+
- Accuracy sous attaque k=4 : 20-40%
- **Diagnostic :** Hautement vulnérable

### Crash Test 2 (Après Phase D2)
- Clean accuracy : 85-90% (trade-off léger, acceptable)
- Accuracy sous attaque k=4 : 80-95% (remontée spectaculaire)
- **Diagnostic :** Robuste et sécurisé

---

## Architekture Finale des Modèles

Les 6 modèles entraînés suivent ce curriculum :

1. **LSTM** - Socle temporel
2. **BiLSTM** - Temporel bidirectionnel
3. **CNN-LSTM** - Spatio-temporel
4. **XGBoost-LSTM** - ML/DL hybride
5. **Transformer** - Attention globale
6. **CNN-BiLSTM-Transformer** - État de l'art

---

## Implémentation (Greedy_new_optimized.ipynb)

**Cellules principales :**
- Setup et chargement des données
- Configuration du curriculum et des phases
- Fonction GreedyAttackSimulator
- Boucles d'entraînement adaptatif par modèle
- Évaluation et visualisation des résultats

Chaque modèle bénéficie du **curriculum complet** pour garantir une robustesse certifiée 
contre les attaques greedy identifiées en advance.

---

## Documentation Mise à Jour

- ✅ `rapport_final/06_entrainement_antagoniste.md` - Détails complets du curriculum
- ✅ `docs/train` - Plan d'entraînement étape par étape
- ✅ Ce fichier - Vue d'ensemble

### Fichiers à consulter pour plus de détails

1. **Phases spécifiques :** `rapport_final/06_entrainement_antagoniste.md` § 6.2 à 6.8
2. **Protocole d'entraînement :** `docs/train` - Sections Phase 0 à D2
3. **Implémentation :** `greedy_new_optimized.ipynb` cells
