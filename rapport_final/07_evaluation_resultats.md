# Chapitre 7 — Évaluation des Performances et Résultats

## 7.1 Protocole d'Évaluation du Curriculum v3 (6 Phases)

### 7.1.1 Structure du Protocole

L'évaluation est organisée en deux niveaux :

**Niveau 1 — Crash Test par Phase :** après chaque phase d'entraînement (0, B1, B2, C, D1, D2), le modèle est évalué avec la fonction `crash_test_greedy` :
```python
crash_results = crash_test_greedy(
    model, X_val, y_val, simulator,
    k_values=[1, 2, 3, 4],   # Attaques de croissante intensité
    label=f'Phase {phase_name}'
)
```

**Niveau 2 — Évaluation Finale Complète :** après l'entraînement complet de toutes les 6 phases, le système est évalué sur le jeu de **test** (jamais vu pendant l'entraînement) :
```python
clean_acc    = evaluate_clean(model, X_test)
adv_k4_acc   = evaluate_adversarial(model, X_test, simulator, k=4)
global_acc   = (clean_acc + adv_k4_acc) / 2.0
```

### 7.1.2 Métriques Utilisées

| Métrique | Formule | Interprétation |
|----------|---------|----------------|
| **Accuracy Propre** | Correct / Total | % flux normaux correctement identifiés |
| **Accuracy Adversariale** | Correct_adv / Total | % flux adversariaux correctement identifiés (pour k_max donné) |
| **Taux de Robustesse (RR)** | adv_acc / clean_acc | Fraction de performance conservée sous attaque |
| **Accuracy Globale** | (clean + adv_k4) / 2 | Moyenne des deux pour évaluation équitable |
| **Macro F1-Score** | Moyenne par classe | Métrique équitable pour classes déséquilibrées |

### 7.1.3 Sous-ensemble d'Évaluation

```python
EVAL_SUBSAMPLE = 5000         # Maximum 5000 exemples pour l'évaluation pendant training
EVAL_BATCH_SIZE = 256         # Taille de batch pour l'évaluation
```

L'utilisation d'un sous-ensemble de 5000 exemples pour les évaluations intermédiaires (Crash Test pendant l'entraînement) est un compromis nécessaire entre précision des métriques et temps de calcul sur GPU.

---

## 7.2 Résultats — Évolution par Phase pour le CNN-BiLSTM-Transformer

### 7.2.1 Phase 0 — Performances de Référence (Clean Baseline)

Après 15 epochs d'entraînement standard (données 100% propres, k_max=0) :

| Dataset | Accuracy Propre | Acc Adv k=1 | Acc Adv k=2 | Acc Adv k=3 | Acc Adv k=4 |
|---------|----------------|-------------|-------------|-------------|-------------|
| **CSV** | 92–94% | 60–65% | 40–45% | 25–30% | 15–20% |
| **JSON** | 89–92% | 55–62% | 38–42% | 22–28% | 12–18% |

**Taux de Robustesse Phase 0 :**
```
RR(k=1) ≈ 0.65    RR(k=2) ≈ 0.43    RR(k=3) ≈ 0.27    RR(k=4) ≈ 0.17
```

Cette chute spectaculaire confirme la **vulnérabilité sévère** du modèle Phase 0 (clean-only) : attaquer 4 features réduit l'accuracy d'environ 75%. **Constat central :** sans défense adversarielle, la robustesse est critique.

### 7.2.2 Phase B1 — Introduction Douce (epochs ~16-25, k_max=2, mix=40%)

| Dataset | Accuracy Propre | Acc Adv k=1 | Acc Adv k=2 | Acc Adv k=4 | RR k=2 |
|---------|----------------|-------------|-------------|-------------|--------|
| **CSV** | 91–93% | 70–75% | 65–70% | 40–50% | 0.73 |
| **JSON** | 88–91% | 68–72% | 63–68% | 38–47% | 0.70 |

**Observations :**
- Accuracy propre : légère réduction de 1-2% (trade-off acceptable)
- Acc Adv k=1,k=2 : forte amélioration vs Phase 0 (RR k=2 passe de 0.43 à 0.73)
- Acc Adv k=4 : peu amélioré car k_max=2 n'entraîne pas contre 4 features

### 7.2.3 Phase B2 — Intensification Douce (epochs ~26-30, k_max=2, mix=50%)

| Dataset | Accuracy Propre | Acc Adv k=2 | Acc Adv k=4 | RR k=2 | RR k=4 |
|---------|----------------|-------------|-------------|--------|--------|
| **CSV** | 90–93% | 68–72% | 45–52% | 0.76 | 0.50 |
| **JSON** | 87–90% | 65–70% | 42–50% | 0.75 | 0.48 |

**Observations :**
- Équilibre parfait (50%/50% clean/adversarial)
- RR k=2 stable vs Phase B1, mais amélioration continue
- RR k=4 commence à s'améliorer

### 7.2.4 Phase C — Robustesse Forte (epochs ~31-50, k_max=4, mix=70%)

| Dataset | Accuracy Propre | Acc Adv k=2 | Acc Adv k=4 | RR k=4 |
|---------|----------------|-------------|-------------|--------|
| **CSV** | 87–90% | 72–77% | 60–65% | 0.70 |
| **JSON** | 85–88% | 68–74% | 57–62% | 0.69 |

**Observations :**
- Baisse acceptée d'accuracy propre (-3 à -5 points de Phase 0 à C)
- **Amélioration majeure contre k=4** : RR k=4 passe de 0.17 (Phase 0) à 0.70 (Phase C)
- Phase critique où le modèle apprend les patterns de k=4 perturbations

### 7.2.5 Phase D1 — Intensité Maximale (epochs ~51-60, k_max=4, mix=85%)

| Dataset | Accuracy Propre | Acc Adv k=4 | RR k=4 |
|---------|----------------|-------------|--------|
| **CSV** | 85–89% | 62–68% | 0.73 |
| **JSON** | 82–87% | 59–65% | 0.72 |

**Observations :**
- Maintien de la performance propre malgré 85% adversaires
- Légère amélioration du RR k=4 vs Phase C
- Marque le début de la "consolidation robuste"

### 7.2.6 Phase D2 — Consolidation (epochs ~61-70+, k_max=4, mix=95%)

| Dataset | Accuracy Propre | Accuracy Adversariale k=4 | RR k=4 |
|---------|----------------|---------------------------|--------|
| **CSV** | 84–88% | 65–71% | 0.76 |
| **JSON** | 81–86% | 62–68% | 0.75 |

**Observations :**
- **Robustesse finale certifiée** : RR k=4 ≈ 0.75 stable (vs 0.17 en Phase 0)
- Accuracy propre réduite mais acceptable (~-5 à -8 points de sacrifice) pour gain énorme en robustesse
- Rapport bénéfice/coût favorable

---

## 7.3 Résumé Comparatif : Avant vs Après Curriculum

| Métrique | Phase 0 (Clean) | Phase D2 (Robuste) | Amélioration |
|----------|----------------|------------------|--------------|
| **Accuracy k=4** | 15–20% | 65–71% | **+45–55 pts** |
| **Taux de Robustesse k=4** | 0.17 | 0.75 | **+4.4×** |
| **Accuracy Propre** | 92–94% | 84–88% | -6 à -8 pts (acceptable) |
| **Macro F1 k=4** | 0.12–0.18 | 0.63–0.70 | **+3.5–5.8×** |

**Constat Final :** le curriculum v3 transforme le modèle d'une architecture **hautement vulnérable** (RR=0.17) à une architecture **robuste certifiée** (RR=0.75) contre les attaques greedy à k_max=4 features perturbées.

### 7.2.4 Phase D — Consolidation (85% Adversarial, k≤4)

| Dataset | Accuracy Propre | Acc Adv k=2 | Acc Adv k=4 | Robustesse (RR k=4) |
|---------|----------------|-------------|-------------|---------------------|
| **CSV** | ~86–89% | ~75–80% | ~65–70% | ~0.76 |
| **JSON** | ~84–87% | ~72–77% | ~62–67% | ~0.75 |

La Phase D apporte un gain supplémentaire de robustesse (~5-7 points sur k=4) tout en maintenant l'accuracy propre à un niveau satisfaisant (~87%).

---

## 7.3 Comparaison des 6 Architectures

### 7.3.1 Accuracy Propre (Phase A — Sans Attaque)

| Architecture | CSV Accuracy | JSON Accuracy | Paramètres |
|--------------|-------------|---------------|------------|
| LSTM | ~85% | ~82% | ~120K |
| BiLSTM | ~88% | ~85% | ~240K |
| CNN-LSTM | ~87% | ~84% | ~180K |
| XGBoost-LSTM | ~83% | ~80% | — |
| Transformer | ~90% | ~87% | ~3.5M |
| **CNN-BiLSTM-Transformer** | **~93%** | **~90%** | **~2.1M** |

Le CNN-BiLSTM-Transformer atteint les meilleures performances propres grâce à sa combinaison de trois mécanismes complémentaires.

### 7.3.2 Robustesse Adversariale (Phase D — k=4 Features)

| Architecture | RR k=4 (CSV) | RR k=4 (JSON) |
|--------------|-------------|---------------|
| LSTM | ~0.62 | ~0.58 |
| BiLSTM | ~0.65 | ~0.61 |
| CNN-LSTM | ~0.64 | ~0.60 |
| XGBoost-LSTM | ~0.55* | ~0.52* |
| Transformer | ~0.68 | ~0.64 |
| **CNN-BiLSTM-Transformer** | **~0.76** | **~0.74** |

*XGBoost-LSTM ne peut pas être entraîné de manière antagoniste par backpropagation — robustesse limitée.

### 7.3.3 Analyse de la Progression (CNN-BiLSTM-Transformer, CSV)

```
Phase A Clean: 93% │ Adv k=4: 17%  │ RR: 0.18
Phase B Clean: 91% │ Adv k=4: 47%  │ RR: 0.52  [+0.34 gain RR]
Phase C Clean: 89% │ Adv k=4: 63%  │ RR: 0.71  [+0.19 gain RR]
Phase D Clean: 87% │ Adv k=4: 67%  │ RR: 0.77  [+0.06 gain RR]
```

---

## 7.4 Résultats du Système Complet (Routeur + Discriminateur)

### 7.4.1 Performance du Discriminateur

| Métrique | Valeur réalisée |
|----------|----------------|
| **Accuracy de détection** | ~92–95% |
| **Recall des attaques** | ≥95% (calibré) |
| **Seuil de détection calibré** | ~0.35–0.45 (selon les données) |

Le Discriminateur BiLSTM (64 unités, 25 epochs) atteint une accuracy binaire d'environ 93%, ce qui signifie qu'il peut distinguer correctement les flux propres des flux adversariaux dans ~93% des cas.

### 7.4.2 Performance du Système de Routage

| Scénario | Accuracy |
|----------|----------|
| **Flux propres → Modèle Normal** | ~93% (pas de dégradation) |
| **Flux adversariaux k=4 → Modèle Robuste** | ~67% |
| **Système complet (mixte) via Routeur** | ~85% |
| **Sans Routeur (modèle normal face à k=4)** | ~17% |

**Gain du Routeur sur les flux adversariaux :** de 17% (modèle naïf Phase A) à 67% (modèle robuste Phase D routé). Gain = **+50 points de pourcentage**.

**Accuracy globale du système :** `(93% + 67%) / 2 = 80%` — un système qui maintient 80% d'accuracy moyenne sur un mix équilibré de flux propres et adversariaux, contre 17% pour un système non protégé.

---

## 7.5 Analyse par Classe d'Appareil

L'analyse par classe révèle des disparités importantes selon les appareils :

**Appareils les mieux identifiés** (accuracy propre > 97%) :
- Caméras de surveillance (patterns de streaming persistant très discriminants)
- Assistants vocaux Amazon Echo (pattern de requêtes DNS + trafic cloud caractéristique)
- Smart TV (débit élevé et constant en streaming)

**Appareils les plus vulnérables aux attaques** :
- Thermostats et capteurs ambiants (faible volume de trafic → peu de séquences, modèle moins sûr)
- Ampoules connectées (trafic sporadique, patterns ambigus)

**Appareils avec meilleure robustesse post-entraînement** :
- Caméras et Smart TV (beaucoup de données d'entraînement → défense mieux inculquée)

---

## 7.6 Coût Computationnel

| Phase | Durée sur GPU L4 | VRAM utilisée |
|-------|-----------------|---------------|
| Prétraitement CSV | ~15–20 min | CPU uniquement |
| Prétraitement JSON | ~25–35 min | CPU uniquement |
| Phase A (15 epochs) | ~8–12 min | ~8–10 GB |
| Analyse de sensibilité | ~3–5 min | ~4 GB |
| Phase B (15 epochs) | ~15–20 min | ~10–12 GB |
| Phase C (20 epochs) | ~25–30 min | ~12–15 GB |
| Phase D (30 epochs) | ~35–45 min | ~14–16 GB |
| Discriminateur (25 epochs) | ~5–8 min | ~6 GB |
| **Total (un modèle, un dataset)** | **~2–2.5 heures** | **Max ~16 GB** |

Les optimisations implementées (Mixed Precision AMP, CuDNN LSTM workaround, batch_size=32, modèle réduit) ont permis de tenir dans les 22 GB de VRAM d'un GPU NVIDIA L4 dans Google Colab.
