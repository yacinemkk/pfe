# Analyse de l'Entraînement Adversarial — LSTM IoT

## 1. Comment les Attaques sont Générées

### 1.1 Attaque Feature-Level (Modification de Caractéristiques)

Un flux réseau IoT possède plusieurs caractéristiques mesurables. L'attaque modifie ces caractéristiques pour tromper le modèle.

**Exemple concret avec un flux réseau:**

| Caractéristique | Valeur Originale | Valeur Modifiée | Stratégie |
|-----------------|------------------|-----------------|-----------|
| packetTotalCount | 150 | 0 | Zero |
| octetTotalCount | 45000 | 85 (moyenne autre appareil) | Mimic_Mean |
| averageInterarrivalTime | 0.05 | 120 (95e percentile) | Mimic_95th |
| reversePacketCount | 50 | 500 | Padding_x10 |

**Les 4 stratégies disponibles:**

- **Zero**: Met la caractéristique à 0
- **Mimic_Mean**: Remplace par la moyenne des flux bénins
- **Mimic_95th**: Remplace par le 95e percentile des flux bénins
- **Padding_x10**: Multiplie la valeur par 10

### 1.2 Analyse de Sensibilité

Le code teste chaque caractéristique une par une pour mesurer son importance:

```
Test: Modifier "packetTotalCount" → Zero
      Accuracy passe de 95% → 45%
      → Cette caractéristique est TRÈS importante!

Test: Modifier "protocolIdentifier" → Zero
      Accuracy passe de 95% → 94%
      → Cette caractéristique est peu importante
```

Les résultats sont classés par "drop" (chute d'accuracy) pour identifier les caractéristiques les plus vulnérables.

### 1.3 Attaque Sequence-Level (Recherche Greedy)

Le code combine les meilleures attaques de manière itérative:

```
Étape 1: Modifier packetTotalCount → Zero
         Accuracy: 95% → 75%

Étape 2: Modifier octetTotalCount → Mimic_Mean
         Accuracy: 75% → 50%

Étape 3: Modifier averageInterarrivalTime → Padding_x10
         Accuracy: 50% → 25%

→ L'attaque finale combine les 3 modifications!
```

L'algorithme s'arrête quand l'accuracy cible est atteinte (par défaut 50%).

---

## 2. Schéma de l'Entraînement Actuel

### Phase 1: Entraînement Normal

```
┌─────────────────────────────────────────────────────────┐
│  Données normales (1,119,349 séquences)                 │
│           ↓                                             │
│  Modèle apprend les patterns normaux                    │
│           ↓                                             │
│  Résultat: 95% accuracy sur données propres             │
└─────────────────────────────────────────────────────────┘
```

### Phase 2: Entraînement Adversarial

```
┌─────────────────────────────────────────────────────────┐
│  Batch de 64 exemples                                   │
│           ↓                                             │
│  60% (38) = Données normales                            │
│  20% (13) = Attaques feature-level                      │
│  20% (13) = Attaques sequence-level                     │
│           ↓                                             │
│  Le modèle apprend à résister aux attaques              │
└─────────────────────────────────────────────────────────┘
```

### Crash Test (Évaluation)

```
┌─────────────────────────────────────────────────────────┐
│  Test 1 - Bénignes: 219,811 séquences normales          │
│           → Accuracy: 96% ✓                             │
│                                                         │
│  Test 2 - Adversaires: 1000 séquences attaquées         │
│           → Accuracy: 5% ✗                              │
│                                                         │
│  Test 3 - Mélangé: 500 normales + 500 attaquées         │
│           → Accuracy: 51%                               │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Problèmes Identifiés

### 3.1 Sensibilité Obsolète

**Le problème:**

```
Phase 2 Epoch 1-2: Attaques basées sur le modèle Phase 1 ❌
Phase 2 Epoch 3-5: Nouvelles attaques basées sur Phase 2 ✓
Phase 2 Epoch 6-8: Nouvelles attaques basées sur Phase 2 ✓
Phase 2 Epoch 9-10: Nouvelles attaques basées sur Phase 2 ✓

Crash Test: NOUVELLES attaques basées sur modèle final ← TROUVE DE NOUVELLES FAILLES!
```

**Explication:** Les attaques utilisées pendant l'entraînement sont basées sur les vulnérabilités du modèle à un instant T. Le modèle évolue, mais les attaques ne sont pas mises à jour assez souvent.

### 3.2 Ratio Adversarial Trop Faible

```
Actuellement: 20% du batch = ~12 exemples adverses/batch
Proposition: 50% du batch = ~32 exemples adverses/batch
```

### 3.3 Phase 2 Trop Courte

```
Actuellement: 10 epochs maximum
Proposition: 20-30 epochs
```

### 3.4 Exemples Adverses Statiques vs Dynamiques

```
Post-Phase 1: 223,869 exemples générés → ÉVALUATION SEULEMENT (jetés après)
Phase 2: Génération dynamique par batch → ENTRAÎNEMENT
```

Les 223k exemples générés après Phase 1 ne sont pas réutilisés pour l'entraînement.

---

## 4. Solutions Proposées

### Solution 1: Mettre à Jour les Attaques Chaque Epoch

**Actuellement:**
```python
if epoch > 0 and epoch % 3 == 0:  # Tous les 3 epochs
    sensitivity_p2 = feature_attack.analyze_sensitivity(...)
```

**Proposition:**
```python
if epoch > 0:  # Chaque epoch
    sensitivity_p2 = feature_attack.analyze_sensitivity(...)
```

**Impact:** ⭐⭐⭐ (Élevé)
**Effort:** Très petit (1 ligne à modifier)

### Solution 2: Augmenter le Ratio Adversarial

**Actuellement:**
```python
adv_ratio = 0.2  # 20%
```

**Proposition:**
```python
adv_ratio = 0.5  # 50%
```

**Impact:** ⭐⭐⭐ (Élevé)
**Effort:** Très petit (1 paramètre à modifier)

### Solution 3: Plus d'Epochs en Phase 2

**Actuellement:**
```python
phase2_epochs = 10
```

**Proposition:**
```python
phase2_epochs = 20  # ou 30
```

**Impact:** ⭐⭐ (Moyen-Élevé)
**Effort:** Très petit (1 paramètre à modifier)

### Solution 4: Augmenter le Nombre d'Exemples de Test

**Actuellement:**
```python
n_eval = min(1000, len(X_test))  # Max 1000 exemples
```

**Proposition:**
```python
n_eval = min(5000, len(X_test))  # Max 5000 exemples
```

**Impact:** ⭐ (Faible - ne résout pas le problème de robustesse, mais donne une évaluation plus fiable)
**Effort:** Très petit

### Solution 5: Loss TRADES

**Actuellement:**
```python
Loss = CrossEntropyLoss
```

**Proposition:**
```python
Loss = CrossEntropyLoss + β × KL_Divergence(prédictions_clean || prédictions_adv)
```

**Impact:** ⭐⭐ (Moyen)
**Effort:** Moyen (modification de la fonction de loss)

---

## 5. Ordre de Priorité

| Priorité | Solution | Impact | Effort |
|----------|----------|--------|--------|
| 1 | Mettre à jour chaque epoch | ⭐⭐⭐ | Petit |
| 2 | Augmenter adv_ratio (20%→50%) | ⭐⭐⭐ | Très petit |
| 3 | Plus d'epochs (10→20) | ⭐⭐ | Très petit |
| 4 | TRADES loss | ⭐⭐ | Moyen |
| 5 | Plus de stratégies d'attaque | ⭐ | Moyen |

---

## 6. Résumé des Résultats Actuels

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Phase    Test                 Loss     Acc      F1       RR            │
├─────────────────────────────────────────────────────────────────────────┤
│  P1       test1_benign         0.1722   0.9533   0.9031   1.0000       │
│  P1       test2_adversarial    26.3233  0.0460   0.0203   0.0483       │
│  P1       test3_mixed          15.1425  0.5070   0.5025   1.0000       │
├─────────────────────────────────────────────────────────────────────────┤
│  P2       test1_benign         0.1506   0.9645   0.9210   1.0000       │
│  P2       test2_adversarial    53.8941  0.0530   0.0305   0.0550       │
│  P2       test3_mixed          30.1223  0.5110   0.5009   1.0000       │
└─────────────────────────────────────────────────────────────────────────┘

Observation: La robustesse (RR) n'a pas amélioré entre P1 et P2!
```

---

## 7. Conclusion

Le modèle actuel n'est **pas robuste** aux attaques adversales. La Phase 2 n'a pas réussi à améliorer la robustesse car:

1. Les attaques utilisées pendant l'entraînement sont basées sur un modèle **obsolète**
2. Le **ratio d'exemples adverses** est trop faible (20%)
3. La **Phase 2 est trop courte** (10 epochs)

Les solutions proposées (mettre à jour les attaques chaque epoch, augmenter le ratio adversarial, plus d'epochs) sont simples à implémenter et devraient améliorer significativement la robustesse.
