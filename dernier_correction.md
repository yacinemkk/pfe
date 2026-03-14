# Plan d'Amélioration : Entraînement par Curriculum & Early Stopping Local

Ce document détaille le plan optimal pour restructurer la boucle d'entraînement (Hybrid Adversarial Training) de votre projet, en y intégrant un Early Stopping par phase et une évaluation granulaire.

---

## 1. Ordre d'Entraînement Optimal (Curriculum Learning)

D'après la littérature scientifique sur l'apprentissage robuste et le *Curriculum Learning* (Apprentissage par Curriculum), l'approche la plus efficace pour éviter le *robust overfitting* (perte de performance sévère sur données normales) est d'aller du "plus simple au plus complexe".

L'ordre optimal s'établit donc en 3 phases progressives :

1. **Phase 1 : Entraînement Normal (Benign First)**  
   *Pourquoi ?* Le modèle doit d'abord apprendre la vraie distribution des classes et extraire les features fondamentales sans aucun bruit cryptique.
2. **Phase 2 : Attaques au niveau des Features (Feature-level Adversarial)**  
   *Pourquoi ?* Ce sont des attaques discrètes, un peu plus bruitées qu'une donnée normale, mais qui ne détruisent pas la dimensionnalité temporelle. C'est le premier palier de perturbation.
3. **Phase 3 : Attaques au niveau des Séquences (Sequence-level PGD / FGSM)**  
   *Pourquoi ?* Ce sont les attaques en "boîte blanche" les plus violentes. Le modèle doit utiliser l'immunité acquise en Phase 2 pour ne pas s'effondrer financièrement ou mathématiquement face à un PGD.

---

## 2. Early Stopping Par Phase

Actuellement, l'early stopping (arrêt prématuré) agit sur l'intégralité du cycle (ex: patience globale = 10). L'amélioration consiste à **réinitialiser et isoler l'Early Stopping** pour chaque phase.

### Logique d'exécution :
- **Phase 1** : S'exécute pour un maximum de `N` epochs. Si `val_acc_clean` ne s'améliore pas pendant `X` epochs, la Phase 1 s'arrête prématurément. Le *meilleur poids* de la Phase 1 est chargé.
- **Phase 2** : Démarre à partir du meilleur poids de la Phase 1. S'arrête prématurément si `val_acc_feature_adv` ou une pondération (ex: 50% clean, 50% adv) ne s'améliore plus. Le *meilleur poids* de la Phase 2 est chargé.
- **Phase 3** : Démarre à partir du meilleur poids de Phase 2. S'arrête prématurément si `val_acc_seq_adv` / robuste ne s'améliore plus.

---

## 3. Matrice d'Évaluation Exhaustive Post-Phase

À la toute fin de **chaque phase**, le modèle doit subir un Crash-Test complet pour garantir qu'il n'a pas subi d’oubli catastrophique (*Catastrophic Forgetting*).

**Données de Tests Automatisées à injecter :**
- `Test_Loader_Normal` (Données pures)
- `Test_Loader_Adv_Feature` (Attaque sur les caractéristiques numériques)
- `Test_Loader_Adv_Seq_PGD` (Attaque PGD sur la série temporelle)
- `Test_Loader_Adv_Seq_FGSM` (Attaque FGSM rapide)

**Métriques exigées pour chaque Loader :**
- `Loss` (Cross Entropy)
- `Accuracy` (Précision globale)
- `Precision`, `Recall`, `F1-Score` (Macro-averaged, pour les classes déséquilibrées)
- `Robustness Ratio` (Précision Adv / Précision Normale)

---

## 4. Application Multi-Modèles

Ce même plan d'évolution sera obligatoirement orchestré à l'identique pour toutes vos architectures :
- **LSTM** : Sensibilité temporelle de base.
- **Transformer** : Sensibilité à l'attention vectorielle (souvent le plus résistant si bien régularisé).
- **CNN-LSTM** : Extraction spatio-temporelle.
- **XGBoost-LSTM** : Utilisation du surrogate PyTorch pour les phases 1-3, puis un fit final de the l'arbre XGBoost avant l'évaluation des métriques de Phase 3 !

---

## Résumé - Le workflow du script sera :

```text
POUR CHAQUE MODÈLE (LSTM, Transformer, etc.) :
  DÉBUT PHASE 1 (Normal)
      Boucle d'epochs + Early Stopping (patience=X)
      Test de validation croisée
  FIN PHASE 1 -> Crash Test (Normal, FeatAdv, SeqAdv) + Sauvegarde Checkpoint P1

  DÉBUT PHASE 2 (Feature Adv)
      Boucle d'epochs + Early Stopping (patience=X)
      Test de validation hybride
  FIN PHASE 2 -> Crash Test (Normal, FeatAdv, SeqAdv) + Sauvegarde Checkpoint P2

  DÉBUT PHASE 3 (Sequence Adv)
      Boucle d'epochs + Early Stopping (patience=X)
      Test de validation hybride profond
  FIN PHASE 3 -> (Si XGBoost -> Fit XGBoost ici) -> Crash Test Final 

  Génération du rapport JSON récapitulatif comparant P1, P2 et P3.
```
