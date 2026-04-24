# Rapport de Fin d'Études — Table des Matières

**Titre :** Identification de Dispositifs IoT par Deep Learning et Robustesse Adversariale  
**Auteur :** [Votre Nom]  
**Encadrant :** [Nom de l'encadrant]  
**Établissement :** [Nom de l'établissement]  
**Année universitaire :** 2025–2026

---

## Plan du Rapport

| N° | Chapitre | Fichier | Pages estimées |
|----|----------|---------|---------------|
| 1 | Introduction et Contexte | `01_introduction.md` | 4–5 |
| 2 | État de l'Art | `02_etat_de_lart.md` | 6–8 |
| 3 | Datasets et Prétraitement des Données | `03_pretraitement.md` | 8–10 |
| 4 | Architectures des Modèles de Deep Learning | `04_architectures.md` | 8–10 |
| 5 | Attaques Adversariales sur les Flux IoT | `05_attaques_adversariales.md` | 6–7 |
| 6 | Curriculum d'Entraînement Antagoniste (4 Phases) | `06_entrainement_antagoniste.md` | 9–11 |
| 7 | Évaluation des Performances et Résultats | `07_evaluation_resultats.md` | 6–7 |
| 8 | Conclusion et Perspectives | `08_conclusion.md` | 3–4 |

**Total estimé :** 50–62 pages

---

## Résumé du Projet

Ce PFE porte sur le problème de l'**identification de dispositifs IoT** à partir de données de flux réseau au format IPFIX, dans un contexte de réseau défini par logiciel (SDN). L'approche proposée combine plusieurs architectures de Deep Learning — dont un modèle hybride **CNN-BiLSTM-Transformer** — avec un **curriculum d'entraînement antagoniste en 4 phases** permettant de renforcer la robustesse des modèles face aux attaques adversariales. Le système intègre un **discriminateur** (LSTM bidirectionnel) pour détecter les flux malveillants et router les paquets suspects vers le modèle robuste.

**Mots-clés :** IoT, Identification d'Appareils, IPFIX, SDN, Deep Learning, CNN, BiLSTM, Transformer, Adversarial Training, Robustesse, GreedyAttack, AFDLoss, Curriculum Learning.

---

## Liste des Figures

- Figure 1 : Architecture générale du pipeline (données → modèle → évaluation)
- Figure 2 : Pipeline de prétraitement CSV en 4 étapes
- Figure 3 : Pipeline de prétraitement JSON en 4 étapes
- Figure 4 : Méthode du coude (Elbow Method) pour la sélection de features
- Figure 5 : Architecture CNN-BiLSTM-Transformer
- Figure 6 : Curriculum d'entraînement en 4 phases (A, B, C, D)
- Figure 7 : Architecture du Discriminateur BiLSTM
- Figure 8 : Architecture du Routeur IoT (IoTRouter)
- Figure 9 : Comparaison des performances des 6 modèles (propre vs adversarial)

## Liste des Tableaux

- Tableau 1 : Caractéristiques des deux datasets (CSV IPFIX, JSON IPFIX)
- Tableau 2 : Features conservées après filtrage SDN (CSV)
- Tableau 3 : Features conservées après filtrage SDN (JSON)
- Tableau 4 : Comparaison des 6 architectures (paramètres, profondeur, mécanisme)
- Tableau 5 : Hyperparamètres du modèle CNN-BiLSTM-Transformer
- Tableau 6 : Stratégies d'attaques Greedy par feature
- Tableau 7 : Configuration des 4 phases d'entraînement
- Tableau 8 : Résultats du Crash Test par modèle et par phase
- Tableau 9 : Taux de Robustesse (RR) avant et après entraînement antagoniste
