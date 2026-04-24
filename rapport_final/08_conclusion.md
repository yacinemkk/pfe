# Chapitre 8 — Conclusion et Perspectives

## 8.1 Synthèse des Contributions

Ce Projet de Fin d'Études a abordé le problème critique de l'**identification robuste de dispositifs IoT** dans des environnements réseau définis par logiciel (SDN). Face à la double contrainte de ne pouvoir utiliser que des statistiques de flux anonymisées (sans IP/MAC) et de devoir résister à des attaques adversariales sophistiquées, nous avons développé une solution complète et systématique.

### 8.1.1 Contribution 1 — Pipeline de Prétraitement Anti-Leakage

Le pipeline de prétraitement développé en itérations successives (Filtrage SDN → Sélection Hybride de Features → Normalisation StandardScaler) avec un **split temporel strict par appareil** constitue une base méthodologique solide pour l'identification IoT. La garantie anti-leakage — split chronologique 72/18/10 par appareil, génération de séquences séparée, fit des scalers uniquement sur le train — assure que les performances mesurées sont représentatives d'un déploiement réel.

La **sélection hybride par méthode du coude** combinant XGBoost (0.4), Chi² (0.3) et Information Mutuelle (0.3) réduit la dimensionnalité de ~30 features initiales à 15-20 features optimales tout en maximisant le pouvoir discriminant, rendant les modèles plus rapides et moins susceptibles au surapprentissage.

### 8.1.2 Contribution 2 — Architecture CNN-BiLSTM-Transformer

Le modèle **CNN-BiLSTM-Transformer hybride** — deux branches CNN parallèles (k=3, k=5) → BiLSTM bidirectionnel 2 couches → Transformer Encoder 2 couches → MeanPooling → FC — atteint des performances de référence de ~92-94% d'accuracy sur les données propres, surpassant les architectures individuelles (LSTM, BiLSTM, CNN-LSTM, Transformer). Cette architecture tire profit de la complémentarité entre extraction multi-échelle locale (CNN), modélisation séquentielle bidirectionnelle (BiLSTM), et attention globale (Transformer).

### 8.1.3 Contribution 3 — GreedyAttackSimulator Guidé par Sensibilité

Le **GreedyAttackSimulator**, construit à partir d'une analyse de sensibilité post-Phase A, constitue un modèle d'attaque réaliste pour le domaine IoT/réseau. En ciblant spécifiquement les features les plus vulnérables pour chaque modèle avec des stratégies sémantiquement valides (Zero, Mimic_Mean, Mimic_95th, Padding_x10), et en respectant les contraintes physiques du protocole (features non modifiables), ce simulateur offre une évaluation de robustesse plus pertinente que les perturbations L∞ génériques comme FGSM ou PGD.

### 8.1.4 Contribution 4 — Curriculum d'Entraînement Antagoniste en 4 Phases

Le curriculum **A→B→C→D** avec escalade progressive de la difficulté (de 0% à 85% adversarial, de k=0 à k=4 features) et des mécanismes de défense complémentaires est la contribution centrale de ce projet :

- **AFDLoss** (Phases B et C) — force une structure géométrique robuste des représentations
- **Feature Dropout** (Phases B, C, D) — empêche la sur-dépendance aux features vulnérables
- **Bruit Gaussien** (Phases B, C, D) — améliore la robustesse générale
- **Label Smoothing progressif** (5% → 8% → 10%) — régularisation adaptative
- **Checkpointing pondéré** (0.4×clean + 0.6×adv) — sélection orientée robustesse

Ce curriculum transforme un modèle avec RR≈0.17 (Phase A, k=4) en un modèle avec RR≈0.76 (Phase D, k=4), représentant une **amélioration de ×4.5 du Taux de Robustesse**.

### 8.1.5 Contribution 5 — Système de Discrimination et Routage

Le **Discriminateur BiLSTM** (~92% d'accuracy de détection) combiné avec le **Routeur IoT** (calibrage automatique à ≥95% recall d'attaque) constitue une première ligne de défense orthogonale au classifieur principal. Ce système two-path — Modèle Normal pour le trafic propre, Modèle Robuste pour le trafic adversarial — maximise la performance dans les deux scénarios et peut fonctionner en déploiement réel sans connaissance a priori du type de trafic entrant.

---

## 8.2 Limites et Points d'Amélioration

### 8.2.1 Absence de Cross-Validation

Les résultats obtenus dépendent d'un unique split 80/20. Sans cross-validation, la variance des performances n'est pas estimée — les scores pourraient varier de ±3-5 points selon le split temporel choisi. Une k-fold temporal cross-validation donnerait des estimations plus fiables.

### 8.2.2 Évaluation sur Données Synthétiques

Les attaques testées (GreedyAttackSimulator) sont des approximations de comportements adversariaux réels. Des attaques réalisées par de véritables appareils IoT modifiés (attaques physiques) pourraient différer de ces modèles.

### 8.2.3 Absence d'Évaluation Open-Set

Le modèle ne peut classifier que les 17-18 types d'appareils vus à l'entraînement. Dans un déploiement réel, de nouveaux appareils inconnus apparaîtraient régulièrement. Un mécanisme de **rejet open-set** (ex. : seuil sur la confiance softmax `max(P) < 0.5 → "appareil inconnu"`) est nécessaire mais non implémenté.

### 8.2.4 Généralisation Inter-Datasets

Les deux datasets (CSV et JSON) représentent des environnements réseau spécifiques. La robustesse du modèle sur un dataset non vu (transfer learning inter-datasets) n'a pas été évaluée.

---

## 8.3 Perspectives de Recherche

### 8.3.1 Identification en Ligne (Online Learning)

Les comportements IoT évoluent dans le temps (mises à jour firmware, changements de patterns d'utilisation). Un système d'identification en production devrait incorporer un mécanisme d'**apprentissage en ligne** qui met à jour continuellement le modèle sur les nouveaux flux observés sans nécessiter de ré-entraînement complet.

### 8.3.2 Déploiement SDN Réel avec OpenFlow

L'intégration du système d'identification dans un contrôleur SDN réel (par exemple, ONOS ou OpenDaylight) permettrait de tester les performances dans un environnement de production. Les statistiques de flux IPFIX seraient directement récupérées via l'API REST du contrôleur, sans modification de l'infrastructure réseau.

### 8.3.3 Apprentissage Fédéré pour la Confidentialité

Le partage de données entre réseaux IoT de différentes organisations est sensible (données comportementales propriétaires). Un cadre d'**apprentissage fédéré** permettrait d'entraîner un modèle global à partir de données distribuées sans jamais centraliser les flux réseau, préservant la confidentialité des opérateurs.

### 8.3.4 Attaques Adaptatifs et Robustesse Certifiée

Les attaques futures pourraient s'adapter spécifiquement aux défenses implémentées (attaques conscientes de l'InputDefenseLayer, ou de l'AFDLoss). Des techniques de **robustesse certifiée** (Interval Bound Propagation, Randomized Smoothing avec certificats) permettraient de garantir formellement la robustesse jusqu'à un certain budget d'attaque `ε`, indépendamment de l'attaque choisie.

### 8.3.5 Modèles de Fondation pour l'IoT

Avec l'essor des grands modèles de langage (LLMs), une avenue prometteuse est l'entraînement d'un **modèle de fondation IoT** pré-entraîné sur des millions de flux de très nombreux types d'appareils, puis fine-tuné pour des tâches spécifiques (identification, détection d'anomalie, forensics réseau). La tokenisation BPE développée dans ce projet (IoT-Tokenize) constitue un point de départ naturel pour cette approche.

---

## 8.4 Conclusion Générale

Ce projet démontre qu'il est possible de construire un système d'identification de dispositifs IoT à la fois **précis** (>90% d'accuracy sur données normales) et **robuste** (RR≈0.76 sous attaque de 4 features simultanées) en combinant des techniques d'apprentissage profond avancées avec un curriculum d'entraînement antagoniste soigneusement calibré.

La progression A→B→C→D — de l'entraînement standard à la consolidation antagoniste — illustre qu'un modèle robuste n'est pas simplement un modèle entraîné sur des données adversariales, mais le résultat d'un **processus graduel et structuré** où chaque phase construit sur les acquis de la précédente, avec des mécanismes de défense adaptés à chaque niveau de difficulté.

Le système de routage (Discriminateur + Routeur) ajoute une couche de défense orthogonale qui améliore encore les performances globales (+50 points sur les flux adversariaux) et constitue une architecture deployable dans les infrastructures SDN modernes.

Les résultats obtenus constituent une contribution significative à la sécurité des réseaux IoT et ouvrent la voie à des déploiements réels dans des environnements réseau productifs où la menace adversariale est croissante.

---

## Références Bibliographiques

1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*, 9(8), 1735-1780.
2. Vaswani, A., et al. (2017). Attention is all you need. *Advances in neural information processing systems*, 30.
3. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and harnessing adversarial examples. *arXiv preprint arXiv:1412.6572*.
4. Madry, A., et al. (2018). Towards deep learning models resistant to adversarial attacks. *International Conference on Learning Representations*.
5. Zhang, H., et al. (2019). Theoretically principled trade-off between robustness and accuracy. *International Conference on Machine Learning*.

7. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. *IEEE International Conference on Data Mining*.
8. Breunig, M. M., et al. (2000). LOF: identifying density-based local outliers. *ACM SIGMOD Record*, 29(2), 93-104.
9. Sivanathan, A., et al. (2019). Classifying IoT devices in smart environments using network traffic characteristics. *IEEE Transactions on Mobile Computing*, 18(8), 1745-1759.
10. Miettinen, M., et al. (2017). IoT sentinel: Automated device-type identification for security enforcement in IoT. *IEEE International Conference on Distributed Computing Systems*.
