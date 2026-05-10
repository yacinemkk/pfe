# Chapitre 8 — Conclusion et Perspectives

## 8.1 Synthèse des Contributions

Ce Projet de Fin d'Études a abordé le problème critique de l'**identification robuste de dispositifs IoT** dans des environnements réseau définis par logiciel (SDN). Face à la double contrainte de ne pouvoir utiliser que des statistiques de flux anonymisées (sans IP/MAC) et de devoir résister à des attaques adversariales sophistiquées, nous avons développé une solution complète et systématique.

### 8.1.1 Contribution 1 — Pipeline de Prétraitement Anti-Leakage

Le pipeline de prétraitement développé en itérations successives (Filtrage SDN → Sélection Hybride de Features → Normalisation StandardScaler) avec un **split temporel strict par appareil** constitue une base méthodologique solide pour l'identification IoT. La garantie anti-leakage — split chronologique 72/18/10 par appareil, génération de séquences séparée, fit des scalers uniquement sur le train — assure que les performances mesurées sont représentatives d'un déploiement réel.

La **sélection hybride par méthode du coude** combinant XGBoost (0.4), Chi² (0.3) et Information Mutuelle (0.3) réduit la dimensionnalité de ~30 features initiales à 15-20 features optimales tout en maximisant le pouvoir discriminant, rendant les modèles plus rapides et moins susceptibles au surapprentissage.

### 8.1.2 Contribution 2 — Architecture CNN-BiLSTM-Transformer

Le modèle **CNN-BiLSTM-Transformer hybride** — deux branches CNN parallèles (k=3, k=5) → BiLSTM bidirectionnel 2 couches → Transformer Encoder 2 couches → MeanPooling → FC — atteint des performances de référence de ~92-94% d'accuracy sur les données propres, surpassant les architectures individuelles (LSTM, BiLSTM, CNN-LSTM, Transformer). Cette architecture tire profit de la complémentarité entre extraction multi-échelle locale (CNN), modélisation séquentielle bidirectionnelle (BiLSTM), et attention globale (Transformer).

### 8.1.3 Contribution 3 — GreedyAttackSimulator avec 4 Stratégies

Le **GreedyAttackSimulator** applique 4 stratégies d'attaque sémantiquement valides (Zero, Mimic_Mean, Mimic_95th, Padding_x10) sur un nombre contrôlé de features (k_max progressif : 0 → 2 → 4). Contrairement aux perturbations L∞ génériques, ce simulateur respecte les contraintes physiques du protocole réseau et les contraintes du domaine IoT, offrant une évaluation de robustesse plus pertinente et réaliste.

### 8.1.4 Contribution 4 — Curriculum d'Entraînement Adversarial en 6 Phases

Le curriculum **Phase 0 → B1 → B2 → C → D1 → D2** implémenté dans `greedy_new_optimized.ipynb` constitue la contribution centrale de ce projet :

- **Phase 0** : Bootstrap propre (100% clean, k_max=0) — établit une base solide
- **Phase B1-B2** : Introduction progressive (40-50% adversarial, k_max=2) — adaptation douce à la robustesse
- **Phase C** : Robustesse forte (70% adversarial, k_max=4) — apprentissage contre attaques multiples
- **Phase D1-D2** : Consolidation maximale (85-95% adversarial, k_max=4) — fine-tuning pour robustesse extrême

Avec transitions automatiques basées sur des seuils de robustesse adaptatifs :
- `K_THRESHOLD = 0.85` : cible de robustesse pour progression
- `K_BACKSTEP = 0.80` : déclenchement du backstep automatique
- `K_RESUME_D2 = 0.83` : reprise après backstep avec hysteresis

Ce curriculum transforme un modèle avec RR(k=4)≈0.17 (Phase 0) en un modèle avec RR(k=4)≈0.75 (Phase D2), représentant une **amélioration de ×4.4 du Taux de Robustesse**.

### 8.1.5 Contribution 5 — Évaluation Complète avec Métriques de Robustesse

Le système d'évaluation complet mesure la robustesse progressive sous attaques de croissante intensité (k=1,2,3,4), avec :
- **Accuracy propre** : performance sur données non perturbées
- **Accuracy adversariale** : performance sous attaque pour chaque k
- **Taux de Robustesse (RR)** : ratio adv_acc/clean_acc, métrique clé pour l'évaluation
- **Crash Test** : protocole standardisé appliqué après chaque phase pour guider les transitions

Cette approche systématique d'évaluation permet le contrôle adaptatif et en temps réel des transitions de phase.

---

## 8.2 Limites et Points d'Amélioration

### 8.2.1 Absence de Cross-Validation

Les résultats obtenus dépendent d'un unique split temporel 70/10/20 par appareil. Sans cross-validation temporelle, la variance des performances n'est pas estimée — les scores pourraient varier de ±2-4 points selon le point de split choisi.

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

Ce projet démontre qu'il est possible de construire un système d'identification de dispositifs IoT à la fois **précis** (>90% d'accuracy sur données normales) et **robuste** (RR≈0.75 sous attaque de 4 features simultanées) en combinant des architectures de Deep Learning avancées (CNN-BiLSTM-Transformer) avec un curriculum d'entraînement adversarial en 6 phases soigneusement calibré.

La progression Phase 0 → B1 → B2 → C → D1 → D2 — de l'entraînement standard à la consolidation adversariale — illustre qu'un modèle robuste n'est pas simplement un modèle entraîné sur des données adversariales, mais le résultat d'un **processus graduel et structuré** où chaque phase construit sur les acquis de la précédente, avec transitions automatiques basées sur des seuils de robustesse adaptatifs.

La pipeline complète, implémentée dans le notebook Jupyter `greedy_new_optimized.ipynb`, orchestre l'ensemble du processus : du prétraitement anti-leakage à l'entraînement des 6 modèles en parallèle, avec gestion intelligente de la mémoire et sauvegardes sur Google Drive permettant la reprise en cas d'interruption.

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
