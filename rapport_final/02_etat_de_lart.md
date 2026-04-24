# Chapitre 2 — État de l'Art

## 2.1 Identification de Dispositifs IoT : Historique et Approches

### 2.1.1 Approches Basées sur les Signatures

Les premières méthodes d'identification des dispositifs IoT reposaient sur des **signatures statiques** : empreintes de systèmes d'exploitation (OS fingerprinting), identificateurs de fabricant dans les adresses MAC (OUI — Organizationally Unique Identifier), ou comportements de communication spécifiques à chaque constructeur. Ces approches présentent plusieurs limites critiques :

- **Fragiles face aux mises à jour firmware :** un changement de comportement réseau suite à une mise à jour invalide la signature.
- **Contournables facilement :** un attaquant peut usurper une adresse MAC ou simuler un OS différent.
- **Non généralisables :** elles nécessitent une maintenance manuelle constante d'une base de données de signatures.

### 2.1.2 Approches par Machine Learning Classique

Face aux limites des signatures statiques, des méthodes d'apprentissage automatique ont été proposées. Les travaux précurseurs de Sivanathan et al. (2019), Miettinen et al. (2017) et d'Ortiz et al. ont établi que des modèles comme les forêts aléatoires (*Random Forest*), les SVM ou XGBoost, entraînés sur des statistiques de flux réseau (débit moyen, durée de flow, taille des paquets, inter-arrivées), pouvaient atteindre des taux d'identification supérieurs à 90% sur des datasets contrôlés.

Cependant, ces approches traitent chaque **flux individuellement** et ne capturent pas les **dynamiques temporelles** du comportement IoT. Or, un appareil IoT présente des motifs temporels caractéristiques — par exemple, une caméra de surveillance envoie des rafales de paquets suivies de périodes de silence, ou un thermostat contacte un serveur cloud à intervalles réguliers. Ces informations séquentielles sont perdues lorsqu'on analyse les flux de manière isolée.

---

## 2.2 Réseaux de Neurones pour la Classification de Trafic

### 2.2.1 Réseaux LSTM pour les Séquences Temporelles

Les **Long Short-Term Memory (LSTM)** (Hochreiter & Schmidhuber, 1997) ont révolutionné le traitement des séquences temporelles. Leur mécanisme de cellule de mémoire avec trois portes (oubli, entrée, sortie) leur permet de capturer des dépendances à longue portée sans souffrir du problème de gradient évanescent qui affecte les RNN simples. Dans le contexte de l'identification IoT, plusieurs travaux ont montré que les LSTM, entraînés sur des séquences de flux consécutifs issus du même appareil, améliorent significativement les performances par rapport aux approches flux-par-flux.

Le **BiLSTM (LSTM Bidirectionnel)** étend ce principe en parcourant la séquence dans les deux sens (avant et arrière), obtenant une représentation contextuelle plus riche pour chaque instant de la séquence.

### 2.2.2 Réseaux CNN pour l'Extraction de Motifs Locaux

Les **réseaux de neurones convolutifs 1D (CNN 1D)** sont particulièrement efficaces pour détecter des motifs locaux dans des séquences. Dans une séquence de flux réseau, certains patterns locaux — par exemple, deux flux successifs présentant des tailles très différentes — sont caractéristiques d'un type d'appareil. Le CNN 1D extrait ces motifs via ses **filtres convolutifs** (noyaux de taille fixe), captures ensuite compressées par le *Max Pooling*.

L'architecture **CNN-LSTM** combine les forces des deux approches : le CNN extrait d'abord les motifs locaux de la séquence, puis le LSTM modélise les dépendances temporelles entre ces motifs.

### 2.2.3 Le Mécanisme d'Attention et les Transformers

Les **Transformers** (Vaswani et al., 2017), initialement développés pour le traitement du langage naturel (NLP), ont montré une efficacité remarquable pour capturer des **dépendances globales** sur des séquences longues. Leur mécanisme d'**attention multi-têtes** (*Multi-Head Self-Attention*) calcule des scores de similarité entre tous les paires de positions dans la séquence, permettant de relier directement des événements éloignés dans le temps sans passer par un mécanisme récurrent.

Dans le contexte IoT, ce mécanisme permet au modèle d'apprendre que, par exemple, le flux numéro 1 d'une séquence et le flux numéro 10 sont fortement corrélés pour identifier un type d'appareil particulier — une relation que le LSTM aurait du mal à maintenir sur 10 pas de temps.

### 2.2.4 Modèles Hybrides : Vers l'Architecture Ultime

Plusieurs travaux récents proposent des architectures hybrides combinant CNN, LSTM/BiLSTM et Transformer. L'idée fondamentale est que ces trois composants sont **complémentaires** :
- Le **CNN** est expert en motifs locaux (haute résolution spatiale)
- Le **BiLSTM** capture les dépendances temporelles à moyen terme de manière séquentielle
- Le **Transformer** capture les dépendances globales via l'attention

Des architectures comme **TC-Net**, **TrCNN** ou **Conformer** explorent ces combinaisons dans différents domaines (reconnaissance vocale, classification de trafic réseau). Notre projet s'inscrit dans cette lignée avec le modèle **CNN-BiLSTM-Transformer**.

---

## 2.3 Attaques Adversariales sur les Réseaux de Neurones

### 2.3.1 Fondements Théoriques

Les attaques adversariales ont été formalisées pour la première fois par Szegedy et al. (2013) et Goodfellow et al. (2014). L'idée centrale est la suivante : pour un exemple d'entrée `x` correctement classifié par un modèle `f` en classe `y`, il existe un exemple perturbé `x' = x + δ` (avec `||δ|| < ε` pour une norme donnée et un budget `ε` petit) tel que `f(x') ≠ y`. Ces perturbations sont souvent imperceptibles pour un humain mais trompent systématiquement le modèle.

Dans le domaine du trafic réseau, les perturbations adversariales doivent respecter des **contraintes sémantiques** supplémentaires : on ne peut pas modifier arbitrairement n'importe quelle feature. Par exemple, dans le contexte SDN/IPFIX, le protocole réseau (`ipProto` ou `protocolIdentifier`) est imposé par la pile réseau et ne peut pas être modifié sans changer fondamentalement la nature du flux. Les directions de paquets (`pkt_dir_*`) sont des données binaires imposées par le protocole.

### 2.3.2 Types d'Attaques dans la Littérature

**FGSM (Fast Gradient Sign Method)** (Goodfellow et al., 2014) : méthode en une seule étape qui perturbe l'entrée dans la direction du gradient de la loss par rapport à l'entrée :
```
x' = x + ε · sign(∇_x L(f(x), y))
```
Simple et rapide, mais relativement peu puissante face à des modèles robustifiés.

**PGD (Projected Gradient Descent)** (Madry et al., 2018) : généralisation itérative du FGSM avec projection du résultat dans la boule L∞ autour de l'entrée originale :
```
x'_(t+1) = Π_{x+S}(x'_t + α · sign(∇_{x'} L(f(x'_t), y)))
```
Considérée comme l'attaque "universelle" de référence dans la communauté.

**Attaques Feature-Level** : spécifiques au domaine réseau/IoT, ces attaques ne perturbent pas les entrées de manière générique mais simulent des comportements réseau réalistes (zéro-trafic, imitation des statistiques d'une autre classe, amplification de certaines features). Ces attaques sont physiquement réalisables par un adversaire contrôlant un équipement réseau.

### 2.3.3 Méthodes de Défense dans la Littérature

**Adversarial Training** (Madry et al., 2018) : la défense la plus efficace consiste à inclure des exemples adversariaux dans les données d'entraînement. Le modèle apprend ainsi à reconnaître correctement non seulement les exemples normaux mais aussi leurs versions perturbées.

**TRADES (Zhang et al., 2019)** : formalise l'adversarial training comme un problème d'optimisation avec un terme de régularisation KL-divergence :
```
L = L_CE(f(x), y) + β · KL(f(x) || f(x'))
```
Le paramètre `β` contrôle le compromis entre précision sur données propres et robustesse adversariale.

**Feature Dropout** : technique de régularisation inspirée du Dropout standard, appliquée au niveau des features d'entrée plutôt qu'aux activations internes, forçant le modèle à ne pas dépendre excessivement d'un sous-ensemble de features vulnérables.

**Input Defense / Feature Squeezing** : preprocessing défensif appliqué en entrée du modèle pour réduire l'impact des perturbations adversariales (clipping, lissage temporel, débruitage).

---

## 2.4 SDN et Identification IoT dans ce Contexte

### 2.4.1 Le Contexte SDN

Les réseaux SDN (*Software-Defined Networking*) séparent le plan de contrôle (les décisions de routage) du plan de données (la transmission effective des paquets). Un contrôleur SDN centralisé (OpenFlow, OpenDaylight, ONOS) reçoit des statistiques agrégées de flux — au format IPFIX ou NetFlow — de la part des équipements réseau (switches, routeurs).

Ces statistiques **ne contiennent pas les contenus des paquets** (confidentialité préservée) mais uniquement des métas-données : adresses source/destination, protocole, timestamps, compteurs de paquets et d'octets, statistiques d'inter-arrivée. C'est sur ces données anonymisées que l'identification IoT doit opérer.

### 2.4.2 Contrainte SDN : Features Non Modifiables

Dans le contexte SDN, certaines features IPFIX sont **physiquement non modifiables** par un attaquant, car elles sont imposées par la stack réseau :
- **`ipProto` (CSV)** et **`protocolIdentifier` (JSON)** : le protocole réseau (TCP=6, UDP=17) est imposé par l'application.
- **`pkt_dir_0` à `pkt_dir_7` (JSON)** : 8 bits binaires encodant la direction des 8 premiers paquets du flux — information imposée par la séquence de handshake TCP/UDP.

Ces contraintes guident la conception des attaques adversariales réalistes dans ce projet.

---

## 2.5 Positionnement de Ce Travail

Par rapport à l'état de l'art, ce projet se distingue par :

1. **L'utilisation d'un GreedyAttackSimulator guidé par analyse de sensibilité**, qui identifie dynamiquement (post-Phase A) quelles features sont les plus vulnérables pour chaque modèle specific, puis construit des attaques ciblées. C'est plus réaliste que des perturbations génériques L∞.

2. **Un curriculum d'entraînement en 4 phases progressives** (A→B→C→D) avec des hyperparamètres finement calibrés pour éviter le problème classique d'effondrement de l'accuracy propre lors de l'entraînement antagoniste.

3. **Un mécanisme de discrimination** orthogonal au classifieur principal : un Discriminateur BiLSTM entraîné séparément pour détecter si un flux a été perturbé, combiné avec un routeur qui choisit dynamiquement le modèle le plus approprié.

4. **L'AFDLoss (Adversarial Feature Defense Loss)** — une fonction de perte custom qui découple les représentations internes des exemples propres et adversariaux, forçant le modèle à maintenir des centres de classes distincts pour les deux types d'entrées.
