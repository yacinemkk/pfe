# Chapitre 1 — Introduction et Contexte

## 1.1 Contexte Général : L'Explosion de l'IoT

Le paysage numérique contemporain est marqué par une croissance exponentielle du nombre de dispositifs connectés à Internet. Sous l'appellation **Internet des Objets (IoT, Internet of Things)**, on regroupe un ensemble hétérogène d'équipements électroniques — caméras de surveillance, thermostats intelligents, ampoules connectées, assistants vocaux, serrures numériques, stations météo domestiques — dont la caractéristique commune est leur capacité à communiquer via un réseau IP. Selon les estimations récentes, le nombre de dispositifs IoT actifs dans le monde dépasse déjà les **15 milliards d'unités** en 2025, avec des projections qui annoncent plus de 25 milliards d'ici 2030.

Cette prolifération soulève des défis de sécurité sans précédent. Contrairement aux ordinateurs personnels et serveurs traditionnels, les dispositifs IoT présentent un profil de vulnérabilité particulier :

- **Ressources limitées :** la majorité des appareils IoT disposent de mémoire RAM, puissance de calcul et autonomie énergétique très contraintes, rendant l'installation d'agents de sécurité sophistiqués impossible directement sur l'équipement.
- **Hétérogénéité :** un réseau domestique ou industriel peut contenir des dizaines de types d'appareils différents, de fabricants différents, utilisant des protocoles propriétaires variés.
- **Durée de vie longue :** les appareils IoT restent souvent déployés pendant de nombreuses années sans mises à jour de firmware, accumulant ainsi des vulnérabilités non corrigées.
- **Absence de mécanismes d'authentification forte :** beaucoup de dispositifs s'appuient sur des mots de passe par défaut ou des communications non chiffrées.

Ces caractéristiques font des réseaux IoT des cibles de choix pour des attaques de type botnet (comme Mirai), des usurpations d'identité (spoofing d'adresse MAC/IP), ou des attaques par déni de service distribué (DDoS).

---

## 1.2 Problématique : Identifier un Appareil Sans Connaître son Identité

Dans ce contexte, la question de l'**identification des dispositifs IoT** — c'est-à-dire la capacité à déterminer avec précision le **type** d'un appareil (caméra, thermostat, ampoule...) à partir de son comportement réseau — devient un enjeu de sécurité critique. Cette capacité permet à un administrateur réseau de :

1. **Détecter les imposteurs :** si un appareil prétend être une ampoule connectée mais génère du trafic identique à celui d'un serveur de commande et contrôle (C&C), il est probablement compromis.
2. **Appliquer des politiques de sécurité granulaires :** différents appareils nécessitent des règles de pare-feu différentes — une caméra n'a pas besoin d'accéder à des serveurs de messagerie.
3. **Détecter des appareils non autorisés :** tout équipement inconnu du réseau est potentiellement une intrusion.

Cependant, cette identification se heurte à une contrainte fondamentale dans les réseaux **SDN (Software-Defined Networking)** modernes : **les adresses IP et MAC ne sont pas des identifiants fiables**. Les appareils peuvent changer d'adresse IP dynamiquement (DHCP), usurper une adresse MAC, ou être remplacés physiquement sans changer leur identifiant logique. Il est donc impératif d'identifier les appareils à partir de **leur comportement réseau intrinsèque** — c'est-à-dire les statistiques de flux (durées, tailles de paquets, inter-arrivées, etc.) — sans s'appuyer sur des informations d'identification qui peuvent être falsifiées.

### 1.2.1 La Menace Adversariale

Une seconde couche de complexité s'ajoute : un attaquant sophistiqué peut délibérément **modifier le comportement réseau de son appareil malveillant** pour imiter le profil d'un appareil légitime connu. Ces attaques, appelées **attaques adversariales**, consistent à perturber minutieusement les statistiques de flux (par exemple, ajuster légèrement les délais inter-paquets, modifier la taille des paquets) de manière à tromper un modèle de classification. Des études récentes montrent que des modèles de Deep Learning ayant atteint plus de 95% d'accuracy sur des données normales peuvent chuter à moins de 20% sous une attaque adversariale bien construite.

Cette vulnérabilité est particulièrement préoccupante dans les infrastructures critiques où les systèmes d'identification IoT jouent un rôle de garde-barrière.

---

## 1.3 Objectifs du Projet de Fin d'Études

Ce PFE vise à répondre à la problématique suivante :

> **Est-il possible de construire un système d'identification de dispositifs IoT à partir de flux réseau IPFIX, capable de maintenir une haute précision même en présence d'attaques adversariales ?**

Pour répondre à cette question, les objectifs spécifiques du projet sont :

1. **Construire un pipeline de prétraitement robuste** pour deux datasets IPFIX (format CSV et JSON), garantissant une absence totale de fuite de données (*data leakage*) entre les ensembles d'entraînement et de test.

2. **Comparer six architectures de Deep Learning** — LSTM, BiLSTM, CNN-LSTM, XGBoost-LSTM, Transformer, et le modèle hybride CNN-BiLSTM-Transformer — sur la tâche de classification multiclasse des appareils IoT.

3. **Démontrer la vulnérabilité** des modèles naïvement entraînés face aux attaques adversariales via un protocole de *Crash Test*.

4. **Concevoir et implémenter un curriculum d'entraînement antagoniste en 4 phases** (A : Fondation, B : Robustesse Douce, C : Robustesse Forte, D : Consolidation) qui renforce progressivement la robustesse des modèles, en utilisant le **GreedyAttackSimulator** — un simulateur d'attaques guidé par l'analyse de sensibilité des features.

5. **Déployer un système de détection et de routage** basé sur un Discriminateur BiLSTM qui identifie les flux adversariaux et les redirige vers le modèle robuste.

---

## 1.4 Datasets Utilisés

Deux datasets IPFIX indépendants sont utilisés dans ce projet, chacun représentant un réseau domestique instrumenté avec des appareils IoT réels :

### Dataset 1 — IPFIX ML Instances (Format CSV)
- **Source :** Capture de trafic réseau dans un environnement domestique
- **Format :** Fichiers CSV (un par session de capture, nommés `home*_labeled.csv`)
- **Classes :** 18 types d'appareils IoT (Eclear, Amazon Echo Dot, Microsoft Atom Cam, Fire TV Stick 4K, Google Nest Hub, Philips Hue, etc.)
- **Features initiales :** ~30 colonnes de statistiques de flux IPFIX
- **Volume :** plusieurs centaines de milliers de flux

### Dataset 2 — IPFIX Records (Format JSON)
- **Source :** Capture IPFIX dans un autre environnement domestique
- **Format :** Fichier JSON unique de grande taille (> 1 Go)
- **Classes :** 17 types d'appareils IoT (Qrio Hub, Philips Hue Bridge, Amazon Echo, Wansview Camera, etc.)
- **Features initiales :** 28 features continues + 8 features binaires de direction de paquets (`pkt_dir_0` à `pkt_dir_7`)
- **Spécificité :** format IPFIX Record de bas niveau, nécessitant un préprocesseur adapté

Ces deux datasets sont traités de manière **entièrement indépendante** (deux pipelines parallèles), reflétant la réalité d'un déploiement sur différents types d'infrastructures réseau.

---

## 1.5 Contributions du Projet

Les principales contributions de ce travail sont :

- **Un pipeline de prétraitement anti-leakage complet**, incluant filtrage SDN, sélection hybride de features (XGBoost + Chi² + Information Mutuelle + méthode du coude), et normalisation StandardScaler avec split temporel strict par appareil.

- **Un GreedyAttackSimulator** guidé par l'analyse de sensibilité des features, qui identifie dynamiquement les features les plus vulnérables après la Phase A et construit des attaques ciblées (Zero, Mimic_Mean, Mimic_95th, Padding_x10) en respectant les contraintes sémantiques du domaine IoT.

- **Un curriculum d'entraînement antagoniste en 4 phases** avec des mécanismes de défense complémentaires : AFDLoss (décorrélation des features), Feature Dropout, ajout de bruit gaussien, et une stratégie de checkpointing pondérée (score = 0.4 × clean_acc + 0.6 × adv_acc) favorisant la robustesse sans sacrifier les performances propres.

- **Un système de discrimination et de routage** combinant un Discriminateur BiLSTM (détection binaire : flux propre vs adversarial) avec un routeur IoTRouter qui dirige chaque flux vers le modèle approprié selon un seuil calibré automatiquement.

---

## 1.6 Organisation du Rapport

Le reste du rapport est structuré comme suit :
- **Chapitre 2** présente l'état de l'art en identification IoT et en robustesse adversariale.
- **Chapitre 3** détaille les deux datasets et le pipeline de prétraitement complet avec les justifications de chaque étape.
- **Chapitre 4** décrit les six architectures de Deep Learning implémentées.
- **Chapitre 5** présente le modèle d'attaques adversariales utilisé dans ce projet.
- **Chapitre 6** expose le curriculum d'entraînement antagoniste en 4 phases et ses mécanismes de défense.
- **Chapitre 7** présente le protocole d'évaluation et analyse les résultats obtenus.
- **Chapitre 8** conclut le rapport et ouvre des perspectives de recherche.
