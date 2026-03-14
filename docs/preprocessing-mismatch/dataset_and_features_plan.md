# Plan d'Action pour Résoudre les Confusions du Dataset IoT

Ce document détaille le plan exhaustif mis en œuvre pour nettoyer les données brutes (JSON) et préparer les **36 attributs essentiels** de notre modèle sans introduire de fuite d'information (*Data Leakage*).

---

## 🗺️ Le Plan en 4 Étapes

Pour corriger les trois failles critiques du pipeline de prétraitement, le code doit appliquer les 4 étapes suivantes de manière séquentielle :

### 1. Séparation stricte des pipelines (CSV vs JSON)
*   **Action :** Ne plus jamais tenter de joindre le CSV expérimental (IoT IPFIX Home) avec le JSON (IPFIX Records). 
*   **Solution :** Les JSON sont traités de manière 100% native (ligne par ligne) par un script dédié (`src/data/json_preprocessor.py`). Ce script remplace `preprocessor.py` pour le pipeline JSON (les deux coexistent via le paramètre `PIPELINE_MODE` dans `config/config.py`).

### 2. Nouveau système d'étiquetage bidirectionnel
*   **Action :** Arrêter de jeter plus de 50% des flux réseaux.
*   **Solution :** Lors de la lecture d'une ligne JSON, vérifier **simultanément** si `sourceMacAddress` OU `destinationMacAddress` correspond à une adresse de notre dictionnaire de 17 appareils IoT (en ignorant le DEFAULT GATEWAY). L'appareil IoT devient le label du flux global, qu'il soit l'émetteur ou le destinataire.

### 3. Curation anti-Data Leakage (Le plus important)
*   **Action :** Vider le dataset de tout identifiant réseau pour forcer l'apprentissage temporel comportemental.
*   **Solution :** Immédiatement après avoir posé l'étiquette (label) en utilisant l'adresse MAC, le système doit **détruire (drop)** les **26 colonnes interdites** : `sourceMacAddress`, `destinationMacAddress`, `sourceIPv4Address`, `destinationIPv4Address`, `sourceTransportPort`, `destinationTransportPort`, `tcpSequenceNumber`, `reverseTcpSequenceNumber`, `collectorName`, `observationDomainId`, `vlanId`, `ingressInterface`, `egressInterface`, `flowAttributes`, `reverseFlowAttributes`, `flowStartMilliseconds`, `flowEndMilliseconds`, `flowEndReason`, `silkAppLabel`, `ipClassOfService`, `active_timeout`, `idle_timeout`, `initialTCPFlags`, `unionTCPFlags`, `reverseInitialTCPFlags`, `reverseUnionTCPFlags`. Sans cette suppression immédiate, le modèle apprendra par cœur les IP et obtiendra 100% de précision sans jamais évaluer le comportement de l'appareil.

### 4. Décodage de la "Signature Binaire"
*   **Action :** Rendre les séquences de direction réseau manipulables par les modèles (LSTM/Transformer) et surtout par les attaquants Deep Learning.
*   **Solution :** Convertir l'attribut hexadécimal `firstEightNonEmptyPacketDirections` (exemple: "0e") en une liste de 8 variables binaires (0 ou 1). Ces attributs sont stockés dans 8 nouvelles colonnes (`pkt_dir_0` jusqu'à `pkt_dir_7`).

---

## 🧬 Les 36 Attributs Essentiels pour l'Entraînement

Une fois les identifiants réseaux supprimés par l'étape 3 du plan, voici les **36 features statistiques** exactes qui forment l'empreinte comportementale propre et légitime de chaque appareil. Vous remarquerez qu'il y a 28 features continues (nombres réels) et 8 features discrètes (0 ou 1).

### ⏱ 1. Le Rythme (Attributs Temporels - Continus)
1. `flowDurationMilliseconds`
2. `reverseFlowDeltaMilliseconds`
3. `averageInterarrivalTime`
4. `standardDeviationInterarrivalTime`
5. `reverseAverageInterarrivalTime`
6. `reverseStandardDeviationInterarrivalTime`

### 📦 2. Le Poids (Volume & Bande Passante - Continus)
7. `packetTotalCount`
8. `octetTotalCount`
9. `reversePacketTotalCount`
10. `reverseOctetTotalCount`
11. `dataByteCount`
12. `reverseDataByteCount`

### 📐 3. La Forme (Profilage des Paquets - Continus)
13. `smallPacketCount`
14. `largePacketCount`
15. `nonEmptyPacketCount`
16. `firstNonEmptyPacketSize`
17. `maxPacketSize`
18. `standardDeviationPayloadLength`
19. `bytesPerPacket`
20. `reverseSmallPacketCount`
21. `reverseLargePacketCount`
22. `reverseNonEmptyPacketCount`
23. `reverseFirstNonEmptyPacketSize`
24. `reverseMaxPacketSize`
25. `reverseStandardDeviationPayloadLength`

### 🗣 4. La Langue (Protocoles et TCP flags - Continus)
26. `protocolIdentifier`
27. `tcpUrgTotalCount`
28. `reverseTcpUrgTotalCount`

### 🤝 5. L'Échange Initial (Séquence Binaire - Discrets)
29. `pkt_dir_0`
30. `pkt_dir_1`
31. `pkt_dir_2`
32. `pkt_dir_3`
33. `pkt_dir_4`
34. `pkt_dir_5`
35. `pkt_dir_6`
36. `pkt_dir_7`

---

## ⚠️ Règle de Prétraitement et d'Adversarial Training

**La séparation Type Continu vs Type Discret est fondamentale.** 

Durant le prétraitement par StandardScaler, l'entraînement, et l'application des attaques PGD/FGSM :
1. **Les 28 attributs continus (1 à 28)** doivent être mis à l'échelle (z-score scaling) et peuvent être bruités numériquement de manière classique (ex: +0.05 de magnitude d'attaque).
2. **Les 8 attributs binaires (29 à 36)** ne doivent **jamais** être mis à l'échelle ni recevoir de perturbation par gradient continue. On applique sur ces caractéristiques une méthode d'attaque de type *Bit-Flipping* guidée par le gradient pour que la donnée reste toujours soit 0 soit 1, conservant une consistance sémantique réseau absolue.

Cette hybridation assure l'obtention d'un modèle invulnérable aux données bruitées sémantiquement impossibles et maintient l'intégrité de l'entraînement par *Curriculum Learning*.
