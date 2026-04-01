# Rapport d'Analyse des Spécifications et du Code

## 1. Objectif

Analyser les spécifications et la documentation du projet pour identifier les contradictions internes, les lacunes et les ambiguïtés. Réaliser une analyse d'écart complète entre les spécifications et le codebase actuel. Fournir un rapport consolidé avec des recommandations actionnables et priorisées.

---

## 2. Instructions Clés

### 2.1 Protocole d'Entraînement
- Suivre strictement le protocole en 2 phases défini dans `docs/train` :
  - **Phase 1** : Standard + Crash Test 1
  - **Phase 2** : Adversarial Training + Crash Test 2
- La méthode `fit()` en 3 phases dans `train_adversarial.py` est une variante expérimentale non documentée. Elle doit être documentée ou dépréciée.

### 2.2 Pipeline Anti-Fuite de Données
- Ordonnancement chronologique strict :
  1. Grouper par appareil (`device`)
  2. Trier par `flow_start`
  3. Split temporel 80/20
  4. Générer les séquences séparément pour train/test
  5. Normaliser avec les paramètres du train uniquement
- **Ne pas utiliser `main.py` pour le split** : il manque le tri chronologique et introduit des fuites de données.

### 2.3 Contraintes sur les Fonctionnalités SDN
- Exclure les adresses IP, MAC et les ports.
- Conserver uniquement les statistiques de flux accessibles via SDN.
- Les fonctionnalités de protocole L7 ne doivent être conservées que si elles sont binaires/dérivées des ports, et non dépendantes du DPI.

### 2.4 Hiérarchie de Documentation
- En cas de contradiction, le codebase et `docs/general`/`docs/train` font foi.
- Mettre à jour `docs/architectures` pour refléter les implémentations réelles.

---

## 3. Découvertes

### 3.1 Contradictions Critiques

| # | Sujet | Spécification A | Spécification B | Code | Verdict |
|---|-------|-----------------|-----------------|------|---------|
| 1 | Ordre CNN-LSTM | LSTM→CNN (`docs/architectures`) | CNN→LSTM (`docs/general`, `docs/structure`) | CNN→LSTM (`cnn_lstm.py`) | Doc erronée |
| 2 | Normalisation | Min-Max **puis** Standardisation (`docs/pretraitement`) | Min-Max uniquement (autres docs) | Applique les deux | Redondance mathématique |
| 3 | Split des données | Split chronologique par appareil | Split naïf 80/20 (`main.py`) | Correct dans `preprocessor.py` | `main.py` obsolète |

### 3.2 Lacunes Identifiées

| # | Lacune | Impact |
|---|--------|--------|
| 1 | Set de validation (`val_size: 0.1` dans `config.yaml`) non documenté | Origine et objectif du set de validation inconnus |
| 2 | `docs/important.md` référencé dans `config.yaml` (ligne 2) mais fichier inexistant | Référence cassée |
| 3 | Méthode `fit()` en 3 phases non documentée | Confusion sur le protocole d'entraînement |
| 4 | Redondance massive entre 6 documents de spécification | Maintenance difficile, incohérences |
| 5 | Pas de cross-références entre les documents | Navigation difficile |
| 6 | Features L7 non spécifiées clairement (DPI vs binaire) | Risque de fuite de données |
| 7 | Origine du `val_size` inconnue (provenant du train ou du test ?) | Ambiguïté dans le pipeline |
| 8 | Pas de documentation sur la gestion des appareils inconnus au test | Comportement non spécifié |
| 9 | Pas de spécification sur le format d'entrée JSON vs CSV | Deux préprocesseurs existent sans guidance |
| 10 | Pas de métriques de performance cibles | Impossible de valider la qualité du modèle |
| 11 | Pas de spécification sur la persistance des modèles sauvegardés | Format et emplacement non documentés |
| 12 | Pas de documentation sur les hyperparamètres optimaux | Valeurs par défaut non justifiées |
| 13 | Pas de plan de déploiement SDN-NAC | Phase 5 mentionnée mais non détaillée |

### 3.3 Ambiguïtés

| # | Ambiguïté |
|---|-----------|
| 1 | Le set de validation est-il extrait du train ou du test ? |
| 2 | Les features L7 sont-elles basées sur DPI ou sur des dérivations de ports ? |
| 3 | Comment gérer les appareils non vus pendant l'entraînement ? |
| 4 | Le format d'entrée préféré est-il JSON ou CSV ? |
| 5 | Quelle est la source de vérité pour l'ordre CNN-LSTM ? |
| 6 | La normalisation doit-elle inclure Standardisation ou non ? |
| 7 | Le Crash Test 1 et 2 ont-ils des seuils de réussite définis ? |
| 8 | La defensive distillation est-elle requise ou optionnelle ? |
| 9 | Comment les séquences sont-elles tronquées/padées ? |
| 10 | Quelle est la tolérance acceptable pour le taux de faux positifs ? |

### 3.4 Problèmes Structurels de Documentation

| # | Problème |
|---|----------|
| 1 | Redondance massive entre 6 documents de spécification |
| 2 | Pas de cross-références entre les documents |
| 3 | Fichier `docs/important.md` manquant |
| 4 | `docs/architectures` contient des erreurs (ordre CNN-LSTM) |
| 5 | Pas de versioning des documents |
| 6 | Pas de date de dernière mise à jour |
| 7 | Pas d'auteur responsable par document |
| 8 | Incohérences de terminologie entre les documents |
| 9 | Pas de glossaire partagé |
| 10 | Pas de schéma d'architecture unifié |

---

## 4. Analyse d'Écart Code vs Spécifications

### 4.1 Résumé de Conformité

| Catégorie | Requis | Implémentés | Partiellement | Non Implémentés | Déviations |
|-----------|--------|-------------|---------------|-----------------|------------|
| Pipeline de données | 8 | 6 | 2 | 0 | 1 |
| Tokenisation | 4 | 4 | 0 | 0 | 0 |
| Sélection de features | 3 | 3 | 0 | 0 | 0 |
| Modèles (6 architectures) | 12 | 12 | 0 | 0 | 0 |
| Protocole d'entraînement | 10 | 8 | 2 | 0 | 0 |
| Attaques adversariales | 10 | 9 | 0 | 1 | 0 |
| **Total** | **47** | **42** | **4** | **1** | **1** |

### 4.2 Détails par Catégorie

#### Pipeline de Données
- **Conformité** : 6/8 entièrement implémentés
- **Partiellement** : Split temporel (`main.py` non conforme), Normalisation (double application)
- **Déviation** : `main.py` utilise un split naïf sans tri chronologique

#### Tokenisation
- **Conformité** : 4/4 entièrement implémentés
- Tokenizers IoT et standard fonctionnels

#### Sélection de Features
- **Conformité** : 3/3 entièrement implémentés
- Exclusion IP/MAC/Ports respectée

#### Modèles (6 architectures)
- **Conformité** : 12/12 entièrement implémentés
- LSTM, BiLSTM, CNN-LSTM, XGBoost-LSTM, Transformer, Hybrid, CNN-BiLSTM tous présents

#### Protocole d'Entraînement
- **Conformité** : 8/10 entièrement implémentés
- **Partiellement** : Validation set non documenté, Phase 3 expérimentale non documentée

#### Attaques Adversariales
- **Conformité** : 9/10 entièrement implémentés
- **Non implémenté** : Defensive distillation (optionnelle, Phase 5)
- BIM est couvert par PGD

---

## 5. Plan d'Action Priorisé

### 5.1 Immédiat (Semaine 1)

| # | Action | Fichiers Concernés | Priorité |
|---|--------|-------------------|----------|
| 1 | Corriger la fuite de données dans `main.py` | `main.py`, `preprocessor.py` | Critique |
| 2 | Résoudre la contradiction CNN-LSTM dans `docs/architectures` | `docs/architectures` | Critique |
| 3 | Corriger la normalisation (supprimer Standardisation redondante) | `docs/pretraitement`, code de normalisation | Haute |
| 4 | Documenter le set de validation | `docs/train`, `config.yaml` | Haute |

### 5.2 Court Terme (Semaines 2-3)

| # | Action | Fichiers Concernés | Priorité |
|---|--------|-------------------|----------|
| 5 | Documenter ou déprécier la Phase 3 de `train_adversarial.py` | `train_adversarial.py`, `docs/train` | Haute |
| 6 | Créer ou supprimer la référence à `docs/important.md` | `config.yaml` | Moyenne |
| 7 | Clarifier les features L7 (DPI vs binaire) | `docs/featureselection` | Moyenne |
| 8 | Ajouter des cross-références entre les documents | Tous les docs | Moyenne |
| 9 | Réduire la redondance entre les 6 documents | Tous les docs | Moyenne |

### 5.3 Long Terme (Mois 1-3)

| # | Action | Priorité |
|---|--------|----------|
| 10 | Implémenter le déploiement SDN-NAC (Phase 5) | Basse |
| 11 | Implémenter la defensive distillation (optionnelle) | Basse |
| 12 | Ajouter des métriques de performance cibles | Moyenne |
| 13 | Créer un glossaire partagé | Moyenne |
| 14 | Versionner les documents | Moyenne |
| 15 | Créer un schéma d'architecture unifié | Moyenne |

---

## 6. Conclusion

L'analyse révèle un projet globalement bien implémenté avec **42 exigences sur 47 entièrement conformes**. Les problèmes identifiés sont principalement :

1. **Documentaires** : Contradictions entre les spécifications, redondance, références cassées
2. **Pipeline de données** : Fuite de données potentielle via `main.py`, normalisation redondante
3. **Architecturaux** : Ordre CNN-LSTM incorrect dans la documentation

Les corrections immédiates se concentrent sur la sécurisation du pipeline de données et la cohérence documentaire. Le codebase lui-même est solide et ne nécessite que des ajustements mineurs.

---

*Date de création : 2026-04-01*
*Basé sur l'analyse de 6 documents de spécification, 1 fichier de configuration, et ~20 fichiers source.*
