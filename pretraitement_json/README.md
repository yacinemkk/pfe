# Prétraitement JSON — Version Kaggle

Version Kaggle du pipeline de prétraitement des données IPFIX Records (JSON).
Ce dossier est **indépendant** et réutilise les modules existants du projet via import dynamique.

## Structure

```
pretraitement_json/
├── config.py                      # Chemins Kaggle et paramètres du pipeline
├── utils.py                       # Logger, vérification JSON, utilitaires
├── run_preprocessing_kaggle.py    # Script principal d'exécution
└── README.md                      # Ce fichier
```

## Prérequis sur Kaggle

### 1. Uploader les données JSON comme Dataset

1. Aller sur [kaggle.com/datasets/new](https://www.kaggle.com/datasets/new)
2. Créer un dataset nommé `ipfix-records-json`
3. Uploader le dossier contenant les fichiers `*.json` (IPFIX Records)
4. Les données seront montées en lecture seule dans `/kaggle/input/ipfix-records-json/`

### 2. Uploader le code source du projet comme Dataset

1. Créer un second dataset nommé `pfe-source-code`
2. Uploader le projet complet (ou au minimum les dossiers `src/` et `config/`)
3. Le code sera monté dans `/kaggle/input/pfe-source-code/`

### 3. Structure attendue sur Kaggle

```
/kaggle/input/
├── ipfix-records-json/          # Dataset JSON
│   └── *.json                   # Fichiers IPFIX Records
└── pfe-source-code/             # Code source du projet
    ├── src/
    │   └── data/
    │       └── json_preprocessor.py
    └── config/
        └── config.py

/kaggle/working/
└── pretraitement_json/          # Sortie (générée automatiquement)
    ├── arrays/                  # X_train.npy, y_train.npy, etc.
    ├── models/                  # preprocessor.pkl
    └── reports/                 # Rapport de statistiques JSON
```

## Configuration

Modifier `kaggle_config.py` selon vos noms de datasets Kaggle :

```python
KAGGLE_DATASET_SLUG = "ipfix-records-json"    # ← Nom de votre dataset JSON
KAGGLE_PROJECT_SLUG = "pfe-source-code"       # ← Nom de votre dataset code source
```

### Paramètres ajustables dans `kaggle_config.py`

| Paramètre | Défaut | Description |
|---|---|---|
| `SEQ_LENGTH` | 10 | Longueur des séquences temporelles |
| `STRIDE` | 10 | Pas entre les fenêtres glissantes |
| `MAX_RECORDS` | None | Nombre max d'enregistrements (None = tous) |
| `CHUNK_SIZE` | 100 000 | Taille des chunks pour le chargement mémoire |
| `MIN_SAMPLES_PER_CLASS` | 500 | Minimum d'échantillons par classe |
| `APPLY_BALANCING` | True | Activer le filtrage du bruit (Isolation Forest, LOF) |
| `APPLY_FEATURE_SELECTION` | True | Activer la sélection hybride de features |
| `TOP_K_FEATURES` | None | Nombre de features (None = coude automatique) |

## Exécution

### Sur Kaggle (Notebook)

```python
# Dans une cellule Kaggle :
import subprocess
subprocess.run(["python", "/kaggle/input/pfe-source-code/pretraitement_json/run_preprocessing_kaggle.py"])
```

Ou copier le contenu de `run_preprocessing_kaggle.py` dans une cellule notebook.

### En local (test)

```bash
cd pretraitement_json/
python run_preprocessing_kaggle.py
```

## Sortie

Après exécution, les fichiers suivants sont générés dans `OUTPUT_DIR` :

| Fichier | Description |
|---|---|
| `arrays/X_train.npy` | Séquences d'entraînement |
| `arrays/X_val.npy` | Séquences de validation |
| `arrays/X_test.npy` | Séquences de test |
| `arrays/y_train.npy` | Labels d'entraînement |
| `arrays/y_val.npy` | Labels de validation |
| `arrays/y_test.npy` | Labels de test |
| `arrays/preprocessor.pkl` | Scaler, LabelEncoder, feature names |
| `reports/preprocessing_report_*.json` | Rapport de statistiques |

## Notes

- **Aucun code existant n'est modifié** — ce dossier est purement additif.
- Le pipeline réutilise `JsonIoTDataProcessor` de `src/data/json_preprocessor.py`.
- Aucune dépendance à Google Drive ni Colab.
- Le GPU Kaggle n'est pas nécessaire pour le prétraitement (CPU suffit).
