"""
Configuration Kaggle pour le prétraitement JSON (IPFIX Records).

Ce fichier centralise TOUS les chemins et paramètres nécessaires
à l'exécution du pipeline de prétraitement sur Kaggle.

Usage Kaggle:
  - Les données JSON doivent être uploadées comme Dataset Kaggle
  - Le code source du projet doit être uploadé comme Dataset Kaggle séparé
  - Les résultats sont écrits dans /kaggle/working/
"""

import os
import sys
from pathlib import Path

# ─── Détection automatique de l'environnement ────────────────────────────────
IS_KAGGLE = os.path.exists("/kaggle")

# ─── Chemins des données ─────────────────────────────────────────────────────
if IS_KAGGLE:
    # Sur Kaggle: le dataset JSON est monté en lecture seule
    # Adapter le nom du dataset ci-dessous selon votre upload Kaggle
    DATA_DIR = Path("/kaggle/input/pretraitement/IPFIX Records (UNSW IoT Analytics)")

    # Comme vous avez placé "src" juste à côté de "run_preprocessing_kaggle.py"
    # le PROJECT_ROOT est simplement le dossier actuel.
    PROJECT_ROOT = Path(__file__).parent.absolute()

    # Répertoire de sortie (lecture/écriture)
    OUTPUT_DIR = Path("/kaggle/working/pretraitement_json")
else:
    # En local: chemins relatifs au projet
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()
    DATA_DIR = PROJECT_ROOT / "data" / "pcap" / "IPFIX Records (UNSW IoT Analytics)"
    OUTPUT_DIR = PROJECT_ROOT / "pretraitement_json" / "output"

# ─── Injection du chemin projet pour les imports ──────────────────────────────
# Permet d'importer src.data.json_preprocessor, config.config, etc.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─── Paramètres du pipeline ──────────────────────────────────────────────────
SEQ_LENGTH = 10          # Longueur des séquences temporelles
STRIDE = 10              # Pas entre les fenêtres glissantes
MAX_RECORDS = None      # None = charger tous les enregistrements
CHUNK_SIZE = 100_000    # Taille des chunks pour le chargement mémoire
MIN_SAMPLES_PER_CLASS = 500  # Minimum d'échantillons par classe

# ─── Paramètres de filtrage du bruit ──────────────────────────────────────────
CONTAMINATION = 0.05    # Taux de contamination pour Isolation Forest / LOF

# ─── Paramètres de sélection de features ──────────────────────────────────────
APPLY_BALANCING = True
APPLY_FEATURE_SELECTION = True
TOP_K_FEATURES = None   # None = méthode du coude automatique

# ─── Sous-dossiers de sortie ─────────────────────────────────────────────────
ARRAYS_DIR = OUTPUT_DIR / "arrays"      # X_train.npy, y_train.npy, etc.
MODELS_DIR = OUTPUT_DIR / "models"      # scaler.pkl, label_encoder.pkl
REPORTS_DIR = OUTPUT_DIR / "reports"     # rapport de statistiques
