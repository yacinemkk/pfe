#!/usr/bin/env python3
"""
Script principal de prétraitement JSON pour Kaggle.

Ce script exécute le pipeline complet de prétraitement des données
IPFIX Records en utilisant les modules existants du projet (JsonIoTDataProcessor).

Usage sur Kaggle:
  1. Uploader les données JSON comme Dataset Kaggle
  2. Uploader le code source du projet comme Dataset Kaggle
  3. Modifier KAGGLE_DATASET_SLUG et KAGGLE_PROJECT_SLUG dans config.py
  4. Exécuter ce script

Résultats sauvegardés dans /kaggle/working/pretraitement_json/
"""

import time
import sys
import numpy as np
from pathlib import Path

# ─── Configuration Kaggle ─────────────────────────────────────────────────────
# Importe et injecte PROJECT_ROOT dans sys.path automatiquement
from kaggle_config import (
    IS_KAGGLE,
    DATA_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    SEQ_LENGTH,
    STRIDE,
    MAX_RECORDS,
    CHUNK_SIZE,
    MIN_SAMPLES_PER_CLASS,
    CONTAMINATION,
    APPLY_BALANCING,
    APPLY_FEATURE_SELECTION,
    TOP_K_FEATURES,
    ARRAYS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
)
from utils import setup_logger, verify_json_files, ensure_output_dirs, save_report

# ─── Setup Logger ─────────────────────────────────────────────────────────────
logger = setup_logger()


def main():
    """Pipeline principal de prétraitement JSON pour Kaggle."""

    start_time = time.time()

    # ─── Bannière ─────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("PRÉTRAITEMENT JSON IPFIX RECORDS — VERSION KAGGLE")
    logger.info("=" * 70)
    logger.info(f"Environnement: {'Kaggle' if IS_KAGGLE else 'Local'}")
    logger.info(f"Données:       {DATA_DIR}")
    logger.info(f"Sortie:        {OUTPUT_DIR}")
    logger.info(f"Projet source: {PROJECT_ROOT}")
    logger.info(f"Paramètres:    seq_length={SEQ_LENGTH}, stride={STRIDE}, "
                f"max_records={MAX_RECORDS}")
    logger.info("-" * 70)

    # ─── Étape 0: Vérification de l'environnement ────────────────────────────
    logger.info("\n[ÉTAPE 0] Vérification de l'environnement...")

    # Vérifier que le code source est accessible
    src_path = PROJECT_ROOT / "src" / "data" / "json_preprocessor.py"
    if not src_path.exists():
        logger.error(f"❌ Code source introuvable: {src_path}")
        logger.error("Assurez-vous que le projet est uploadé comme Dataset Kaggle.")
        logger.error(f"Chemin attendu: {PROJECT_ROOT}")
        sys.exit(1)
    logger.info(f"✅ Code source trouvé: {src_path}")

    # Vérifier les fichiers JSON
    json_stats = verify_json_files(DATA_DIR, logger)
    if not json_stats["valid"]:
        logger.error("❌ Aucun fichier JSON valide trouvé. Arrêt.")
        sys.exit(1)

    # Créer les dossiers de sortie
    dirs = ensure_output_dirs(OUTPUT_DIR, logger)

    # ─── Étape 1: Import du processeur ────────────────────────────────────────
    logger.info("\n[ÉTAPE 1] Import de JsonIoTDataProcessor...")
    try:
        from src.data.json_preprocessor import JsonIoTDataProcessor
        logger.info("✅ JsonIoTDataProcessor importé avec succès")
    except ImportError as e:
        logger.error(f"❌ Erreur d'import: {e}")
        logger.error("Vérifiez que PROJECT_ROOT est correct dans config.py")
        logger.error(f"PROJECT_ROOT = {PROJECT_ROOT}")
        logger.error(f"sys.path inclut: {[p for p in sys.path if 'kaggle' in p or 'pfe' in p]}")
        sys.exit(1)

    # ─── Étape 2: Exécution du pipeline ───────────────────────────────────────
    logger.info("\n[ÉTAPE 2] Exécution du pipeline de prétraitement...")
    logger.info(f"  • Données:              {DATA_DIR}")
    logger.info(f"  • Sauvegarde:            {ARRAYS_DIR}")
    logger.info(f"  • Séquence:              length={SEQ_LENGTH}, stride={STRIDE}")
    logger.info(f"  • Max records:           {MAX_RECORDS or 'tous'}")
    logger.info(f"  • Équilibrage:           {'oui' if APPLY_BALANCING else 'non'}")
    logger.info(f"  • Sélection features:    {'oui' if APPLY_FEATURE_SELECTION else 'non'}")

    processor = JsonIoTDataProcessor()

    result = processor.process_all(
        data_dir=DATA_DIR,
        save_path=ARRAYS_DIR,
        max_records=MAX_RECORDS,
        seq_length=SEQ_LENGTH,
        stride=STRIDE,
        min_samples=MIN_SAMPLES_PER_CLASS,
        apply_balancing=APPLY_BALANCING,
        apply_feature_selection=APPLY_FEATURE_SELECTION,
        top_k_features=TOP_K_FEATURES,
    )

    # Décomposer les résultats
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        feature_names, scaler, label_encoder,
    ) = result

    # ─── Étape 3: Rapport de statistiques ─────────────────────────────────────
    logger.info("\n[ÉTAPE 3] Génération du rapport de statistiques...")

    stats = {
        "environment": "kaggle" if IS_KAGGLE else "local",
        "data_dir": str(DATA_DIR),
        "output_dir": str(OUTPUT_DIR),
        "parameters": {
            "seq_length": SEQ_LENGTH,
            "stride": STRIDE,
            "max_records": MAX_RECORDS,
            "min_samples_per_class": MIN_SAMPLES_PER_CLASS,
            "apply_balancing": APPLY_BALANCING,
            "apply_feature_selection": APPLY_FEATURE_SELECTION,
            "top_k_features": TOP_K_FEATURES,
        },
        "json_files": json_stats,
        "results": {
            "X_train_shape": list(X_train.shape),
            "X_val_shape": list(X_val.shape),
            "X_test_shape": list(X_test.shape),
            "y_train_shape": list(y_train.shape),
            "y_val_shape": list(y_val.shape),
            "y_test_shape": list(y_test.shape),
            "n_classes": len(label_encoder.classes_),
            "classes": list(label_encoder.classes_),
            "feature_names": feature_names,
            "n_features": len(feature_names) if feature_names else 0,
        },
        "class_distribution": {
            "train": {
                str(label_encoder.classes_[i]): int(np.sum(y_train == i))
                for i in range(len(label_encoder.classes_))
            },
            "val": {
                str(label_encoder.classes_[i]): int(np.sum(y_val == i))
                for i in range(len(label_encoder.classes_))
            },
            "test": {
                str(label_encoder.classes_[i]): int(np.sum(y_test == i))
                for i in range(len(label_encoder.classes_))
            },
        },
    }

    save_report(stats, OUTPUT_DIR, logger)

    # ─── Résumé final ─────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("PRÉTRAITEMENT TERMINÉ AVEC SUCCÈS")
    logger.info("=" * 70)
    logger.info(f"⏱️  Durée totale:    {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    logger.info(f"📊 Train:           {X_train.shape}")
    logger.info(f"📊 Validation:      {X_val.shape}")
    logger.info(f"📊 Test:            {X_test.shape}")
    logger.info(f"🏷️  Classes:         {len(label_encoder.classes_)}")
    logger.info(f"📁 Fichiers sauvés: {ARRAYS_DIR}")
    logger.info("")

    logger.info("Fichiers générés:")
    for f in sorted(OUTPUT_DIR.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 ** 2)
            logger.info(f"  📄 {f.relative_to(OUTPUT_DIR)} ({size_mb:.1f} MB)")

    logger.info("\n✅ Prêt pour l'entraînement !")
    return result


if __name__ == "__main__":
    main()
