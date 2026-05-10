#!/usr/bin/env python3
"""
Exemple d'utilisation du checkpoint étape 2.

Ce script montre comment utiliser le cache étape 2 pour:
1. Première exécution : Générer et sauvegarder le checkpoint
2. Réexécution : Charger depuis le checkpoint pour sauter 2h30 de traitement
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.json_preprocessor import JsonIoTDataProcessor
from config.config import (
    JSON_DATA_DIR, DATASET_SAVE_PATH, JSON_PREPROCESSED_DIR,
    SEQ_LENGTH, STRIDE, MAX_RECORDS
)

# ============================================================================
# Configuration du checkpoint
# ============================================================================

CHECKPOINT_DIR = Path("/home/pc/Desktop/pfe/results/preprocessed/json_checkpoint_stage2")
DRIVE_CACHE_DIR = "/content/drive/MyDrive/PFE/results/preprocessed/json_checkpoint_stage2"

# ============================================================================
# Scénario 1 : Première exécution (Génère le checkpoint)
# ============================================================================

def scenario_1_generate_checkpoint():
    """
    🔧 Première exécution : Générer le checkpoint étape 2
    
    Timeline:
    - Étape 1: Load JSON + Split    (20 min)
    - Étape 2: IF + LOF             (150 min) ← Sauvegardé ici!
    - Étape 3: Adversarial Select   (5 min)
    - Étape 4: Scaling + Sequences  (10 min)
    ---
    Total: ~185 min
    
    Résultat: Checkpoint sauvegardé en /home/pc/.../json_checkpoint_stage2/
    """
    print("=" * 80)
    print("SCÉNARIO 1 : Générer Checkpoint Étape 2 (PREMIÈRE EXÉCUTION)")
    print("=" * 80)
    print()
    
    processor = JsonIoTDataProcessor()
    
    print("📦 Paramètres:")
    print(f"  - Data dir       : {JSON_DATA_DIR}")
    print(f"  - Checkpoint dir : {CHECKPOINT_DIR}")
    print(f"  - Seq length     : {SEQ_LENGTH}")
    print(f"  - Stride         : {STRIDE}")
    print()
    
    print("🚀 Exécution (cela prendra ~185 minutes)...")
    print()
    
    # ✅ Première exécution : apply_balancing=True → exécute étape 2 complète
    X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq = processor.process_all(
        data_dir=JSON_DATA_DIR,
        save_path=DATASET_SAVE_PATH,
        seq_length=SEQ_LENGTH,
        stride=STRIDE,
        max_records=MAX_RECORDS,
        
        # 📥 Cache configuration
        apply_balancing=True,                        # Exécute étape 2
        apply_feature_selection=True,                # Exécute étape 3
        step2_cache_dir=CHECKPOINT_DIR,              # 💾 Sauvegarde ici
        drive_cache_dir=DRIVE_CACHE_DIR if "COLAB" in sys.version else None,
    )
    
    print()
    print("✅ Résultats générés et checkpoint sauvegardé!")
    print(f"   Train sequences  : {len(X_train_seq):,}")
    print(f"   Val sequences    : {len(X_val_seq):,}")
    print(f"   Test sequences   : {len(X_test_seq):,}")
    print()


# ============================================================================
# Scénario 2 : Réexécution (Charge depuis checkpoint)
# ============================================================================

def scenario_2_load_checkpoint():
    """
    ⚡ Réexécution : Charger depuis checkpoint étape 2
    
    Timeline:
    - Étape 1+2: SKIPPED (loaded from cache) ← 💾 Charge depuis checkpoint!
    - Étape 3: Adversarial Select               (5 min)
    - Étape 4: Scaling + Sequences              (10 min)
    ---
    Total: ~35 min (vs 185 min avant!)
    
    Gain: 150 minutes! 🎉
    """
    print()
    print("=" * 80)
    print("SCÉNARIO 2 : Charger depuis Checkpoint (RÉEXÉCUTION)")
    print("=" * 80)
    print()
    
    processor = JsonIoTDataProcessor()
    
    print("📦 Paramètres:")
    print(f"  - Data dir       : {JSON_DATA_DIR}")
    print(f"  - Checkpoint dir : {CHECKPOINT_DIR}")
    print(f"  - Seq length     : {SEQ_LENGTH}")
    print(f"  - Stride         : {STRIDE}")
    print()
    
    print("🚀 Exécution (cela prendra ~35 minutes)...")
    print("  ⚡ Étapes 1+2 SKIPPED (chargées depuis checkpoint!)")
    print()
    
    # ✅ Réexécution : checkpoint trouvé → étapes 1+2 skipped
    X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq = processor.process_all(
        data_dir=JSON_DATA_DIR,
        save_path=DATASET_SAVE_PATH,
        seq_length=SEQ_LENGTH,
        stride=STRIDE,
        max_records=MAX_RECORDS,
        
        # 📥 Cache configuration (même que avant)
        apply_balancing=True,                        # Cache trouvé → SKIP étape 2
        apply_feature_selection=True,
        step2_cache_dir=CHECKPOINT_DIR,              # 📥 Charge depuis ici
        drive_cache_dir=DRIVE_CACHE_DIR if "COLAB" in sys.version else None,
    )
    
    print()
    print("✅ Résultats générés à partir du checkpoint!")
    print(f"   Train sequences  : {len(X_train_seq):,}")
    print(f"   Val sequences    : {len(X_val_seq):,}")
    print(f"   Test sequences   : {len(X_test_seq):,}")
    print()
    print("⏱️  Gain de temps : ~150 minutes économisées! 🎉")
    print()


# ============================================================================
# Scénario 3 : Correction de Bug (sans refaire étape 2)
# ============================================================================

def scenario_3_fix_bug_without_refitting():
    """
    🐛 Correction de bug : Redémarrer à étape 3 sans refaire étape 2
    
    Cas d'usage:
    - Vous avez une erreur à étape 3 ou 4
    - Vous corrigez le bug
    - Vous voulez relancer le traitement
    - Mais vous ne voulez pas répéter 2h30 de LOF!
    
    Solution: Charger depuis checkpoint avec les MÊMES paramètres
    """
    print()
    print("=" * 80)
    print("SCÉNARIO 3 : Corriger Bug sans Refaire Étape 2")
    print("=" * 80)
    print()
    
    processor = JsonIoTDataProcessor()
    
    print("Situation:")
    print("  ❌ Erreur à étape 3: UnboundLocalError: X_train_balanced is None")
    print("  ✅ Bug CORRIGÉ!")
    print("  → Relancer avec checkpoint pour éviter 2h30 de refonte")
    print()
    
    print("🚀 Exécution (cela prendra ~35 minutes)...")
    print("  ⚡ Étapes 1+2 SKIPPED (utilisent le checkpoint existant)")
    print()
    
    # ✅ Même configuration que avant → charge depuis checkpoint
    X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq = processor.process_all(
        data_dir=JSON_DATA_DIR,
        save_path=DATASET_SAVE_PATH,
        seq_length=SEQ_LENGTH,
        stride=STRIDE,
        max_records=MAX_RECORDS,
        apply_balancing=True,
        apply_feature_selection=True,
        step2_cache_dir=CHECKPOINT_DIR,
        drive_cache_dir=DRIVE_CACHE_DIR if "COLAB" in sys.version else None,
    )
    
    print()
    print("✅ Bug corrigé et traitement terminé!")
    print(f"   Train sequences  : {len(X_train_seq):,}")
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Exemple d'utilisation du checkpoint étape 2"
    )
    parser.add_argument(
        "scenario",
        choices=["1", "2", "3", "all"],
        help="Scénario à exécuter (1=Generate, 2=Load, 3=BugFix, all=tout)"
    )
    
    args = parser.parse_args()
    
    if args.scenario in ["1", "all"]:
        scenario_1_generate_checkpoint()
    
    if args.scenario in ["2", "all"]:
        # Vérifier que le checkpoint existe
        if not (CHECKPOINT_DIR / "step2_cache_ready").exists():
            print("⚠️  Checkpoint non trouvé! Exécutez d'abord le scénario 1.")
            sys.exit(1)
        scenario_2_load_checkpoint()
    
    if args.scenario in ["3", "all"]:
        # Vérifier que le checkpoint existe
        if not (CHECKPOINT_DIR / "step2_cache_ready").exists():
            print("⚠️  Checkpoint non trouvé! Exécutez d'abord le scénario 1.")
            sys.exit(1)
        scenario_3_fix_bug_without_refitting()


# ============================================================================
# Utilisation
# ============================================================================
"""
Exécution:

1. Première fois (générer checkpoint):
   $ python checkpoint_etape2_example.py 1
   
2. Deuxième fois (charger depuis checkpoint):
   $ python checkpoint_etape2_example.py 2
   
3. Corriger bug et relancer:
   $ python checkpoint_etape2_example.py 3
   
4. Exécuter tous les scénarios:
   $ python checkpoint_etape2_example.py all

Résultats:

Scénario 1 (Generate):
  - Temps: ~185 min
  - Sortie: Checkpoint sauvegardé + Séquences d'entraînement

Scénario 2 (Load):
  - Temps: ~35 min (150 min économisées! 🎉)
  - Sortie: Mêmes séquences d'entraînement

Scénario 3 (BugFix):
  - Temps: ~35 min
  - Sortie: Résultats corrects sans refaire étape 2
"""
