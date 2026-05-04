"""
Utilitaires pour le prétraitement JSON sur Kaggle.

Fonctions:
  - setup_logger()        : Logger formaté avec timestamps
  - verify_json_files()   : Vérification et inventaire des fichiers JSON
  - ensure_output_dirs()  : Création automatique des dossiers de sortie
  - save_report()         : Sauvegarde d'un rapport de statistiques JSON
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name: str = "pretraitement_json", level=logging.INFO) -> logging.Logger:
    """
    Configure un logger avec timestamp et formatage clair.

    Args:
        name: Nom du logger
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Logger configuré
    """
    logger = logging.getLogger(name)

    # Éviter les handlers dupliqués si appelé plusieurs fois
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Handler console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def verify_json_files(data_dir: Path, logger: logging.Logger = None) -> dict:
    """
    Vérifie et inventorie les fichiers JSON dans le répertoire de données.

    Args:
        data_dir: Répertoire racine contenant les fichiers JSON
        logger: Logger optionnel (utilise print si absent)

    Returns:
        Dict avec les statistiques:
          - total_files: nombre de fichiers JSON trouvés
          - total_size_gb: taille totale en GB
          - files: liste de dicts {path, size_mb}
          - valid: True si au moins un fichier trouvé
    """
    log = logger.info if logger else print

    data_dir = Path(data_dir)
    if not data_dir.exists():
        msg = f"❌ Répertoire introuvable: {data_dir}"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        return {"total_files": 0, "total_size_gb": 0, "files": [], "valid": False}

    json_files = sorted(data_dir.rglob("*.json"))

    if not json_files:
        msg = f"❌ Aucun fichier JSON trouvé dans {data_dir}"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        return {"total_files": 0, "total_size_gb": 0, "files": [], "valid": False}

    files_info = []
    total_size = 0

    log(f"📂 Répertoire de données: {data_dir}")
    log(f"📄 {len(json_files)} fichier(s) JSON trouvé(s):")

    for f in json_files:
        size_bytes = f.stat().st_size
        size_mb = size_bytes / (1024 ** 2)
        total_size += size_bytes

        try:
            rel_path = str(f.relative_to(data_dir))
        except ValueError:
            rel_path = str(f)

        log(f"   • {rel_path} ({size_mb:.1f} MB)")
        files_info.append({"path": rel_path, "size_mb": round(size_mb, 2)})

    total_gb = total_size / (1024 ** 3)
    log(f"📊 Taille totale: {total_gb:.2f} GB")

    # Vérification rapide du format (première ligne du premier fichier)
    first_file = json_files[0]
    try:
        with open(first_file, "r") as fh:
            first_line = fh.readline().strip()
            if first_line:
                record = json.loads(first_line)
                has_flows = "flows" in record
                log(f"✅ Format JSON valide (clé 'flows': {'oui' if has_flows else 'non'})")
            else:
                log("⚠️ Première ligne vide dans le premier fichier")
    except (json.JSONDecodeError, Exception) as e:
        log(f"⚠️ Erreur lors de la vérification du format: {e}")

    return {
        "total_files": len(json_files),
        "total_size_gb": round(total_gb, 2),
        "files": files_info,
        "valid": True,
    }


def ensure_output_dirs(output_dir: Path, logger: logging.Logger = None) -> dict:
    """
    Crée tous les sous-dossiers de sortie nécessaires.

    Args:
        output_dir: Répertoire racine de sortie

    Returns:
        Dict avec les chemins créés
    """
    log = logger.info if logger else print

    subdirs = {
        "arrays": output_dir / "arrays",
        "models": output_dir / "models",
        "reports": output_dir / "reports",
    }

    for name, path in subdirs.items():
        path.mkdir(parents=True, exist_ok=True)
        log(f"📁 Dossier '{name}': {path}")

    return subdirs


def save_report(stats: dict, output_dir: Path, logger: logging.Logger = None) -> Path:
    """
    Sauvegarde un rapport de statistiques en JSON.

    Args:
        stats: Dictionnaire de statistiques à sauvegarder
        output_dir: Répertoire de sortie
        logger: Logger optionnel

    Returns:
        Chemin du fichier rapport sauvegardé
    """
    log = logger.info if logger else print

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"preprocessing_report_{timestamp}.json"

    # Convertir les types numpy en types Python natifs
    def _convert(obj):
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    clean_stats = json.loads(json.dumps(stats, default=_convert))

    with open(report_path, "w") as f:
        json.dump(clean_stats, f, indent=2, ensure_ascii=False)

    log(f"📋 Rapport sauvegardé: {report_path}")
    return report_path
