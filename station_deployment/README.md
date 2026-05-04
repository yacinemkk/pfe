# Station Deployment

Ce dossier contient tout le code nécessaire pour lancer l'entraînement adversarial de votre projet sur la station MobaXterm (avec GPU).

## Structure
- `train.py` : Script principal pour lancer l'entraînement, directement généré depuis votre notebook.
- `requirements.txt` : Bibliothèques Python requises.
- `src/` : Modules locaux (prétraitement, modèles, pipeline).
- `data/` : Dossier pour placer vos jeux de données (CSV et JSON).
- `results/` : Dossier où les résultats, modèles, et analyses seront sauvegardés.

## Étape 1 : Préparation des données
Veuillez recréer vos répertoires de données à la racine de ce dossier :
```bash
mkdir -p data/IPFIX_Records
mkdir -p data/IPFIX_ML_Instances
```
Placez vos fichiers `.json` dans `data/IPFIX_Records` et vos fichiers `.csv` dans `data/IPFIX_ML_Instances`.

## Étape 2 : Installation des dépendances
Sur votre station (idéalement dans un environnement virtuel) :
```bash
pip install -r requirements.txt
```

## Étape 3 : Lancement de l'Entraînement
Assurez-vous que les données sont en place et lancez simplement le script :
```bash
python train.py
```
Les modèles générés par époques et les métriques seront disponibles dans le dossier `results/`.
