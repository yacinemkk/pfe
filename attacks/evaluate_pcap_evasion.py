import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import dgl
import joblib
from train_ctu_gnn import EGraphSAGE, load_graph
from sklearn.metrics import recall_score, classification_report

# Paths
MODEL_PATH = "results_v2/CTU_Lite/models/egraphsage/model.pth"
SCALER_PATH = "results_v2/CTU_Lite/models/egraphsage/scaler.skl"
BENIGN_CSV = "ctu_benign_features_massive.csv"
EVASION_CSVS = {
    "Temporal": "adversarial_features_temporal.csv",
    "Protocol": "adversarial_features_protocol.csv",
    "LowRate": "adversarial_features_lowrate.csv"
}

def evaluate(variant_name, attack_csv):
    print(f"\n[***] Évaluation de l'évasion : {variant_name} [***]")
    try:
        attack_df = pd.read_csv(attack_csv)
    except FileNotFoundError:
        print(f"[!] Fichier {attack_csv} non trouvé. Saut...")
        return
        
    print(f"[*] Chargement de {len(attack_df)} flux d'attaque...")
    
    # Load benign for background graph context (same size as attack or same as test set)
    benign_df = pd.read_csv(BENIGN_CSV).sample(15000, random_state=42)
    
    # Combine
    combined_df = pd.concat([benign_df, attack_df], ignore_index=True)
    combined_df.to_csv("temp_eval.csv", index=False)
    
    # Load Scaler
    scaler = joblib.load(SCALER_PATH)
    
    # Load Graph
    g, in_feats, _ = load_graph("temp_eval.csv", scaler)
    
    # Load Model
    # Use the same architecture as in train_ctu_gnn.py: (ndim, 128, ndim, F.relu, 0.2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EGraphSAGE(in_feats, 128, in_feats, F.relu, 0.2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # Move graph to device
    g = g.to(device)
    features_n = g.ndata['h'].to(device)
    features_e = g.edata['h'].to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(g, features_n, features_e)
        _, indices = torch.max(logits, dim=1)
        
    # Metrics
    # The attack flows are at the end of combined_df
    # Note: DGL graph edges are bidirectional, so we check the labels of the original edges
    # Standard load_graph stores labels in G.edata['label']
    attack_labels = g.edata['label'].cpu().numpy()
    preds = indices.cpu().numpy()
    
    # Identify indices of attack flows (labels == 1)
    attack_indices = np.where(attack_labels == 1)[0]
    
    if len(attack_indices) == 0:
        print("[!] Aucun flux d'attaque détecté dans le graphe.")
        return

    y_true = attack_labels[attack_indices]
    y_pred = preds[attack_indices]
    
    recall = recall_score(y_true, y_pred, pos_label=1)
    print(f"✅ Résultat {variant_name} -> Rappel (Botnet): {recall:.2%}")
    
    if recall < 0.1:
        print(f"🔥 ÉVASION RÉUSSIE ! Le modèle est aveugle à cette attaque.")
    elif recall < 0.5:
        print(f"⚠️ ÉVASION PARTIELLE. La détection est fortement dégradée.")
    else:
        print(f"❌ ÉVASION ÉCHOUÉE. Le modèle détecte encore l'attaque.")

if __name__ == "__main__":
    for name, path in EVASION_CSVS.items():
        evaluate(name, path)
