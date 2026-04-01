import pandas as pd
import numpy as np
import torch as th
import torch.nn.functional as F
import dgl
import os
import joblib
import argparse
from EGraphSAGE import EGraphSAGE
from sklearn.metrics import recall_score

def load_data_and_prep(csv_path, dataset_type):
    df = pd.read_csv(csv_path)
    if dataset_type == "ctu":
        label_col = 'label'
        metadata_cols = ['src_ip', 'dst_ip', 'label', 'Label', 'flow_id', 'timestamp', 
                         'protocol', 'delta_start', 'handshake_duration']
        # Robust label mapping
        df['Label'] = df[label_col].map({'Normal': 0, 'Botnet': 1})
        src_col, dst_col = 'src_ip', 'dst_ip'
    else: # cic
        label_col = 'Label'
        metadata_cols = ['Src IP', 'Dst IP', 'Label', 'timestamp', 'Timestamp']
        src_col, dst_col = 'Src IP', 'Dst IP'
    
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    # Clean features
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df, feature_cols, src_col, dst_col, 'Label'

def create_dgl_graph(df, feature_cols, src_col, dst_col, label_col, scaler=None):
    all_ips = pd.concat([df[src_col], df[dst_col]]).unique()
    ip2id = {ip: i for i, ip in enumerate(all_ips)}
    n_nodes = len(ip2id)
    
    src_ids = th.tensor([ip2id[ip] for ip in df[src_col]], dtype=th.long)
    dst_ids = th.tensor([ip2id[ip] for ip in df[dst_col]], dtype=th.long)
    
    G = dgl.graph((th.cat([src_ids, dst_ids]), th.cat([dst_ids, src_ids])), num_nodes=n_nodes)
    
    feats = df[feature_cols].values
    if scaler:
        feats = scaler.transform(feats)
    
    edge_feats = th.tensor(feats, dtype=th.float32)
    G.edata['h'] = th.cat([edge_feats, edge_feats], dim=0).unsqueeze(1)
    
    labels = th.tensor(df[label_col].values, dtype=th.long)
    G.edata['label'] = th.cat([labels, labels])
    G.ndata['h'] = th.ones(n_nodes, 1, len(feature_cols))
    
    return G

def evaluate(model, G):
    model.eval()
    with th.no_grad():
        logits = model(G, G.ndata['h'], G.edata['h'])
        preds = logits.argmax(1).cpu().numpy()
        labels = G.edata['label'].cpu().numpy()
        # Recall for class 1 (Botnet)
        recall = recall_score(labels, preds, pos_label=1, zero_division=0)
    return recall

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["ctu", "cic"], required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--scaler_path", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    df, feature_cols, src_col, dst_col, label_col = load_data_and_prep(args.test_csv, args.dataset)
    ndim = len(feature_cols)
    
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    model = EGraphSAGE(ndim, 128, ndim, F.relu, 0.2).to(device)
    model.load_state_dict(th.load(args.model_path, map_location=device))
    
    scaler = None
    if args.scaler_path:
        scaler = joblib.load(args.scaler_path)
    
    # Baseline
    G_base = create_dgl_graph(df, feature_cols, src_col, dst_col, label_col, scaler).to(device)
    base_recall = evaluate(model, G_base)
    print(f"[*] Baseline Recall: {base_recall:.4f}")
    
    results = []
    
    # Analyze each feature
    # We only perturb ATTACK samples to see if they evade
    df_attack = df[df[label_col] == 1].copy()
    
    for i, col in enumerate(feature_cols):
        print(f"[*] Analyzing feature {i+1}/{len(feature_cols)}: {col}")
        
        # Strategy: Mimicry (set to benign mean)
        # We need benign stats
        benign_val = df[df[label_col] == 0][col].mean()
        camou_val = df[df[label_col] == 0][col].quantile(0.95)
        
        strategies = {
            "Zero": 0,
            "Mimic_Mean": benign_val,
            "Mimic_95th": camou_val,
            "Padding_x10": None # special case
        }
        
        for strat_name, val in strategies.items():
            df_perturbed = df.copy()
            # Ensure column is float to avoid "Invalid value for dtype int64"
            df_perturbed[col] = df_perturbed[col].astype(float)
            
            if strat_name == "Padding_x10":
                df_perturbed.loc[df[label_col] == 1, col] = df.loc[df[label_col] == 1, col] * 10
            else:
                df_perturbed.loc[df[label_col] == 1, col] = val
            
            G_p = create_dgl_graph(df_perturbed, feature_cols, src_col, dst_col, label_col, scaler).to(device)
            recall_p = evaluate(model, G_p)
            drop = base_recall - recall_p
            
            results.append({
                "feature": col,
                "strategy": strat_name,
                "recall": recall_p,
                "drop": drop
            })
            
            if drop > 0.05:
                print(f"  ➜ {strat_name}: Recall dropped to {recall_p:.4f} (Drop: {drop:.4f})")

    res_df = pd.DataFrame(results)
    res_df.to_csv(args.output, index=False)
    print(f"✅ Résultats sauvegardés dans {args.output}")

if __name__ == "__main__":
    main()
