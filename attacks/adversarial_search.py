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
        df['Label'] = df[label_col].map({'Normal': 0, 'Botnet': 1})
        src_col, dst_col = 'src_ip', 'dst_ip'
    else: # cic
        label_col = 'Label'
        metadata_cols = ['Src IP', 'Dst IP', 'Label', 'timestamp', 'Timestamp']
        src_col, dst_col = 'Src IP', 'Dst IP'
    
    feature_cols = [c for c in df.columns if c not in metadata_cols]
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
    G.edata['label'] = th.cat([th.tensor(df[label_col].values, dtype=th.long)] * 2)
    G.ndata['h'] = th.ones(n_nodes, 1, len(feature_cols))
    return G

def evaluate(model, G):
    model.eval()
    with th.no_grad():
        logits = model(G, G.ndata['h'], G.edata['h'])
        preds = logits.argmax(1).cpu().numpy()
        labels = G.edata['label'].cpu().numpy()
        recall = recall_score(labels, preds, pos_label=1, zero_division=0)
    return recall

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--scaler_path", type=str, default=None)
    parser.add_argument("--sensitivity_csv", type=str, required=True)
    parser.add_argument("--target_recall", type=float, default=0.5)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    df, feature_cols, src_col, dst_col, label_col = load_data_and_prep(args.test_csv, args.dataset)
    ndim = len(feature_cols)
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    
    model = EGraphSAGE(ndim, 128, ndim, F.relu, 0.2).to(device)
    model.load_state_dict(th.load(args.model_path, map_location=device))
    scaler = joblib.load(args.scaler_path) if args.scaler_path else None
    
    # Load sensitivity results and pick best strategies
    sens_df = pd.read_csv(args.sensitivity_csv)
    # Sort by drop
    sens_df = sens_df.sort_values(by="drop", ascending=False)
    
    current_df = df.copy()
    G_base = create_dgl_graph(current_df, feature_cols, src_col, dst_col, label_col, scaler).to(device)
    current_recall = evaluate(model, G_base)
    print(f"[*] Baseline Recall: {current_recall:.4f}")
    
    perturbed_features = []
    
    # Greedy approach: add one feature-strategy pair at a time
    for _, row in sens_df.iterrows():
        feat = row['feature']
        strat = row['strategy']
        
        # Don't perturb the same feature twice
        if feat in [p['feature'] for p in perturbed_features]:
            continue
            
        print(f"[*] Testing {feat} with {strat} (Previous Recall: {current_recall:.4f})")
        
        # Apply perturbation to ATTACK samples
        temp_df = current_df.copy()
        temp_df[feat] = temp_df[feat].astype(float)
        
        if strat == "Zero": val = 0
        elif strat == "Mimic_Mean": val = df[df[label_col] == 0][feat].mean()
        elif strat == "Mimic_95th": val = df[df[label_col] == 0][feat].quantile(0.95)
        elif strat == "Padding_x10": val = current_df.loc[df[label_col] == 1, feat] * 10
        
        temp_df.loc[df[label_col] == 1, feat] = val
        
        G_p = create_dgl_graph(temp_df, feature_cols, src_col, dst_col, label_col, scaler).to(device)
        new_recall = evaluate(model, G_p)
        
        if new_recall < current_recall:
            current_recall = new_recall
            current_df = temp_df
            perturbed_features.append({"feature": feat, "strategy": strat, "recall": new_recall})
            print(f"  ➜ ADDED: Recall dropped to {new_recall:.4f}")
        else:
            print(f"  ➜ SKIPPED: No improvement (Recall: {new_recall:.4f})")
            
        if current_recall <= args.target_recall:
            print(f"🎯 Mission Accomplished! Recall ({current_recall:.4f}) <= Target ({args.target_recall:.4f})")
            break

    # Save final set
    res_df = pd.DataFrame(perturbed_features)
    res_df.to_csv(args.output, index=False)
    print(f"✅ Minimal Evasion Set saved in {args.output}")

if __name__ == "__main__":
    main()
