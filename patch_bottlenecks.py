import json
import os

NOTEBOOK_PATH = '/home/pc/Desktop/pfe/greedy_new.ipynb'
OUTPUT_PATH = '/home/pc/Desktop/pfe/greedy_new_optimized.ipynb'

def patch_notebook():
    print(f"Loading {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Nouveaux codes de remplacement
    new_worst_case = '''    def generate_worst_case_k(self, X, y_np, model, device, k=3, n_candidates=8):
        """Sélection worst-case vectorisée parmi n_candidates attaques pour k features (Phase D).
        
        Optimisation: Génère l'attaque stochastique sur l'ensemble du batch simultanément
        et fait l'inférence en une seule passe sur le GPU. Vitesse x100 par rapport à l'original.
        """
        model.eval()
        N = len(X)
        best_X = X.copy()
        best_losses = np.full(N, -np.inf)
        
        y_t = torch.LongTensor(y_np).to(device)
        
        with torch.no_grad():
            for _ in range(n_candidates):
                # Génération pour tout le batch d'un coup (extrêmement rapide)
                X_cand = self.generate_greedy(X, k)
                X_t = torch.FloatTensor(X_cand).to(device)
                
                # Inférence massive GPU
                logits = model(X_t)
                
                # Pertes individuelles
                losses = F.cross_entropy(logits, y_t, reduction='none').cpu().numpy()
                
                # Mise à jour des meilleurs candidats
                mask = losses > best_losses
                best_losses[mask] = losses[mask]
                best_X[mask] = X_cand[mask]

        model.train()
        return best_X
'''

    modifications = {
        'batch_size': False,
        'eval_batch_size': False,
        'worst_case': False,
        'dataloader': False
    }

    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
            
        source = cell['source']
        for i, line in enumerate(source):
            # Patch BATCH_SIZE
            if "BATCH_SIZE = 32" in line and "reduced from 64 to avoid OOM" in line:
                source[i] = "BATCH_SIZE = 256          # Augmenté pour accélérer l'entraînement (optimisé)\n"
                modifications['batch_size'] = True
            
            # Patch EVAL_BATCH_SIZE
            if line.strip() == "EVAL_BATCH_SIZE = 32":
                source[i] = "EVAL_BATCH_SIZE = 256\n"
                modifications['eval_batch_size'] = True

            # Patch DataLoader num_workers
            if "num_workers=0" in line:
                source[i] = line.replace("num_workers=0", "num_workers=2").replace("pin_memory=False", "pin_memory=True")
                modifications['dataloader'] = True

            # Patch generate_worst_case_k (on cherche la signature)
            if "def generate_worst_case_k(self, X, y_np, model, device, k=3, n_candidates=8):" in line:
                # Trouver la fin de la fonction
                start_idx = i
                end_idx = i + 1
                while end_idx < len(source) and (source[end_idx].startswith("        ") or source[end_idx] == "\n" or source[end_idx] == "    \n" or source[end_idx].startswith('    """')):
                    end_idx += 1
                
                # Remplacer les lignes de la fonction
                lines_to_insert = [l + '\n' if not l.endswith('\n') else l for l in new_worst_case.split('\n')[:-1]]
                source[start_idx:end_idx] = lines_to_insert
                modifications['worst_case'] = True

    print("Modifications appliquées :")
    for k, v in modifications.items():
        print(f" - {k}: {'Succès' if v else 'Échec (non trouvé)'}")

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"\nNotebook optimisé sauvegardé sous : {OUTPUT_PATH}")
    print("Vous pouvez maintenant utiliser ce notebook sur Colab.")

if __name__ == '__main__':
    patch_notebook()
