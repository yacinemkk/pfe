import re

with open('build_greedy_notebook.py', 'r') as f:
    content = f.read()

# 1. Inject Discriminator and Router definitions
disc_code = """# =====================================================================
# Discriminator and Router
# =====================================================================

class Discriminator(nn.Module):
    def __init__(self, input_size, seq_length, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h_cat = torch.cat([h[0], h[1]], dim=1)
        return self.head(h_cat).squeeze(1)
    
    def predict_proba(self, x):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

def train_discriminator(discriminator, X_train, simulator, device=None, epochs=25, batch_size=64, lr=1e-3, save_path='discriminator.pt'):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\\n{'='*65}\\n  ENTRAÎNEMENT DU DISCRIMINATEUR\\n{'='*65}")
    discriminator = discriminator.to(device)
    optimizer = torch.optim.AdamW(discriminator.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    n = len(X_train)
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        discriminator.train()
        n_half = n // 2
        idx = np.random.permutation(n)
        idx_clean = idx[:n_half]
        idx_adv = idx[n_half:n_half*2]
        k_values = np.random.randint(1, 4, size=n_half)
        X_adv_list = []
        for orig_i, k in zip(idx_adv, k_values):
            X_adv_list.append(simulator.generate_greedy(X_train[[orig_i]], k=k))
        X_adv_ep = np.concatenate(X_adv_list, axis=0)
        X_combined = np.concatenate([X_train[idx_clean], X_adv_ep], axis=0)
        labels_bin = np.array([0.0]*n_half + [1.0]*n_half, dtype=np.float32)
        perm = np.random.permutation(len(X_combined))
        X_combined = X_combined[perm]
        labels_bin = labels_bin[perm]
        loader = DataLoader(TensorDataset(torch.FloatTensor(X_combined), torch.FloatTensor(labels_bin)), batch_size=batch_size, shuffle=True)
        total_loss, total_correct, total_n = 0.0, 0, 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = discriminator(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            optimizer.step()
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == yb).sum().item()
            total_loss += loss.item() * len(yb)
            total_n += len(yb)
        acc = total_correct / total_n
        print(f"  Epoch {epoch:3d}/{epochs}  Loss={total_loss/total_n:.4f}  Acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({'model_state_dict': discriminator.state_dict(), 'accuracy': acc}, save_path)
    print(f"\\n  Discriminateur — meilleure accuracy : {best_acc:.4f}\\n  Sauvegardé → {save_path}")
    ckpt = torch.load(save_path, map_location=device)
    discriminator.load_state_dict(ckpt['model_state_dict'])
    return discriminator, best_acc

class IoTRouter(nn.Module):
    def __init__(self, normal_model, adversarial_model, discriminator, threshold=0.5):
        super().__init__()
        self.normal = normal_model
        self.adversarial = adversarial_model
        self.discriminator = discriminator
        self.threshold = threshold
    
    @torch.no_grad()
    def predict(self, X):
        self.normal.eval()
        self.adversarial.eval()
        self.discriminator.eval()
        attack_scores = self.discriminator.predict_proba(X)
        is_attacked = (attack_scores >= self.threshold)
        logits_normal = self.normal(X)
        logits_adv = self.adversarial(X)
        pred_normal = logits_normal.argmax(1)
        pred_adv = logits_adv.argmax(1)
        predictions = torch.where(is_attacked, pred_adv, pred_normal)
        routes = is_attacked.long()
        return predictions, routes, attack_scores
    
    def calibrate_threshold(self, X_clean, X_attacked, target_recall=0.95):
        with torch.no_grad():
            scores_clean = self.discriminator.predict_proba(X_clean).cpu().numpy()
            scores_attacked = self.discriminator.predict_proba(X_attacked).cpu().numpy()
        all_scores = np.concatenate([scores_clean, scores_attacked])
        all_labels = np.array([0]*len(scores_clean) + [1]*len(scores_attacked))
        thresholds = np.linspace(0.0, 1.0, 200)
        best_t, best_f1 = 0.5, 0.0
        for t in thresholds:
            preds = (all_scores >= t).astype(int)
            tp = ((preds == 1) & (all_labels == 1)).sum()
            fp = ((preds == 1) & (all_labels == 0)).sum()
            fn = ((preds == 0) & (all_labels == 1)).sum()
            recall = tp / (tp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            if recall >= target_recall and f1 > best_f1:
                best_f1 = f1
                best_t = t
        self.threshold = best_t
        print(f"  Seuil calibré : {best_t:.3f}  (recall attaques ≥ {target_recall:.0%})")
        return best_t

# =====================================================================
# train_model_greedy — orchestrate full 3-phase training for one model
# =====================================================================
"""

content = content.replace("""# =====================================================================
# train_model_greedy — orchestrate full 3-phase training for one model
# =====================================================================
""", disc_code)

# 2. Inject evaluation and discriminator training
eval_code = """    # ─── PHASE D: Discriminator ──────────────────────────────────────────
    disc_path = f'{save_dir}/discriminator.pt'
    disc = Discriminator(input_size=input_size, seq_length=10, hidden_size=64)
    if os.path.exists(disc_path):
        print(f"\\n  Discriminator model found in Drive. Loading...")
        ckpt = torch.load(disc_path, map_location=device)
        disc.load_state_dict(ckpt['model_state_dict'])
        disc_acc = ckpt.get('accuracy', 0.95)
        disc = disc.to(device)
        print(f"  Loaded Discriminator (acc={disc_acc:.4f})")
    else:
        disc, disc_acc = train_discriminator(
            discriminator=disc,
            X_train=X_train,
            simulator=simulator,
            device=device,
            epochs=25,
            batch_size=batch_size,
            save_path=disc_path
        )
    
    # ─── Final evaluation on test set with Router ────────────────────────
    print(f"\\n{'='*80}")
    print(f"  RÉSULTATS ATTENDUS APRÈS ENTRAÎNEMENT")
    print(f"{'='*80}")
    model.eval() # Adversarial model
    disc.eval()

    # Create & load normal model
    normal_model_path = f'{DRIVE_RESULTS_DIR}/models/{model_type}_{dataset_type}/best_val_model.pt'
    normal_model = create_model(model_type, input_size, num_classes)
    if os.path.exists(normal_model_path):
        try:
            ckpt_nor = torch.load(normal_model_path, map_location=device)
            if 'model_state_dict' in ckpt_nor:
                normal_model.load_state_dict(ckpt_nor['model_state_dict'])
            else:
                normal_model.load_state_dict(ckpt_nor)
            print(f"  Normal model loaded from {normal_model_path}")
        except Exception as e:
            print(f"  Could not load normal model: {e}. Using random weights.")
    else:
        print(f"  WARN: Normal model not found at {normal_model_path}. Using random weights.")
    normal_model = normal_model.to(device)
    normal_model.eval()

    router = IoTRouter(normal_model, model, disc, threshold=0.5)
    
    # Calibrate router limit
    X_val_clean_sub = torch.FloatTensor(X_val[:min(len(X_val), 1000)]).to(device)
    X_val_adv_sub = torch.FloatTensor(simulator.generate_greedy(X_val[:min(len(X_val), 1000)], k=4)).to(device)
    router.calibrate_threshold(X_val_clean_sub, X_val_adv_sub, target_recall=0.95)
    
    # Evaluate Clean
    test_dataset = IoTSequenceDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE)
    correct_clean = 0
    total = 0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            pred, _, _ = router.predict(X_b)
            total += y_b.size(0)
            correct_clean += pred.eq(y_b).sum().item()
    clean_acc = correct_clean / total

    # Evaluate pure models
    correct_normal_clean = 0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            _, pred = normal_model(X_b).max(1)
            correct_normal_clean += pred.eq(y_b).sum().item()
    normal_clean_acc = correct_normal_clean / total

    # Evaluate Adv k=4 on Adv Model
    adv_results = {}
    correct_adv_model_k4 = 0
    correct_router_adv_k4 = 0
    total_adv = 0
    
    clean_results = crash_test_greedy(model, X_test, y_test, simulator=None, device=device, label='Final Test - No Attack')
    
    for k in [1, 2, 3, 4]:
        c_adv_k = 0
        t_adv_k = 0
        for i in range(0, min(len(X_test), EVAL_SUBSAMPLE), EVAL_BATCH_SIZE):
            batch_end = min(i + EVAL_BATCH_SIZE, len(X_test), EVAL_SUBSAMPLE)
            X_sub = X_test[i:batch_end]
            y_sub = y_test[i:batch_end]
            
            X_adv = simulator.generate_greedy(X_sub, k=k)
            X_adv_t = torch.FloatTensor(X_adv).to(device)
            y_sub_t = torch.LongTensor(y_sub).to(device)
            
            with torch.no_grad():
                _, pred_adv = model(X_adv_t).max(1)
                
            c_adv_k += pred_adv.eq(y_sub_t).sum().item()
            t_adv_k += len(y_sub)
        adv_results[f'k{k}'] = c_adv_k / max(t_adv_k, 1)
        if k == 4:
            adv_model_k4_acc = adv_results[f'k{k}']
            # router test on k4
            for i in range(0, min(len(X_test), EVAL_SUBSAMPLE), EVAL_BATCH_SIZE):
                batch_end = min(i + EVAL_BATCH_SIZE, len(X_test), EVAL_SUBSAMPLE)
                X_sub = X_test[i:batch_end]
                y_sub = y_test[i:batch_end]
                X_adv = simulator.generate_greedy(X_sub, k=4)
                X_adv_t = torch.FloatTensor(X_adv).to(device)
                y_sub_t = torch.LongTensor(y_sub).to(device)
                with torch.no_grad():
                    pred_router, _, _ = router.predict(X_adv_t)
                correct_router_adv_k4 += pred_router.eq(y_sub_t).sum().item()
            router_k4_acc = correct_router_adv_k4 / max(t_adv_k, 1)

    global_acc = (clean_acc + router_k4_acc) / 2.0

    print(f"──────────────────────────────────────")
    print(f"    Modèle normal     → Clean accuracy     : {normal_clean_acc*100:.1f}%")
    print(f"    Modèle antagoniste→ Adversarial acc k4 : {adv_model_k4_acc*100:.1f}%")
    print(f"    Discriminateur    → Détection attaque  : {disc_acc*100:.1f}%")
    print(f"    Système complet   → Accuracy globale   : {global_acc*100:.1f}%")
    print(f"================================================================================")
    
    results = {
        'model_type': model_type,
        'dataset_type': dataset_type,
        'clean_accuracy': clean_acc,
        'adversarial_accuracies': adv_results,
        'crash_tests': all_crash_results,
        'input_size': input_size,
        'num_classes': num_classes,
        'phases': {
            'A': {'epochs': f'1-{PHASE_A_EPOCHS}', 'mix_ratio': PHASE_A_MIX_RATIO, 'k_max': 0},
            'B': {'epochs': f'{PHASE_A_EPOCHS+1}-{PHASE_B_EPOCHS}', 'mix_ratio': PHASE_B_MIX_RATIO, 'k_max': PHASE_B_K_MAX},
            'C': {'epochs': f'{PHASE_B_EPOCHS+1}-{PHASE_C_EPOCHS}', 'mix_ratio': PHASE_C_MIX_RATIO, 'k_max': PHASE_C_K_MAX},
        }
    }

    with open(f'{save_dir}/greedy_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    aggressive_cleanup()
    return results

"""

content = content.replace("""    # ─── Final evaluation on test set ──────────────────────────────────────
    model.eval()
    test_dataset = IoTSequenceDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE)

    correct_clean = 0
    total = 0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            _, pred = model(X_b).max(1)
            total += y_b.size(0)
            correct_clean += pred.eq(y_b).sum().item()
    clean_acc = correct_clean / total

    adv_results = {}
    for k in [1, 2, 3, 4]:
        correct_adv = 0
        total_adv = 0
        for i in range(0, min(len(X_test), EVAL_SUBSAMPLE), EVAL_BATCH_SIZE):
            batch_end = min(i + EVAL_BATCH_SIZE, len(X_test), EVAL_SUBSAMPLE)
            X_sub = X_test[i:batch_end]
            y_sub = y_test[i:batch_end]
            X_adv = simulator.generate_greedy(X_sub, k=k)
            X_adv_t = torch.FloatTensor(X_adv).to(device)
            y_sub_t = torch.LongTensor(y_sub).to(device)
            with torch.no_grad():
                _, pred = model(X_adv_t).max(1)
            correct_adv += pred.eq(y_sub_t).sum().item()
            total_adv += len(y_sub)
        adv_results[f'k{k}'] = correct_adv / max(total_adv, 1)

    results = {
        'model_type': model_type,
        'dataset_type': dataset_type,
        'clean_accuracy': clean_acc,
        'adversarial_accuracies': adv_results,
        'crash_tests': all_crash_results,
        'input_size': input_size,
        'num_classes': num_classes,
        'phases': {
            'A': {'epochs': f'1-{PHASE_A_EPOCHS}', 'mix_ratio': PHASE_A_MIX_RATIO, 'k_max': 0},
            'B': {'epochs': f'{PHASE_A_EPOCHS+1}-{PHASE_B_EPOCHS}', 'mix_ratio': PHASE_B_MIX_RATIO, 'k_max': PHASE_B_K_MAX},
            'C': {'epochs': f'{PHASE_B_EPOCHS+1}-{PHASE_C_EPOCHS}', 'mix_ratio': PHASE_C_MIX_RATIO, 'k_max': PHASE_C_K_MAX},
        }
    }

    with open(f'{save_dir}/greedy_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\\n{'='*80}")
    print(f"  FINAL RESULTS — {model_type.upper()} on {dataset_type.upper()}")
    print(f"{'='*80}")
    print(f"  Clean Accuracy: {clean_acc:.4f}")
    for k, acc in adv_results.items():
        rr = acc / max(clean_acc, 1e-8)
        print(f"  Adversarial {k}: {acc:.4f} (RR={rr:.3f})")
    print(f"  Results saved to {save_dir}/greedy_results.json")

    aggressive_cleanup()

    return results

""", eval_code)

with open('build_greedy_notebook.py', 'w') as f:
    f.write(content)
