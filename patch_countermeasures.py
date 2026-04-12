import re

with open('build_greedy_notebook.py', 'r') as f:
    text = f.read()

# 1. Import AFDLoss
text = text.replace(
    "from src.training.trainer import IoTSequenceDataset",
    "from src.training.trainer import IoTSequenceDataset\\nfrom src.adversarial.robust_losses import AFDLoss"
)

# 2. Update train_greedy_phase function signature
text = text.replace(
    """def train_greedy_phase(
    model, X_train, y_train, X_val, y_val,
    phase, start_epoch, end_epoch,
    mix_ratio, k_max,
    simulator=None, device=None,
    lr=5e-4, batch_size=64, save_path=None,
):""",
    """def train_greedy_phase(
    model, X_train, y_train, X_val, y_val,
    phase, start_epoch, end_epoch,
    mix_ratio, k_max,
    p_drop=0.0, sigma_noise=0.0, afd_lambda=0.0,
    simulator=None, device=None,
    lr=5e-4, batch_size=64, save_path=None,
):"""
)

# 3. Update the loop inside train_greedy_phase
old_loop = """        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            X_np = X_batch.numpy()

            if mix_ratio > 0 and simulator is not None:
                X_mixed, _ = simulator.generate_training_batch(X_np, k_max=k_max, mix_ratio=mix_ratio)
                X_input = torch.FloatTensor(X_mixed).to(device)
            else:
                X_input = X_batch.to(device)

            y_input = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_input)
            loss = criterion(logits, y_input)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()"""

new_loop = """        if afd_lambda > 0:
            num_classes = len(np.unique(y_train))
            afd_criterion = AFDLoss(num_classes, num_classes, lambda_intra=1.0, lambda_inter=0.5).to(device)

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            X_np = X_batch.numpy()
            y_input = y_batch.to(device)
            optimizer.zero_grad()
            
            if mix_ratio > 0 and simulator is not None:
                if afd_lambda > 0:
                    X_clean_t = X_batch.to(device)
                    X_adv_mixed, _ = simulator.generate_training_batch(X_np, k_max=k_max, mix_ratio=mix_ratio)
                    X_adv_t = torch.FloatTensor(X_adv_mixed).to(device)
                    
                    if p_drop > 0:
                        mask_c = (torch.rand(X_clean_t.shape[0], 1, X_clean_t.shape[2], device=device) > p_drop).float()
                        X_clean_t = X_clean_t * mask_c / (1.0 - p_drop)
                        mask_a = (torch.rand(X_adv_t.shape[0], 1, X_adv_t.shape[2], device=device) > p_drop).float()
                        X_adv_t = X_adv_t * mask_a / (1.0 - p_drop)
                        
                    if sigma_noise > 0:
                        X_clean_t = X_clean_t + torch.randn_like(X_clean_t) * sigma_noise
                        X_adv_t = X_adv_t + torch.randn_like(X_adv_t) * sigma_noise
                        
                    logits_clean = model(X_clean_t)
                    logits_adv = model(X_adv_t)
                    
                    loss_ce = criterion(logits_adv, y_input)
                    loss_afd = afd_criterion(logits_clean, logits_adv, y_input)
                    loss = loss_ce + afd_lambda * loss_afd
                    logits = logits_adv
                else:
                    X_mixed, _ = simulator.generate_training_batch(X_np, k_max=k_max, mix_ratio=mix_ratio)
                    X_input = torch.FloatTensor(X_mixed).to(device)
                    if p_drop > 0:
                        mask = (torch.rand(X_input.shape[0], 1, X_input.shape[2], device=device) > p_drop).float()
                        X_input = X_input * mask / (1.0 - p_drop)
                    if sigma_noise > 0:
                        X_input = X_input + torch.randn_like(X_input) * sigma_noise
                    logits = model(X_input)
                    loss = criterion(logits, y_input)
            else:
                X_input = X_batch.to(device)
                if p_drop > 0:
                    mask = (torch.rand(X_input.shape[0], 1, X_input.shape[2], device=device) > p_drop).float()
                    X_input = X_input * mask / (1.0 - p_drop)
                if sigma_noise > 0:
                    X_input = X_input + torch.randn_like(X_input) * sigma_noise
                logits = model(X_input)
                loss = criterion(logits, y_input)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()"""

text = text.replace(old_loop, new_loop)

# 4. Update the training calls in train_model_greedy
call_A_old = """        model = train_greedy_phase(
            model, X_train, y_train, X_val, y_val,
            phase='A', start_epoch=1, end_epoch=PHASE_A_EPOCHS,
            mix_ratio=PHASE_A_MIX_RATIO, k_max=0,
            simulator=None, device=device, lr=lr,
            batch_size=batch_size, save_path=phase_a_path,
        )"""
call_A_new = """        model = train_greedy_phase(
            model, X_train, y_train, X_val, y_val,
            phase='A', start_epoch=1, end_epoch=PHASE_A_EPOCHS,
            mix_ratio=PHASE_A_MIX_RATIO, k_max=0,
            p_drop=0.0, sigma_noise=0.0, afd_lambda=0.0,
            simulator=None, device=device, lr=lr,
            batch_size=batch_size, save_path=phase_a_path,
        )"""
text = text.replace(call_A_old, call_A_new)

call_B_old = """        model = train_greedy_phase(
            model, X_train, y_train, X_val, y_val,
            phase='B', start_epoch=PHASE_A_EPOCHS + 1, end_epoch=PHASE_B_EPOCHS,
            mix_ratio=PHASE_B_MIX_RATIO, k_max=PHASE_B_K_MAX,
            simulator=simulator, device=device, lr=lr,
            batch_size=batch_size, save_path=phase_b_path,
        )"""
call_B_new = """        model = train_greedy_phase(
            model, X_train, y_train, X_val, y_val,
            phase='B', start_epoch=PHASE_A_EPOCHS + 1, end_epoch=PHASE_B_EPOCHS,
            mix_ratio=PHASE_B_MIX_RATIO, k_max=PHASE_B_K_MAX,
            p_drop=0.1, sigma_noise=0.01, afd_lambda=0.5,
            simulator=simulator, device=device, lr=lr,
            batch_size=batch_size, save_path=phase_b_path,
        )"""
text = text.replace(call_B_old, call_B_new)

call_C_old = """        model = train_greedy_phase(
            model, X_train, y_train, X_val, y_val,
            phase='C', start_epoch=PHASE_B_EPOCHS + 1, end_epoch=PHASE_C_EPOCHS,
            mix_ratio=PHASE_C_MIX_RATIO, k_max=PHASE_C_K_MAX,
            simulator=simulator, device=device, lr=lr,
            batch_size=batch_size, save_path=phase_c_path,
        )"""
call_C_new = """        model = train_greedy_phase(
            model, X_train, y_train, X_val, y_val,
            phase='C', start_epoch=PHASE_B_EPOCHS + 1, end_epoch=PHASE_C_EPOCHS,
            mix_ratio=PHASE_C_MIX_RATIO, k_max=PHASE_C_K_MAX,
            p_drop=0.2, sigma_noise=0.01, afd_lambda=1.0,
            simulator=simulator, device=device, lr=lr,
            batch_size=batch_size, save_path=phase_c_path,
        )"""
text = text.replace(call_C_old, call_C_new)

with open('build_greedy_notebook.py', 'w') as f:
    f.write(text)
print("Patching successful.")
