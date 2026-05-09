import json

def patch_file(filename):
    with open(filename, "r") as f:
        nb = json.load(f)

    old_code = """            with torch.amp.autocast('cuda', enabled=use_amp):
                if mix_ratio > 0 and simulator is not None:
                    # ── Génération adversariale stratifiée ou stochastique ───────────
                    if adv_method == 'stratified' and phase in ('B', 'C', 'D'):
                        y_np_batch = y_batch.numpy()
                        X_mixed, y_mixed_np = simulator.generate_training_batch_stratified(
                            X_np, y_np_batch, model, device,
                            phase=phase, k_max=k_max, mix_ratio=mix_ratio,
                            n_candidates=n_candidates, top_n_k2=top_n_k2,
                        )
                        y_input = torch.LongTensor(y_mixed_np).to(device)
                    else:
                        X_mixed, _ = simulator.generate_training_batch(X_np, k_max=k_max, mix_ratio=mix_ratio)

                    if is_nlp:
                        X_input = torch.LongTensor(tokenizer.transform(X_mixed, features)).to(device)
                    else:
                        X_input = torch.FloatTensor(X_mixed).to(device)
                    if p_drop > 0 and not is_nlp:
                        mask = (torch.rand(X_input.shape[0], 1, X_input.shape[2], device=device) > p_drop).float()
                        X_input = X_input * mask / (1.0 - p_drop)
                    if sigma_noise > 0:
                        X_input = X_input + torch.randn_like(X_input) * sigma_noise

                    logits = model(X_input)
                    loss = criterion(logits, y_input)
                else:
                    if is_nlp:
                        X_input = torch.LongTensor(tokenizer.transform(X_np, features)).to(device)
                    else:
                        X_input = X_batch.to(device)
                    if p_drop > 0 and not is_nlp:
                        mask = (torch.rand(X_input.shape[0], 1, X_input.shape[2], device=device) > p_drop).float()
                        X_input = X_input * mask / (1.0 - p_drop)
                    if sigma_noise > 0:
                        X_input = X_input + torch.randn_like(X_input) * sigma_noise



                    logits = model(X_input)
                    loss = criterion(logits, y_input)

            # ── anti-NaN batch skip ───────────────────────────────
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                optimizer.zero_grad()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(y_input)
            total_correct += (logits.argmax(1) == y_input).sum().item()
            total_n += len(y_input)"""

    new_code = """            if mix_ratio > 0 and simulator is not None:
                # ── Génération adversariale stratifiée ou stochastique ───────────
                if adv_method == 'stratified' and phase in ('B', 'C', 'D'):
                    y_np_batch = y_batch.numpy()
                    # It's better to NOT use autocast during generation inside this loop to avoid nested/missed casts unless it's within the inner model forward, but we'll leave the autocast context to generation itself.
                    # Wait, generation calls model(...) which expects the correct types.
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        X_mixed, y_mixed_np = simulator.generate_training_batch_stratified(
                            X_np, y_np_batch, model, device,
                            phase=phase, k_max=k_max, mix_ratio=mix_ratio,
                            n_candidates=n_candidates, top_n_k2=top_n_k2,
                        )
                    y_input = torch.LongTensor(y_mixed_np).to(device)
                else:
                    X_mixed, _ = simulator.generate_training_batch(X_np, k_max=k_max, mix_ratio=mix_ratio)
                    y_input = y_batch.to(device)
            else:
                X_mixed = X_np
                y_input = y_batch.to(device)

            # ── Memory-efficient chunked forward/backward ───────────────────
            chunk_sz = batch_size
            n_chunks = (len(X_mixed) + chunk_sz - 1) // chunk_sz
            
            batch_loss = 0.0
            batch_correct = 0
            skip_batch = False
            
            for c_idx in range(n_chunks):
                start = c_idx * chunk_sz
                end = min(start + chunk_sz, len(X_mixed))
                X_chunk = X_mixed[start:end]
                y_chunk_input = y_input[start:end]
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    if is_nlp:
                        X_input_chunk = torch.LongTensor(tokenizer.transform(X_chunk, features)).to(device)
                    else:
                        X_input_chunk = torch.FloatTensor(X_chunk).to(device)
                        
                    if p_drop > 0 and not is_nlp:
                        mask = (torch.rand(X_input_chunk.shape[0], 1, X_input_chunk.shape[2], device=device) > p_drop).float()
                        X_input_chunk = X_input_chunk * mask / (1.0 - p_drop)
                    if sigma_noise > 0:
                        X_input_chunk = X_input_chunk + torch.randn_like(X_input_chunk) * sigma_noise
                        
                    logits_chunk = model(X_input_chunk)
                    loss_chunk = criterion(logits_chunk, y_chunk_input) * (len(X_chunk) / len(X_mixed))
                
                if torch.isnan(loss_chunk) or torch.isinf(loss_chunk):
                    skip_batch = True
                    break
                    
                scaler.scale(loss_chunk).backward()
                
                batch_loss += loss_chunk.item() * (len(X_mixed) / max(1, len(X_chunk))) * (len(X_chunk) / len(X_mixed)) # which is loss_chunk.item() without scaling down? Wait.
                # Actually, loss_chunk is already scaled down. So its value is the contribution to the total mean loss.
                # If we sum it, we get the total mean loss for the whole X_mixed batch.
                batch_loss += loss_chunk.item() 
                batch_correct += (logits_chunk.argmax(1) == y_chunk_input).sum().item()

            if skip_batch:
                nan_batches += 1
                optimizer.zero_grad()
                continue
                
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            total_loss += batch_loss * len(X_mixed)
            total_correct += batch_correct
            total_n += len(X_mixed)"""

    found = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            if old_code in source:
                source = source.replace(old_code, new_code)
                cell['source'] = [line + '\n' for line in source.split('\n')]
                cell['source'][-1] = cell['source'][-1].rstrip('\n')
                found = True

    if found:
        with open(filename, "w") as f:
            json.dump(nb, f, indent=1)
        print(f"Patched {filename}")
    else:
        print(f"Old code not found in {filename}")

patch_file("greedy_new_optimized.ipynb")
