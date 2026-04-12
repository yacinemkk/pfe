import json

nb_path = '/home/pc/Desktop/pfe/IOT_nor.ipynb'
with open(nb_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        
        # Patch 1: _crash_test modifications
        if 'def _crash_test(' in source:
            source = source.replace("active_attack, is_multi_attack, label=''):\n", "active_attack, is_multi_attack, label='', input_defense=None):\n")
            
            source = source.replace(
                "X, y = X.to(device), y.to(device)\n            out = model(X)",
                "X, y = X.to(device), y.to(device)\n            if input_defense is not None:\n                X = input_defense(X)\n            out = model(X)"
            )
            
            source = source.replace(
                "X, y = X.to(device), y.to(device)\n        with torch.no_grad():\n            X_adv = pgd_attack.generate(model, X, y, device)",
                "X, y = X.to(device), y.to(device)\n        if input_defense is not None:\n            X = input_defense(X)\n        with torch.no_grad():\n            X_adv = pgd_attack.generate(model, X, y, device)"
            )
            
            source = source.replace(
                "X, y = X.to(device), y.to(device)\n            with torch.no_grad():\n                result = active_attack.generate(model, X, y, device)",
                "X, y = X.to(device), y.to(device)\n            if input_defense is not None:\n                X = input_defense(X)\n            with torch.no_grad():\n                result = active_attack.generate(model, X, y, device)"
            )

        # Patch 2: validation loop
        if '# ── Validation ─────────────────────────────────────────────────────' in source:
            source = source.replace(
                "X_b, y_b = X_b.to(device), y_b.to(device)\n                val_loss += criterion_eval(model(X_b), y_b).item()",
                "X_b, y_b = X_b.to(device), y_b.to(device)\n                if input_defense is not None:\n                    X_b = input_defense(X_b)\n                val_loss += criterion_eval(model(X_b), y_b).item()"
            )

        # Patch 3: crash test inside loop
        if 'ct = _crash_test(' in source:
            source = source.replace(
                "label=f'Phase{phase}')",
                "label=f'Phase{phase}', input_defense=input_defense)"
            )

        # Patch 4: final evaluation
        if '# ── Final Evaluation ──' in source:
            source = source.replace(
                "X_batch, y_batch = X_batch.to(device), y_batch.to(device)\n            outputs = model(X_batch)",
                "X_batch, y_batch = X_batch.to(device), y_batch.to(device)\n            if input_defense is not None:\n                X_batch = input_defense(X_batch)\n            outputs = model(X_batch)"
            )
            
            source = source.replace(
                "X_batch, y_batch = X_batch.to(device), y_batch.to(device)\n        with torch.no_grad():\n            result = active_attack.generate(model, X_batch, y_batch, device)",
                "X_batch, y_batch = X_batch.to(device), y_batch.to(device)\n        if input_defense is not None:\n            X_batch = input_defense(X_batch)\n        with torch.no_grad():\n            result = active_attack.generate(model, X_batch, y_batch, device)"
            )

        # Apply back
        lines = []
        for line in source.split('\n'):
            lines.append(line + '\n')
            
        if lines and lines[-1] == '\n':
            lines.pop()
        else:
            lines[-1] = lines[-1][:-1]

        if len(lines) > 0:
            cell['source'] = lines

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook patched successfully!")
