import json

nb_path = '/home/pc/Desktop/pfe/IOT_nor.ipynb'
with open(nb_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Determine dataset type from either cell contents or some lines
        src = "".join(cell.get('source', []))
        dset = 'json' if 'json' in src.lower() else 'csv'
        
        lines = cell.get('source', [])
        new_lines = []
        in_call = False
        replaced = False
        
        for line in lines:
            if 'results = train_model_with_countermeasures(' in line and not replaced:
                in_call = True
                
                # Append the exact correct block!
                new_lines.append("        results = train_model_with_countermeasures(\n")
                new_lines.append("            model_type=MODEL,\n")
                new_lines.append(f"            dataset_type='{dset}',\n")
                new_lines.append("            data_dict=data,\n")
                new_lines.append("            pgd_steps=TRADES_PGD_STEPS,\n")
                new_lines.append("            batch_size=BATCH_SIZE,\n")
                new_lines.append("            lr=LEARNING_RATE,\n")
                new_lines.append("            use_cutmix=USE_CUTMIX,\n")
                new_lines.append("            use_afd=USE_AFD,\n")
                new_lines.append("            use_randomized_smoothing=USE_RS,\n")
                new_lines.append("            rs_sigma=RS_SIGMA,\n")
                new_lines.append("            rs_n_samples=RS_N_SAMPLES,\n")
                new_lines.append("            use_multi_attack=USE_MULTI_ATTACK,\n")
                new_lines.append("            multi_attack_strategies=MULTI_ATTACK_STRATEGIES,\n")
                new_lines.append(f"            save_dir=f'{{DRIVE_RESULTS_DIR}}/models/{{MODEL}}_countermeasures_{dset}',\n")
                new_lines.append("            use_class_weights=True,\n")
                new_lines.append("            use_ibp=USE_IBP,\n")
                new_lines.append("            ibp_epsilon=IBP_EPSILON,\n")
                new_lines.append("            lambda_ibp=LAMBDA_IBP,\n")
                new_lines.append("            ibp_method=IBP_METHOD,\n")
                new_lines.append("            ibp_warmup_epochs=IBP_WARMUP_EPOCHS,\n")
                new_lines.append("            ibp_epsilon_start=IBP_EPSILON_START,\n")
                new_lines.append("            total_epochs=TOTAL_EPOCHS,\n")
                new_lines.append("            use_input_defense=USE_INPUT_DEFENSE,\n")
                new_lines.append("        )\n")
                replaced = True
                
            elif in_call:
                # We are skipping the old parameters. 
                # When we see a closing parenthesis that belongs to this call (a line mostly with `)` )
                if line.strip() == ')' or line.strip() == '),\n' or line.strip() == ')':
                    in_call = False
                # Just skip
                continue
            else:
                new_lines.append(line)
        
        cell['source'] = new_lines

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Flawlessly rebuilt parameters for all cells!")
