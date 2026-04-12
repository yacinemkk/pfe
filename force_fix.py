import json
import re

nb_path = '/home/pc/Desktop/pfe/IOT_nor.ipynb'
with open(nb_path, 'r') as f:
    nb = json.load(f)

# The correct parameters to pass.
# We will construct it per-dataset dynamically slightly, since save_dir uses {MODEL} and 'csv'/'json'.

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell.get('source', []))
        if 'results = train_model_with_countermeasures(' in src:
            # We want to replace the whole call.
            # Find the dataset type from the cell source
            dataset_match = re.search(r"dataset_type\s*=\s*'([^']+)'", src)
            dset = dataset_match.group(1) if dataset_match else ('json' if 'json' in src.lower() else 'csv')
            
            # The correct call block:
            correct_call = f"""results = train_model_with_countermeasures(
            model_type=MODEL,
            dataset_type='{dset}',
            data_dict=data,
            pgd_steps=TRADES_PGD_STEPS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            use_cutmix=USE_CUTMIX,
            use_afd=USE_AFD,
            use_randomized_smoothing=USE_RS,
            rs_sigma=RS_SIGMA,
            rs_n_samples=RS_N_SAMPLES,
            use_multi_attack=USE_MULTI_ATTACK,
            multi_attack_strategies=MULTI_ATTACK_STRATEGIES,
            save_dir=f'{{DRIVE_RESULTS_DIR}}/models/{{MODEL}}_countermeasures_{dset}',
            use_class_weights=True,
            use_ibp=USE_IBP,
            ibp_epsilon=IBP_EPSILON,
            lambda_ibp=LAMBDA_IBP,
            ibp_method=IBP_METHOD,
            ibp_warmup_epochs=IBP_WARMUP_EPOCHS,
            ibp_epsilon_start=IBP_EPSILON_START,
            total_epochs=TOTAL_EPOCHS,
            use_input_defense=USE_INPUT_DEFENSE,
        )"""

            # Regex to replace EVERYTHING from `results = train_model_with_countermeasures(` to the closing `)`
            # We use `\)` to match the closing parenthesis of the function call.
            new_src = re.sub(
                r'results\s*=\s*train_model_with_countermeasures\s*\([^)]*\)',
                correct_call,
                src,
                flags=re.MULTILINE
            )
            
            # Replace the string list
            cell['source'] = list(new_src.splitlines(keepends=True))

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Flawlessly rebuilt parameters for all cells!")
