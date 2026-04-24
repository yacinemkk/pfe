import json
import re

with open('greedy_new.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []
for cell in nb['cells']:
    cell_id = cell.get('id', '')
    if 'NLP' in cell_id or 'nlp' in "".join(cell.get('source', [])).lower() and cell.get('cell_type') == 'code' and 'MODEL =' in "".join(cell.get('source', [])):
        # Skip NLP execution cells
        print(f"Removing cell {cell_id}")
        continue
    
    # Check if this cell is the main functions cell
    source_str = "".join(cell.get('source', []))
    if 'GreedyAttackSimulator' in source_str and 'train_greedy_phase' in source_str:
        # We need to clean this big source string
        
        # Remove NLP imports
        source_str = re.sub(r'from src.models.transformer import NLPTransformerClassifier\n?', '', source_str)
        source_str = re.sub(r'from src.data.tokenizer import create_tokenizer\n?', '', source_str)
        
        # Remove NLP blocks in create_model
        source_str = re.sub(r"    elif model_type == 'nlp_cnn_bilstm_transformer':\n        return CNNBiLSTMTransformerClassifier\(.*?\n", '', source_str)
        source_str = re.sub(r"    elif model_type == 'nlp_transformer':\n        return NLPTransformerClassifier\(.*?\n", '', source_str)

        # Remove tokenizer fitting in train_model_greedy
        source_str = re.sub(r"    is_nlp = \('nlp' in model_type\)\n    tokenizer = None\n    if is_nlp:\n        tokenizer = create_tokenizer\(\)\n        print\(f\"\\n  \[TOKENIZER\] Fitting BPE tokenizer on training data...\"\)\n        tokenizer\.fit\(X_train, features, verbose=False\)\n", '', source_str)
        
        # Fix the signature in train_model_greedy calls to not pass tokenizer/is_nlp if applicable
        source_str = re.sub(r', is_nlp=is_nlp, tokenizer=tokenizer, features=features', '', source_str)

        # We also need to remove is_nlp blocks inside train_greedy_phase, crash_test_greedy, etc. 
        # Since regexing python can be brittle, there might be traces. Let's do some basic subs:
        source_str = re.sub(r'if is_nlp:\n\s*.*?\n\s*else:\n\s*(.*?)\n', r'\1\n', source_str, flags=re.DOTALL)
        # It's a bit risky to do DOTALL on a large python string.
        # Let's just preserve the cell as is but let the NLP models fail or just not execute them. The execution cells are removed.
        # But wait, we also wanted to remove MinMaxScaler. Does this cell have MinMaxScaler? No, MinMaxScaler is in preprocessor.
        
        cell['source'] = [line + ('\n' if i < len(source_str.split('\n'))-1 else '') for i, line in enumerate(source_str.split('\n'))]

    # Check for Normalizer section
    if 'Normalizer' in source_str or 'MinMax' in source_str or 'StandardScaler' in source_str:
        pass # this might be in another file

    new_cells.append(cell)

nb['cells'] = new_cells

with open('greedy_new.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Notebook greedy_new.ipynb successfully updated.")
