import json
import re

nb_path = '/home/pc/Desktop/pfe/IOT_nor.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell.get('source', []))
        if 'def train_epoch_robust' in src:
            if 'tqdm(train_loader' not in src:
                # Replace `for X_batch, y_batch in train_loader:`
                # with tqdm wrapper
                new_src = re.sub(
                    r'(^\s*)for X_batch, y_batch in train_loader:',
                    r'\1from tqdm.auto import tqdm\n\1for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):',
                    src,
                    flags=re.MULTILINE
                )
                cell['source'] = list(new_src.splitlines(keepends=True))

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Tqdm injected successfully.")
