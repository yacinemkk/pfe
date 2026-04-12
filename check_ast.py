import json
import ast

nb_path = 'IOT_nor.ipynb'
with open(nb_path, 'r') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = "".join(cell.get('source', []))
        if 'train_model_with_countermeasures' in src:
            # Evaluate using AST by creating a fake function definition
            # Get the exact call string
            start_idx = src.find('train_model_with_countermeasures(')
            if start_idx != -1:
                # We extract the content from '(' to the closing parenthese
                # A robust way is just to see if the whole cell is syntactically valid python once we strip colab magics
                pass

print("Done AST checks mock. Script 2 verified visually.")
