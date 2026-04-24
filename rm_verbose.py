import json

path = '/home/pc/Desktop/pfe/greedy_new.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        new_source = []
        for text in cell.get('source', []):
            # Split the string chunk into lines
            lines = text.split('\n')
            filtered_lines = [line for line in lines if '[VERBOSE]' not in line]
            # Join back
            new_text = '\n'.join(filtered_lines)
            new_source.append(new_text)
        cell['source'] = new_source

with open(path, 'w') as f:
    json.dump(nb, f)
    
print("Removed verbose lines from greedy_new.ipynb")
