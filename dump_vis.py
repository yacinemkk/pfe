import json

with open("iot_adversarial_colab.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'Cell 6: Visualise Results' in "".join(cell.get('source', [])):
        print("".join(cell['source']))
