import json

with open("/home/pc/Desktop/pfe/iot_adversarial_colab.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the index of the "Phase 7 : Visualisation" cell
vis_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and len(cell['source']) > 0 and 'Phase 7' in cell['source'][0]:
        vis_idx = i
        break

if vis_idx == -1:
    print("Could not find Phase 7 cell!")
    exit(1)

def create_md(title, desc):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"# {title}\n", "\n", f"{desc}"]
    }

def create_code(model_name, cell_num):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            f"# ─── Cell {cell_num}: Train {model_name} ───────────────────────────────────────────────\n",
            "!python train_adversarial.py \\\n",
            f"    --model {model_name} \\\n",
            "    --seq_length {SEQ_LENGTH} \\\n",
            "    --adv_method {ADV_METHOD} \\\n",
            "    --adv_ratio {ADV_RATIO} \\\n",
            "    --epochs {EPOCHS} \\\n",
            "    --batch_size {BATCH_SIZE} \\\n",
            "    --data_dir \"{JSON_DATA_DIR}\" \\\n",
            "    --results_dir \"{DRIVE_RESULTS_DIR}\" \\\n",
            "    --phase_checkpoints"
        ]
    }

new_cells = [
    create_md("🧩 Phase 6.1 : Entraînement du BiLSTM", "Architecture LSTM bidirectionnelle."),
    create_code("bilstm", "6_1"),
    create_md("🧩 Phase 6.2 : Entraînement du CNN-BiLSTM", "Architecture combinant extraction spatiale et temporelle bidirectionnelle."),
    create_code("cnn_bilstm", "6_2"),
    create_md("🏆 Phase 6.3 : Entraînement du CNN-BiLSTM-Transformer", "Architecture hybride finale state-of-the-art."),
    create_code("cnn_bilstm_transformer", "6_3")
]

nb['cells'] = nb['cells'][:vis_idx] + new_cells + nb['cells'][vis_idx:]

with open("/home/pc/Desktop/pfe/iot_adversarial_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    
print("Notebook updated successfully!")
