#!/usr/bin/env python3
"""Patch greedy_new_optimized.ipynb: replace the short JSON load cell
with the full loading + distribution analysis cell, and insert a new
SMOTE verification cell right after it."""

import json, copy

NB_PATH = "/home/pc/Desktop/pfe/greedy_new_optimized.ipynb"

# ── New cell 1: Load JSON Dataset (full) ─────────────────────────────────
LOAD_JSON_SOURCE = r'''# ─── Cell: Load JSON Dataset ─────────────────────────────────────────────
import os
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

if DATASETS in ['json', 'both']:
    JSON_BASE_DIR = f"{DRIVE_RESULTS_DIR}/preprocessed/json"
    JSON_BALANCED_DIR = f"{DRIVE_RESULTS_DIR}/preprocessed/json_smote"
    JSON_DIR = JSON_BALANCED_DIR if os.path.exists(f"{JSON_BALANCED_DIR}/json_ready") else JSON_BASE_DIR

    print(f"Loading JSON dataset from: {JSON_DIR}")

    X_train = np.load(f"{JSON_DIR}/X_train.npy")
    X_val = np.load(f"{JSON_DIR}/X_val.npy")
    X_test = np.load(f"{JSON_DIR}/X_test.npy")
    y_train = np.load(f"{JSON_DIR}/y_train.npy")
    y_val = np.load(f"{JSON_DIR}/y_val.npy")
    y_test = np.load(f"{JSON_DIR}/y_test.npy")

    with open(f"{JSON_DIR}/json_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    print(f"  X_train : {X_train.shape} | y_train : {y_train.shape}")
    print(f"  X_val   : {X_val.shape}   | y_val   : {y_val.shape}")
    print(f"  X_test  : {X_test.shape}  | y_test  : {y_test.shape}")

    print("\nMetadata keys:", list(metadata.keys()))

    if 'class_names' in metadata:
        label_names = metadata['class_names']
    elif 'label_encoder' in metadata:
        label_names = metadata['label_encoder'].classes_
    elif 'classes' in metadata:
        label_names = metadata['classes']
    else:
        label_names = None
        print("  [!] Aucun label_names trouvé dans metadata — affichage class_0, class_1...")

    print(f"  Classes : {label_names}")

    print("=" * 70)
    print("  CLASS DISTRIBUTION ANALYSIS — JSON DATASET")
    print("=" * 70)

    splits = {
        'TRAIN': y_train,
        'VAL': y_val,
        'TEST': y_test,
    }

    for split_name, y in splits.items():
        counter = Counter(y)
        total = len(y)
        n_cls = len(counter)

        print(f"\n{'─'*70}")
        print(f"  {split_name} SET — {total:,} samples | {n_cls} classes")
        print(f"{'─'*70}")
        print(f"  {'ID':>4} | {'Label':<35} | {'Count':>8} | {'%':>6} | Bar")
        print(f"  {'─'*4}-+-{'─'*35}-+-{'─'*8}-+-{'─'*6}-+-{'─'*25}")

        for cls_id, count in sorted(counter.items()):
            pct = 100.0 * count / total
            name = (
                label_names[cls_id]
                if label_names is not None and cls_id < len(label_names)
                else f"class_{cls_id}"
            )
            bar = "█" * int(pct / 2)
            print(f"  {cls_id:>4} | {name:<35} | {count:>8,} | {pct:>5.1f}% | {bar}")

        counts = np.array([counter[c] for c in sorted(counter)])
        ratio = counts.max() / (counts.min() + 1e-9)
        entropy = -np.sum((counts / total) * np.log2(counts / total + 1e-12))
        balance = entropy / np.log2(n_cls)

        print(f"\n  ► Total samples      : {total:,}")
        print(f"  ► Min samples/class  : {counts.min():,}  → class {np.argmin(counts)}"
              + (f" ({label_names[np.argmin(counts)]})" if label_names is not None else ""))
        print(f"  ► Max samples/class  : {counts.max():,}  → class {np.argmax(counts)}"
              + (f" ({label_names[np.argmax(counts)]})" if label_names is not None else ""))
        print(f"  ► Imbalance ratio    : {ratio:.1f}x  "
              f"({'✅ OK' if ratio < 5 else '⚠️  Moderate' if ratio < 20 else '🔴 Severe'})")
        print(f"  ► Balance score      : {balance:.3f}  "
              f"({'✅ OK' if balance > 0.85 else '⚠️  Moderate' if balance > 0.70 else '🔴 Severe'})")

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle("Class Distribution per Split — JSON Dataset", fontsize=14, fontweight='bold')

    colors = ['steelblue', 'darkorange', 'seagreen']

    for ax, (split_name, y), color in zip(axes, splits.items(), colors):
        counter = Counter(y)
        classes = sorted(counter.keys())
        counts = [counter[c] for c in classes]
        x_labels = (
            [label_names[c] for c in classes]
            if label_names is not None
            else [f"cls_{c}" for c in classes]
        )

        bars = ax.bar(range(len(classes)), counts, color=color, alpha=0.8, edgecolor='white')
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        ax.set_title(f"{split_name}  ({len(y):,} samples)", fontweight='bold')
        ax.set_ylabel("Number of samples")
        ax.set_xlabel("Class")

        mean_count = np.mean(counts)
        ax.axhline(mean_count, color='red', linestyle='--', linewidth=1.2, label=f"Mean = {mean_count:,.0f}")
        ax.legend(fontsize=8)

        for bar, cnt in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f"{cnt:,}",
                ha='center',
                va='bottom',
                fontsize=6,
                rotation=90,
            )

    plt.tight_layout()
    save_path = f"{JSON_DIR}/class_distribution_json.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  [Saved] {save_path}")

    json_data = {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'features': metadata.get('features'), 'scaler': metadata.get('scaler'),
        'label_encoder': metadata.get('label_encoder'), 'n_continuous': metadata.get('n_continuous')
    }
else:
    json_data = None
    print('Skipping JSON dataset')
'''

# ── New cell 2: Verify JSON Dataset After SMOTE ──────────────────────────
VERIFY_JSON_SOURCE = r'''# ─── Cell: Verify JSON Dataset After SMOTE ─────────────────────────────────────
import os
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

if DATASETS in ['json', 'both']:
    JSON_BASE_DIR    = f"{DRIVE_RESULTS_DIR}/preprocessed/json"
    JSON_BALANCED_DIR = f"{DRIVE_RESULTS_DIR}/preprocessed/json_smote"

    smote_available = os.path.exists(f"{JSON_BALANCED_DIR}/X_train.npy")

    # ── Load ORIGINAL (pre-SMOTE)
    y_train_orig = np.load(f"{JSON_BASE_DIR}/y_train.npy")
    y_val_orig   = np.load(f"{JSON_BASE_DIR}/y_val.npy")
    y_test_orig  = np.load(f"{JSON_BASE_DIR}/y_test.npy")

    # ── Load SMOTE-balanced
    if smote_available:
        X_train = np.load(f"{JSON_BALANCED_DIR}/X_train.npy")
        X_val   = np.load(f"{JSON_BALANCED_DIR}/X_val.npy")
        X_test  = np.load(f"{JSON_BALANCED_DIR}/X_test.npy")
        y_train = np.load(f"{JSON_BALANCED_DIR}/y_train.npy")
        y_val   = np.load(f"{JSON_BALANCED_DIR}/y_val.npy")
        y_test  = np.load(f"{JSON_BALANCED_DIR}/y_test.npy")

        with open(f"{JSON_BALANCED_DIR}/json_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        JSON_DIR = JSON_BALANCED_DIR
        print(f"✅ SMOTE cache found — loading from: {JSON_BALANCED_DIR}")
    else:
        X_train, X_val, X_test = [None]*3
        y_train, y_val, y_test = y_train_orig, y_val_orig, y_test_orig
        with open(f"{JSON_BASE_DIR}/json_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        JSON_DIR = JSON_BASE_DIR
        print("⚠️  No SMOTE cache found — showing original only.")

    # ── Label names
    if 'class_names' in metadata:
        label_names = metadata['class_names']
    elif 'label_encoder' in metadata:
        label_names = metadata['label_encoder'].classes_
    elif 'classes' in metadata:
        label_names = metadata['classes']
    else:
        label_names = None

    def balance_metrics(counter, total):
        counts = np.array([counter[c] for c in sorted(counter)])
        n_cls  = len(counts)
        ratio  = counts.max() / (counts.min() + 1e-9)
        entropy = -np.sum((counts / total) * np.log2(counts / total + 1e-12))
        balance = entropy / np.log2(n_cls)
        return ratio, balance, counts

    print("\n" + "=" * 75)
    print("  SMOTE VERIFICATION — TRAIN SPLIT (only train is resampled)")
    print("=" * 75)

    counter_orig = Counter(y_train_orig)
    counter_smote = Counter(y_train)
    all_classes = sorted(set(counter_orig) | set(counter_smote))
    total_orig  = len(y_train_orig)
    total_smote = len(y_train)

    print(f"\n  {'ID':>4} | {'Label':<30} | {'Before':>8} | {'After':>8} | {'Δ Added':>8} | {'% Before':>8} | {'% After':>8}")
    print(f"  {'─'*4}-+-{'─'*30}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}")

    for cls_id in all_classes:
        orig_n  = counter_orig.get(cls_id, 0)
        smote_n = counter_smote.get(cls_id, 0)
        delta   = smote_n - orig_n
        pct_o   = 100.0 * orig_n  / total_orig
        pct_s   = 100.0 * smote_n / total_smote
        name    = (label_names[cls_id] if label_names is not None and cls_id < len(label_names)
                   else f"class_{cls_id}")
        flag    = " ✚" if delta > 0 else ""
        print(f"  {cls_id:>4} | {name:<30} | {orig_n:>8,} | {smote_n:>8,} | {delta:>+8,} | {pct_o:>7.2f}% | {pct_s:>7.2f}%{flag}")

    ratio_o, bal_o, _ = balance_metrics(counter_orig,  total_orig)
    ratio_s, bal_s, _ = balance_metrics(counter_smote, total_smote)

    print(f"\n  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │  Metric              │    Before SMOTE  │   After SMOTE  │")
    print(f"  ├─────────────────────────────────────────────────────────┤")
    print(f"  │  Total samples       │ {total_orig:>15,}  │ {total_smote:>13,}  │")
    print(f"  │  Imbalance ratio     │ {ratio_o:>15.1f}x │ {ratio_s:>13.1f}x │")
    print(f"  │  Balance score       │ {bal_o:>15.3f}  │ {bal_s:>13.3f}  │")
    print(f"  └─────────────────────────────────────────────────────────┘")

    imb_icon = lambda r: '✅ OK' if r < 5 else '⚠️  Moderate' if r < 20 else '🔴 Severe'
    bal_icon = lambda b: '✅ OK' if b > 0.85 else '⚠️  Moderate' if b > 0.70 else '🔴 Severe'
    print(f"\n  Imbalance  : {imb_icon(ratio_o)} → {imb_icon(ratio_s)}")
    print(f"  Balance    : {bal_icon(bal_o)} → {bal_icon(bal_s)}")

    fig = plt.figure(figsize=(24, 10))
    fig.suptitle("Class Distribution — Before vs After SMOTE (JSON)", fontsize=15, fontweight='bold')

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    plot_configs = [
        (0, 0, y_train_orig, f"TRAIN — Before SMOTE  ({total_orig:,})",  "steelblue"),
        (0, 1, y_train,      f"TRAIN — After SMOTE   ({total_smote:,})", "seagreen"),
        (1, 0, y_val_orig,   f"VAL   — Unchanged     ({len(y_val):,})",  "darkorange"),
        (1, 1, y_test_orig,  f"TEST  — Unchanged     ({len(y_test):,})", "tomato"),
    ]

    for row, col, y_data, title, color in plot_configs:
        ax = fig.add_subplot(gs[row, col])
        counter = Counter(y_data)
        classes = sorted(counter.keys())
        counts  = [counter[c] for c in classes]
        xlabels = (
            [label_names[c] for c in classes] if label_names is not None
            else [f"cls_{c}" for c in classes]
        )
        bars = ax.bar(range(len(classes)), counts, color=color, alpha=0.82, edgecolor='white')
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=7)
        ax.set_title(title, fontweight='bold', fontsize=9)
        ax.set_ylabel("Samples")
        mean_c = np.mean(counts)
        ax.axhline(mean_c, color='black', linestyle='--', linewidth=1, label=f"Mean={mean_c:,.0f}")
        ax.legend(fontsize=7)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    f"{cnt:,}", ha='center', va='bottom', fontsize=5.5, rotation=90)

    ax_delta = fig.add_subplot(gs[:, 2])
    deltas   = [counter_smote.get(c, 0) - counter_orig.get(c, 0) for c in all_classes]
    xlabels  = (
        [label_names[c] for c in all_classes] if label_names is not None
        else [f"cls_{c}" for c in all_classes]
    )
    bar_colors = ["seagreen" if d > 0 else "lightgray" for d in deltas]
    ax_delta.barh(range(len(all_classes)), deltas, color=bar_colors, edgecolor='white', alpha=0.85)
    ax_delta.set_yticks(range(len(all_classes)))
    ax_delta.set_yticklabels(xlabels, fontsize=8)
    ax_delta.set_xlabel("Synthetic samples added")
    ax_delta.set_title("Δ Synthetic Samples Added\n(SMOTE — train only)", fontweight='bold', fontsize=9)
    ax_delta.axvline(0, color='black', linewidth=0.8)
    for i, (d, cls_id) in enumerate(zip(deltas, all_classes)):
        if d > 0:
            ax_delta.text(d + max(deltas)*0.01, i, f"+{d:,}", va='center', fontsize=7)

    save_path = f"{JSON_DIR}/smote_verification_json.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  [Saved] {save_path}")

    if smote_available:
        print("\n  Shapes after SMOTE:")
        print(f"    X_train : {X_train.shape} | y_train : {y_train.shape}")
        print(f"    X_val   : {X_val.shape}   | y_val   : {y_val.shape}")
        print(f"    X_test  : {X_test.shape}  | y_test  : {y_test.shape}")
'''


def source_to_lines(src):
    """Convert a multi-line string into the JSON list-of-lines format notebooks use."""
    lines = src.split('\n')
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + '\n')
        else:
            if line:  # skip trailing empty
                result.append(line + '\n')
    return result


def make_code_cell(source_str, cell_id):
    return {
        "cell_type": "code",
        "source": source_to_lines(source_str),
        "metadata": {"id": cell_id},
        "execution_count": None,
        "outputs": []
    }


# ── Main ─────────────────────────────────────────────────────────────────
with open(NB_PATH, 'r') as f:
    nb = json.load(f)

cells = nb['cells']
target_idx = None

for i, cell in enumerate(cells):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if '# ─── Cell: Load JSON Dataset' in src and 'load_dataset_from_drive' in src:
            target_idx = i
            break

if target_idx is None:
    # Try finding by metadata id
    for i, cell in enumerate(cells):
        if cell.get('metadata', {}).get('id') == 'CELL_JSON_LOAD':
            target_idx = i
            break

if target_idx is None:
    print("ERROR: Could not find the JSON load cell to replace!")
    exit(1)

print(f"Found target cell at index {target_idx}")

new_load_cell = make_code_cell(LOAD_JSON_SOURCE, "CELL_JSON_LOAD")
new_verify_cell = make_code_cell(VERIFY_JSON_SOURCE, "CELL_JSON_VERIFY_SMOTE")

# Replace the old cell and insert the verify cell after it
cells[target_idx] = new_load_cell
cells.insert(target_idx + 1, new_verify_cell)

nb['cells'] = cells

with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Patched {NB_PATH}: replaced cell {target_idx}, inserted verify cell at {target_idx+1}")
print(f"Total cells: {len(cells)}")
