"""
Patch IOT_nor.ipynb to fix: PGD/TRADES attack called inside with torch.no_grad().

The fix removes `with torch.no_grad():` from around `active_attack.generate`.
PGD requires autograd to compute gradients for adversarial perturbation.
"""

import json
import re

NOTEBOOK_PATH = "IOT_nor.ipynb"


def fix_cell_source(source: str) -> str:
    """
    Fix two patterns in the training and evaluation loops.

    Pattern 1 (training loop):
        # Couche 2: TRADES adversarial generation (Worst-of-K if active)
        with torch.no_grad():
            result = active_attack.generate(model, X_batch, y_batch, device)
            if is_multi_attack:
                X_adv, _ = result
            else:
                X_adv = result

    Pattern 2 (evaluation loop), same issue but slightly different leading indent.
    """

    # ── Pattern 1: training loop ─────────────────────────────────────────────
    old_train = (
        "            # Couche 2: TRADES adversarial generation (Worst-of-K if active)\n"
        "            with torch.no_grad():\n"
        "                result = active_attack.generate(model, X_batch, y_batch, device)\n"
        "                if is_multi_attack:\n"
        "                    X_adv, _ = result\n"
        "                else:\n"
        "                    X_adv = result\n"
    )
    new_train = (
        "            # Couche 2: TRADES adversarial generation (Worst-of-K if active)\n"
        "            # NOTE: must NOT be wrapped in torch.no_grad(); PGD needs autograd\n"
        "            result = active_attack.generate(model, X_batch, y_batch, device)\n"
        "            if is_multi_attack:\n"
        "                X_adv, _ = result\n"
        "            else:\n"
        "                X_adv = result\n"
    )
    source = source.replace(old_train, new_train)

    # ── Pattern 2: evaluation / crash-test loop ───────────────────────────────
    # The test loop has an outer with torch.no_grad(): wrapping the generate call,
    # but the second no_grad() for model inference should stay.
    # Before:
    #         with torch.no_grad():
    #             result = active_attack.generate(model, X_batch, y_batch, device)
    #             if is_multi_attack:
    #                 X_adv, _ = result
    #             else:
    #                 X_adv = result
    #         with torch.no_grad():
    #             outputs = model(X_adv)
    old_eval = (
        "        with torch.no_grad():\n"
        "            result = active_attack.generate(model, X_batch, y_batch, device)\n"
        "            if is_multi_attack:\n"
        "                X_adv, _ = result\n"
        "            else:\n"
        "                X_adv = result\n"
        "        with torch.no_grad():\n"
        "            outputs = model(X_adv)\n"
    )
    new_eval = (
        "        # NOTE: must NOT be wrapped in torch.no_grad(); PGD needs autograd\n"
        "        result = active_attack.generate(model, X_batch, y_batch, device)\n"
        "        if is_multi_attack:\n"
        "            X_adv, _ = result\n"
        "        else:\n"
        "            X_adv = result\n"
        "        with torch.no_grad():\n"
        "            outputs = model(X_adv)\n"
    )
    source = source.replace(old_eval, new_eval)

    # ── Pattern 3: same in crash-test block (no indent level change) ──────────
    old_crash = (
        "            with torch.no_grad():\n"
        "                result = active_attack.generate(model, _ct_X, _ct_y, device)\n"
        "                _ct_X_adv = result[0] if isinstance(result, tuple) else result\n"
    )
    new_crash = (
        "            # NOTE: must NOT be wrapped in torch.no_grad(); PGD needs autograd\n"
        "            result = active_attack.generate(model, _ct_X, _ct_y, device)\n"
        "            _ct_X_adv = result[0] if isinstance(result, tuple) else result\n"
    )
    source = source.replace(old_crash, new_crash)

    # ── Pattern 4 (_ct_result variant in crash-test before Phase 2) ──────────
    old_crash2 = (
        "            with torch.no_grad():\n"
        "                _ct_result = active_attack.generate(model, _ct_X, _ct_y, device)\n"
        "                _ct_X_adv = _ct_result[0] if isinstance(_ct_result, tuple) else _ct_result\n"
    )
    new_crash2 = (
        "            # NOTE: must NOT be wrapped in torch.no_grad(); PGD needs autograd\n"
        "            _ct_result = active_attack.generate(model, _ct_X, _ct_y, device)\n"
        "            _ct_X_adv = _ct_result[0] if isinstance(_ct_result, tuple) else _ct_result\n"
    )
    source = source.replace(old_crash2, new_crash2)

    return source


def patch_notebook(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    total_changes = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        original = "".join(cell["source"])
        patched = fix_cell_source(original)
        if patched != original:
            # count replacements made in this cell
            total_changes += 1
            cell["source"] = patched.splitlines(keepends=True)
            print(f"  [PATCHED] cell (first 80 chars): {repr(patched[:80])}")

    if total_changes == 0:
        print("⚠️  No patterns matched. Check if the notebook has already been fixed or if indentation differs.")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"\n✅ Notebook patched successfully. {total_changes} cell(s) modified.")


if __name__ == "__main__":
    print(f"Patching {NOTEBOOK_PATH} …")
    patch_notebook(NOTEBOOK_PATH)
