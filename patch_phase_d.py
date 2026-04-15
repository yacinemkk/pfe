#!/usr/bin/env python3
import re

with open("build_greedy_notebook.py", "r") as f:
    content = f.read()

# 1. Update PHASE_D_MIX_RATIO
content = content.replace("PHASE_D_MIX_RATIO = 0.95", "PHASE_D_MIX_RATIO = 1.0")
content = content.replace("mix_ratio=0.95", "mix_ratio=1.0")

# 2. Update PHASE_D_K_MAX
content = content.replace("PHASE_D_K_MAX = 4", "PHASE_D_K_MAX = 5")

# 3. Disable AFDLoss in Phase D
old_phase_d_train = "p_drop=0.2, sigma_noise=0.01, afd_lambda=1.5,"
new_phase_d_train = "p_drop=0.2, sigma_noise=0.01, afd_lambda=0.0,"
content = content.replace(old_phase_d_train, new_phase_d_train)

# Update log message in Notebook Phase D header
content = content.replace("'D': 'Consolidation (95% adv, k_max=4)'", "'D': 'Consolidation Extreme (100% adv, k_max=5)'")

with open("build_greedy_notebook.py", "w") as f:
    f.write(content)
print("Phase D parameters successfully patched.")
