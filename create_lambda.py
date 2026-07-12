import json

with open("notebooks/train_evaluate_colab.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Modify cells
new_cells = []
for cell in nb["cells"]:
    if cell["cell_type"] != "code" and cell["cell_type"] != "markdown":
        new_cells.append(cell)
        continue
        
    src = "".join(cell.get("source", []))
    
    # Remove Colab text
    src = src.replace("Google Colab", "Lambda/Slurm")
    src = src.replace("Colab", "Lambda/Slurm")
    
    # 1. User config: remove USE_GOOGLE_DRIVE
    if "USE_GOOGLE_DRIVE = True" in src:
        src = src.replace("USE_GOOGLE_DRIVE = True", "")
    
    # 2. Repo setup: we assume they run it FROM the repo
    if 'REPO_DIR = Path("/content/AgeGender")' in src:
        src = """# ============================================================
# Repository setup
# ============================================================
import os
import sys
from pathlib import Path

REPO_DIR = Path(".").resolve()
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
print(f"Repository ready at {REPO_DIR}")
"""

    # 3. Workspace setup: use local runs folder
    if 'WORKSPACE_DIR = Path("/content/agegender_runs")' in src:
        src = src.replace('WORKSPACE_DIR = Path("/content/agegender_runs")', 'WORKSPACE_DIR = REPO_DIR / "runs"')
    
    # 4. Remove Google Drive mounting
    if 'drive.mount("/content/drive")' in src:
        continue # Skip this cell entirely
        
    # Remove drive references
    if "copy_tree_merge(RUN_DIR, dest)" in src and "def sync_after_phase" in src:
        src = """# ============================================================
# (No Google Drive syncing on Lambda)
# ============================================================
def sync_after_phase(phase_label):
    pass  # No Google Drive sync on Lambda
"""
    
    # 5. Kaggle logic
    if "KAGGLE_DATASET_SLUG" in src and "userdata" in src:
        src = """# ============================================================
# Kaggle credentials
# ============================================================
if KAGGLE_DATASET_SLUG:
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        print("Please export KAGGLE_USERNAME and KAGGLE_KEY in your SLURM script!")
        print("e.g. export KAGGLE_USERNAME='your_username'")
        raise RuntimeError("Missing Kaggle credentials")
    print("Kaggle credentials found in environment.")
"""

    cell["source"] = [line + "\n" for line in src.split("\n")]
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n")
        
    new_cells.append(cell)

nb["cells"] = new_cells

with open("notebooks/train_evaluate_lambda.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print("Created notebooks/train_evaluate_lambda.ipynb")
