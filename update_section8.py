import json
import re

notebook_path = "notebooks/train_evaluate_colab.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code" and any("Data validation and deterministic split preparation" in src for src in cell.get("source", [])) or (cell["cell_type"] == "code" and any("prepare_data.py" in src for src in cell.get("source", []))):
        
        # We know this is Section 8
        new_source = """import sys
import yaml
import pandas as pd
from pathlib import Path
import json

# 1. Dynamically find the dataset CSV and its columns
raw_dir = REPO_DIR / "data" / "raw"
csv_files = list(raw_dir.rglob("*.csv"))

if not csv_files:
    raise RuntimeError("Could not find the dataset CSV file!")

csv_path = csv_files[0]
df = pd.read_csv(csv_path)

# Auto-detect columns
image_col = next((col for col in df.columns if df[col].astype(str).str.endswith('.jpg').any() or 'image' in col.lower() or 'file' in col.lower()), df.columns[0])
age_col = next((col for col in df.columns if 'age' in col.lower()), 'age')
gender_col = next((col for col in df.columns if 'gender' in col.lower()), 'gender')

print(f"Found CSV: {csv_path.name}")
print(f"Mapped Columns -> Image: '{image_col}', Age: '{age_col}', Gender: '{gender_col}'")

# 2. Build a map of filename to its relative path in data/raw
print("Mapping images to their subfolders...")
image_paths = list(raw_dir.rglob("*.jpg"))
filename_to_relpath = {p.name: str(p.relative_to(raw_dir).as_posix()) for p in image_paths}

df["mapped_image_path"] = df[image_col].map(lambda x: filename_to_relpath.get(Path(str(x)).name))
n_missing = df["mapped_image_path"].isna().sum()
if n_missing > 0:
    print(f"WARNING: {n_missing} images in CSV were not found on disk!")
df = df.dropna(subset=["mapped_image_path"])

# Save the fixed CSV
(RUN_DIR / "data_quality").mkdir(parents=True, exist_ok=True)
fixed_csv_path = RUN_DIR / "data_quality" / "fixed_metadata.csv"
df.to_csv(fixed_csv_path, index=False)

# 3. Persist overrides to configs/data.yaml
data_cfg_path = REPO_DIR / "configs" / "data.yaml"
if data_cfg_path.exists():
    data_cfg = yaml.safe_load(data_cfg_path.read_text())
    data_cfg["dataset"]["source"] = "csv"
    data_cfg["dataset"]["image_root"] = "data/raw"
    data_cfg["dataset"]["csv"] = {
        "metadata_csv": str(fixed_csv_path.relative_to(REPO_DIR).as_posix()),
        "image_path_column": "mapped_image_path",
        "age_column": age_col,
        "gender_label_column": gender_col,
        "split_column": None,
        "subject_id_column": None,
        "label_mapping": {v: 0 if str(v).strip().lower().startswith(("m", "0")) else 1 for v in df[gender_col].dropna().unique()}
    }
    data_cfg_path.write_text(yaml.dump(data_cfg))
    print("\\n✅ Updated configs/data.yaml to use CACD CSV adapter globally.")

# 4. Run the data preparation script
prepare_overrides = [
    "--set", f"paths.splits_dir={(RUN_DIR / 'data_quality').as_posix()}",
    "--set", f"validation.report_dir={(RUN_DIR / 'data_quality').as_posix()}",
]

print("\\nRunning prepare_data.py...")
split_csv_path = RUN_DIR / "data_quality" / "full_metadata_with_splits.csv"
run_command(
    [sys.executable, "scripts/prepare_data.py"] + prepare_overrides,
    cwd=REPO_DIR, log_path=RUN_DIR / "logs" / "prepare_data.log",
)
print("Data preparation complete!\\n")

# 5. Post-preparation reporting
quality_report = load_json(RUN_DIR / "data_quality" / "data_quality_report.json")
split_hash = sha256_file(split_csv_path)
print("Data quality report:")
print(json.dumps(quality_report, indent=2))
print(f"\\nSplit file SHA-256: {split_hash}")
write_manifest(
    RUN_DIR / "manifests" / "data_manifest.json",
    {**quality_report, "split_file_sha256": split_hash, "split_file": str(split_csv_path)},
)
sync_after_phase("data_preparation")
"""
        
        # Split by lines and add newlines to mimic Jupyter format
        cell["source"] = [line + "\n" for line in new_source.split("\n")]
        cell["source"][-1] = cell["source"][-1].rstrip("\n") # Last line has no newline

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Updated Section 8 in notebook.")
