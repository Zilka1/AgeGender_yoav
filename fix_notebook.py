import json
with open('notebooks/train_evaluate_lambda.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        if any('REPO_DIR = Path(".").resolve()' in line for line in cell['source']):
            cell['source'] = [line.replace('REPO_DIR = Path(".").resolve()', 'REPO_DIR = Path("..").resolve()') for line in cell['source']]
with open('notebooks/train_evaluate_lambda.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
