import json
from pathlib import Path

path = Path('coupled_pendulum_lesson.ipynb')
nb = json.loads(path.read_text())

old = "The parameter $\\lambda_p$ controls how strongly the physics loss contributes to the total loss. In this lesson, the default value is\n\n$$\n\\lambda_p = 0.5.\n$$\n\n"
new = "The parameter $\\lambda_p$ controls how strongly the physics loss contributes to the total loss.\n\n"

changed = False
for cell in nb['cells']:
    if cell.get('cell_type') != 'markdown':
        continue
    text = ''.join(cell.get('source', []))
    if old in text:
        text = text.replace(old, new, 1)
        cell['source'] = text.splitlines(keepends=True)
        changed = True
        break

if not changed:
    raise RuntimeError('Target lambda_p default sentence not found')

path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
