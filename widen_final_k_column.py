import json
from pathlib import Path

files = [
    Path('coupled_pendulum_lesson.ipynb'),
    Path('coupled_pendulum_no_damping_lesson.ipynb'),
]

old_header = '| $\\lambda_p$ | Training data loss | Validation loss | Physics loss | Final $K=k/m$ |'
new_header = '| $\\lambda_p$ | Training data loss | Validation loss | Physics loss | Final $K=k/m$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |'

for path in files:
    nb = json.loads(path.read_text())
    changed = False

    for cell in nb['cells']:
        if cell.get('cell_type') != 'markdown':
            continue
        text = ''.join(cell.get('source', []))
        if '## ❓ Question 3: How Much Should We Trust the Physics?' in text and old_header in text:
            text = text.replace(old_header, new_header, 1)
            cell['source'] = text.splitlines(keepends=True)
            changed = True
            break

    if not changed:
        raise RuntimeError(f'Question 3 table header not found in {path}')

    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
