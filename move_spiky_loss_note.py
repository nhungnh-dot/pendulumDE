import json
from pathlib import Path

path = Path('coupled_pendulum_lesson.ipynb')
nb = json.loads(path.read_text())

heading = '### Why can the loss graphs look spiky?'
block = [
    '### Why can the loss graphs look spiky?\n',
    '\n',
    'Do not expect every loss curve to decrease smoothly at every epoch. The Adam optimizer can move slightly up and down while searching for better parameters. In **Stage 2**, the physics loss is especially likely to fluctuate because the code draws a new random set of collocation points at every epoch. The logarithmic vertical scale also makes small fluctuations look more dramatic.\n',
    '\n',
    'Focus on the **overall trend**, not on whether every individual step decreases. A decreasing training loss means the model is fitting the known data better; a decreasing physics loss means the ODE residuals are becoming smaller; and a small validation loss means the model is predicting the held-out measurements well.\n',
    '\n',
    'The validation loss does not have to decrease whenever the training loss decreases. That is exactly why we keep validation data separate.\n'
]

# Remove the existing block wherever it currently appears.
for cell in nb['cells']:
    if cell.get('cell_type') != 'markdown':
        continue
    src = cell.get('source', [])
    text = ''.join(src)
    if heading in text:
        start = text.index(heading)
        # Remove from heading through the final validation-loss sentence.
        end_marker = 'The validation loss does not have to decrease whenever the training loss decreases. That is exactly why we keep validation data separate.'
        end = text.index(end_marker, start) + len(end_marker)
        # Also remove one trailing newline if present.
        if end < len(text) and text[end] == '\n':
            end += 1
        new_text = text[:start].rstrip() + '\n'
        if text[end:].strip():
            new_text += '\n' + text[end:].lstrip()
        cell['source'] = new_text.splitlines(keepends=True)
        break

# Insert immediately after the loss-graph code cell and before Question 3.
insert_at = None
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code' and cell.get('metadata', {}).get('id') == 'EgTLFmzSU2u4':
        insert_at = i + 1
        break

if insert_at is None:
    raise RuntimeError('Loss graph cell not found')

note_cell = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': block,
}
nb['cells'].insert(insert_at, note_cell)

path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
