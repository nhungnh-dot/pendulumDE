import json
from pathlib import Path

path = Path('coupled_pendulum_lesson.ipynb')
nb = json.loads(path.read_text())

explanation = r'''## Understanding the loss plots

The three plots below show different ways of measuring how well the model is doing. Each loss has a different purpose.

### Plot 1: Stage 1 — Training vs. Validation Loss

In **Stage 1**, the neural network learns only from the measured data. The differential equations are not yet included in the training.

- **Training data loss** measures how closely the predicted angles match the measurements in the **first 80% of the experiment**. These are the measurements used to update the network during training. A smaller value means the model is fitting the training measurements more closely.

- **Validation data loss** measures the same type of prediction error on the **final 20% of the experiment**, which is not used to update the network. It combines the errors from both the left and right pendulums into one number. A small validation loss means the model is also predicting the held-out measurements well.

The star marks the epoch where the validation loss is smallest. This is useful because the training loss may continue to decrease even after the model has stopped improving on the held-out data.

### Plot 2: Stage 2 — Training vs. Validation Data Loss

In **Stage 2**, the network is trained using both the experimental data and the differential equations.

- **Training data loss** still measures how closely the prediction matches the first 80% of the measured angles.

- **Validation data loss** still measures how closely the prediction matches the held-out final 20%.

The important difference from Stage 1 is that the network is now also being influenced by the physics loss. Because the optimizer is trying to fit the data **and** satisfy the ODEs at the same time, the training data loss does not necessarily decrease as smoothly or as quickly as it did in Stage 1.

Again, the star marks the epoch with the smallest validation loss.

### Plot 3: Stage 2 — Training Losses

The third plot shows the quantities involved directly in Stage 2 training.

- **Data loss**

  $$
  L_{\text{data}}
  $$

  measures how closely the neural network matches the training measurements.

- **Physics loss**

  $$
  L_{\text{physics}}
  $$

  measures how closely the predicted functions satisfy the two coupled-pendulum differential equations. A small physics loss means the ODE residuals are close to zero.

- **Total PINN loss**

  $$
  L_{\text{total}}
  =
  L_{\text{data}}
  +
  \lambda_p L_{\text{physics}}
  $$

  is the quantity that the optimizer actually minimizes during Stage 2.

The parameter $\lambda_p$ controls how strongly the physics loss contributes to the total loss. In this lesson, the default value is

$$
\lambda_p = 0.5.
$$

So Stage 2 is trying to accomplish two goals at the same time:

$$
\boxed{\text{match the measured data}}
\qquad\text{and}\qquad
\boxed{\text{satisfy the differential equations}}.
$$

**Important:** the validation loss is **not** part of the total loss. It is only used to evaluate how well the trained model predicts measurements that were not used directly for training.
'''

# Remove any previous short loss-comparison explanation or earlier version of this section.
filtered = []
for cell in nb['cells']:
    if cell.get('cell_type') == 'markdown':
        text = ''.join(cell.get('source', []))
        if text.startswith('## Understanding the loss plots'):
            continue
        if text.startswith('We will compare:') or text.startswith('We will compare four different loss values'):
            continue
    filtered.append(cell)
nb['cells'] = filtered

# Insert immediately before the plotting code cell for section 16.
insert_at = None
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        text = ''.join(cell.get('source', []))
        if '# 16. TRAINING + VALIDATION LOSS COMPARISON' in text:
            insert_at = i
            break

if insert_at is None:
    raise RuntimeError('Section 16 loss-plot code cell not found')

nb['cells'].insert(insert_at, {
    'cell_type': 'markdown',
    'metadata': {},
    'source': explanation.splitlines(keepends=True),
})

path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
