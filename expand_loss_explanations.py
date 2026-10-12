import json
from pathlib import Path

path = Path('coupled_pendulum_lesson.ipynb')
nb = json.loads(path.read_text())

data_text = r'''## 10. Measure how well the prediction matches the data

We need a numerical way to measure the difference between the model prediction and the experimental measurements. This is the **data loss**.

At each training time $t_j$, the neural network predicts

$$
\widehat{\theta}_1(t_j),
\qquad
\widehat{\theta}_2(t_j),
$$

and we compare those predictions with the measured angles

$$
\theta_{1,j},
\qquad
\theta_{2,j}.
$$

For each pendulum, we first compute the prediction error. For example, for the left pendulum,

$$
\widehat{\theta}_1(t_j)-\theta_{1,j}.
$$

If this difference is small, the prediction is close to the measurement. We square the differences so that positive and negative errors do not cancel each other, and then average them over all training points.

In simplified form,

$$
L_{\text{data}}
=
\frac{1}{N}\sum_j
\left(
\frac{\widehat{\theta}_1(t_j)-\theta_{1,j}}
{\theta_{\text{scale}}}
\right)^2
+
\frac{1}{N}\sum_j
\left(
\frac{\widehat{\theta}_2(t_j)-\theta_{2,j}}
{\theta_{\text{scale}}}
\right)^2.
$$

So the data loss asks one main question:

> **How closely does the neural network match the measured motion?**

A small data loss means the predicted curves are close to the experimental points. A large data loss means the predictions are still far from the measurements.

The division by $\theta_{\text{scale}}$ simply rescales the error so that the loss is dimensionless and easier to work with numerically.

Only the **first 80% of the measurements** are used in this training loss. The final 20% are not used to update the network. Instead, we compute the same type of error on those held-out points and call it the **validation loss**.

The validation loss combines the errors from **both the left and right pendulums** into one number. It is used only to evaluate how well the model predicts data it did not use directly for training.
'''

physics_text = r'''## 11. Measure how well the prediction satisfies the physics

Matching the measured data is only one goal. We also want the predicted curves to behave like solutions of the coupled-pendulum differential equations. This is what the **physics loss** measures.

The neural network gives smooth predicted functions

$$
\widehat{\theta}_1(t),
\qquad
\widehat{\theta}_2(t).
$$

Because these are differentiable functions of time, PyTorch can compute

$$
\widehat{\theta}_1',
\quad
\widehat{\theta}_1'',
\quad
\widehat{\theta}_2',
\quad
\widehat{\theta}_2''.
$$

We then substitute the predictions and their derivatives into the two differential equations.

For the left pendulum,

$$
R_1(t)
=
\widehat{\theta}_1''
+d_1\widehat{\theta}_1'
+\frac{g}{l_1}\widehat{\theta}_1
+\frac{K}{l_1}
\left(l_1\widehat{\theta}_1-l_2\widehat{\theta}_2\right).
$$

For the right pendulum,

$$
R_2(t)
=
\widehat{\theta}_2''
+d_2\widehat{\theta}_2'
+\frac{g}{l_2}\widehat{\theta}_2
-\frac{K}{l_2}
\left(l_1\widehat{\theta}_1-l_2\widehat{\theta}_2\right).
$$

These quantities are called **ODE residuals**. If the neural-network predictions satisfied the differential equations perfectly, then

$$
R_1(t)=0,
\qquad
R_2(t)=0
$$

for every time $t$.

The physics loss therefore measures how far these residuals are from zero. In simplified form,

$$
L_{\text{physics}}
\approx
\text{average of }R_1(t)^2
+
\text{average of }R_2(t)^2.
$$

So the physics loss asks a different question from the data loss:

> **Do the predicted curves behave like solutions of the differential equations?**

A small physics loss means the ODE residuals are small, so the predicted motion is close to satisfying the coupled-pendulum model.

The code evaluates the physics loss at many **collocation points**. These are times chosen only to check the differential equations; they do not have to be measurement times. This allows the physics to guide the shape of the predicted curves even between measured data points.

The code also divides each residual by a characteristic acceleration scale,

$$
\frac{g}{l_1}\theta_{\text{scale}}
\qquad\text{and}\qquad
\frac{g}{l_2}\theta_{\text{scale}},
$$

so that the physics loss is dimensionless and numerically comparable to the data loss.

It is possible for a flexible neural network to fit the measured points well but still have derivatives that do not satisfy the ODEs. That would give a small data loss but a larger physics loss. This is why Stage 2 uses both kinds of information.
'''

for cell in nb['cells']:
    if cell.get('cell_type') != 'markdown':
        continue
    text = ''.join(cell.get('source', []))
    if text.startswith('## 10. Measure how well the prediction matches the data'):
        cell['source'] = data_text.splitlines(keepends=True)
    elif text.startswith('## 11. Measure how well the prediction satisfies the physics'):
        cell['source'] = physics_text.splitlines(keepends=True)

path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
