import json
from copy import deepcopy
from pathlib import Path

SRC = Path('coupled_pendulum_lesson.ipynb')
DST = Path('coupled_pendulum_no_damping_lesson.ipynb')

nb = json.loads(SRC.read_text())
out = deepcopy(nb)


def set_md(cell, text):
    cell['source'] = text.strip('\n').splitlines(keepends=True)
    if cell['source'] and not cell['source'][-1].endswith('\n'):
        cell['source'][-1] += '\n'


def set_code(cell, text):
    cell['source'] = text.strip('\n').splitlines(keepends=True)
    if cell['source'] and not cell['source'][-1].endswith('\n'):
        cell['source'][-1] += '\n'
    cell['execution_count'] = None
    cell['outputs'] = []

# Clear stale outputs everywhere while retaining the exact cell order and metadata.
for cell in out['cells']:
    if cell.get('cell_type') == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []

for cell in out['cells']:
    ctype = cell.get('cell_type')
    cid = cell.get('metadata', {}).get('id')
    text = ''.join(cell.get('source', []))

    # ------------------------------------------------------------------
    # Intro/model cell: same structure as damped lesson, undamped physics.
    # ------------------------------------------------------------------
    if cid == 'R60srUUyxGUy':
        set_md(cell, r'''
# Coupled Pendulums Without Damping: Differential Equations and Physics-Informed Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nhungnh-dot/pendulumDE/blob/main/coupled_pendulum_no_damping_lesson.ipynb)

> by Andrew Bennett and Nhung Nguyen

## Learning goals

In this notebook, we study a physical system made of **two pendulums connected by a spring**. We use a simplified model that ignores damping so that we can focus on gravity, spring coupling, and parameter estimation.

By the end of the notebook, you should be able to:

- identify the terms in a second-order differential equation;
- explain how spring coupling affects the motion;
- understand why two connected pendulums lead to a **system of differential equations**;
- compare a model with measured data;
- understand the basic idea of a **Physics-Informed Neural Network (PINN)**; and
- see how an unknown physical parameter can be estimated from data.

---

## 1. The differential-equation model

Let

- $\theta_1(t)$ = angle of the **left** pendulum,
- $\theta_2(t)$ = angle of the **right** pendulum.

The two pendulums have lengths $l_1$ and $l_2$. They are connected by a spring, so the motion of one pendulum affects the other.

For small angles, we use the approximation

$$
\sin(\theta)\approx \theta.
$$

In this version of the lesson, we ignore damping. The equations of motion are

$$
\theta_1''
+\frac{g}{l_1}\theta_1
+\frac{K}{l_1}(l_1\theta_1-l_2\theta_2)
=0,
$$

$$
\theta_2''
+\frac{g}{l_2}\theta_2
-\frac{K}{l_2}(l_1\theta_1-l_2\theta_2)
=0.
$$

Here a prime means differentiation with respect to time. For example,

$$
\theta_1'=\frac{d\theta_1}{dt},
\qquad
\theta_1''=\frac{d^2\theta_1}{dt^2}.
$$

### What do the terms mean?

Each equation contains familiar pieces:

- $\theta_i''$: **angular acceleration**;
- $\dfrac{g}{l_i}\theta_i$: the restoring effect of **gravity**;
- the term containing $K$: the effect of the **spring connecting the two pendulums**.

The constant

$$
K=\frac{k}{m}
$$

combines the spring constant $k$ and the pendulum-bob mass $m$. We assume the two bobs have the same mass.

Because the equations contain both $\theta_1$ and $\theta_2$, they are **coupled**. We cannot solve for one pendulum without also considering the other.

### What will we do?

We have experimental measurements of the pendulums. We will use those measurements to estimate

$$
\theta_1(t),\qquad \theta_2(t),
$$

and the unknown physical parameter

$$
K.
$$

Instead of using only a traditional ODE solver, we will introduce a PINN. The important idea is simple:

> We want a predicted motion that both **matches the experimental data** and **approximately satisfies the differential equations**.

### How to record your work

Use the questions, notes, and tables in this notebook to guide your investigation. Follow the **Canvas assignment page** for exactly which responses and numerical results you need to submit and where to record them.
''')

    elif ctype == 'markdown' and text.startswith('---\n\n## ❓ Question 1: What Does Coupling Do?'):
        set_md(cell, r'''
---

## ❓ Question 1: What Does Coupling Do?

Look again at the two differential equations:

$$
\theta_1''
+\frac{g}{l_1}\theta_1
+\frac{K}{l_1}(l_1\theta_1-l_2\theta_2)=0,
$$

$$
\theta_2''
+\frac{g}{l_2}\theta_2
-\frac{K}{l_2}(l_1\theta_1-l_2\theta_2)=0.
$$

### Your Task

Imagine that the spring is removed. This corresponds to setting

$$
K=0.
$$

Without doing any computation, simplify both differential equations.

### Questions to Answer

1. After setting $K=0$, does the equation for $\theta_1$ still contain $\theta_2$?
2. Does the equation for $\theta_2$ still contain $\theta_1$?
3. What does this tell you about the meaning of the word **coupled**?

**Your answer:**  
*Write here.*

---
''')

    elif cid == 'n8cPWxK7bMne':
        set_md(cell, r'''
## 8. Unknown physical parameter

The differential equations contain one quantity that we want to estimate:

$$
K=\frac{k}{m}.
$$

The parameter $K$ measures the strength of the spring coupling per unit mass.

The computer will adjust this value while it adjusts the predicted functions $\widehat{\theta}_1$ and $\widehat{\theta}_2$.

### Why force the parameter to be positive?

For this model, a negative spring strength would not have the physical meaning we want. Therefore, the code represents $K$ using a positive transformation called the **softplus** function:

$$
\operatorname{softplus}(x)=\log(1+e^x).
$$

Since

$$
\operatorname{softplus}(x)>0,
$$

the estimate of $K$ remains positive during training.

The starting guess is

$$
K=1.0.
$$

This is only a starting value. The computer will update it later.

For this lesson, the key point is:

> The physical constant is treated as an **unknown parameter in the differential equations**, and we estimate it from the observed motion.
''')

    elif cid == 'vuuKFScOUF5F':
        set_code(cell, r'''
# ============================================================
# 9. TRAINABLE PHYSICAL PARAMETER
# ============================================================
#
# Instead of assuming k = 1, estimate k/m.
#
# Let
#
#     K = k/m
#
# softplus guarantees:
#
#     K > 0
#
# ============================================================

def inverse_softplus(x):

    x = torch.tensor(
        x,
        dtype=torch.float32,
        device=device
    )

    return torch.log(
        torch.exp(x) - 1.0
    )


# Reasonable starting guess

raw_K = nn.Parameter(
    inverse_softplus(1.0)
)


def physical_parameters():

    K = F.softplus(raw_K)

    return K
''')

    elif cid == '8Psd725Hp_gp':
        set_md(cell, r'''
## 9. Check whether the prediction satisfies the ODEs

This is where the differential equations become part of the learning process.

The neural network gives predicted functions

$$
\widehat{\theta}_1(t),
\qquad
\widehat{\theta}_2(t).
$$

PyTorch can differentiate these functions automatically, so it can compute

$$
\widehat{\theta}_1',
\quad
\widehat{\theta}_1'',
\quad
\widehat{\theta}_2',
\quad
\widehat{\theta}_2''.
$$

Now substitute the predictions into the two differential equations.

For the left pendulum, define

$$
R_1(t)
=
\widehat{\theta}_1''
+\frac{g}{l_1}\widehat{\theta}_1
+\frac{K}{l_1}
(l_1\widehat{\theta}_1-l_2\widehat{\theta}_2).
$$

For the right pendulum,

$$
R_2(t)
=
\widehat{\theta}_2''
+\frac{g}{l_2}\widehat{\theta}_2
-\frac{K}{l_2}
(l_1\widehat{\theta}_1-l_2\widehat{\theta}_2).
$$

The quantities $R_1$ and $R_2$ are called **residuals**.

If the predicted functions satisfied the ODEs exactly, then

$$
R_1(t)=0,
\qquad
R_2(t)=0
$$

for every time $t$.

So a small residual means that the predicted motion is close to obeying the differential-equation model.
''')

    elif cid == 'atRrrzeKUPks':
        set_code(cell, r'''
# ============================================================
# 10. AUTOMATIC DIFFERENTIATION + PHYSICS
# ============================================================

def physics_residual(model, t):

    # Need derivatives with respect to physical t
    t = t.clone().detach().requires_grad_(True)

    output = model(t)

    theta1 = output[:, 0:1]
    theta2 = output[:, 1:2]


    # ========================================================
    # First derivatives
    # ========================================================

    theta1_t = torch.autograd.grad(
        theta1,
        t,
        grad_outputs=torch.ones_like(theta1),
        create_graph=True
    )[0]

    theta2_t = torch.autograd.grad(
        theta2,
        t,
        grad_outputs=torch.ones_like(theta2),
        create_graph=True
    )[0]


    # ========================================================
    # Second derivatives
    # ========================================================

    theta1_tt = torch.autograd.grad(
        theta1_t,
        t,
        grad_outputs=torch.ones_like(theta1_t),
        create_graph=True
    )[0]

    theta2_tt = torch.autograd.grad(
        theta2_t,
        t,
        grad_outputs=torch.ones_like(theta2_t),
        create_graph=True
    )[0]


    # ========================================================
    # Physical parameter
    # ========================================================

    K = physical_parameters()


    # ========================================================
    # Spring extension
    #
    # l1 theta1 - l2 theta2
    # ========================================================

    coupling = (
        l1 * theta1
        -
        l2 * theta2
    )


    # ========================================================
    # ODE #1
    #
    # theta1''
    # + g/l1 theta1
    # + K/l1 (l1 theta1-l2 theta2)
    # = 0
    # ========================================================

    residual1 = (
        theta1_tt
        +
        (g/l1) * theta1
        +
        (K/l1) * coupling
    )


    # ========================================================
    # ODE #2
    # ========================================================

    residual2 = (
        theta2_tt
        +
        (g/l2) * theta2
        -
        (K/l2) * coupling
    )


    return (
        residual1,
        residual2,
        theta1,
        theta2,
        theta1_t,
        theta2_t
    )
''')

    elif cid == 'sC4wC8R7q6x4':
        set_md(cell, r'''
## 11. Measure how well the prediction satisfies the physics

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
+\frac{g}{l_1}\widehat{\theta}_1
+\frac{K}{l_1}
\left(l_1\widehat{\theta}_1-l_2\widehat{\theta}_2\right).
$$

For the right pendulum,

$$
R_2(t)
=
\widehat{\theta}_2''
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
''')

    elif cid == 'm41lvtgls48X':
        text2 = text.replace('- $K$, $d_1$, and $d_2$ are not changed yet.', '- $K$ is not changed yet.')
        cell['source'] = text2.splitlines(keepends=True)

    elif cid == 'GoNNnzizuOMA':
        text2 = text.replace('- the coupling parameter $K$,\n- the damping parameters $d_1$ and $d_2$.', '- the coupling parameter $K$.')
        cell['source'] = text2.splitlines(keepends=True)

    elif cid == '_p26IxaFUuRX':
        set_code(cell, r'''
# ============================================================
# 15. STAGE 2:
#     JOINT TRAINING DATA + PHYSICS
# ============================================================

physics_epochs = 10000
validation_every_stage2 = 100


optimizer_pinn = torch.optim.Adam(
    list(model.parameters())
    +
    [raw_K],
    lr=5e-4
)

total_history = []
data_history = []
physics_history = []
validation_epochs = []
validation_history = []
K_history = []

# ============================================================
# STUDENT SETTING: PHYSICS-LOSS WEIGHT
# Default value for the main run. Change this only when Question 3 asks you to.
# ============================================================
lambda_p = 0.5

print()
print("========================================")
print("STAGE 2: PINN TRAINING")
print("========================================")

for epoch in range(physics_epochs):

    optimizer_pinn.zero_grad()

    # ----------------------------------------
    # Training observations only
    # ----------------------------------------
    loss_d = data_loss()

    # ----------------------------------------
    # Physics over the full horizon (NO future measured angles)
    # ----------------------------------------
    loss_p = physics_loss(
        n_physics=512
    )

    total_loss = (
        loss_d
        +
        lambda_p * loss_p
    )

    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(
        list(model.parameters())
        +
        [raw_K],
        max_norm=10.0
    )

    optimizer_pinn.step()

    # ----------------------------------------
    # Save training histories
    # ----------------------------------------
    K_now = physical_parameters()

    total_history.append(total_loss.item())
    data_history.append(loss_d.item())
    physics_history.append(loss_p.item())

    K_history.append(K_now.item())

    # ----------------------------------------
    # Held-out validation history
    # ----------------------------------------
    if (
        epoch == 0
        or (epoch + 1) % validation_every_stage2 == 0
        or epoch + 1 == physics_epochs
    ):
        loss_v = validation_loss()
        validation_epochs.append(epoch + 1)
        validation_history.append(loss_v.item())


    # ----------------------------------------
    # Print progress every 1000 epochs
    # ----------------------------------------
    if (epoch + 1) % 1000 == 0:

        loss_v_print = validation_loss().item()

        print(
            f"Epoch {(epoch + 1):5d} | "
            f"Training data loss = {loss_d.item():.3e} | "
            f"Physics loss = {loss_p.item():.3e} | "
            f"Total training loss = {total_loss.item():.3e} | "
            f"Validation loss = {loss_v_print:.3e}"
        )

        print(
            f"        Coupling parameter K = k/m = {K_now.item():.5f}"
        )
''')

    elif cid == 'bbBK_QCMvxw9':
        set_md(cell, r'''
## 15. Watch the estimated physical parameter

While Stage 2 runs, the model repeatedly updates

$$
K=\frac{k}{m}.
$$

The next plot shows this estimate as a function of the training epoch.

If the parameter curve begins to level off, that suggests the estimate is stabilizing.

If it continues to change greatly near the end, the estimation may not yet have converged.

This is an example of an **inverse problem**:

> Instead of knowing the coefficient of the differential equation and solving for the motion, we observe the motion and try to determine the coefficient.
''')

    elif cid == 'fYkoI-IAU61R':
        set_code(cell, r'''
# ============================================================
# 17. PHYSICAL-PARAMETER HISTORY
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    K_history,
    label=r"$K=k/m$"
)

plt.xlabel("Epoch")
plt.ylabel(r"$k/m$ [s$^{-2}$]")
plt.title("Estimated Coupling Parameter")
plt.grid(True)
plt.legend()

plt.show()
''')

    elif cid == '7sZs1-WDwbeG':
        set_md(cell, r'''
# 💡 Final Results and Main Takeaway

Finally, we print the estimated value of

$$
K=\frac{k}{m}.
$$

Its units are

$$
K:\ \mathrm{s}^{-2}.
$$

This number summarizes the spring-coupling strength inferred from the experiment.

---

## Main takeaway

In a traditional differential equations problem, the coefficients are usually given and we solve for the unknown function.

Here, one coefficient is unknown and the measurements are imperfect. The PINN uses two sources of information:

$$
\boxed{\text{experimental data}}
\qquad+\qquad
\boxed{\text{differential equations}}.
$$

The goal is to find functions $\widehat{\theta}_1(t)$ and $\widehat{\theta}_2(t)$ that fit the observations while also behaving like solutions of the coupled ODE system.

This gives us a modern example of how differential equations can be used not only to **predict motion**, but also to **learn an unknown physical parameter from data**.
''')

    elif cid == 'qVgq8Ai4GhRD':
        set_code(cell, r'''
# ============================================================
# 22. FINAL PARAMETER ESTIMATE
# ============================================================

K_final = physical_parameters()


print()
print("========================================")
print("FINAL ESTIMATED PARAMETER")
print("========================================")

print(
    "Coupling parameter K = k/m =",
    K_final.item(),
    "s^-2"
)
''')

    elif ctype == 'markdown' and text.startswith('---\n\n## ⭐ Optional Extension: Do the Initial Parameter Guesses Matter?'):
        set_md(cell, r'''
---

## ⭐ Optional Extension: Do the Initial Parameter Guesses Matter?

If time permits, investigate whether the final parameter estimate depends strongly on the starting guess. The core lesson is complete before this extension.

Keep the better values of `N_FREQ` and $\lambda_p$ that you selected in Questions 2 and 3.

### Your Task

Compare the following initial guesses for $K=k/m$:

$$
K_0=0.2,\qquad 1.0,\qquad 2.0.
$$

For each initial value of $K$:

1. **Go back to the neural-network definition cell** and make sure it uses your selected `N_FREQ`.
2. Re-run the neural-network definition and **Stage 1**.
3. Go back to the parameter-definition cell and change the starting value of $K$:
   ```python
   raw_K = nn.Parameter(inverse_softplus(0.2))
   ```
   or
   ```python
   raw_K = nn.Parameter(inverse_softplus(1.0))
   ```
   or
   ```python
   raw_K = nn.Parameter(inverse_softplus(2.0))
   ```
4. In Stage 2, keep the better value of $\lambda_p$ that you selected in Question 3.
5. Run Stage 2 and record the final results.

Record your results:

| Initial $K_0$ | Final $K$ | Validation loss |
|---:|---:|---:|
| 0.2 |  |  |
| 1.0 |  |  |
| 2.0 |  |  |

### Questions to Answer

1. Do the different starting guesses lead to similar final values of $K$?
2. How does the validation loss compare in each case?
3. What does this experiment tell you about estimating an unknown parameter from data?

**Your observations:**  
*Write here.*

---
''')

# Safety checks: no trainable damping parameters or damping terms remain.
all_text = '\n'.join(''.join(c.get('source', [])) for c in out['cells'])
for forbidden in [
    'raw_d1', 'raw_d2', 'd1_history', 'd2_history',
    'd1 * theta1_t', 'd2 * theta2_t',
    'Damping coefficient d1', 'Damping coefficient d2',
    'Final $d_1$', 'Final $d_2$'
]:
    if forbidden in all_text:
        raise RuntimeError(f'Undamped notebook still contains forbidden text: {forbidden}')

# Confirm the major graph/instruction cells from the damped lesson are retained.
required_ids = [
    'eDpHgbPAUktr',   # Stage 1 left/right plots
    'EgTLFmzSU2u4',   # three loss plots
    'JEOV-cvxYIWj',   # final left plot
    'xcS-XwksYJ2_',   # final right plot
    'fYkoI-IAU61R',   # K history
]
ids = {c.get('metadata', {}).get('id') for c in out['cells']}
for cid in required_ids:
    if cid not in ids:
        raise RuntimeError(f'Missing expected graph cell: {cid}')

# Compile every code cell to catch syntax errors before committing.
for i, cell in enumerate(out['cells']):
    if cell.get('cell_type') == 'code':
        compile(''.join(cell.get('source', [])), f'cell_{i}', 'exec')

DST.write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')
print(f'Wrote {DST} with {len(out["cells"])} cells.')
