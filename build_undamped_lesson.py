import json
from pathlib import Path

OUT = Path('coupled_pendulum_no_damping_lesson.ipynb')

def md(text):
    return {'cell_type':'markdown','metadata':{},'source':text.strip('\n').splitlines(keepends=True)}

def code(text):
    return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':text.strip('\n').splitlines(keepends=True)}

cells = []

cells.append(md(r'''
# Coupled Pendulums Without Damping: Differential Equations and Physics-Informed Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nhungnh-dot/pendulumDE/blob/main/coupled_pendulum_no_damping_lesson.ipynb)

> by Andrew Bennett and Nhung Nguyen

## Learning goals

In this notebook, we study two pendulums connected by a spring using a simplified **undamped model**. We combine ideas from differential equations, experimental data, and a Physics-Informed Neural Network (PINN).

By the end of the notebook, you should be able to:

- identify the terms in a second-order differential equation;
- explain how spring coupling affects the motion;
- understand why two connected pendulums lead to a system of differential equations;
- compare a model with measured data;
- understand the basic idea of a PINN; and
- see how the unknown coupling parameter can be estimated from data.

---

## 1. The differential-equation model

Let

- $\theta_1(t)$ = angle of the **left** pendulum,
- $\theta_2(t)$ = angle of the **right** pendulum.

The pendulums have lengths $l_1$ and $l_2$ and are connected by a spring. For small angles, $\sin(\theta)\approx\theta$.

In this lesson we ignore damping, so the model is

$$
\theta_1''+\frac{g}{l_1}\theta_1
+\frac{K}{l_1}(l_1\theta_1-l_2\theta_2)=0,
$$

$$
\theta_2''+\frac{g}{l_2}\theta_2
-\frac{K}{l_2}(l_1\theta_1-l_2\theta_2)=0.
$$

Here

- $\theta_i''$ is angular acceleration;
- $\dfrac{g}{l_i}\theta_i$ is the restoring effect of gravity;
- the term containing $K$ describes the spring coupling.

The constant

$$
K=\frac{k}{m}
$$

combines the spring constant $k$ and bob mass $m$.

Because each equation contains both $\theta_1$ and $\theta_2$, the equations are **coupled**.

Our goal is to estimate

$$
\theta_1(t),\qquad \theta_2(t),\qquad K
$$

from the measurements while encouraging the predicted motion to satisfy the two ODEs.

### How to record your work

Use the questions, notes, and tables in this notebook to guide your investigation. Follow the **Canvas assignment page** for exactly which responses and numerical results you need to submit.
'''))

cells.append(md(r'''
## ❓ Question 1: What does coupling do?

Set $K=0$ in the two equations.

1. Simplify both differential equations.
2. Does the equation for $\theta_1$ still contain $\theta_2$?
3. Does the equation for $\theta_2$ still contain $\theta_1$?
4. What does this tell you about the meaning of **coupled**?

**Your answer:**  
*Write here.*
'''))

cells.append(code(r'''
# ============================================================
# 1. IMPORTS AND REPRODUCIBILITY
# ============================================================
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42

def reset_random_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

reset_random_seed()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
'''))

cells.append(md(r'''
## 2. Load the experimental data

The data table contains time and measured coordinates for the two pendulums. We will use the horizontal coordinates `a_left` and `a_right`.
'''))

cells.append(code(r'''
# ============================================================
# 2. LOAD EXPERIMENTAL DATA
# ============================================================
csv_url = (
    'https://raw.githubusercontent.com/'
    'ngochuytruong/data/main/data-diff-length.csv'
)

df = pd.read_csv(csv_url, header=None)
df.columns = ['t', 'a_right', 'b_right', 'a_left', 'b_left']
df = df.iloc[2:].reset_index(drop=True)
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna().reset_index(drop=True)
df = df.sort_values('t').reset_index(drop=True)

print(df.head())
print('Number of experimental points:', len(df))
'''))

cells.append(md(r'''
## 3. Convert horizontal displacement to angle

We use

$$
l_1=1.00\text{ m},\qquad l_2=1.05\text{ m},\qquad g=9.8\text{ m/s}^2.
$$

The equilibrium position is estimated using the median horizontal position. After subtracting it, we use the small-angle relation

$$
\theta\approx \frac{x}{l}.
$$
'''))

cells.append(code(r'''
# ============================================================
# 3. PREPARE ANGLE DATA
# ============================================================
t = df['t'].to_numpy(dtype=np.float32)
a_right = df['a_right'].to_numpy(dtype=np.float32)
a_left = df['a_left'].to_numpy(dtype=np.float32)

l1 = 1.00
l2 = 1.05
g = 9.8

# Shift time so the first observation is t = 0
t = t - t[0]

# Remove equilibrium offsets
x1 = a_left - np.median(a_left)
x2 = a_right - np.median(a_right)

# Small-angle conversion: theta approximately x/l
theta1 = x1 / l1
theta2 = x2 / l2

theta_scale = max(np.max(np.abs(theta1)), np.max(np.abs(theta2)), 1e-6)

print('theta scale =', theta_scale)
'''))

cells.append(code(r'''
plt.figure(figsize=(11,5))
plt.plot(t, theta1, '.', markersize=4, label=r'Left: $\theta_1$')
plt.plot(t, theta2, '.', markersize=4, label=r'Right: $\theta_2$')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title('Measured pendulum angles')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''))

cells.append(md(r'''
## 4. Training and validation data

We use the **first 80%** of the measurements for training and keep the **final 20%** for validation.

The training data are used to update the network. The validation data are never used in the optimizer; they tell us how well the model predicts held-out measurements.
'''))

cells.append(code(r'''
# ============================================================
# 4. TRAIN / VALIDATION SPLIT
# ============================================================
N = len(t)
N_train = int(0.8*N)

train_idx = np.arange(N_train)
val_idx = np.arange(N_train, N)

t_train = t[train_idx]
t_val = t[val_idx]

theta1_train = theta1[train_idx]
theta2_train = theta2[train_idx]
theta1_val = theta1[val_idx]
theta2_val = theta2[val_idx]

def tensor_col(x):
    return torch.tensor(x, dtype=torch.float32, device=device).reshape(-1,1)

t_tensor = tensor_col(t_train)
theta1_tensor = tensor_col(theta1_train)
theta2_tensor = tensor_col(theta2_train)
t_val_tensor = tensor_col(t_val)
theta1_val_tensor = tensor_col(theta1_val)
theta2_val_tensor = tensor_col(theta2_val)

t_min = float(t.min())
t_max = float(t.max())
'''))

cells.append(md(r'''
## 5. Neural-network model with Fourier features

The network takes time $t$ as input and predicts two outputs,

$$
\widehat\theta_1(t),\qquad \widehat\theta_2(t).
$$

Because pendulum motion is oscillatory, we first represent time using sine and cosine functions at several frequencies. These are called **Fourier features**.

Change only `N_FREQ` when you experiment with the number of frequencies.
'''))

cells.append(code(r'''
# ============================================================
# STUDENT SETTING: NUMBER OF FOURIER FREQUENCIES
# ============================================================
N_FREQ = 30

class FourierPINN(nn.Module):
    def __init__(self, n_freq, hidden=64):
        super().__init__()
        self.n_freq = n_freq
        in_dim = 1 + 2*n_freq
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2)
        )

    def features(self, t):
        tau = 2.0*(t - t_min)/(t_max - t_min) - 1.0
        pieces = [tau]
        for n in range(1, self.n_freq + 1):
            pieces.append(torch.sin(n*np.pi*tau))
            pieces.append(torch.cos(n*np.pi*tau))
        return torch.cat(pieces, dim=1)

    def forward(self, t):
        return theta_scale*self.net(self.features(t))

reset_random_seed()
model = FourierPINN(n_freq=N_FREQ).to(device)
print(model)
'''))

cells.append(md(r'''
## 6. Data loss

The **data loss** measures how closely the network predictions match the measured angles at the training times.

At each training time $t_j$, the network predicts $\widehat\theta_1(t_j)$ and $\widehat\theta_2(t_j)$. We compare them with the measured values $\theta_{1,j}$ and $\theta_{2,j}$.

In simplified form,

$$
L_{\text{data}}
=
\frac1N\sum_j\left(\frac{\widehat\theta_1(t_j)-\theta_{1,j}}{\theta_{\text{scale}}}\right)^2
+
\frac1N\sum_j\left(\frac{\widehat\theta_2(t_j)-\theta_{2,j}}{\theta_{\text{scale}}}\right)^2.
$$

A smaller data loss means the predicted curves are closer to the measurements.
'''))

cells.append(code(r'''
def data_loss_on(t_in, y1, y2):
    pred = model(t_in)
    e1 = (pred[:,0:1] - y1)/theta_scale
    e2 = (pred[:,1:2] - y2)/theta_scale
    return torch.mean(e1**2) + torch.mean(e2**2)

def training_data_loss():
    return data_loss_on(t_tensor, theta1_tensor, theta2_tensor)

def validation_data_loss():
    with torch.no_grad():
        return data_loss_on(t_val_tensor, theta1_val_tensor, theta2_val_tensor)
'''))

cells.append(md(r'''
## 7. The unknown physical parameter

The only physical parameter we estimate in this undamped model is

$$
K=\frac{k}{m}>0.
$$

The code uses a softplus transformation so that the estimated value of $K$ stays positive during training.
'''))

cells.append(code(r'''
def inverse_softplus(x):
    x = torch.tensor(float(x), dtype=torch.float32, device=device)
    return torch.log(torch.exp(x) - 1.0)

raw_K = nn.Parameter(inverse_softplus(1.0))

def physical_parameter():
    return F.softplus(raw_K)
'''))

cells.append(md(r'''
## 8. Physics loss

Matching the measurements is only one goal. We also want the predicted functions to behave like solutions of the undamped coupled-pendulum equations.

The network gives differentiable predictions $\widehat\theta_1(t)$ and $\widehat\theta_2(t)$. PyTorch can therefore compute their first and second derivatives.

We substitute the predictions into the ODEs and define the residuals

$$
R_1=\widehat\theta_1''+\frac{g}{l_1}\widehat\theta_1
+\frac{K}{l_1}(l_1\widehat\theta_1-l_2\widehat\theta_2),
$$

$$
R_2=\widehat\theta_2''+\frac{g}{l_2}\widehat\theta_2
-\frac{K}{l_2}(l_1\widehat\theta_1-l_2\widehat\theta_2).
$$

If the predictions satisfy the ODEs perfectly, then $R_1=R_2=0$.

The **physics loss** measures the average squared residuals at randomly selected **collocation points**. These points do not have to be measurement times, so the physics can guide the curve between measurements.
'''))

cells.append(code(r'''
def physics_residual(t_phys):
    t_phys = t_phys.clone().detach().requires_grad_(True)
    pred = model(t_phys)
    th1 = pred[:,0:1]
    th2 = pred[:,1:2]

    th1_t = torch.autograd.grad(th1, t_phys, torch.ones_like(th1), create_graph=True)[0]
    th2_t = torch.autograd.grad(th2, t_phys, torch.ones_like(th2), create_graph=True)[0]
    th1_tt = torch.autograd.grad(th1_t, t_phys, torch.ones_like(th1_t), create_graph=True)[0]
    th2_tt = torch.autograd.grad(th2_t, t_phys, torch.ones_like(th2_t), create_graph=True)[0]

    K = physical_parameter()
    coupling = l1*th1 - l2*th2

    r1 = th1_tt + (g/l1)*th1 + (K/l1)*coupling
    r2 = th2_tt + (g/l2)*th2 - (K/l2)*coupling
    return r1, r2

def physics_loss(n_physics=512):
    t_phys = t_min + (t_max-t_min)*torch.rand(n_physics,1,device=device)
    r1, r2 = physics_residual(t_phys)

    scale1 = (g/l1)*theta_scale
    scale2 = (g/l2)*theta_scale
    return torch.mean((r1/scale1)**2) + torch.mean((r2/scale2)**2)
'''))

cells.append(md(r'''
## 9. Stage 1 — fit the measured data first

In Stage 1, we temporarily ignore the physics loss and train only on the measured angles. This gives the network a reasonable approximation before we ask it to satisfy the ODEs.
'''))

cells.append(code(r'''
# ============================================================
# STAGE 1: DATA-ONLY WARMUP
# ============================================================
warmup_epochs = 10000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

warmup_history = []
warmup_val_history = []
warmup_val_epochs = []

for epoch in range(1, warmup_epochs+1):
    optimizer.zero_grad()
    loss = training_data_loss()
    loss.backward()
    optimizer.step()
    warmup_history.append(loss.item())

    if epoch == 1 or epoch % 200 == 0:
        val = validation_data_loss().item()
        warmup_val_epochs.append(epoch)
        warmup_val_history.append(val)

    if epoch % 2000 == 0:
        print(f'Stage 1 epoch {epoch:5d} | training data loss = {loss.item():.3e}')
'''))

cells.append(code(r'''
with torch.no_grad():
    t_all_tensor = tensor_col(t)
    pred_stage1 = model(t_all_tensor).cpu().numpy()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(t, theta1, '.', markersize=3, label='Measurements')
plt.plot(t, pred_stage1[:,0], linewidth=1.8, label='Prediction')
plt.axvline(t[N_train], linestyle='--', alpha=0.7, label='Validation begins')
plt.xlabel('Time (s)'); plt.ylabel(r'$\theta_1$'); plt.title('Left pendulum'); plt.legend(); plt.grid(alpha=0.3)

plt.subplot(1,2,2)
plt.plot(t, theta2, '.', markersize=3, label='Measurements')
plt.plot(t, pred_stage1[:,1], linewidth=1.8, label='Prediction')
plt.axvline(t[N_train], linestyle='--', alpha=0.7, label='Validation begins')
plt.xlabel('Time (s)'); plt.ylabel(r'$\theta_2$'); plt.title('Right pendulum'); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()
'''))

cells.append(md(r'''
### Questions to consider after Stage 1

- In the **left plot**, how closely does the predicted $\theta_1$ curve follow the training measurements? What happens in the validation interval?
- In the **right plot**, how closely does the predicted $\theta_2$ curve follow the measurements?
- Is the quality of the fit similar for the two pendulums?
- Do both predicted curves continue to oscillate reasonably in the validation interval?
'''))

cells.append(md(r'''
## ❓ Question 2: How many Fourier frequencies do we need?

Change **only** `N_FREQ`, reconstruct the model, and rerun Stage 1.

Try, for example,

```python
N_FREQ = 0
N_FREQ = 5
N_FREQ = 30
```

Compare the fit and validation behavior. What do the Fourier features help the neural network represent?
'''))

cells.append(md(r'''
## 10. Stage 2 — add the differential equations

Now we train the network using both sources of information:

$$
L_{\text{total}}=L_{\text{data}}+\lambda_pL_{\text{physics}}.
$$

The parameter $\lambda_p$ controls how strongly the ODEs influence training.
'''))

cells.append(code(r'''
# ============================================================
# STUDENT SETTING: PHYSICS-LOSS WEIGHT
# ============================================================
lambda_p = 0.5
physics_epochs = 10000

optimizer = torch.optim.Adam(
    list(model.parameters()) + [raw_K],
    lr=5e-4
)

total_history = []
data_history = []
physics_history = []
K_history = []
validation_history = []
validation_epochs = []

for epoch in range(1, physics_epochs+1):
    optimizer.zero_grad()

    L_data = training_data_loss()
    L_phys = physics_loss()
    L_total = L_data + lambda_p*L_phys

    L_total.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [raw_K], max_norm=10.0)
    optimizer.step()

    total_history.append(L_total.item())
    data_history.append(L_data.item())
    physics_history.append(L_phys.item())
    K_history.append(physical_parameter().item())

    if epoch == 1 or epoch % 200 == 0:
        validation_epochs.append(epoch)
        validation_history.append(validation_data_loss().item())

    if epoch % 2000 == 0:
        print(
            f'Stage 2 epoch {epoch:5d} | total = {L_total.item():.3e} | '
            f'data = {L_data.item():.3e} | physics = {L_phys.item():.3e} | '
            f'K = {physical_parameter().item():.5f}'
        )
'''))

cells.append(md(r'''
## 11. Final prediction and estimated coupling

The network now balances two goals: fitting the measurements and approximately satisfying the undamped ODE model.
'''))

cells.append(code(r'''
with torch.no_grad():
    pred_final = model(t_all_tensor).cpu().numpy()

K_final = physical_parameter().item()
print('Estimated coupling parameter K = k/m =', K_final)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(t, theta1, '.', markersize=3, label='Measurements')
plt.plot(t, pred_final[:,0], linewidth=1.8, label='PINN prediction')
plt.axvline(t[N_train], linestyle='--', alpha=0.7, label='Validation begins')
plt.xlabel('Time (s)'); plt.ylabel(r'$\theta_1$'); plt.title('Left pendulum'); plt.legend(); plt.grid(alpha=0.3)

plt.subplot(1,2,2)
plt.plot(t, theta2, '.', markersize=3, label='Measurements')
plt.plot(t, pred_final[:,1], linewidth=1.8, label='PINN prediction')
plt.axvline(t[N_train], linestyle='--', alpha=0.7, label='Validation begins')
plt.xlabel('Time (s)'); plt.ylabel(r'$\theta_2$'); plt.title('Right pendulum'); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()
'''))

cells.append(md(r'''
## Understanding the loss plots

The three plots below show different ways of measuring how well the model is doing.

### Plot 1: Stage 1 — Training vs. Validation Loss

In Stage 1, the network learns only from measured data.

- **Training data loss** measures the error on the first 80% of the measurements, which are used to update the network.
- **Validation data loss** measures the same type of error on the held-out final 20%. It combines errors from both pendulums.

The star marks the epoch with the smallest validation loss.

### Plot 2: Stage 2 — Training vs. Validation Data Loss

In Stage 2, the network is influenced by both the measurements and the ODEs.

- **Training data loss** still measures the fit to the first 80%.
- **Validation data loss** still measures the fit to the held-out final 20%.

Because the optimizer is now trying to fit the data **and** satisfy the ODEs, the data loss may not decrease as smoothly as in Stage 1.

### Plot 3: Stage 2 — Training Losses

- **Data loss** $L_{\text{data}}$ measures agreement with the training measurements.
- **Physics loss** $L_{\text{physics}}$ measures how closely the predictions satisfy the two undamped ODEs.
- **Total PINN loss**

$$
L_{\text{total}}=L_{\text{data}}+\lambda_pL_{\text{physics}}
$$

is the quantity minimized by the optimizer in Stage 2.

**Important:** the validation loss is not part of the total loss. It is used only to evaluate predictions on held-out measurements.
'''))

cells.append(code(r'''
# ============================================================
# 12. TRAINING + VALIDATION LOSS COMPARISON
# ============================================================
# Stage 1
plt.figure(figsize=(11,5))
stage1_epochs = np.arange(1, len(warmup_history)+1)
plt.semilogy(stage1_epochs, warmup_history, linewidth=1.8, label='Training data loss')
plt.semilogy(warmup_val_epochs, warmup_val_history, 'o-', markersize=4, linewidth=1.5, label='Validation data loss')
best_i = int(np.argmin(warmup_val_history))
plt.scatter([warmup_val_epochs[best_i]], [warmup_val_history[best_i]], s=90, marker='*', zorder=5, label=f'Best validation epoch = {warmup_val_epochs[best_i]}')
plt.xlabel('Epoch'); plt.ylabel('Normalized data loss'); plt.title('Stage 1: Training vs Validation Loss')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

# Stage 2 data/validation
plt.figure(figsize=(11,5))
stage2_epochs = np.arange(1, len(data_history)+1)
plt.semilogy(stage2_epochs, data_history, linewidth=1.6, label='Training data loss')
plt.semilogy(validation_epochs, validation_history, 'o-', markersize=4, linewidth=1.5, label='Validation data loss')
best_i = int(np.argmin(validation_history))
plt.scatter([validation_epochs[best_i]], [validation_history[best_i]], s=90, marker='*', zorder=5, label=f'Best validation epoch = {validation_epochs[best_i]}')
plt.xlabel('Epoch'); plt.ylabel('Normalized data loss'); plt.title('Stage 2: Training vs Validation Data Loss')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

# Stage 2 training losses
plt.figure(figsize=(11,5))
plt.semilogy(stage2_epochs, total_history, linewidth=1.6, label='Total PINN loss')
plt.semilogy(stage2_epochs, physics_history, linewidth=1.6, label='Physics loss')
plt.semilogy(stage2_epochs, data_history, linewidth=1.6, label='Data loss')
plt.xlabel('Epoch'); plt.ylabel('Normalized loss'); plt.title('Stage 2: Training Losses')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

# Estimated K
plt.figure(figsize=(11,4))
plt.plot(stage2_epochs, K_history, linewidth=1.6)
plt.xlabel('Epoch'); plt.ylabel(r'$K=k/m$'); plt.title('Estimated coupling parameter during Stage 2')
plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
'''))

cells.append(md(r'''
### Why can the loss graphs look spiky?

Do not expect every loss curve to decrease smoothly at every epoch. Adam can move slightly up and down while searching for better parameters. In **Stage 2**, the physics loss is especially likely to fluctuate because the code draws a new random set of collocation points at every epoch. The logarithmic vertical scale also makes small fluctuations look more dramatic.

Focus on the **overall trend**, not on whether every individual step decreases. A decreasing training loss means the model is fitting the known data better; a decreasing physics loss means the ODE residuals are becoming smaller; and a small validation loss means the model is predicting the held-out measurements well.

The validation loss does not have to decrease whenever the training loss decreases. That is exactly why we keep validation data separate.
'''))

cells.append(md(r'''
## ❓ Question 3: What does $\lambda_p$ do?

Repeat Stage 2 using different physics weights, for example

$$
\lambda_p=0,\qquad 0.5,\qquad 2.
$$

For each value, compare

- training data loss,
- validation data loss,
- physics loss,
- estimated $K$,
- and the predicted trajectories.

What tradeoff do you observe between fitting the measurements and satisfying the ODEs?
'''))

cells.append(md(r'''
## ⭐ Optional Extension: Does the initial guess for $K$ matter?

If time permits, rerun Stage 2 with several starting values for the coupling parameter, such as

$$
K_0=0.2,\qquad 1.0,\qquad 2.0.
$$

Change the value in

```python
raw_K = nn.Parameter(inverse_softplus(1.0))
```

and record the final estimated $K$, validation loss, and physics loss.

Does the training return to approximately the same value of $K$, or is the result sensitive to the initial guess?
'''))

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
        'language_info': {'name':'python','version':'3.x'},
        'colab': {'provenance': []}
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

# Validate Python syntax before writing.
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        compile(''.join(cell['source']), f'cell_{i}', 'exec')

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
print(f'Wrote {OUT} with {len(cells)} cells')
