import json
from pathlib import Path

NOTEBOOK = Path("coupled_pendulum_lesson.ipynb")

with NOTEBOOK.open("r", encoding="utf-8") as f:
    nb = json.load(f)


def text(cell):
    src = cell.get("source", [])
    return "".join(src) if isinstance(src, list) else src


def set_text(cell, value):
    cell["source"] = value.splitlines(keepends=True)


# -----------------------------------------------------------------------------
# 1. Add a short assignment/submission note near the beginning.
# -----------------------------------------------------------------------------
intro = nb["cells"][0]
intro_text = text(intro)
submission_note = (
    "\n\n### How to record your work\n\n"
    "Use the questions, notes, and tables in this notebook to guide your investigation. "
    "Follow the **Canvas assignment page** for exactly which responses and numerical results "
    "you need to submit and where to record them.\n"
)
if "### How to record your work" not in intro_text:
    set_text(intro, intro_text + submission_note)


# -----------------------------------------------------------------------------
# 2. Make repeated model construction reproducible.
# -----------------------------------------------------------------------------
for cell in nb["cells"]:
    s = text(cell)
    if cell.get("cell_type") == "code" and "# 1. IMPORTS" in s:
        old = """# Reproducibility\nnp.random.seed(42)\ntorch.manual_seed(42)\n"""
        new = """# Reproducibility\n# Resetting the seed before each new model is constructed makes repeated\n# experiments start from the same neural-network initialization.\nimport random\n\nSEED = 42\n\ndef reset_random_seed():\n    random.seed(SEED)\n    np.random.seed(SEED)\n    torch.manual_seed(SEED)\n    if torch.cuda.is_available():\n        torch.cuda.manual_seed_all(SEED)\n\nreset_random_seed()\n\n# Reduce run-to-run variation on CUDA when possible.\ntorch.backends.cudnn.deterministic = True\ntorch.backends.cudnn.benchmark = False\n"""
        if old in s:
            s = s.replace(old, new)
        set_text(cell, s)
        break


# -----------------------------------------------------------------------------
# 3. Clarify validation loss and add a compact loss glossary.
# -----------------------------------------------------------------------------
for cell in nb["cells"]:
    s = text(cell)
    if cell.get("cell_type") == "markdown" and "## 10. Measure how well the prediction matches the data" in s:
        s = s.replace(
            "We also calculate the same type of error on the final 20%, called the **validation loss**, but we do not use that value to update the model.",
            "We also calculate the same type of error on the final 20%, called the **validation loss**, but we do not use that value to update the model. The reported validation loss is **one combined value for both pendulums**: it includes the squared error for the left pendulum and the squared error for the right pendulum."
        )
        s += (
            "\n\n### Loss glossary\n\n"
            "- **Training data loss:** error on the measured angles used to train the network.\n"
            "- **Validation loss:** error on the held-out final 20% of the measured angles; it combines the left- and right-pendulum errors and is not used for backpropagation.\n"
            "- **Physics loss:** measures how closely the predicted motion satisfies the two differential equations.\n"
            "- **Total training loss:** in Stage 2, $L_{\\text{total}}=L_{\\text{data}}+\\lambda_pL_{\\text{physics}}$.\n"
        )
        set_text(cell, s)
        break


# -----------------------------------------------------------------------------
# 4. Make N_FREQ the single student-facing Fourier-feature setting.
# -----------------------------------------------------------------------------
for cell in nb["cells"]:
    s = text(cell)
    if cell.get("cell_type") == "markdown" and "# 🔵 A Neural-Network Approximation" in s:
        s = s.replace(
            "With `n_freq = 30`, the network receives",
            "The number of Fourier frequencies is controlled by the single student setting `N_FREQ`. With `N_FREQ = 30`, the network receives"
        )
        s = s.replace(
            "The purpose is not to change the differential equation.",
            "When you experiment with the number of Fourier features later, change only `N_FREQ` in the marked code cell. The purpose is not to change the differential equation."
        )
        set_text(cell, s)

    if cell.get("cell_type") == "code" and "class FourierPINN(nn.Module):" in s:
        s = s.replace(
            "class FourierPINN(nn.Module):",
            "# ============================================================\n# STUDENT SETTING: NUMBER OF FOURIER FREQUENCIES\n# Change ONLY this value for the Fourier-feature experiment.\n# ============================================================\nN_FREQ = 30\n\n\nclass FourierPINN(nn.Module):"
        )
        s = s.replace("        n_freq=30,\n", "        n_freq,\n")
        s = s.replace(
            "model = FourierPINN(\n",
            "# Reset the seed so re-running this cell gives the same initialization.\nreset_random_seed()\n\nmodel = FourierPINN(\n"
        )
        s = s.replace("    n_freq=30,\n", "    n_freq=N_FREQ,\n")
        set_text(cell, s)


# -----------------------------------------------------------------------------
# 5. Clarify exactly which Stage 1 curves students should inspect.
# -----------------------------------------------------------------------------
for cell in nb["cells"]:
    s = text(cell)
    if cell.get("cell_type") == "markdown" and "### Check the Stage 1 fit" in s:
        new = """### Check the Stage 1 fit

The next two plots compare the measurements with the neural-network prediction after data-only training.

Look separately at the **left-pendulum plot** and the **right-pendulum plot**. In each plot, distinguish among:

- the training points, which the network used;
- the validation points, which the network did not use;
- the continuous predicted curve.

Questions to consider:

1. **Left-pendulum plot:** How closely does the predicted $\\theta_1(t)$ curve follow the training measurements? How well does it match the later validation measurements?
2. **Right-pendulum plot:** How closely does the predicted $\\theta_2(t)$ curve follow the training measurements? How well does it match the later validation measurements?
3. Do you notice a difference between the quality of the left- and right-pendulum fits?
4. Do the predicted curves continue to oscillate in a physically reasonable way in the validation interval?

At this stage, the network has learned from data, but we have not yet required it to satisfy the coupled-pendulum ODEs.
"""
        set_text(cell, new)
        break


# -----------------------------------------------------------------------------
# 6. Rewrite Fourier-feature experiment so there is only one place to edit.
# -----------------------------------------------------------------------------
for cell in nb["cells"]:
    s = text(cell)
    if cell.get("cell_type") == "markdown" and "## ❓ Question 2: How Many Fourier Features Do We Need?" in s:
        new = """---

## ❓ Question 2: How Many Fourier Features Do We Need?

The neural network uses Fourier features to help represent oscillatory motion. The number of frequencies is controlled by the single setting

```python
N_FREQ = 30
```

in the code cell labeled **STUDENT SETTING: NUMBER OF FOURIER FREQUENCIES**.

### Your Task

Change **only `N_FREQ`** in that marked cell. Do not edit the `FourierPINN` class itself.

Try these values one at a time:

```python
N_FREQ = 0
N_FREQ = 5
N_FREQ = 30
```

For each choice, re-run the neural-network definition cell and then re-run **Stage 1**. Because the random seed is reset when a new model is created, each comparison starts from the same random initialization.

Record the **final training loss** and **final validation loss**:

| `N_FREQ` | Final training loss | Final validation loss |
|---:|---:|---:|
| 0 |  |  |
| 5 |  |  |
| 30 |  |  |

### Questions to Answer

1. Which value of `N_FREQ` gives the smallest training loss?
2. Which value gives the smallest validation loss?
3. Does increasing `N_FREQ` always improve the validation loss?
4. In this experiment, how does `N_FREQ = 5` compare with `N_FREQ = 30`?
5. Why is the validation loss especially useful when deciding how many Fourier features to use?

### What to Notice

A small training loss tells us how well the network fits the measurements it has seen. The validation loss tells us how well it predicts the held-out final 20% of the experiment. The validation loss shown here is a **combined error for both pendulums**.

**Your observations:**  
*Write here.*

---
"""
        set_text(cell, new)

    if cell.get("cell_type") == "markdown" and "### Before You Continue" in s and "n_freq" in s:
        s = s.replace("`n_freq`", "`N_FREQ`")
        s = s.replace("go back to the neural-network definition cell**, set `N_FREQ`", "go back to the marked Fourier-feature setting and set `N_FREQ`")
        set_text(cell, s)


# -----------------------------------------------------------------------------
# 7. Fix lambda_p inconsistency: the default is 0.5 everywhere.
# -----------------------------------------------------------------------------
for cell in nb["cells"]:
    s = text(cell)
    if cell.get("cell_type") == "markdown" and "# 🔵 Stage 2: Adding the Differential Equations" in s:
        s = s.replace("\\lambda_p=2.", "\\lambda_p=0.5.")
        s = s.replace(
            "The number $\\lambda_p$ tells us how much weight to give the differential equations.",
            "The number $\\lambda_p$ tells us how much weight to give the differential equations. The default value used in the Stage 2 code is $\\lambda_p=0.5$; later you will compare this with other choices."
        )
        set_text(cell, s)

    if cell.get("cell_type") == "code" and "# 15. STAGE 2:" in s:
        # Put lambda_p in one obvious location above the loop instead of redefining it every epoch.
        s = s.replace(
            "d2_history = []\n\nprint()",
            "d2_history = []\n\n# ============================================================\n# STUDENT SETTING: PHYSICS-LOSS WEIGHT\n# Default value for the main run. Change this only when Question 3 asks you to.\n# ============================================================\nlambda_p = 0.5\n\nprint()"
        )
        s = s.replace("\n    lambda_p = 0.5\n\n    total_loss", "\n    total_loss")
        set_text(cell, s)


# -----------------------------------------------------------------------------
# 8. Explain spiky loss curves and remind students what each curve means.
# -----------------------------------------------------------------------------
for cell in nb["cells"]:
    s = text(cell)
    if cell.get("cell_type") == "markdown" and "## 14. Compare the errors during training" in s:
        new = """## 14. Compare the errors during training

The next plots show how the errors change as the model is trained. An **epoch** is one update step in the optimization process.

We will compare:

- **training data loss:** fit to the first 80% of the measured angles;
- **validation loss:** combined left- and right-pendulum error on the held-out final 20%;
- **physics loss:** how closely the predictions satisfy the two ODEs;
- **total loss:** the quantity minimized in Stage 2, $L_{\\text{data}}+\\lambda_pL_{\\text{physics}}$.

### Why can the loss graphs look spiky?

Do not expect every loss curve to decrease smoothly at every epoch. The Adam optimizer can move slightly up and down while searching for better parameters. In **Stage 2**, the physics loss is especially likely to fluctuate because the code draws a new random set of collocation points at every epoch. The logarithmic vertical scale also makes small fluctuations look more dramatic.

Focus on the **overall trend**, not on whether every individual step decreases. A decreasing training loss means the model is fitting the known data better; a decreasing physics loss means the ODE residuals are becoming smaller; and a small validation loss means the model is predicting the held-out measurements well.

The validation loss does not have to decrease whenever the training loss decreases. That is exactly why we keep validation data separate.
"""
        set_text(cell, new)
        break


# -----------------------------------------------------------------------------
# 9. Clarify Question 3 instructions and validation loss.
# -----------------------------------------------------------------------------
for cell in nb["cells"]:
    s = text(cell)
    if cell.get("cell_type") == "markdown" and "## ❓ Question 3: How Much Should We Trust the Physics?" in s:
        s = s.replace("Keep the better value of `n_freq`", "Keep the better value of `N_FREQ`")
        s = s.replace("uses the `n_freq` value", "uses the `N_FREQ` value")
        s = s.replace(
            "6. Record the final training data loss, validation loss, physics loss, and estimated $K=k/m$.",
            "6. Record the final training data loss, validation loss, physics loss, and estimated $K=k/m$. Remember that the validation loss is one combined value for the left and right pendulums."
        )
        set_text(cell, s)


# -----------------------------------------------------------------------------
# 10. Make the initial-guess study optional so the core lesson can fit the class period.
# -----------------------------------------------------------------------------
for cell in nb["cells"]:
    s = text(cell)
    if cell.get("cell_type") == "markdown" and "## ❓ Question 4: Do the Initial Parameter Guesses Matter?" in s:
        s = s.replace(
            "## ❓ Question 4: Do the Initial Parameter Guesses Matter?",
            "## ⭐ Optional Extension: Do the Initial Parameter Guesses Matter?"
        )
        s = s.replace(
            "Now investigate whether the final parameter estimates depend strongly on the starting guess.",
            "If time permits, investigate whether the final parameter estimates depend strongly on the starting guess. The core lesson is complete before this extension."
        )
        s = s.replace("`n_freq`", "`N_FREQ`")
        set_text(cell, s)


# -----------------------------------------------------------------------------
# 11. Remove the final comparison question that requires another lesson.
# -----------------------------------------------------------------------------
nb["cells"] = [
    cell for cell in nb["cells"]
    if "## ❓ Final Comparison: How Do the Two Methods Compare?" not in text(cell)
]


with NOTEBOOK.open("w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("Updated", NOTEBOOK)
