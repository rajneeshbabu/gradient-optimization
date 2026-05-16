# Gradient-Based Optimization: Convergence Analysis

[![GitHub Pages](https://img.shields.io/badge/🌐_Project_Page-GitHub_Pages-6366f1?style=for-the-badge)](https://rajneeshbabu.github.io/gradient-optimization/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy)](https://numpy.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)

**Author:** Rajneesh Babu  
**Project Page:** [rajneeshbabu.github.io/gradient-optimization](https://rajneeshbabu.github.io/gradient-optimization/)

---

## What this is

I built this project to deeply understand what gradient-based optimizers are actually doing — not just using `torch.optim.Adam` as a black box, but implementing every update rule from scratch in NumPy and watching how each one behaves on different loss landscapes.

The project covers 7 optimizers, 3 loss landscapes, and 3 learning rate schedules. Every optimizer is benchmarked on the same starting point and plotted on the same axes so comparisons are fair.

---

## Optimizers Implemented

| Optimizer | Key Idea |
|-----------|----------|
| Gradient Descent | Baseline — pure gradient step |
| SGD (mini-batch) | Noisy gradient, faster in practice |
| Momentum | Velocity accumulation — dampens oscillations |
| Nesterov AGD | Look-ahead before gradient step — O(1/k²) rate |
| AdaGrad | Per-parameter adaptive lr — good for sparse gradients |
| RMSProp | Fixes AdaGrad's vanishing lr using EMA |
| Adam | Momentum + RMSProp + bias correction |

---

## Loss Landscapes

- **Well-conditioned quadratic** (κ ≈ 2) — all optimizers converge fast, baseline comparison
- **Ill-conditioned quadratic** (κ = 1000) — exposes GD's zigzagging, shows where momentum and Adam win
- **Rosenbrock** — non-convex banana-shaped valley, tests ability to follow a curved narrow path
- **Logistic regression** — real ML loss, supports mini-batch SGD

---

## Key Findings

- On **ill-conditioned problems**, Nesterov and Adam converge in ~50 iterations vs ~500 for GD
- **Adam** reaches a good solution fastest but can overshoot on simple convex problems
- **AdaGrad** works well early but stalls on dense gradients as its lr decays to near zero
- **Cosine annealing** consistently outperforms fixed lr and step decay across all methods
- **Momentum β=0.9** cuts iteration count by ~3× on quadratics with κ=1000

---

## Project Structure

```
gradient-optimization/
├── src/
│   ├── optimizers.py     # All 7 optimizers + 3 LR schedules (NumPy)
│   └── landscapes.py     # Quadratic, Rosenbrock, Logistic loss functions
├── notebooks/
│   ├── 01_convex_analysis.ipynb       # Convergence on quadratic landscapes
│   ├── 02_optimizer_comparison.ipynb  # Head-to-head comparison of all 7
│   ├── 03_lr_schedules.ipynb          # Fixed vs cosine vs step decay
│   └── 04_dl_benchmarks.ipynb         # PyTorch MLP training comparison
├── results/
│   └── figures/          # Saved convergence plots
├── requirements.txt
└── README.md
```

---

## Run It

```bash
git clone https://github.com/rajneeshbabu/gradient-optimization.git
cd gradient-optimization
pip install -r requirements.txt

# Run all notebooks in sequence
jupyter notebook notebooks/
```

Start with `01_convex_analysis.ipynb` — it builds intuition before the harder comparisons.

---

## Quick Code Example

```python
from src.landscapes import make_ill_conditioned_quadratic
from src.optimizers import run_all
import numpy as np
import matplotlib.pyplot as plt

landscape = make_ill_conditioned_quadratic(n=10, kappa=1000)
x0 = np.zeros(10)

results = run_all(landscape, x0, n_iter=300)

for name, r in results.items():
    plt.semilogy(r.f_history, label=name)

plt.xlabel("Iteration")
plt.ylabel("Loss (log scale)")
plt.title("Convergence on Ill-Conditioned Quadratic (κ=1000)")
plt.legend()
plt.show()
```

---

## Dependencies

| Package | Version | Use |
|---------|---------|-----|
| numpy | ≥ 1.23 | All optimizer implementations |
| matplotlib | ≥ 3.6 | Convergence plots |
| torch | ≥ 2.0 | DL benchmark notebook |
| jupyter | ≥ 1.0 | Notebooks |

---

*© 2025 Rajneesh Babu*
