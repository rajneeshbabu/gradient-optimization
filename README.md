# Gradient-Based Optimization: Convergence Analysis of Numerical Methods in ML

[![GitHub Pages](https://img.shields.io/badge/🌐_Project_Page-GitHub_Pages-6366f1?style=for-the-badge)](https://rajneeshbabu.github.io/gradient-optimization/)
[![IISc](https://img.shields.io/badge/IISc-CDS_Course_Project-003580?style=for-the-badge)](https://cds.iisc.ac.in)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python)](https://python.org)

---

## Overview

A course project for **M.Tech Computational and Data Sciences, IISc Bengaluru** analysing the convergence behaviour, stability, and practical performance of gradient-based numerical optimisation algorithms used in machine learning — connecting classical numerical analysis theory to modern deep learning practice.

---

## Algorithms Studied

| Optimiser | Type | Key Property |
|-----------|------|--------------|
| Gradient Descent (GD) | First-order | Baseline; fixed step size |
| Stochastic GD (SGD) | First-order | Noisy gradients; online learning |
| Momentum (Heavy Ball) | First-order + memory | Dampens oscillations |
| Nesterov Accelerated GD | First-order + lookahead | O(1/k²) convergence |
| AdaGrad | Adaptive | Per-parameter learning rate |
| RMSProp | Adaptive | Exponential moving average |
| Adam | Adaptive + momentum | Combines best of both |

---

## Key Questions Explored

- How do step size (learning rate) and curvature of the loss landscape affect convergence rate?
- When does adaptive learning rate help vs. hurt generalisation?
- What is the relationship between Lipschitz smoothness, strong convexity, and convergence guarantees?
- How do momentum-based methods compare to classical accelerated gradient methods?

---

## Key Findings

- **SGD + Momentum** converges 2–4× faster than vanilla GD on ill-conditioned problems
- **Adam** reaches lower loss faster but can generalise worse than SGD on smooth convex problems
- **Nesterov AGD** achieves optimal O(1/k²) rate on strongly convex functions, confirming theory
- **AdaGrad** excels on sparse gradient landscapes but suffers from learning rate decay on dense problems
- **Learning rate schedules** (cosine, step decay) consistently outperform fixed rates across all methods

---

## Tech Stack

- Python 3.10+
- NumPy (custom optimiser implementations)
- PyTorch (deep learning benchmarks)
- Matplotlib / Plotly (convergence curve visualisations)
- Jupyter Notebook

---

## Project Structure

```
gradient-optimization/
├── notebooks/
│   ├── 01_convex_analysis.ipynb       # Convex loss landscapes
│   ├── 02_optimizer_comparison.ipynb  # GD vs SGD vs Adam etc.
│   ├── 03_lr_schedules.ipynb          # Learning rate scheduling
│   └── 04_dl_benchmarks.ipynb         # Neural network training experiments
├── src/
│   ├── optimizers.py     # Numpy implementations of all optimisers
│   └── landscapes.py     # Rosenbrock, quadratic, ill-conditioned functions
├── results/
│   └── figures/
├── report/
│   └── gradient_optimization_report.pdf
├── requirements.txt
└── README.md
```

---

## Run Locally

```bash
git clone https://github.com/rajneeshbabu/gradient-optimization.git
cd gradient-optimization
pip install -r requirements.txt
jupyter notebook notebooks/
```

---

## Course

**DS 285 — Numerical Methods for Data Science**
M.Tech Computational and Data Sciences · IISc Bengaluru · 2024–25

---

*© 2025 Rajneesh Babu · IISc Bengaluru*
