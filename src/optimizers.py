"""
optimizers.py
-------------
NumPy implementations of 7 gradient-based optimization algorithms.

Optimizers:
  1. Gradient Descent (GD)
  2. Stochastic Gradient Descent (SGD)
  3. Momentum (Heavy Ball)
  4. Nesterov Accelerated Gradient Descent (NAG)
  5. AdaGrad
  6. RMSProp
  7. Adam

Each optimizer returns a history of (x, f(x)) pairs for convergence analysis.
"""

import numpy as np


# ─── BASE CLASS ───────────────────────────────────────────────────────────────

class OptimResult:
    """Container for optimization run history."""

    def __init__(self, name):
        self.name = name
        self.x_history = []    # parameter values
        self.f_history = []    # function values
        self.g_norm_history = []  # gradient norms

    def record(self, x, f, g_norm):
        self.x_history.append(x.copy())
        self.f_history.append(float(f))
        self.g_norm_history.append(float(g_norm))

    @property
    def n_iters(self):
        return len(self.f_history)


# ─── 1. GRADIENT DESCENT ──────────────────────────────────────────────────────

def gradient_descent(landscape, x0, lr=0.01, n_iter=500, tol=1e-8):
    """
    Vanilla Gradient Descent.

    x_{k+1} = x_k - lr * grad f(x_k)

    Parameters
    ----------
    landscape : object with .f(x) and .grad(x) methods
    x0        : initial point (array)
    lr        : learning rate (step size)
    n_iter    : max iterations
    tol       : gradient norm tolerance for early stopping

    Returns
    -------
    result : OptimResult
    """
    result = OptimResult("Gradient Descent")
    x = np.array(x0, dtype=np.float64)

    for _ in range(n_iter):
        g = landscape.grad(x)
        g_norm = np.linalg.norm(g)
        result.record(x, landscape.f(x), g_norm)
        if g_norm < tol:
            break
        x = x - lr * g

    return result


# ─── 2. SGD ───────────────────────────────────────────────────────────────────

def sgd(landscape, x0, lr=0.01, n_iter=500, batch_size=32,
        lr_schedule=None, tol=1e-8, seed=42):
    """
    Stochastic Gradient Descent with optional learning rate schedule.

    Parameters
    ----------
    landscape   : object with .stochastic_grad(x, batch_size, rng) method
    lr_schedule : callable(step) -> lr, or None for constant lr
    seed        : random seed for reproducibility

    Returns
    -------
    result : OptimResult
    """
    result = OptimResult("SGD")
    x = np.array(x0, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for k in range(n_iter):
        lr_k = lr_schedule(k) if lr_schedule else lr
        g = landscape.stochastic_grad(x, batch_size=batch_size, rng=rng)
        g_norm = np.linalg.norm(g)
        result.record(x, landscape.f(x), g_norm)
        if g_norm < tol:
            break
        x = x - lr_k * g

    return result


# ─── 3. MOMENTUM (HEAVY BALL) ────────────────────────────────────────────────

def momentum(landscape, x0, lr=0.01, beta=0.9, n_iter=500, tol=1e-8):
    """
    Gradient Descent with Heavy Ball (Polyak) Momentum.

    v_{k+1} = beta * v_k - lr * grad f(x_k)
    x_{k+1} = x_k + v_{k+1}

    Parameters
    ----------
    beta : momentum coefficient (0.9 is typical)

    Returns
    -------
    result : OptimResult
    """
    result = OptimResult("Momentum")
    x = np.array(x0, dtype=np.float64)
    v = np.zeros_like(x)

    for _ in range(n_iter):
        g = landscape.grad(x)
        g_norm = np.linalg.norm(g)
        result.record(x, landscape.f(x), g_norm)
        if g_norm < tol:
            break
        v = beta * v - lr * g
        x = x + v

    return result


# ─── 4. NESTEROV ACCELERATED GRADIENT DESCENT ────────────────────────────────

def nesterov(landscape, x0, lr=0.01, beta=0.9, n_iter=500, tol=1e-8):
    """
    Nesterov Accelerated Gradient Descent.

    y_{k+1} = x_k + beta * (x_k - x_{k-1})   [lookahead]
    x_{k+1} = y_{k+1} - lr * grad f(y_{k+1})

    Achieves optimal O(1/k^2) convergence on strongly convex functions.

    Returns
    -------
    result : OptimResult
    """
    result = OptimResult("Nesterov AGD")
    x = np.array(x0, dtype=np.float64)
    x_prev = x.copy()

    for _ in range(n_iter):
        g = landscape.grad(x)
        g_norm = np.linalg.norm(g)
        result.record(x, landscape.f(x), g_norm)
        if g_norm < tol:
            break
        # Lookahead step
        y = x + beta * (x - x_prev)
        g_y = landscape.grad(y)
        x_prev = x.copy()
        x = y - lr * g_y

    return result


# ─── 5. ADAGRAD ───────────────────────────────────────────────────────────────

def adagrad(landscape, x0, lr=0.1, eps=1e-8, n_iter=500, tol=1e-8):
    """
    AdaGrad — Adaptive Gradient Algorithm.

    G_k = G_{k-1} + grad f(x_k) ⊙ grad f(x_k)   [accumulated squared gradients]
    x_{k+1} = x_k - (lr / sqrt(G_k + eps)) * grad f(x_k)

    Per-parameter adaptive learning rate; excels on sparse gradients.

    Returns
    -------
    result : OptimResult
    """
    result = OptimResult("AdaGrad")
    x = np.array(x0, dtype=np.float64)
    G = np.zeros_like(x)  # accumulated squared gradients

    for _ in range(n_iter):
        g = landscape.grad(x)
        g_norm = np.linalg.norm(g)
        result.record(x, landscape.f(x), g_norm)
        if g_norm < tol:
            break
        G += g**2
        x = x - (lr / (np.sqrt(G) + eps)) * g

    return result


# ─── 6. RMSPROP ───────────────────────────────────────────────────────────────

def rmsprop(landscape, x0, lr=0.01, rho=0.9, eps=1e-8, n_iter=500, tol=1e-8):
    """
    RMSProp — Root Mean Square Propagation.

    v_k = rho * v_{k-1} + (1 - rho) * grad f(x_k)^2   [EMA of squared grads]
    x_{k+1} = x_k - (lr / sqrt(v_k + eps)) * grad f(x_k)

    Prevents AdaGrad's monotonically decreasing learning rates.

    Parameters
    ----------
    rho : decay rate for EMA (0.9 is typical)

    Returns
    -------
    result : OptimResult
    """
    result = OptimResult("RMSProp")
    x = np.array(x0, dtype=np.float64)
    v = np.zeros_like(x)  # EMA of squared gradients

    for _ in range(n_iter):
        g = landscape.grad(x)
        g_norm = np.linalg.norm(g)
        result.record(x, landscape.f(x), g_norm)
        if g_norm < tol:
            break
        v = rho * v + (1 - rho) * g**2
        x = x - (lr / (np.sqrt(v) + eps)) * g

    return result


# ─── 7. ADAM ──────────────────────────────────────────────────────────────────

def adam(landscape, x0, lr=0.001, beta1=0.9, beta2=0.999,
         eps=1e-8, n_iter=500, tol=1e-8):
    """
    Adam — Adaptive Moment Estimation.

    m_k = beta1 * m_{k-1} + (1 - beta1) * grad       [1st moment / momentum]
    v_k = beta2 * v_{k-1} + (1 - beta2) * grad^2     [2nd moment / RMSProp]
    m̂_k = m_k / (1 - beta1^k)                         [bias correction]
    v̂_k = v_k / (1 - beta2^k)                         [bias correction]
    x_{k+1} = x_k - lr * m̂_k / (sqrt(v̂_k) + eps)

    Parameters
    ----------
    beta1 : 1st moment decay (0.9 default)
    beta2 : 2nd moment decay (0.999 default)

    Returns
    -------
    result : OptimResult
    """
    result = OptimResult("Adam")
    x = np.array(x0, dtype=np.float64)
    m = np.zeros_like(x)  # 1st moment
    v = np.zeros_like(x)  # 2nd moment

    for k in range(1, n_iter + 1):
        g = landscape.grad(x)
        g_norm = np.linalg.norm(g)
        result.record(x, landscape.f(x), g_norm)
        if g_norm < tol:
            break
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**k)
        v_hat = v / (1 - beta2**k)
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)

    return result


# ─── LEARNING RATE SCHEDULES ──────────────────────────────────────────────────

def cosine_schedule(lr_init, n_iter, lr_min=1e-6):
    """Returns a callable: step -> lr using cosine annealing."""
    def schedule(k):
        return lr_min + 0.5 * (lr_init - lr_min) * (
            1 + np.cos(np.pi * k / n_iter))
    return schedule


def step_decay_schedule(lr_init, drop=0.5, every=100):
    """Returns a callable: step -> lr using step decay."""
    def schedule(k):
        return lr_init * (drop ** (k // every))
    return schedule


def warmup_cosine_schedule(lr_init, warmup_steps, n_iter):
    """Linear warmup followed by cosine decay."""
    def schedule(k):
        if k < warmup_steps:
            return lr_init * k / max(1, warmup_steps)
        progress = (k - warmup_steps) / max(1, n_iter - warmup_steps)
        return lr_init * 0.5 * (1 + np.cos(np.pi * progress))
    return schedule


# ─── CONVENIENCE: RUN ALL ─────────────────────────────────────────────────────

def run_all_optimizers(landscape, x0, n_iter=500,
                       gd_lr=0.01, sgd_lr=0.05, mom_lr=0.01,
                       nag_lr=0.01, ada_lr=0.1, rms_lr=0.01, adam_lr=0.001):
    """Run all 7 optimizers and return a dict of OptimResult objects."""
    results = {}
    results['GD']       = gradient_descent(landscape, x0, lr=gd_lr,  n_iter=n_iter)
    results['Momentum'] = momentum(landscape,         x0, lr=mom_lr, n_iter=n_iter)
    results['Nesterov'] = nesterov(landscape,         x0, lr=nag_lr, n_iter=n_iter)
    results['AdaGrad']  = adagrad(landscape,          x0, lr=ada_lr, n_iter=n_iter)
    results['RMSProp']  = rmsprop(landscape,          x0, lr=rms_lr, n_iter=n_iter)
    results['Adam']     = adam(landscape,             x0, lr=adam_lr, n_iter=n_iter)
    return results
