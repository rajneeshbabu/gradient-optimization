"""
optimizers.py

Implementations of 7 gradient-based optimizers from scratch using NumPy.
I wrote these to understand what's actually happening under the hood --
no PyTorch autograd, just the raw update rules.

Optimizers covered:
    - Gradient Descent
    - SGD (with mini-batch + LR schedule support)
    - Momentum (Heavy Ball)
    - Nesterov Accelerated GD
    - AdaGrad
    - RMSProp
    - Adam

Each function returns an OptimResult object that stores the full trajectory,
so you can plot convergence curves and compare them easily.
"""

import numpy as np


# --------------------------------------------------------------------------- #
#  Result container                                                             #
# --------------------------------------------------------------------------- #

class OptimResult:
    """Stores the trajectory of an optimization run."""

    def __init__(self, name):
        self.name = name
        self.x_history = []        # parameter at each step
        self.f_history = []        # loss at each step
        self.g_norm_history = []   # gradient norm (useful for diagnosing convergence)

    def record(self, x, f_val, g_norm):
        self.x_history.append(x.copy())
        self.f_history.append(float(f_val))
        self.g_norm_history.append(float(g_norm))

    @property
    def n_iters(self):
        return len(self.f_history)

    def __repr__(self):
        return (f"OptimResult(name='{self.name}', "
                f"iters={self.n_iters}, "
                f"final_loss={self.f_history[-1]:.6f})")


# --------------------------------------------------------------------------- #
#  1. Gradient Descent                                                          #
# --------------------------------------------------------------------------- #

def gradient_descent(landscape, x0, lr=0.01, n_iter=500, tol=1e-8):
    """
    Plain gradient descent -- the simplest possible optimizer.

    At each step we move in the direction opposite to the gradient:
        x = x - lr * grad(f(x))

    Works well when the loss is convex and well-conditioned, but can be
    painfully slow on ill-conditioned problems (elongated valleys).

    Args:
        landscape : any object with .f(x) and .grad(x) methods
        x0        : starting point
        lr        : step size (learning rate)
        n_iter    : maximum number of steps
        tol       : stop early if gradient norm drops below this

    Returns:
        OptimResult with full trajectory
    """
    result = OptimResult("Gradient Descent")
    x = np.array(x0, dtype=float)

    for _ in range(n_iter):
        g = landscape.grad(x)
        g_norm = np.linalg.norm(g)
        result.record(x, landscape.f(x), g_norm)

        # stop if we are close enough to a critical point
        if g_norm < tol:
            break

        x = x - lr * g

    return result


# --------------------------------------------------------------------------- #
#  2. SGD with optional LR schedule                                             #
# --------------------------------------------------------------------------- #

def sgd(landscape, x0, lr=0.01, n_iter=500, batch_size=32,
        lr_schedule=None, tol=1e-8, seed=42):
    """
    Stochastic Gradient Descent.

    Instead of the full gradient, we use a noisy estimate computed from
    a random mini-batch. This is the workhorse of deep learning training --
    the noise actually helps escape sharp local minima.

    Optionally pass a lr_schedule callable (see the schedule functions below)
    to decay the learning rate over training.

    Args:
        landscape   : must have a .stochastic_grad(x, batch_size, rng) method
        lr_schedule : callable(step) -> float, or None for constant lr
        seed        : for reproducibility

    Returns:
        OptimResult with full trajectory
    """
    result = OptimResult("SGD")
    x = np.array(x0, dtype=float)
    rng = np.random.default_rng(seed)

    for k in range(n_iter):
        # use decayed lr if a schedule was provided
        current_lr = lr_schedule(k) if lr_schedule is not None else lr

        g = landscape.stochastic_grad(x, batch_size=batch_size, rng=rng)
        g_norm = np.linalg.norm(g)
        result.record(x, landscape.f(x), g_norm)

        if g_norm < tol:
            break

        x = x - current_lr * g

    return result


# --------------------------------------------------------------------------- #
#  3. Momentum (Heavy Ball)                                                     #
# --------------------------------------------------------------------------- #

def momentum(landscape, x0, lr=0.01, beta=0.9, n_iter=500, tol=1e-8):
    """
    Gradient descent with momentum (Polyak's Heavy Ball method).

    Instead of stepping purely in the gradient direction, we accumulate
    a velocity vector that builds up in consistent directions and dampens
    oscillations across ravine walls.

        v = beta * v - lr * grad(f(x))
        x = x + v

    beta=0.9 means 90% of the old velocity is kept -- the ball keeps rolling.

    Args:
        beta : momentum coefficient, usually 0.9

    Returns:
        OptimResult with full trajectory
    """
    result = OptimResult("Momentum")
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)   # velocity starts at rest

    for _ in range(n_iter):
        g = landscape.grad(x)
        g_norm = np.linalg.norm(g)
        result.record(x, landscape.f(x), g_norm)

        if g_norm < tol:
            break

        v = beta * v - lr * g   # update velocity
        x = x + v               # move with it

    return result


# --------------------------------------------------------------------------- #
#  4. Nesterov Accelerated Gradient Descent                                     #
# --------------------------------------------------------------------------- #

def nesterov(landscape, x0, lr=0.01, beta=0.9, n_iter=500, tol=1e-8):
    """
    Nesterov accelerated gradient method.

    The key insight over plain momentum: instead of evaluating the gradient
    at the current position, peek ahead to where momentum would take us,
    then evaluate the gradient there. This "look before you leap" trick
    gives the optimal O(1/k^2) convergence rate on convex problems.

        y      = x + beta * (x - x_prev)   <- look-ahead point
        x_new  = y - lr * grad(f(y))       <- step from look-ahead

    Args:
        beta : momentum coefficient

    Returns:
        OptimResult with full trajectory
    """
    result = OptimResult("Nesterov AGD")
    x = np.array(x0, dtype=float)
    x_prev = x.copy()

    for _ in range(n_iter):
        g = landscape.grad(x)
        g_norm = np.linalg.norm(g)
        result.record(x, landscape.f(x), g_norm)

        if g_norm < tol:
            break

        # compute look-ahead point using extrapolation from previous step
        y = x + beta * (x - x_prev)
        g_y = landscape.grad(y)

        x_prev = x.copy()
        x = y - lr * g_y

    return result


# --------------------------------------------------------------------------- #
#  5. AdaGrad                                                                   #
# --------------------------------------------------------------------------- #

def adagrad(landscape, x0, lr=0.1, eps=1e-8, n_iter=500, tol=1e-8):
    """
    AdaGrad -- Adaptive Gradient Algorithm.

    Scales the learning rate for each parameter individually based on the
    history of gradients. Parameters with frequent large updates get a
    smaller effective lr; rarely updated parameters keep a larger lr.
    Great for sparse gradients, but the accumulated denominator grows
    forever -- lr eventually shrinks to near zero.

        G = G + grad^2                      <- cumulative squared gradients
        x = x - (lr / sqrt(G + eps)) * grad

    Args:
        eps : small constant to avoid division by zero

    Returns:
        OptimResult with full trajectory
    """
    result = OptimResult("AdaGrad")
    x = np.array(x0, dtype=float)
    G = np.zeros_like(x)   # running sum of squared gradients

    for _ in range(n_iter):
        g = landscape.grad(x)
        g_norm = np.linalg.norm(g)
        result.record(x, landscape.f(x), g_norm)

        if g_norm < tol:
            break

        G += g ** 2   # keep accumulating -- this eventually kills the lr
        x = x - (lr / (np.sqrt(G) + eps)) * g

    return result


# --------------------------------------------------------------------------- #
#  6. RMSProp                                                                   #
# --------------------------------------------------------------------------- #

def rmsprop(landscape, x0, lr=0.01, rho=0.9, eps=1e-8, n_iter=500, tol=1e-8):
    """
    RMSProp -- Root Mean Square Propagation (Hinton, 2012).

    Fix for AdaGrad's vanishing lr: use an exponential moving average of
    squared gradients instead of summing them all. Old gradients gradually
    expire so the effective lr stays reasonable throughout training.

        v = rho * v + (1 - rho) * grad^2   <- EMA of squared gradients
        x = x - (lr / sqrt(v + eps)) * grad

    Args:
        rho : EMA decay rate (0.9 standard -- forgets ~10% per step)

    Returns:
        OptimResult with full trajectory
    """
    result = OptimResult("RMSProp")
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)

    for _ in range(n_iter):
        g = landscape.grad(x)
        g_norm = np.linalg.norm(g)
        result.record(x, landscape.f(x), g_norm)

        if g_norm < tol:
            break

        v = rho * v + (1 - rho) * g ** 2
        x = x - (lr / (np.sqrt(v) + eps)) * g

    return result


# --------------------------------------------------------------------------- #
#  7. Adam                                                                      #
# --------------------------------------------------------------------------- #

def adam(landscape, x0, lr=0.001, beta1=0.9, beta2=0.999,
         eps=1e-8, n_iter=500, tol=1e-8):
    """
    Adam -- Adaptive Moment Estimation (Kingma & Ba, 2015).

    Combines momentum (1st moment) with RMSProp (2nd moment), plus bias
    correction to handle the cold-start problem in the first few steps.

        m = beta1 * m + (1 - beta1) * grad       <- 1st moment (mean)
        v = beta2 * v + (1 - beta2) * grad^2     <- 2nd moment (uncentered var)
        m_hat = m / (1 - beta1^k)                <- bias-corrected
        v_hat = v / (1 - beta2^k)
        x = x - lr * m_hat / (sqrt(v_hat) + eps)

    Adam is the default choice for most DL projects. Fast convergence,
    relatively tolerant of lr choice. Can overfit vs SGD on some problems.

    Args:
        beta1 : 1st moment decay, controls momentum (default 0.9)
        beta2 : 2nd moment decay, controls adaptive scaling (default 0.999)
        eps   : numerical stability

    Returns:
        OptimResult with full trajectory
    """
    result = OptimResult("Adam")
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)   # 1st moment
    v = np.zeros_like(x)   # 2nd moment

    for k in range(1, n_iter + 1):
        g = landscape.grad(x)
        g_norm = np.linalg.norm(g)
        result.record(x, landscape.f(x), g_norm)

        if g_norm < tol:
            break

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2

        # bias correction matters most in the first ~10 steps
        m_hat = m / (1 - beta1 ** k)
        v_hat = v / (1 - beta2 ** k)

        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)

    return result


# --------------------------------------------------------------------------- #
#  Learning rate schedules                                                      #
# --------------------------------------------------------------------------- #

def cosine_schedule(lr_init, n_iter, lr_min=1e-6):
    """
    Cosine annealing: smoothly decays lr from lr_init down to lr_min.
    Usually better than step decay in practice -- no sudden jumps.

    Usage:
        schedule = cosine_schedule(0.01, n_iter=500)
        sgd(landscape, x0, lr_schedule=schedule, ...)
    """
    def schedule(k):
        progress = k / n_iter
        return lr_min + 0.5 * (lr_init - lr_min) * (1 + np.cos(np.pi * progress))
    return schedule


def step_decay_schedule(lr_init, drop=0.5, every=100):
    """
    Step decay: multiply lr by `drop` every `every` steps.
    Simple and interpretable but causes abrupt lr changes.

    Usage:
        schedule = step_decay_schedule(0.1, drop=0.5, every=100)
    """
    def schedule(k):
        return lr_init * (drop ** (k // every))
    return schedule


def warmup_cosine_schedule(lr_init, warmup_steps, n_iter):
    """
    Linear warmup then cosine decay.
    Helpful when training is unstable at the start -- ramp up lr slowly,
    then anneal. Common in transformer training.

    Usage:
        schedule = warmup_cosine_schedule(0.01, warmup_steps=50, n_iter=500)
    """
    def schedule(k):
        if k < warmup_steps:
            return lr_init * k / max(1, warmup_steps)
        progress = (k - warmup_steps) / max(1, n_iter - warmup_steps)
        return lr_init * 0.5 * (1 + np.cos(np.pi * progress))
    return schedule


# --------------------------------------------------------------------------- #
#  Convenience: run all optimizers at once for comparison                       #
# --------------------------------------------------------------------------- #

def run_all(landscape, x0, n_iter=500,
            gd_lr=0.01, mom_lr=0.01, nag_lr=0.01,
            ada_lr=0.1, rms_lr=0.01, adam_lr=0.001):
    """
    Run all 6 deterministic optimizers on the same problem.
    Returns a dict {name: OptimResult} ready for plotting.

    Quick example:
        from landscapes import make_ill_conditioned_quadratic
        land = make_ill_conditioned_quadratic()
        results = run_all(land, x0=np.zeros(10))
        for name, r in results.items():
            plt.semilogy(r.f_history, label=name)
        plt.legend(); plt.xlabel("Iteration"); plt.ylabel("Loss")
        plt.show()
    """
    return {
        "GD":       gradient_descent(landscape, x0, lr=gd_lr,   n_iter=n_iter),
        "Momentum": momentum(landscape,         x0, lr=mom_lr,  n_iter=n_iter),
        "Nesterov": nesterov(landscape,         x0, lr=nag_lr,  n_iter=n_iter),
        "AdaGrad":  adagrad(landscape,          x0, lr=ada_lr,  n_iter=n_iter),
        "RMSProp":  rmsprop(landscape,          x0, lr=rms_lr,  n_iter=n_iter),
        "Adam":     adam(landscape,             x0, lr=adam_lr, n_iter=n_iter),
    }

# alias kept for notebook compatibility
run_all_optimizers = run_all
