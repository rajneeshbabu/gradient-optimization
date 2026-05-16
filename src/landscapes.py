"""
landscapes.py

Loss landscape functions for testing and comparing optimizers.
These are the functions we actually minimize -- each one tests a
different property of an optimizer (convexity, conditioning, non-convexity).

Landscapes:
    - QuadraticLandscape      : convex, closed-form optimum, tunable conditioning
    - RosenbrockLandscape     : classic non-convex banana-shaped valley
    - LogisticLandscape       : logistic regression loss (convex, ML context)
"""

import numpy as np


# --------------------------------------------------------------------------- #
#  Quadratic functions                                                          #
# --------------------------------------------------------------------------- #

class QuadraticLandscape:
    """
    f(x) = 0.5 * x^T A x - b^T x

    This is the classic convex quadratic -- gradient descent on this
    is fully understood theoretically, so it's the best starting point
    for seeing how different optimizers behave.

    The condition number kappa = L / mu tells you how hard the problem is:
    - kappa ~ 1   : well-conditioned, GD converges fast
    - kappa ~ 1000: ill-conditioned, GD zigzags, momentum helps a lot

    The optimal x* = A^{-1} b is precomputed so we can track suboptimality.
    """

    def __init__(self, A, b):
        """
        Args:
            A : (n, n) positive definite matrix -- defines the shape of the bowl
            b : (n,) vector -- shifts the minimum away from the origin
        """
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)

        # precompute the optimum
        self.x_star = np.linalg.solve(A, b)
        self.f_star = self.f(self.x_star)

        # Lipschitz constant L and strong convexity mu from eigenvalues
        eigvals = np.linalg.eigvalsh(A)
        self.L = float(np.max(eigvals))
        self.mu = float(np.min(eigvals))
        self.kappa = self.L / self.mu   # condition number

    def f(self, x):
        x = np.asarray(x, dtype=float)
        return 0.5 * x @ self.A @ x - self.b @ x

    def grad(self, x):
        x = np.asarray(x, dtype=float)
        return self.A @ x - self.b

    def suboptimality(self, x):
        """How far above the minimum we are -- useful for convergence plots."""
        return self.f(x) - self.f_star

    def __repr__(self):
        return f"QuadraticLandscape(n={len(self.b)}, kappa={self.kappa:.1f})"


def make_well_conditioned_quadratic(n=10, seed=42):
    """
    Creates a quadratic with condition number ~2 (easy to optimize).
    GD, Momentum, Adam all converge quickly here.
    """
    rng = np.random.default_rng(seed)
    eigvals = np.linspace(1.0, 2.0, n)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = Q @ np.diag(eigvals) @ Q.T
    b = rng.standard_normal(n)
    return QuadraticLandscape(A, b)


def make_ill_conditioned_quadratic(n=10, kappa=1000, seed=42):
    """
    Creates a quadratic with condition number ~kappa (hard to optimize).
    Vanilla GD oscillates badly; momentum and Adam shine here.
    """
    rng = np.random.default_rng(seed)
    eigvals = np.linspace(1.0, float(kappa), n)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = Q @ np.diag(eigvals) @ Q.T
    b = rng.standard_normal(n)
    return QuadraticLandscape(A, b)


# --------------------------------------------------------------------------- #
#  Rosenbrock                                                                   #
# --------------------------------------------------------------------------- #

class RosenbrockLandscape:
    """
    f(x, y) = (a - x)^2 + b * (y - x^2)^2

    The Rosenbrock function is the classic non-convex test -- it has a
    narrow, curved banana-shaped valley that's easy to find but very hard
    to follow to the minimum. The global minimum is at (a, a^2) where f=0.

    With a=1, b=100 (defaults): the valley curves steeply, and the
    gradient nearly vanishes along the valley floor but points steeply
    across it. This is where adaptive methods like Adam really help.
    """

    def __init__(self, a=1.0, b=100.0):
        self.a = a
        self.b = b
        self.x_star = np.array([a, a ** 2])
        self.f_star = 0.0

    def f(self, x):
        x = np.asarray(x, dtype=float)
        return (self.a - x[0]) ** 2 + self.b * (x[1] - x[0] ** 2) ** 2

    def grad(self, x):
        x = np.asarray(x, dtype=float)
        df_dx = -2 * (self.a - x[0]) - 4 * self.b * x[0] * (x[1] - x[0] ** 2)
        df_dy = 2 * self.b * (x[1] - x[0] ** 2)
        return np.array([df_dx, df_dy])

    def suboptimality(self, x):
        return self.f(x) - self.f_star

    def __repr__(self):
        return f"RosenbrockLandscape(a={self.a}, b={self.b})"


# --------------------------------------------------------------------------- #
#  Logistic Regression                                                          #
# --------------------------------------------------------------------------- #

class LogisticLandscape:
    """
    Binary logistic regression loss with L2 regularization.

    f(w) = (1/n) * sum_i log(1 + exp(-y_i * x_i^T w))  +  (lam/2) * ||w||^2

    This is a convex, smooth loss that shows up in real ML tasks. It's
    a good middle ground between the clean theory of quadratics and the
    complexity of neural network losses. Also supports stochastic gradients
    (mini-batch), so SGD can be benchmarked here.
    """

    def __init__(self, X, y, lam=1e-3):
        """
        Args:
            X   : (n, d) feature matrix
            y   : (n,) labels in {-1, +1}
            lam : L2 regularization strength
        """
        self.X = np.array(X, dtype=float)
        self.y = np.array(y, dtype=float)
        self.lam = lam
        self.n, self.d = X.shape

    def _sigmoid(self, z):
        # numerically stable sigmoid
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    def f(self, w):
        w = np.asarray(w, dtype=float)
        z = self.y * (self.X @ w)
        loss = np.mean(np.log1p(np.exp(-z)))
        return loss + 0.5 * self.lam * np.dot(w, w)

    def grad(self, w):
        w = np.asarray(w, dtype=float)
        z = self.y * (self.X @ w)
        s = self._sigmoid(-z)
        return (-1 / self.n) * (self.X.T @ (self.y * s)) + self.lam * w

    def stochastic_grad(self, w, batch_size=32, rng=None):
        """Mini-batch gradient -- needed for SGD."""
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(self.n, size=batch_size, replace=False)
        X_b, y_b = self.X[idx], self.y[idx]
        z = y_b * (X_b @ w)
        s = self._sigmoid(-z)
        return (-1 / batch_size) * (X_b.T @ (y_b * s)) + self.lam * w

    def __repr__(self):
        return f"LogisticLandscape(n={self.n}, d={self.d}, lam={self.lam})"


def make_logistic_landscape(n=500, d=20, seed=42):
    """
    Generate a synthetic binary classification problem.
    Labels are determined by a random linear separator with some noise.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w_true = rng.standard_normal(d) * 0.5
    y = np.sign(X @ w_true + 0.1 * rng.standard_normal(n))
    return LogisticLandscape(X, y)
