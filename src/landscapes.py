"""
landscapes.py
-------------
Loss landscape functions for optimizer benchmarking.

Includes:
  - Quadratic (convex, well-conditioned)
  - Ill-conditioned Quadratic
  - Rosenbrock (non-convex)
  - Logistic Regression loss (for ML context)
  - Simple neural network loss surface
"""

import numpy as np


# ─── QUADRATIC FUNCTIONS ──────────────────────────────────────────────────────

class QuadraticLandscape:
    """
    f(x) = 0.5 * x^T A x - b^T x
    gradient: A x - b
    minimum: x* = A^{-1} b
    """

    def __init__(self, A, b):
        """
        Parameters
        ----------
        A : (n, n) positive definite matrix
        b : (n,) vector
        """
        self.A = np.array(A, dtype=np.float64)
        self.b = np.array(b, dtype=np.float64)
        self.x_star = np.linalg.solve(A, b)
        self.f_star = self.f(self.x_star)

        eigvals = np.linalg.eigvalsh(A)
        self.L = float(np.max(eigvals))      # Lipschitz constant
        self.mu = float(np.min(eigvals))     # strong convexity constant
        self.kappa = self.L / self.mu        # condition number

    def f(self, x):
        x = np.array(x, dtype=np.float64)
        return 0.5 * x @ self.A @ x - self.b @ x

    def grad(self, x):
        x = np.array(x, dtype=np.float64)
        return self.A @ x - self.b

    def suboptimality(self, x):
        return self.f(x) - self.f_star


def make_well_conditioned_quadratic(n=10, seed=42):
    """Condition number ≈ 2."""
    rng = np.random.default_rng(seed)
    eigvals = np.linspace(1.0, 2.0, n)
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    A = Q @ np.diag(eigvals) @ Q.T
    b = rng.standard_normal(n)
    return QuadraticLandscape(A, b)


def make_ill_conditioned_quadratic(n=10, kappa=1000, seed=42):
    """Condition number ≈ kappa."""
    rng = np.random.default_rng(seed)
    eigvals = np.linspace(1.0, kappa, n)
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    A = Q @ np.diag(eigvals) @ Q.T
    b = rng.standard_normal(n)
    return QuadraticLandscape(A, b)


# ─── ROSENBROCK ───────────────────────────────────────────────────────────────

class RosenbrockLandscape:
    """
    f(x, y) = (a - x)^2 + b*(y - x^2)^2
    Global minimum at (a, a^2), f* = 0.
    Classic non-convex test function.
    """

    def __init__(self, a=1.0, b=100.0):
        self.a = a
        self.b = b
        self.x_star = np.array([a, a**2])
        self.f_star = 0.0

    def f(self, x):
        x = np.asarray(x, dtype=np.float64)
        return (self.a - x[0])**2 + self.b * (x[1] - x[0]**2)**2

    def grad(self, x):
        x = np.asarray(x, dtype=np.float64)
        dfdx = -2*(self.a - x[0]) - 4*self.b*x[0]*(x[1] - x[0]**2)
        dfdy = 2*self.b*(x[1] - x[0]**2)
        return np.array([dfdx, dfdy])

    def suboptimality(self, x):
        return self.f(x) - self.f_star


# ─── LOGISTIC REGRESSION ──────────────────────────────────────────────────────

class LogisticLandscape:
    """
    Binary logistic regression loss (convex, smooth).
    f(w) = (1/n) sum_i log(1 + exp(-y_i * x_i^T w)) + (lambda/2) ||w||^2
    """

    def __init__(self, X, y, lam=1e-3):
        self.X = np.array(X, dtype=np.float64)
        self.y = np.array(y, dtype=np.float64)
        self.lam = lam
        self.n, self.d = X.shape
        self.x_star = None  # no closed form

    def _sigmoid(self, z):
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    def f(self, w):
        w = np.array(w, dtype=np.float64)
        z = self.y * (self.X @ w)
        loss = np.mean(np.log1p(np.exp(-z)))
        return loss + 0.5 * self.lam * np.dot(w, w)

    def grad(self, w):
        w = np.array(w, dtype=np.float64)
        z = self.y * (self.X @ w)
        s = self._sigmoid(-z)
        return (-1/self.n) * (self.X.T @ (self.y * s)) + self.lam * w

    def stochastic_grad(self, w, batch_size=32, rng=None):
        """Mini-batch stochastic gradient."""
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(self.n, size=batch_size, replace=False)
        X_b, y_b = self.X[idx], self.y[idx]
        z = y_b * (X_b @ w)
        s = self._sigmoid(-z)
        return (-1/batch_size) * (X_b.T @ (y_b * s)) + self.lam * w


def make_logistic_landscape(n=500, d=20, seed=42):
    """Generate a synthetic binary classification dataset."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w_true = rng.standard_normal(d) * 0.5
    y = np.sign(X @ w_true + 0.1 * rng.standard_normal(n))
    return LogisticLandscape(X, y)
