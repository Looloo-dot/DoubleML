# DML-PLIV + R-learner (standalone)
# Requirements: numpy, scikit-learn

import numpy as np
from typing import Tuple
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor

def set_seed(seed: int = 123) -> None:
    np.random.seed(seed)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def clone_hgbr(model: HistGradientBoostingRegressor) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**model.get_params())

def cross_fit_oof(
    model: HistGradientBoostingRegressor,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 2,
    random_state: int = 123
) -> np.ndarray:
    n = X.shape[0]
    oof = np.zeros(n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr, te in kf.split(X):
        m = clone_hgbr(model)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof

def generate_data(n: int = 2000, p: int = 10, seed: int = 42) -> Tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)

    # Features with controlled scale
    X = rng.normal(loc=0.0, scale=1.0, size=(n, p)).astype(np.float64)
    X /= np.sqrt(np.maximum(1.0, np.var(X, axis=0, ddof=1)))  # column-scale guard

    # Unobserved confounder
    U = rng.normal(size=n)

    # Helper to create unit-norm vectors
    def unit_vec(k):
        v = rng.normal(size=k).astype(np.float64)
        norm = np.linalg.norm(v)
        return v / (norm + 1e-12)

    gamma = unit_vec(p) * 0.8   # instrument strength controlled
    beta  = unit_vec(p) * 0.8
    w     = unit_vec(p)

    # Safe linear forms
    Xg = X @ gamma
    Xb = X @ beta
    Xw = X @ w

    # Clip linear forms to avoid extreme tails from BLAS
    Xg = np.clip(Xg, -12.0, 12.0)
    Xb = np.clip(Xb, -12.0, 12.0)
    Xw = np.clip(Xw, -12.0, 12.0)

    # Instrument
    Z = Xg + rng.normal(scale=1.0, size=n)

    # Endogenous treatment
    D = 0.8 * Z + Xb + 0.7 * U + rng.normal(scale=1.0, size=n)

    # Heterogeneous treatment effect, bounded via sigmoid
    def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
    tau_x = 0.3 + 1.2 * sigmoid(Xw) + 0.5 * (X[:, 0] > 0).astype(float)

    # Baseline outcome
    g_x = 2 * np.sin(np.clip(X[:, 0], -12, 12)) \
        + 0.5 * X[:, 1] ** 2 \
        - 0.3 * X[:, 2] * X[:, 3] \
        + 0.1 * X[:, 4]

    Y = g_x + tau_x * D + U + rng.normal(scale=1.0, size=n)

    # Final sanity: replace any non-finite values (belt-and-braces)
    for arr in (X, Y, D, Z, tau_x):
        if not np.all(np.isfinite(arr)):
            raise RuntimeError("Non-finite values detected in generated data.")
    true_ate = float(tau_x.mean())
    return X, Y, D, Z, tau_x, true_ate

class DMLPLIV:
    def __init__(
        self,
        n_splits: int = 2,
        random_state: int = 123,
        params_g: dict = None,
        params_m: dict = None,
        params_r: dict = None
    ):
        base = dict(loss='squared_error', max_depth=6, learning_rate=0.05, max_iter=350, random_state=random_state)
        self.model_g = HistGradientBoostingRegressor(**(params_g or base))
        self.model_m = HistGradientBoostingRegressor(**(params_m or {**base, "random_state": random_state + 1}))
        self.model_r = HistGradientBoostingRegressor(**(params_r or {**base, "random_state": random_state + 2}))
        self.n_splits = n_splits
        self.random_state = random_state
        self.theta_ = None
        self.se_ = None
        self.ci_ = None

    def fit(self, X: np.ndarray, Y: np.ndarray, D: np.ndarray, Z: np.ndarray) -> "DMLPLIV":
        g_hat = cross_fit_oof(
            self.model_g, X, Y, n_splits=self.n_splits, random_state=self.random_state
        )
        m_hat = cross_fit_oof(
            self.model_m, X, D, n_splits=self.n_splits, random_state=self.random_state + 1
        )
        r_hat = cross_fit_oof(
            self.model_r, X, Z, n_splits=self.n_splits, random_state=self.random_state + 2
        )

        y_tilde = Y - g_hat
        d_tilde = D - m_hat
        z_tilde = Z - r_hat

        num = float(np.dot(z_tilde, y_tilde))
        den = float(np.dot(z_tilde, d_tilde))
        self.theta_ = num / den

        psi = z_tilde * (y_tilde - self.theta_ * d_tilde)
        n = len(Y)
        Mhat = den / n
        S2 = np.mean(psi ** 2)
        self.se_ = float(np.sqrt(S2 / (n * (Mhat ** 2))))

        zcrit = 1.959963984540054
        self.ci_ = (self.theta_ - zcrit * self.se_, self.theta_ + zcrit * self.se_)
        return self

class RLearnerCATE:
    def __init__(
        self,
        n_splits: int = 2,
        random_state: int = 123,
        params_m: dict = None,
        params_e: dict = None,
        params_tau: dict = None
    ):
        base = dict(loss='squared_error', max_depth=6, learning_rate=0.05, max_iter=350, random_state=random_state)
        self.model_m = HistGradientBoostingRegressor(**(params_m or base))
        self.model_e = HistGradientBoostingRegressor(**(params_e or {**base, "random_state": random_state + 1}))
        self.model_tau = HistGradientBoostingRegressor(**(params_tau or {**base, "max_iter": 550, "random_state": random_state + 2}))
        self.n_splits = n_splits
        self.random_state = random_state
        self._mhat = None
        self._ehat = None
        self._fitted = False

    def fit(self, X: np.ndarray, Y: np.ndarray, D: np.ndarray) -> "RLearnerCATE":
        m_hat = cross_fit_oof(
            self.model_m, X, Y, n_splits=self.n_splits, random_state=self.random_state
        )
        e_hat = cross_fit_oof(
            self.model_e, X, D, n_splits=self.n_splits, random_state=self.random_state + 1
        )

        denom = D - e_hat
        eps = 1e-3
        small = np.abs(denom) < eps
        denom = denom + np.sign(denom) * eps * small + eps * (denom == 0)

        y_tilde = (Y - m_hat) / denom
        w = denom ** 2

        model = clone_hgbr(self.model_tau)
        model.fit(X, y_tilde, sample_weight=w)

        self.model_tau = model
        self._mhat = m_hat
        self._ehat = e_hat
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        return self.model_tau.predict(X)

if __name__ == "__main__":
    set_seed(77)
    X, Y, D, Z, tau_true_x, true_ate = generate_data(n=2500, p=12, seed=7)

    dml = DMLPLIV(n_splits=3, random_state=77).fit(X, Y, D, Z)
    theta = float(dml.theta_)
    se = float(dml.se_)
    ci_low, ci_high = float(dml.ci_[0]), float(dml.ci_[1])

    rlearner = RLearnerCATE(n_splits=3, random_state=77).fit(X, Y, D)
    tau_hat_x = rlearner.predict(X)

    corr_true_pred = float(np.corrcoef(tau_true_x, tau_hat_x)[0, 1])
    ate_from_tau = float(np.mean(tau_hat_x))

    q75 = float(np.quantile(tau_hat_x, 0.75))
    top_idx = tau_hat_x >= q75
    ate_top_true = float(np.mean(tau_true_x[top_idx]))
    ate_top_pred = float(np.mean(tau_hat_x[top_idx]))

    print("\n=== DML-PLIV (ATE) ===")
    print(f"True ATE:                 {true_ate: .6f}")
    print(f"Estimated ATE (theta):    {theta: .6f}")
    print(f"Std. Error (IF-based):    {se: .6f}")
    print(f"95% CI:                   [{ci_low: .6f}, {ci_high: .6f}]")

    print("\n=== R-learner CATE ===")
    print(f"Corr(true, pred):         {corr_true_pred: .4f}")
    print(f"ATE from CATE mean:       {ate_from_tau: .6f}")

    print("\n=== Targeting (Top 25% by predicted CATE) ===")
    print(f"Avg true CATE (top 25%):  {ate_top_true: .6f}")
    print(f"Avg pred CATE (top 25%):  {ate_top_pred: .6f}")

    assert np.isfinite(theta) and np.isfinite(se)
    assert np.isfinite(corr_true_pred)
    assert tau_hat_x.shape[0] == X.shape[0]



# Visualization

import matplotlib.pyplot as plt
import os

# Create figs folder if not exists
os.makedirs("figs", exist_ok=True)

# 1. Scatter: True vs Predicted CATE
plt.figure(figsize=(6,6))
plt.scatter(tau_true_x, tau_hat_x, alpha=0.3, s=10)
plt.xlabel("True CATE")
plt.ylabel("Predicted CATE")
plt.title("True vs Predicted Individual Treatment Effects")
plt.axline((0,0),(1,1), color="red", linestyle="--")
plt.savefig("figs/scatter_true_vs_pred_cate.png", dpi=300)
plt.close()

# 2. Distribution of CATE
plt.figure(figsize=(8,4))
plt.hist(tau_true_x, bins=30, alpha=0.5, label="True CATE")
plt.hist(tau_hat_x, bins=30, alpha=0.5, label="Predicted CATE")
plt.axvline(dml.theta_, color="red", linestyle="--", label="Estimated ATE")
plt.axvline(true_ate, color="green", linestyle="--", label="True ATE")
plt.legend()
plt.title("Distribution of Treatment Effects")
plt.savefig("figs/distribution_cate.png", dpi=300)
plt.close()

# 3. Policy targeting curve
sorted_idx = np.argsort(-tau_hat_x)  # rank by predicted benefit
cumulative_true = np.cumsum(tau_true_x[sorted_idx]) / np.arange(1, len(tau_true_x)+1)

plt.figure(figsize=(8,5))
plt.plot(np.linspace(0,100,len(cumulative_true)), cumulative_true, label="Cumulative avg true effect")
plt.axhline(y=true_ate, color="red", linestyle="--", label="Population ATE")
plt.xlabel("Top % of population treated (by predicted CATE rank)")
plt.ylabel("Cumulative avg true CATE")
plt.title("Policy Targeting Curve")
plt.legend()
plt.savefig("figs/policy_targeting_curve.png", dpi=300)
plt.close()

print("\nFigures saved in ./figs/")
