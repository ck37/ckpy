"""
Simulation-based coverage tests for nadeau_bengio_cv_ci().

Each test generates data from a known distribution where the true AUC
is analytically computable, runs repeated k-fold cross-validation, and
checks that the empirical coverage of the confidence interval is near
the nominal level.

Data-generating process:
    Y ~ Bernoulli(prevalence)
    X | Y ~ Normal(delta * Y, 1)
    True AUC = Phi(delta / sqrt(2))

where Phi is the standard normal CDF. This follows from the fact that
the AUC equals P(X1 > X0) where X1 ~ N(delta, 1) and X0 ~ N(0, 1),
and the difference X1 - X0 ~ N(delta, sqrt(2)).

References
----------
Nadeau, C. and Bengio, Y. (2003). Inference for the generalization
error. Machine Learning, 52(3), 239-281.

Bouckaert, R.R. and Frank, E. (2004). Evaluating the replicability
of significance tests for comparing learning algorithms. In PAKDD,
Lecture Notes in Computer Science, 3056, 3-12.
"""

import numpy as np
import pytest
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

from ckpy.stats import nadeau_bengio_cv_ci

pytestmark = pytest.mark.slow

# With 300 sims the Monte Carlo SE of coverage is ~0.013 at 0.95,
# so [0.90, 1.00] is a generous ~4 SE band.
COVERAGE_MIN = 0.90
N_SIMS = 300


def _run_coverage_simulation(
    N,
    delta,
    prevalence=0.3,
    n_sims=N_SIMS,
    n_splits=5,
    n_repeats=5,
    alpha=0.05,
    seed=42,
):
    """
    Run a Monte Carlo coverage simulation for nadeau_bengio_cv_ci().

    Parameters
    ----------
    N : int
        Sample size per simulation replicate.
    delta : float
        Signal strength. True AUC = Phi(delta / sqrt(2)).
    prevalence : float
        P(Y = 1).
    n_sims : int
        Number of Monte Carlo replications.
    n_splits : int
        Number of CV folds.
    n_repeats : int
        Number of CV repeats.
    alpha : float
        Significance level for the CI.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        true_auc, N, n_sims_completed,
        logit_coverage, raw_coverage,
        logit_mean_width, raw_mean_width,
        raw_below_0, raw_above_1, boundary_folds, total_folds
    """
    rng = np.random.default_rng(seed)
    true_auc = stats.norm.cdf(delta / np.sqrt(2))

    logit_covers = 0
    raw_covers = 0
    logit_widths = []
    raw_widths = []
    raw_below_0 = 0
    raw_above_1 = 0
    boundary_folds = 0
    total_folds = 0
    completed = 0

    for _ in range(n_sims):
        y = rng.binomial(1, prevalence, size=N)

        # Skip degenerate samples where stratified CV would fail
        if y.sum() < 2 or (N - y.sum()) < 2:
            continue

        X = rng.normal(delta * y, 1.0, size=N).reshape(-1, 1)

        cv = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=int(rng.integers(1e9)),
        )
        aucs, trs, tes = [], [], []
        for train_idx, test_idx in cv.split(X, y):
            model = LogisticRegression(solver="lbfgs", max_iter=500)
            model.fit(X[train_idx], y[train_idx])
            probs = model.predict_proba(X[test_idx])[:, 1]
            aucs.append(roc_auc_score(y[test_idx], probs))
            trs.append(len(train_idx))
            tes.append(len(test_idx))

        boundary_folds += sum(1 for a in aucs if a == 0.0 or a == 1.0)
        total_folds += len(aucs)

        # Logit-transformed CI
        _, lo_l, hi_l, _ = nadeau_bengio_cv_ci(
            aucs, trs, tes, alpha=alpha, logit_transform=True
        )
        logit_covers += int(lo_l <= true_auc <= hi_l)
        logit_widths.append(hi_l - lo_l)

        # Raw (untransformed) CI
        _, lo_r, hi_r, _ = nadeau_bengio_cv_ci(
            aucs, trs, tes, alpha=alpha, logit_transform=False
        )
        raw_covers += int(lo_r <= true_auc <= hi_r)
        raw_widths.append(hi_r - lo_r)
        raw_below_0 += int(lo_r < 0)
        raw_above_1 += int(hi_r > 1)

        completed += 1

    return {
        "true_auc": true_auc,
        "N": N,
        "n_sims_completed": completed,
        "logit_coverage": logit_covers / completed if completed > 0 else float("nan"),
        "raw_coverage": raw_covers / completed if completed > 0 else float("nan"),
        "logit_mean_width": float(np.mean(logit_widths)) if logit_widths else float("nan"),
        "raw_mean_width": float(np.mean(raw_widths)) if raw_widths else float("nan"),
        "raw_below_0": raw_below_0,
        "raw_above_1": raw_above_1,
        "boundary_folds": boundary_folds,
        "total_folds": total_folds,
    }


# ── Coverage simulation tests ────────────────────────────────────────


def test_coverage_moderate_auc_n500():
    """Moderate AUC (~0.76), N=500. Logit coverage should be >= 0.90."""
    res = _run_coverage_simulation(N=500, delta=1.0, seed=42)
    assert res["logit_coverage"] >= COVERAGE_MIN, (
        f"logit coverage {res['logit_coverage']:.3f} < {COVERAGE_MIN} "
        f"(true AUC={res['true_auc']:.4f}, n_sims={res['n_sims_completed']})"
    )


def test_coverage_high_auc_n500():
    """High AUC (~0.92), N=500. Logit coverage should be >= 0.90."""
    res = _run_coverage_simulation(N=500, delta=2.0, seed=456)
    assert res["logit_coverage"] >= COVERAGE_MIN, (
        f"logit coverage {res['logit_coverage']:.3f} < {COVERAGE_MIN} "
        f"(true AUC={res['true_auc']:.4f}, n_sims={res['n_sims_completed']})"
    )


def test_coverage_moderate_auc_n200():
    """Moderate AUC (~0.76), N=200. Logit coverage should be >= 0.90."""
    res = _run_coverage_simulation(N=200, delta=1.0, seed=123)
    assert res["logit_coverage"] >= COVERAGE_MIN, (
        f"logit coverage {res['logit_coverage']:.3f} < {COVERAGE_MIN} "
        f"(true AUC={res['true_auc']:.4f}, n_sims={res['n_sims_completed']})"
    )


def test_coverage_very_high_auc_n200():
    """
    Very high AUC (~0.98), N=200. Stress test: many folds hit AUC=1.0.
    Logit coverage should be >= 0.90.
    """
    res = _run_coverage_simulation(N=200, delta=3.0, seed=888)
    assert res["logit_coverage"] >= COVERAGE_MIN, (
        f"logit coverage {res['logit_coverage']:.3f} < {COVERAGE_MIN} "
        f"(true AUC={res['true_auc']:.4f}, n_sims={res['n_sims_completed']})"
    )


# ── Logit bounds validity ────────────────────────────────────────────


def test_logit_bounds_always_valid_random():
    """Logit-transformed CI must always lie in (0, 1) across random inputs."""
    rng = np.random.default_rng(777)
    for i in range(100):
        k = rng.integers(5, 30)
        metrics = rng.beta(0.5, 0.5, size=k)
        train_sizes = rng.integers(50, 500, size=k).astype(float)
        test_sizes = rng.integers(10, 100, size=k).astype(float)

        _, lo, hi, _ = nadeau_bengio_cv_ci(
            metrics, train_sizes, test_sizes, logit_transform=True
        )
        assert 0 < lo < hi < 1, (
            f"Iteration {i}: logit bounds out of (0,1): [{lo:.6f}, {hi:.6f}]"
        )
