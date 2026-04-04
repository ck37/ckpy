"""Statistical utilities for cross-validation."""

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.special import logit, expit


def nadeau_bengio_cv_ci(
    fold_metrics: ArrayLike,
    train_sizes: ArrayLike,
    test_sizes: ArrayLike,
    alpha: float = 0.05,
    logit_transform: bool = True,
) -> tuple[float, float, float, float]:
    """
    Nadeau-Bengio corrected confidence interval for a single performance
    metric estimated via repeated k-fold cross-validation.

    Accounts for the positive correlation among fold-level estimates
    induced by overlapping training sets, which causes naive intervals
    to undercover. When fold sizes are not exactly equal (e.g., due to
    stratification or indivisible N), per-fold test/train ratios are
    averaged rather than computing the ratio of mean sizes, yielding a
    slightly more conservative correction via Jensen's inequality.

    The correction is metric-agnostic: it operates on the fold-level
    values regardless of what performance measure they represent (AUC,
    accuracy, Brier score, log-loss, F1, etc.). The t-based interval
    assumes approximate normality of the fold-level estimates, which
    holds well for smooth metrics at moderate sample sizes but may be
    strained for discrete or heavily bounded metrics in very small folds.

    When ``logit_transform=True`` (the default), the corrected variance
    is first computed on the raw scale, then mapped to the logit scale
    via the delta method and back-transformed with the expit function.
    This guarantees bounds in (0, 1) and produces appropriately
    asymmetric intervals when the point estimate is near the boundary.
    The delta method approach is more robust than applying the logit to
    each fold-level metric individually, which can fail when many folds
    produce metrics at exactly 0 or 1 (e.g., perfect AUC in small
    test folds). Set ``logit_transform=False`` to recover the original
    untransformed interval, which may fall outside [0, 1].

    Note that because this method treats each fold's metric as an opaque
    scalar, it does not exploit the internal structure of the metric
    itself. For the AUC specifically, observation-level approaches can
    yield tighter intervals:

    - LeDell, Petersen, and van der Laan (2015) derive an influence-
      curve-based variance estimator that operates on per-observation
      placement values rather than fold-level summaries, gaining
      substantial efficiency. Their R package ``cvAUC`` implements
      this via ``ci.cvAUC()``. Coverage is near nominal for N >= 5,000
      but can dip to 92-93% around N = 1,000 due to the nonlinearity
      of the AUC functional.

    - Benkeser, Petersen, and van der Laan (2020) address that small-
      sample undercoverage by estimating nuisance parameters (class-
      conditional score distributions) on the training fold and using
      the validation fold only for bias correction, via cross-validated
      TMLE, one-step estimation, or estimating equations. Their R
      package ``nlpred`` implements this via ``cv_auc()``. Nested
      cross-validation within training folds is recommended when using
      aggressive learners (e.g., random forests, gradient boosting).

    Both observation-level methods require access to the raw predicted
    probabilities and labels, not just fold-level metric summaries. The
    Nadeau-Bengio correction is the appropriate choice when only fold-
    level summaries are available, when a metric-agnostic interval is
    desired, or when conservative coverage is preferred over efficiency
    in moderate samples.

    Parameters
    ----------
    fold_metrics : ArrayLike of shape (k * r,)
        Per-fold performance metric values, one per fold across all
        repeats. For 5-fold CV with 5 repeats, this has length 25.
    train_sizes : ArrayLike of shape (k * r,)
        Number of observations in the training set for each fold.
        Must align element-wise with fold_metrics.
    test_sizes : ArrayLike of shape (k * r,)
        Number of observations in the test set for each fold.
        Must align element-wise with fold_metrics.
    alpha : float, optional
        Significance level for the confidence interval. Default is 0.05,
        yielding a 95% interval.
    logit_transform : bool, optional
        If True (default), map the corrected SE to the logit scale via
        the delta method and back-transform to ensure bounds lie in
        (0, 1). Appropriate for metrics bounded in [0, 1] such as AUC,
        accuracy, and Brier score. Set to False for unbounded metrics
        (e.g., log-loss, mean squared error) or to recover the original
        untransformed Nadeau-Bengio interval.

    Returns
    -------
    mean_metric : float
        Mean of the fold-level metric values (on the original scale).
    ci_lower : float
        Lower bound of the corrected confidence interval.
    ci_upper : float
        Upper bound of the corrected confidence interval.
    se : float
        Corrected standard error on the raw (untransformed) scale.
        This is the square root of the Nadeau-Bengio corrected variance
        and is always reported on the original metric scale regardless
        of the logit_transform setting, to facilitate downstream use
        in power calculations and non-inferiority testing.

    Raises
    ------
    ValueError
        If the input arrays have mismatched lengths, if fewer than two
        fold-level estimates are provided, or if any train or test size
        is not positive.

    Examples
    --------
    Compute a 95% CI for cross-validated AUC from 5-fold, 2-repeat CV
    on a dataset of N = 500 (400 training, 100 test per fold):

    >>> fold_aucs = [
    ...     0.81, 0.78, 0.83, 0.79, 0.80,  # repeat 1
    ...     0.82, 0.77, 0.81, 0.80, 0.79,  # repeat 2
    ... ]
    >>> train_sizes = [400] * 10
    >>> test_sizes = [100] * 10
    >>> mean_auc, ci_lo, ci_hi, se = nadeau_bengio_cv_ci(
    ...     fold_aucs, train_sizes, test_sizes
    ... )
    >>> print(f"AUC = {mean_auc:.3f} (95% CI: {ci_lo:.3f}, {ci_hi:.3f})")
    AUC = 0.800 (95% CI: 0.774, 0.823)

    For an unbounded metric like log-loss, disable the logit transform:

    >>> fold_logloss = [0.45, 0.48, 0.44, 0.47, 0.46] * 2
    >>> mean_ll, ll_lo, ll_hi, se_ll = nadeau_bengio_cv_ci(
    ...     fold_logloss, train_sizes, test_sizes,
    ...     logit_transform=False,
    ... )

    References
    ----------
    Nadeau, C. and Bengio, Y. (2003). Inference for the generalization
    error. Machine Learning, 52(3), 239-281.

    Bouckaert, R.R. and Frank, E. (2004). Evaluating the replicability
    of significance tests for comparing learning algorithms. In Advances
    in Knowledge Discovery and Data Mining (PAKDD), Lecture Notes in
    Computer Science, 3056, 3-12.

    Dudoit, S. and van der Laan, M.J. (2005). Asymptotics of cross-
    validated risk estimation in estimator selection and performance
    assessment. Statistical Methodology, 2(2), 131-154.

    LeDell, E., Petersen, M., and van der Laan, M. (2015).
    Computationally efficient confidence intervals for cross-validated
    area under the ROC curve estimates. Electronic Journal of
    Statistics, 9(1), 1583-1607.

    Benkeser, D., Petersen, M., and van der Laan, M.J. (2020). Improved
    small-sample estimation of nonlinear cross-validated prediction
    metrics. Journal of the American Statistical Association, 115(532),
    1917-1932.
    """
    # Convert inputs to numpy arrays for vectorized operations
    fold_metrics = np.asarray(fold_metrics, dtype=float)
    train_sizes = np.asarray(train_sizes, dtype=float)
    test_sizes = np.asarray(test_sizes, dtype=float)

    # --- Input validation ---
    if not (len(fold_metrics) == len(train_sizes) == len(test_sizes)):
        raise ValueError(
            f"Input length mismatch: fold_metrics has {len(fold_metrics)}, "
            f"train_sizes has {len(train_sizes)}, "
            f"test_sizes has {len(test_sizes)}. All must be equal."
        )
    if len(fold_metrics) < 2:
        raise ValueError(
            f"At least 2 fold-level estimates are required to compute a "
            f"variance; got {len(fold_metrics)}."
        )
    if np.any(train_sizes <= 0) or np.any(test_sizes <= 0):
        raise ValueError("All train_sizes and test_sizes must be positive.")

    # Total number of fold-level estimates (k folds * r repeats)
    k = len(fold_metrics)

    # Point estimate: simple average of fold-level metrics
    mean_metric = float(np.mean(fold_metrics))

    # Unbiased sample variance of fold-level metrics (dividing by k - 1)
    s2 = float(np.var(fold_metrics, ddof=1))

    # Per-fold ratio of test size to train size;
    # averaging these is slightly more conservative than
    # mean(test_sizes) / mean(train_sizes) by Jensen's inequality
    mean_ratio = float(np.mean(test_sizes / train_sizes))

    # Nadeau-Bengio corrected variance on the raw scale:
    #   1/k captures the usual averaging-over-folds reduction,
    #   mean_ratio captures the irreducible covariance due to
    #   overlapping training sets across folds
    corrected_var = (1.0 / k + mean_ratio) * s2

    # Corrected standard error on the raw scale
    se = float(np.sqrt(corrected_var))

    # Critical value from the t-distribution
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=k - 1))

    if logit_transform:
        # Bound the point estimate away from {0, 1} to keep the logit
        # finite; epsilon is half the reciprocal of the mean test fold
        # size, analogous to a continuity correction
        mean_n_test = float(np.mean(test_sizes))
        eps = 0.5 / mean_n_test
        mean_bounded = float(np.clip(mean_metric, eps, 1.0 - eps))

        # Delta method: d/dp logit(p) = 1 / (p * (1 - p)), so
        # Var(logit(p_hat)) ≈ Var(p_hat) / (p * (1 - p))^2
        deriv = 1.0 / (mean_bounded * (1.0 - mean_bounded))
        se_logit = se * deriv

        # Symmetric interval on the logit scale
        logit_mean = float(logit(mean_bounded))
        margin_logit = t_crit * se_logit

        # Back-transform to [0, 1] via expit
        ci_lower = float(expit(logit_mean - margin_logit))
        ci_upper = float(expit(logit_mean + margin_logit))
    else:
        # Untransformed symmetric interval on the raw scale
        margin = t_crit * se
        ci_lower = mean_metric - margin
        ci_upper = mean_metric + margin

    return mean_metric, ci_lower, ci_upper, se
