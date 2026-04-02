"""Tests for ckpy.stats.nadeau_bengio_cv_ci."""

import numpy as np
import pytest

from ckpy.stats import nadeau_bengio_cv_ci


# --- Input validation ---


def test_mismatched_lengths():
    with pytest.raises(ValueError, match="Input length mismatch"):
        nadeau_bengio_cv_ci([0.8, 0.7], [400, 400, 400], [100, 100, 100])


def test_too_few_folds():
    with pytest.raises(ValueError, match="At least 2"):
        nadeau_bengio_cv_ci([0.8], [400], [100])


def test_non_positive_train_size():
    with pytest.raises(ValueError, match="positive"):
        nadeau_bengio_cv_ci([0.8, 0.7], [0, 400], [100, 100])


def test_non_positive_test_size():
    with pytest.raises(ValueError, match="positive"):
        nadeau_bengio_cv_ci([0.8, 0.7], [400, 400], [100, -1])


# --- Known-value regression ---


FOLD_AUCS = [0.81, 0.78, 0.83, 0.79, 0.80, 0.82, 0.77, 0.81, 0.80, 0.79]
TRAIN_SIZES = [400] * 10
TEST_SIZES = [100] * 10


def test_known_value_logit():
    mean, lo, hi, se = nadeau_bengio_cv_ci(FOLD_AUCS, TRAIN_SIZES, TEST_SIZES)
    assert mean == pytest.approx(0.8)
    assert lo == pytest.approx(0.7744447705711514, abs=1e-12)
    assert hi == pytest.approx(0.8233203874820713, abs=1e-12)
    assert se == pytest.approx(0.010801234497346424, abs=1e-12)


def test_known_value_raw():
    mean, lo, hi, se = nadeau_bengio_cv_ci(
        FOLD_AUCS, TRAIN_SIZES, TEST_SIZES, logit_transform=False
    )
    assert mean == pytest.approx(0.8)
    assert lo == pytest.approx(0.7755659100147647, abs=1e-12)
    assert hi == pytest.approx(0.8244340899852354, abs=1e-12)
    assert se == pytest.approx(0.010801234497346424, abs=1e-12)


# --- Property / invariant tests ---


def test_ci_contains_mean():
    mean, lo, hi, _ = nadeau_bengio_cv_ci(FOLD_AUCS, TRAIN_SIZES, TEST_SIZES)
    assert lo <= mean <= hi


def test_logit_bounds_in_unit_interval():
    mean, lo, hi, _ = nadeau_bengio_cv_ci(FOLD_AUCS, TRAIN_SIZES, TEST_SIZES)
    assert 0.0 < lo < 1.0
    assert 0.0 < hi < 1.0


def test_wider_ci_with_fewer_folds():
    """Fewer folds → higher t critical value and larger 1/k → wider CI."""
    _, lo10, hi10, _ = nadeau_bengio_cv_ci(
        FOLD_AUCS, TRAIN_SIZES, TEST_SIZES, logit_transform=False
    )
    _, lo5, hi5, _ = nadeau_bengio_cv_ci(
        FOLD_AUCS[:5], TRAIN_SIZES[:5], TEST_SIZES[:5], logit_transform=False
    )
    assert (hi5 - lo5) >= (hi10 - lo10)


def test_identical_metrics_zero_se():
    """When all fold metrics are identical, variance and SE are zero."""
    metrics = [0.85] * 10
    mean, lo, hi, se = nadeau_bengio_cv_ci(
        metrics, TRAIN_SIZES, TEST_SIZES, logit_transform=False
    )
    assert mean == pytest.approx(0.85)
    assert se == pytest.approx(0.0, abs=1e-15)
    assert lo == pytest.approx(0.85, abs=1e-15)
    assert hi == pytest.approx(0.85, abs=1e-15)


def test_two_folds_minimum():
    """Exactly 2 folds should work (df=1, very wide CI)."""
    mean, lo, hi, se = nadeau_bengio_cv_ci(
        [0.80, 0.85], [400, 400], [100, 100], logit_transform=False
    )
    assert lo < mean < hi
    assert se > 0
