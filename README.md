# ckpy

Personal Python utility library.

## Installation

Install from GitHub:

```bash
pip install git+https://github.com/ck37/ckpy.git
```

For development:

```bash
git clone https://github.com/ck37/ckpy.git
cd ckpy
uv pip install -e .
```

## Functions

### Statistics

- `nadeau_bengio_cv_ci` - Nadeau-Bengio corrected confidence interval for a performance metric estimated via repeated k-fold cross-validation. Accounts for the positive correlation among fold-level estimates induced by overlapping training sets. Supports an optional logit transform (via the delta method) to guarantee bounds in (0, 1) for bounded metrics.

## Examples

### Nadeau-Bengio corrected CV confidence interval

Compute a 95% CI for cross-validated AUC from 5-fold, 2-repeat CV on a dataset of N = 500 (400 training, 100 test per fold):

```python
from ckpy.stats import nadeau_bengio_cv_ci

fold_aucs = [
    0.81, 0.78, 0.83, 0.79, 0.80,  # repeat 1
    0.82, 0.77, 0.81, 0.80, 0.79,  # repeat 2
]
train_sizes = [400] * 10
test_sizes = [100] * 10

mean_auc, ci_lo, ci_hi, se = nadeau_bengio_cv_ci(
    fold_aucs, train_sizes, test_sizes
)
print(f"AUC = {mean_auc:.3f} (95% CI: {ci_lo:.3f}, {ci_hi:.3f})")
#> AUC = 0.800 (95% CI: 0.774, 0.823)
```

For unbounded metrics like log-loss, disable the logit transform:

```python
fold_logloss = [0.45, 0.48, 0.44, 0.47, 0.46] * 2

mean_ll, ll_lo, ll_hi, se_ll = nadeau_bengio_cv_ci(
    fold_logloss, train_sizes, test_sizes,
    logit_transform=False,
)
```

## References

Nadeau, C. and Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52(3), 239-281.

Bouckaert, R.R. and Frank, E. (2004). Evaluating the replicability of significance tests for comparing learning algorithms. In *Advances in Knowledge Discovery and Data Mining* (PAKDD), Lecture Notes in Computer Science, 3056, 3-12.

LeDell, E., Petersen, M., and van der Laan, M. (2015). Computationally efficient confidence intervals for cross-validated area under the ROC curve estimates. *Electronic Journal of Statistics*, 9(1), 1583-1607.

Benkeser, D., Petersen, M., and van der Laan, M.J. (2020). Improved small-sample estimation of nonlinear cross-validated prediction metrics. *Journal of the American Statistical Association*, 115(532), 1917-1932.
