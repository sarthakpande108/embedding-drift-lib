"""
Maximum Mean Discrepancy (MMD) — primary drift metric.

MMD measures the distance between two distributions without assuming
they follow any particular shape (unlike KL divergence).

MMD² = 0   → distributions are identical
MMD² > 0   → distributions differ (higher = more different)
"""

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances


def _validate_samples(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize and validate two embedding matrices before kernel work."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays with shape (n_samples, n_features)")
    if X.shape[0] == 0 or Y.shape[0] == 0:
        raise ValueError("X and Y must each contain at least one sample")
    if X.shape[1] == 0 or Y.shape[1] == 0:
        raise ValueError("X and Y must each contain at least one feature")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"Feature dimension mismatch: X={X.shape[1]}, Y={Y.shape[1]}")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("X and Y must contain only finite values")

    return X, Y


def _validate_gamma(gamma: float | None) -> float | None:
    if gamma is None:
        return None
    if isinstance(gamma, (bool, np.bool_)):
        raise ValueError("gamma must be a finite positive number")
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be a finite positive number")
    return gamma


def _compute_mmd_validated(X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
    """Compute biased MMD² for already validated samples and a fixed kernel."""
    XX = rbf_kernel(X, X, gamma=gamma)
    YY = rbf_kernel(Y, Y, gamma=gamma)
    XY = rbf_kernel(X, Y, gamma=gamma)

    mmd2 = XX.mean() + YY.mean() - 2.0 * XY.mean()
    return float(max(mmd2, 0.0))


def compute_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: float | None = None,
) -> float:
    """
    Compute biased MMD² between reference sample X and current sample Y.

    Args:
        X: Reference vectors, shape (n_ref, n_dims).
        Y: Current vectors, shape (n_cur, n_dims).
        gamma: Positive RBF kernel bandwidth. None uses a pooled median heuristic.

    Returns:
        Non-negative MMD² score.
    """
    X, Y = _validate_samples(X, Y)
    gamma = _validate_gamma(gamma)
    if gamma is None:
        gamma = _median_heuristic_gamma(np.vstack((X, Y)))
    return _compute_mmd_validated(X, Y, gamma)


def is_drift_significant(
    X: np.ndarray,
    Y: np.ndarray,
    n_permutations: int = 200,
    alpha: float = 0.05,
    random_state: int | None = 42,
) -> dict:
    """
    Test whether the observed MMD is larger than expected under equal distributions.

    The RBF bandwidth is estimated once from the pooled, label-independent sample
    and held fixed for the observed statistic and every permutation. Re-estimating
    it after shuffling labels changes the test statistic between permutations and
    does not produce a valid permutation null distribution.

    The returned Monte Carlo p-value uses the plus-one correction, so it can never
    be zero merely because the finite set of permutations did not exceed the
    observed statistic.

    Null hypothesis: X and Y are drawn from the same distribution.
    """
    X, Y = _validate_samples(X, Y)
    if isinstance(n_permutations, (bool, np.bool_)) or not isinstance(
        n_permutations, (int, np.integer)
    ) or n_permutations < 1:
        raise ValueError("n_permutations must be a positive integer")
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError("alpha must be between 0 and 1 (exclusive)")
    try:
        alpha = float(alpha)
    except (TypeError, ValueError):
        raise ValueError("alpha must be between 0 and 1 (exclusive)") from None
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1 (exclusive)")
    if random_state is not None and (
        isinstance(random_state, (bool, np.bool_))
        or not isinstance(random_state, (int, np.integer))
    ):
        raise ValueError("random_state must be an integer or None")

    combined = np.vstack((X, Y))
    n_x = len(X)
    gamma = _median_heuristic_gamma(combined)
    observed_mmd = _compute_mmd_validated(X, Y, gamma)

    rng = np.random.default_rng(random_state)
    null_mmds = np.empty(n_permutations, dtype=np.float64)

    for i in range(n_permutations):
        permuted = combined[rng.permutation(len(combined))]
        X_perm = permuted[:n_x]
        Y_perm = permuted[n_x:]
        null_mmds[i] = _compute_mmd_validated(X_perm, Y_perm, gamma)

    exceedances = int(np.count_nonzero(null_mmds >= observed_mmd))
    p_value = (exceedances + 1) / (n_permutations + 1)

    return {
        "observed_mmd": round(observed_mmd, 6),
        "p_value": round(float(p_value), 4),
        "is_significant": bool(p_value < alpha),
        "null_mmd_mean": round(float(null_mmds.mean()), 6),
        "null_mmd_p95": round(float(np.percentile(null_mmds, 95)), 6),
    }


def _median_heuristic_gamma(pooled: np.ndarray) -> float:
    """Estimate RBF gamma from a label-independent sample of pooled vectors."""
    sample_size = min(1000, len(pooled))
    if sample_size < 2:
        return 1.0

    rng = np.random.default_rng(seed=0)
    sample = pooled[rng.choice(len(pooled), sample_size, replace=False)]
    sq_dists = euclidean_distances(sample, sample, squared=True)
    upper = sq_dists[np.triu_indices_from(sq_dists, k=1)]
    positive_distances = upper[upper > 0]

    if positive_distances.size == 0:
        return 1.0

    median_sq = float(np.median(positive_distances))
    return 1.0 / (2.0 * median_sq)
