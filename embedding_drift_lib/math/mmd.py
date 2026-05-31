"""
Maximum Mean Discrepancy (MMD) — primary drift metric.

MMD measures the distance between two distributions without assuming
they follow any particular shape (unlike KL divergence).

MMD² = 0   → distributions are identical
MMD² > 0   → distributions differ (higher = more different)
"""

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances


def compute_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: float | None = None,
) -> float:
    """
    Compute MMD² between reference sample X and current sample Y.

    Args:
        X: Reference vectors, shape (n_ref, n_dims)
        Y: Current vectors,   shape (n_cur, n_dims)
        gamma: RBF kernel bandwidth. None = median heuristic (recommended).

    Returns:
        MMD² score ≥ 0. Values > 0.08 warrant investigation.
    """
    if gamma is None:
        gamma = _median_heuristic_gamma(X, Y)

    XX = rbf_kernel(X, X, gamma=gamma)
    YY = rbf_kernel(Y, Y, gamma=gamma)
    XY = rbf_kernel(X, Y, gamma=gamma)

    mmd2 = XX.mean() + YY.mean() - 2.0 * XY.mean()
    return float(max(mmd2, 0.0))  # clamp: numerical noise can give tiny negatives


def is_drift_significant(
    X: np.ndarray,
    Y: np.ndarray,
    n_permutations: int = 200,
    alpha: float = 0.05,
) -> dict:
    """
    Bootstrap permutation test to confirm detected drift isn't sampling noise.

    Null hypothesis: X and Y are drawn from the same distribution.
    """
    observed_mmd = compute_mmd(X, Y)
    combined = np.vstack([X, Y])
    n_x = len(X)

    rng = np.random.default_rng(seed=42)
    null_mmds = []

    for _ in range(n_permutations):
        perm = rng.permutation(len(combined))
        X_perm = combined[perm[:n_x]]
        Y_perm = combined[perm[n_x:]]
        null_mmds.append(compute_mmd(X_perm, Y_perm))

    null_array = np.array(null_mmds)
    p_value = float(np.mean(null_array >= observed_mmd))

    return {
        "observed_mmd":   round(observed_mmd, 6),
        "p_value":        round(p_value, 4),
        "is_significant": p_value < alpha,
        "null_mmd_mean":  round(float(null_array.mean()), 6),
        "null_mmd_p95":   round(float(np.percentile(null_array, 95)), 6),
    }


def _median_heuristic_gamma(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Set gamma = 1 / (2 * median pairwise distance²).
    Optimized to prevent (N, N, D) memory allocation explosions.
    """
    sample_size = min(500, len(X), len(Y))
    rng = np.random.default_rng(seed=0)

    x_sample = X[rng.choice(len(X), sample_size, replace=False)]
    y_sample = Y[rng.choice(len(Y), sample_size, replace=False)]
    all_vecs = np.vstack([x_sample, y_sample])

    # Memory Safe: Computes (N, N) directly in C without allocating an intermediate (N, N, D) array
    sq_dists = euclidean_distances(all_vecs, all_vecs, squared=True)

    # Take upper triangle (avoid zero diagonal)
    upper = sq_dists[np.triu_indices_from(sq_dists, k=1)]
    median_sq = float(np.median(upper[upper > 0]))

    return 1.0 / (2.0 * median_sq) if median_sq > 0 else 1.0