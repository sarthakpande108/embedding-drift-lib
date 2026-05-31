import numpy as np
from scipy.stats import wasserstein_distance


def compute_wasserstein_per_dim(
    X_projected: np.ndarray,
    Y_projected: np.ndarray,
    top_n: int = 5,
) -> dict:
    """
    Compute 1D Wasserstein distance per PCA component to diagnose drift root-causes.

    Args:
        X_projected: Reference vectors after PCA, shape (n_ref, n_components)
        Y_projected: Current vectors after PCA,   shape (n_cur, n_components)
        top_n:       How many top-drifted dimensions to isolate

    Returns:
        Dict filled with structural drift telemetry and a concentration ratio.
    """
    n_dims = X_projected.shape[1]
    per_dim_scores = []

    # Calculate Earth Mover's Distance along each individual semantic axis
    for dim in range(n_dims):
        w = wasserstein_distance(
            X_projected[:, dim],
            Y_projected[:, dim],
        )
        per_dim_scores.append((dim, round(float(w), 6)))

    scores_only = [s for _, s in per_dim_scores]
    mean_w = float(np.mean(scores_only))
    max_w  = float(np.max(scores_only))

    # Sort to isolate the absolute worst-performing semantic dimensions
    top_dims = sorted(per_dim_scores, key=lambda x: -x[1])[:top_n]

    return {
        "mean":             round(mean_w, 6),
        "max":              round(max_w, 6),
        "per_dim":          per_dim_scores,
        "top_drifted_dims": top_dims,
        # Concentration > 3 → drift is localized (likely silent model update)
        # Concentration ≈ 1 → drift is spread thin (likely real-world data staleness)
        "concentration":    round(max_w / mean_w, 2) if mean_w > 0 else 0.0,
    }