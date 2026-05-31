import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_coverage_gap(
    index_vectors: np.ndarray,
    query_vectors: np.ndarray,
    k: int = 3,
    threshold: float = 0.65,
) -> dict:
    """
    For each query, find its k nearest neighbours in the index.
    If best similarity < threshold, the query is "uncovered."

    Args:
        index_vectors:  Sampled vectors from your corpus, shape (n_docs, dim)
        query_vectors:  Recent query vectors,             shape (n_queries, dim)
        k:              How many neighbours to check
        threshold:      Min cosine similarity to count as "covered"
                        Start at 0.65 for general models, lower for
                        specialised domains (legal, medical).

    Returns:
        Dict with telemetry tracking the coverage gap rate and worst-performing percentiles.
    """
    # Compute all pairwise cosine similarities: shape (n_queries, n_docs)
    sims = cosine_similarity(query_vectors, index_vectors)

    # For each query, get the top-k similarity scores
    top_k = np.sort(sims, axis=1)[:, -k:]
    best_sims = top_k[:, -1]  # highest similarity per query

    uncovered_mask = best_sims < threshold

    return {
        "coverage_gap_rate":     round(float(uncovered_mask.mean()), 4),
        "mean_best_similarity": round(float(best_sims.mean()), 4),
        "p10_best_similarity":  round(float(np.percentile(best_sims, 10)), 4),
        "uncovered_count":      int(uncovered_mask.sum()),
        "total_queries":        len(query_vectors),
        # Sample of indices where coverage failed — useful for debugging
        # what topics are uncovered
        "uncovered_indices":    np.where(uncovered_mask)[0][:20].tolist(),
    }