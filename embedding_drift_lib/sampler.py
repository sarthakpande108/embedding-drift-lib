from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


class StratifiedReservoirSampler:
    """
    Maintain a bounded, approximately uniform sample within each semantic cluster.

    Each cluster has its own reservoir counter. This is important: using a
    global observation count would make frequently observed clusters suppress
    replacement in less frequent clusters and bias their samples toward older
    observations. When the reservoir size is not divisible by the cluster
    count, the remaining slots are distributed across the first cluster IDs.
    """

    def __init__(
        self,
        reservoir_size: int = 2000,
        n_clusters: int = 20,
        random_state: int | None = None,
    ):
        if isinstance(reservoir_size, bool) or not isinstance(reservoir_size, int) or reservoir_size <= 0:
            raise ValueError("reservoir_size must be a positive integer")
        if isinstance(n_clusters, bool) or not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, int)
        ):
            raise ValueError("random_state must be an integer or None")

        self.reservoir_size = reservoir_size
        self.n_clusters = n_clusters
        self.per_cluster_quota = reservoir_size // n_clusters
        self.buckets: dict[int, list] = defaultdict(list)
        self.cluster_model = None
        self._total_seen = 0
        self._cluster_seen: dict[int, int] = {}
        self._quotas: dict[int, int] = {}
        self._embedding_dim: int = 0
        self._rng = random.Random(random_state)

    def fit_clusters(self, vectors: np.ndarray) -> None:
        from sklearn.cluster import MiniBatchKMeans

        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
            raise ValueError("vectors must be a non-empty 2D array")
        if not np.isfinite(vectors).all():
            raise ValueError("vectors must contain only finite values")

        # At least one slot per fitted cluster; otherwise some clusters would
        # be assigned a zero-capacity reservoir and could never be represented.
        actual_clusters = min(self.n_clusters, len(vectors), self.reservoir_size)
        self.cluster_model = MiniBatchKMeans(
            n_clusters=actual_clusters,
            random_state=42,
            batch_size=1000,
            n_init="auto",
        )
        self.cluster_model.fit(vectors)

        base_quota, remainder = divmod(self.reservoir_size, actual_clusters)
        self._quotas = {
            cluster_id: base_quota + (1 if cluster_id < remainder else 0)
            for cluster_id in range(actual_clusters)
        }
        self.per_cluster_quota = base_quota
        self.buckets.clear()
        self._cluster_seen = {cluster_id: 0 for cluster_id in range(actual_clusters)}
        self._total_seen = 0
        self._embedding_dim = vectors.shape[1]

    def add(self, vector: np.ndarray, metadata: dict | None = None) -> None:
        if self.cluster_model is None:
            return

        vector = np.asarray(vector, dtype=np.float64)
        if vector.ndim != 1:
            raise ValueError("vector must be a 1D array")
        if vector.shape[0] != self._embedding_dim:
            raise ValueError(
                f"vector dimension mismatch: expected {self._embedding_dim}, got {vector.shape[0]}"
            )
        if not np.isfinite(vector).all():
            raise ValueError("vector must contain only finite values")

        cluster_id = int(self.cluster_model.predict(vector.reshape(1, -1))[0])
        if cluster_id not in self._quotas:
            raise ValueError(f"cluster model returned unknown cluster id: {cluster_id}")

        self._total_seen += 1
        seen_in_cluster = self._cluster_seen[cluster_id] + 1
        self._cluster_seen[cluster_id] = seen_in_cluster

        quota = self._quotas[cluster_id]
        bucket = self.buckets[cluster_id]
        if len(bucket) < quota:
            bucket.append((vector, metadata))
            return

        # Algorithm R, applied independently to this cluster's stream.
        replacement_index = self._rng.randrange(seen_in_cluster)
        if replacement_index < quota:
            bucket[replacement_index] = (vector, metadata)

    def get_sample(self) -> np.ndarray:
        vecs = [vector for bucket in self.buckets.values() for vector, _ in bucket]
        if not vecs:
            return (
                np.empty((0, self._embedding_dim))
                if self._embedding_dim
                else np.empty((0,))
            )
        return np.asarray(vecs)

    @property
    def size(self) -> int:
        return sum(len(bucket) for bucket in self.buckets.values())


class AdaptiveSamplingScheduler:
    """
    Increases sampling rate when early drift signals appear.
    Backs off gradually when readings stabilise.
    """

    def __init__(
        self,
        baseline_rate: float = 0.05,
        elevated_rate: float = 0.30,
        mmd_warning_threshold: float = 0.04,
    ):
        assert 0.0 <= baseline_rate <= 1.0
        assert 0.0 <= elevated_rate <= 1.0
        self.baseline_rate = baseline_rate
        self.elevated_rate = elevated_rate
        self.threshold = mmd_warning_threshold
        self.current_rate = baseline_rate
        self._recent: list[float] = []

    def update(self, latest_mmd: float) -> None:
        self._recent.append(latest_mmd)
        if len(self._recent) > 10:
            self._recent.pop(0)
        avg = sum(self._recent) / len(self._recent)
        if avg > self.threshold:
            self.current_rate = self.elevated_rate
        else:
            self.current_rate = max(self.baseline_rate, self.current_rate * 0.9)

    def should_sample(self) -> bool:
        return random.random() < self.current_rate
