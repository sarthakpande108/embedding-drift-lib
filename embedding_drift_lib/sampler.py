from __future__ import annotations
import random
import numpy as np
from collections import defaultdict


class StratifiedReservoirSampler:
    """
    Maintains a fixed-size sample that stays representative of your index
    across all semantic clusters — not just the most common topics.
    """

    def __init__(self, reservoir_size: int = 2000, n_clusters: int = 20):
        self.reservoir_size = reservoir_size
        self.n_clusters = n_clusters
        self.per_cluster_quota = reservoir_size // n_clusters
        self.buckets: dict[int, list] = defaultdict(list)
        self.cluster_model = None
        self._total_seen = 0
        self._embedding_dim: int = 0

    def fit_clusters(self, vectors: np.ndarray) -> None:
        from sklearn.cluster import MiniBatchKMeans
        actual_clusters = min(self.n_clusters, len(vectors))
        self.cluster_model = MiniBatchKMeans(
            n_clusters=actual_clusters,
            random_state=42,
            batch_size=1000,
            n_init="auto",
        )
        self.cluster_model.fit(vectors.astype(np.float64))
        self._embedding_dim = vectors.shape[1]

    def add(self, vector: np.ndarray, metadata: dict | None = None) -> None:
        if self.cluster_model is None:
            return
        self._total_seen += 1
        cluster_id = int(
            self.cluster_model.predict(
                vector.astype(np.float64).reshape(1, -1)
            )[0]
        )
        bucket = self.buckets[cluster_id]
        if len(bucket) < self.per_cluster_quota:
            bucket.append((vector, metadata))
        else:
            # Correct reservoir sampling: uniform draw from [0, total_seen)
            j = random.randrange(self._total_seen)
            if j < self.per_cluster_quota:
                bucket[j % self.per_cluster_quota] = (vector, metadata)

    def get_sample(self) -> np.ndarray:
        vecs = [v for bucket in self.buckets.values() for v, _ in bucket]
        if not vecs:
            return np.empty((0, self._embedding_dim)) if self._embedding_dim else np.empty((0,))
        return np.array(vecs)

    @property
    def size(self) -> int:
        return sum(len(b) for b in self.buckets.values())


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