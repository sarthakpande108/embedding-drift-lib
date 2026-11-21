import numpy as np
import pytest

from embedding_drift_lib.sampler import StratifiedReservoirSampler


class FixedClusterModel:
    def predict(self, vectors):
        return vectors[:, 0].astype(int)


class RecordingRandom:
    def __init__(self):
        self.stops = []

    def randrange(self, stop):
        self.stops.append(stop)
        return stop - 1


def test_reservoir_uses_each_clusters_own_observation_count():
    sampler = StratifiedReservoirSampler(reservoir_size=2, n_clusters=2)
    sampler.cluster_model = FixedClusterModel()
    sampler._embedding_dim = 1
    sampler._quotas = {0: 1, 1: 1}
    sampler._cluster_seen = {0: 0, 1: 0}
    sampler._rng = RecordingRandom()

    sampler.add(np.array([0.0]))
    sampler.add(np.array([0.0]))
    sampler.add(np.array([1.0]))
    sampler.add(np.array([1.0]))

    # Each cluster has seen two items. A global counter would incorrectly
    # use 2 for the first cluster and 4 for the second cluster.
    assert sampler._rng.stops == [2, 2]
    assert sampler._cluster_seen == {0: 2, 1: 2}


def test_fit_distributes_all_reservoir_slots_and_resets_previous_state(monkeypatch):
    import sklearn.cluster

    class FakeMiniBatchKMeans:
        def __init__(self, n_clusters, **kwargs):
            self.n_clusters = n_clusters

        def fit(self, vectors):
            return self

        def predict(self, vectors):
            return vectors[:, 0].astype(int)

    monkeypatch.setattr(sklearn.cluster, "MiniBatchKMeans", FakeMiniBatchKMeans)
    sampler = StratifiedReservoirSampler(
        reservoir_size=7, n_clusters=3, random_state=7
    )
    baseline = np.array([[0.0], [1.0], [2.0]])

    sampler.fit_clusters(baseline)
    assert sampler._quotas == {0: 3, 1: 2, 2: 2}

    for cluster_id in range(3):
        for offset in range(5):
            sampler.add(np.array([cluster_id + offset / 100]))

    assert sampler.size == 7
    assert [len(sampler.buckets[i]) for i in range(3)] == [3, 2, 2]

    sampler.fit_clusters(baseline)
    assert sampler.size == 0
    assert sampler._cluster_seen == {0: 0, 1: 0, 2: 0}
    assert sampler._total_seen == 0


def test_fitted_cluster_count_cannot_exceed_reservoir_capacity(monkeypatch):
    import sklearn.cluster

    fitted_counts = []

    class FakeMiniBatchKMeans:
        def __init__(self, n_clusters, **kwargs):
            fitted_counts.append(n_clusters)

        def fit(self, vectors):
            return self

    monkeypatch.setattr(sklearn.cluster, "MiniBatchKMeans", FakeMiniBatchKMeans)
    sampler = StratifiedReservoirSampler(reservoir_size=2, n_clusters=5)
    sampler.fit_clusters(np.arange(5, dtype=float).reshape(-1, 1))

    assert fitted_counts == [2]
    assert sum(sampler._quotas.values()) == 2
    assert all(quota >= 1 for quota in sampler._quotas.values())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reservoir_size": 0},
        {"reservoir_size": -1},
        {"reservoir_size": True},
        {"n_clusters": 0},
        {"n_clusters": -1},
        {"n_clusters": True},
        {"random_state": 1.5},
    ],
)
def test_constructor_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        StratifiedReservoirSampler(**kwargs)


@pytest.mark.parametrize(
    "vectors",
    [
        np.array([1.0, 2.0]),
        np.empty((0, 2)),
        np.empty((2, 0)),
        np.array([[np.nan, 1.0]]),
        np.array([[np.inf, 1.0]]),
    ],
)
def test_fit_rejects_invalid_vectors(vectors):
    sampler = StratifiedReservoirSampler()
    with pytest.raises(ValueError):
        sampler.fit_clusters(vectors)


def test_add_rejects_wrong_dimension_and_non_finite_values():
    sampler = StratifiedReservoirSampler(reservoir_size=2, n_clusters=1)
    sampler.cluster_model = FixedClusterModel()
    sampler._embedding_dim = 2
    sampler._quotas = {0: 2}
    sampler._cluster_seen = {0: 0}

    with pytest.raises(ValueError, match="dimension mismatch"):
        sampler.add(np.array([0.0]))
    with pytest.raises(ValueError, match="finite"):
        sampler.add(np.array([0.0, np.nan]))
