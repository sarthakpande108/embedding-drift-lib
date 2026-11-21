import numpy as np
import pytest

from embedding_drift_lib.math import mmd


def test_compute_mmd_is_symmetric_and_zero_for_identical_samples():
    rng = np.random.default_rng(4)
    samples = rng.normal(size=(40, 8))

    assert mmd.compute_mmd(samples, samples) == pytest.approx(0.0, abs=1e-12)
    assert mmd.compute_mmd(samples[:20], samples[20:]) == pytest.approx(
        mmd.compute_mmd(samples[20:], samples[:20]), abs=1e-12
    )


def test_permutation_test_detects_a_clear_distribution_shift():
    rng = np.random.default_rng(12)
    reference = rng.normal(0, 1, size=(60, 6))
    current = rng.normal(1.5, 1, size=(60, 6))

    result = mmd.is_drift_significant(reference, current, n_permutations=199)

    assert result["is_significant"] is True
    assert result["p_value"] <= 0.05


def test_permutation_test_holds_one_bandwidth_fixed_for_all_statistics(monkeypatch):
    rng = np.random.default_rng(8)
    reference = rng.normal(size=(24, 5))
    current = rng.normal(0.8, size=(24, 5))
    observed_gammas = []
    original = mmd._compute_mmd_validated

    def recording_compute(X, Y, gamma):
        observed_gammas.append(gamma)
        return original(X, Y, gamma)

    monkeypatch.setattr(mmd, "_compute_mmd_validated", recording_compute)
    mmd.is_drift_significant(reference, current, n_permutations=17)

    assert len(observed_gammas) == 18
    assert len(set(observed_gammas)) == 1


def test_permutation_p_value_uses_plus_one_correction_and_is_reproducible():
    rng = np.random.default_rng(15)
    reference = rng.normal(size=(40, 4))
    current = rng.normal(2.5, size=(40, 4))

    first = mmd.is_drift_significant(
        reference, current, n_permutations=19, random_state=123
    )
    second = mmd.is_drift_significant(
        reference, current, n_permutations=19, random_state=123
    )

    assert first == second
    assert first["p_value"] >= 1 / 20
    assert first["p_value"] <= 1.0


def test_constant_vectors_have_a_finite_bandwidth_and_score():
    samples = np.ones((8, 3))

    assert mmd.compute_mmd(samples, samples) == 0.0
    result = mmd.is_drift_significant(samples, samples, n_permutations=9)
    assert result["p_value"] == 1.0
    assert result["is_significant"] is False


@pytest.mark.parametrize(
    "X,Y,message",
    [
        (np.array([1.0, 2.0]), np.ones((2, 2)), "2D arrays"),
        (np.empty((0, 2)), np.ones((2, 2)), "at least one sample"),
        (np.ones((2, 0)), np.ones((2, 0)), "at least one feature"),
        (np.ones((2, 2)), np.ones((2, 3)), "dimension mismatch"),
        (np.array([[np.nan]]), np.ones((1, 1)), "finite values"),
        (np.array([[np.inf]]), np.ones((1, 1)), "finite values"),
    ],
)
def test_compute_mmd_rejects_invalid_samples(X, Y, message):
    with pytest.raises(ValueError, match=message):
        mmd.compute_mmd(X, Y)


@pytest.mark.parametrize("gamma", [0, -1, np.inf, np.nan, True])
def test_compute_mmd_rejects_invalid_gamma(gamma):
    samples = np.ones((2, 2))
    with pytest.raises(ValueError, match="gamma"):
        mmd.compute_mmd(samples, samples, gamma=gamma)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"n_permutations": 0}, "positive integer"),
        ({"n_permutations": 1.5}, "positive integer"),
        ({"alpha": 0}, "between 0 and 1"),
        ({"alpha": 1}, "between 0 and 1"),
        ({"random_state": 1.5}, "integer or None"),
    ],
)
def test_significance_rejects_invalid_parameters(kwargs, message):
    samples = np.ones((3, 2))
    with pytest.raises(ValueError, match=message):
        mmd.is_drift_significant(samples, samples, **kwargs)
