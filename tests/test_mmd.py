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



def test_kernel_block_mean_chunks_rows_without_changing_the_mean():
    kernel = np.arange(600 * 700, dtype=np.float64).reshape(600, 700)
    rows = np.arange(0, 600, 2)
    columns = np.arange(1, 700, 3)

    actual = mmd._mean_kernel_block(kernel, rows, columns)
    expected = kernel[np.ix_(rows, columns)].mean()

    assert actual == pytest.approx(expected, abs=1e-10)


def test_permutation_test_detects_a_clear_distribution_shift():
    rng = np.random.default_rng(12)
    reference = rng.normal(0, 1, size=(60, 6))
    current = rng.normal(1.5, 1, size=(60, 6))

    result = mmd.is_drift_significant(reference, current, n_permutations=199)

    assert result["is_significant"] is True
    assert result["p_value"] <= 0.05


def test_permutation_test_precomputes_one_pooled_kernel_and_matches_reference(monkeypatch):
    rng = np.random.default_rng(8)
    reference = rng.normal(size=(24, 5))
    current = rng.normal(0.8, size=(24, 5))
    n_permutations = 17
    random_state = 123

    # Build the reference result using the direct, fixed-bandwidth calculation.
    combined = np.vstack((reference, current))
    gamma = mmd._median_heuristic_gamma(combined)
    expected_observed = mmd._compute_mmd_validated(reference, current, gamma)
    reference_rng = np.random.default_rng(random_state)
    expected_null = np.empty(n_permutations, dtype=np.float64)
    for index in range(n_permutations):
        permuted = combined[reference_rng.permutation(len(combined))]
        expected_null[index] = mmd._compute_mmd_validated(
            permuted[:len(reference)], permuted[len(reference):], gamma
        )
    expected_p = (
        np.count_nonzero(expected_null >= expected_observed) + 1
    ) / (n_permutations + 1)

    kernel_calls = []
    original_kernel = mmd.rbf_kernel

    def recording_kernel(X, Y, gamma):
        kernel_calls.append((X.shape, Y.shape, gamma))
        return original_kernel(X, Y, gamma=gamma)

    monkeypatch.setattr(mmd, "rbf_kernel", recording_kernel)
    result = mmd.is_drift_significant(
        reference,
        current,
        n_permutations=n_permutations,
        random_state=random_state,
    )

    assert len(kernel_calls) == 1
    assert kernel_calls[0][0] == (48, 5)
    assert kernel_calls[0][1] == (48, 5)
    assert kernel_calls[0][2] == pytest.approx(gamma)
    assert result["observed_mmd"] == round(expected_observed, 6)
    assert result["p_value"] == round(expected_p, 4)
    assert result["null_mmd_mean"] == round(float(expected_null.mean()), 6)
    assert result["null_mmd_p95"] == round(float(np.percentile(expected_null, 95)), 6)


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
