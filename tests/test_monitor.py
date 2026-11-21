import asyncio

import numpy as np
import pytest

from embedding_drift_lib.config import DriftConfig
from embedding_drift_lib.monitor import DriftMonitor


class FakeReport:
    drift_type = "none"
    mmd_score = 0.01
    mmd_p_value = 1.0
    coverage_gap_rate = 0.0


class RecordingAnalyser:
    def __init__(self, error=None):
        self.calls = []
        self.error = error
        self.report = FakeReport()

    def analyse(self, reference, current):
        if self.error:
            raise self.error
        self.calls.append((reference.copy(), current.copy()))
        return self.report


class PendingTask:
    def done(self):
        return False


def make_monitor(tmp_path, *, batch_size=4, max_buffer_size=8):
    config = DriftConfig(
        index_name="monitor-test",
        sample_rate=1.0,
        analysis_batch_size=batch_size,
        max_query_buffer_size=max_buffer_size,
        snapshot_dir=str(tmp_path),
    )
    monitor = DriftMonitor(config)
    monitor.scheduler.should_sample = lambda: True
    return monitor


def test_config_requires_a_positive_batch_and_sufficient_buffer():
    with pytest.raises(ValueError, match="analysis_batch_size"):
        DriftConfig(analysis_batch_size=0)
    with pytest.raises(ValueError, match="max_query_buffer_size"):
        DriftConfig(max_query_buffer_size=0)
    with pytest.raises(ValueError, match="greater than or equal"):
        DriftConfig(analysis_batch_size=10, max_query_buffer_size=9)


@pytest.mark.asyncio
async def test_wrap_bounds_recent_query_buffer_and_counts_evictions(tmp_path):
    monitor = make_monitor(tmp_path, batch_size=2, max_buffer_size=3)
    monitor._snapshot_available = False
    first_batch = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    batches = [
        first_batch,
        np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
    ]

    async def embed():
        return batches.pop(0)

    wrapped = monitor.wrap(embed)
    expected_response = await wrapped()
    assert expected_response is first_batch
    # The monitor must own its buffered copy, not a mutable view of the response.
    first_batch[:] = -1
    await wrapped()

    buffered = np.asarray(list(monitor._query_buffer))
    np.testing.assert_array_equal(
        buffered, np.array([[1, 1], [2, 2], [3, 3]], dtype=np.float32)
    )
    assert monitor.current_sample_size == 3
    assert monitor.dropped_query_vector_count == 1


@pytest.mark.asyncio
async def test_background_analysis_triggers_are_coalesced(tmp_path, monkeypatch):
    monitor = make_monitor(tmp_path, batch_size=2, max_buffer_size=8)
    monitor._snapshot_available = True
    scheduled = []

    def fake_create_task(coro):
        coro.close()
        scheduled.append(True)
        return PendingTask()

    monkeypatch.setattr("embedding_drift_lib.monitor.create_task", fake_create_task)

    async def embed():
        return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    wrapped = monitor.wrap(embed)
    await wrapped()
    await wrapped()

    assert len(scheduled) == 1
    assert monitor.current_sample_size == 4


@pytest.mark.asyncio
async def test_missing_snapshot_disables_repeated_background_retries(tmp_path, monkeypatch):
    monitor = make_monitor(tmp_path, batch_size=2, max_buffer_size=8)
    monitor.snapshot_manager.load_latest = lambda: None

    async def embed():
        return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    await monitor.wrap(embed)()
    await asyncio.sleep(0)
    assert monitor._snapshot_available is False

    scheduled = []

    def fake_create_task(coro):
        coro.close()
        scheduled.append(True)
        return PendingTask()

    monkeypatch.setattr("embedding_drift_lib.monitor.create_task", fake_create_task)
    await monitor.wrap(embed)()

    assert scheduled == []
    assert monitor.current_sample_size == 4


@pytest.mark.asyncio
async def test_force_analysis_processes_a_partial_batch(tmp_path):
    monitor = make_monitor(tmp_path, batch_size=4, max_buffer_size=8)
    analyser = RecordingAnalyser()
    monitor.analyser = analyser
    reference = np.full((5, 2), 10.0, dtype=np.float32)
    monitor.snapshot_manager.load_latest = lambda: reference

    async def embed():
        return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    await monitor.wrap(embed)()
    report = await monitor.force_analysis()

    assert report is analyser.report
    assert monitor.current_sample_size == 0
    assert len(analyser.calls) == 1
    np.testing.assert_array_equal(analyser.calls[0][0], reference)
    np.testing.assert_array_equal(
        analyser.calls[0][1],
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )


@pytest.mark.asyncio
async def test_failed_analysis_keeps_batch_available_for_retry(tmp_path):
    monitor = make_monitor(tmp_path, batch_size=4, max_buffer_size=8)
    monitor.analyser = RecordingAnalyser(error=RuntimeError("temporary failure"))
    monitor.snapshot_manager.load_latest = lambda: np.ones((5, 2), dtype=np.float32)

    async def embed():
        return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    await monitor.wrap(embed)()
    report = await monitor.force_analysis()

    assert report is None
    assert monitor.current_sample_size == 2
    assert monitor.dropped_query_vector_count == 0
