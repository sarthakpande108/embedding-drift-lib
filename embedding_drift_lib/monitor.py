"""
DriftMonitor — main entry point.

Wraps any async embedding function and runs drift analysis in the background.
No external services required — no Supabase, no alerts, no API keys.
"""

import functools
import logging
from asyncio import create_task
from collections import deque
from typing import Callable, Optional
import numpy as np

from .config import DriftConfig
from .analyser import DriftAnalyser, DriftReport
from .sampler import StratifiedReservoirSampler, AdaptiveSamplingScheduler
from .storage.snapshots import SnapshotManager

logger = logging.getLogger("embedding_drift_lib")


class DriftMonitor:
    """
    Wraps your embedding function to silently monitor for drift.

    Example:
        from openai import AsyncOpenAI
        from embedding_drift_lib import DriftMonitor, DriftConfig

        client = AsyncOpenAI()
        monitor = DriftMonitor(DriftConfig(index_name="my-index"))

        embed = monitor.wrap(client.embeddings.create)
        await monitor.create_snapshot(my_index_vectors, label="v1-baseline")

        # Use embed() normally — drift analysis runs automatically
        response = await embed(model="text-embedding-3-small", input=["my query"])

        # Check results anytime
        report = monitor.last_report
    """

    def __init__(self, config: DriftConfig):
        self.config = config
        self.analyser = DriftAnalyser(config)
        self.sampler = StratifiedReservoirSampler(
            reservoir_size=config.reservoir_size,
            n_clusters=config.n_clusters,
        )
        self.scheduler = AdaptiveSamplingScheduler(
            baseline_rate=config.sample_rate,
            elevated_rate=config.elevated_sample_rate,
            mmd_warning_threshold=config.mmd_mild_threshold,
        )
        self.snapshot_manager = SnapshotManager(config.snapshot_dir)
        # Keep a recent, bounded window even if no baseline exists or analysis fails.
        # deque drops the oldest observation when the configured capacity is reached.
        self._query_buffer: deque[np.ndarray] = deque(maxlen=config.max_query_buffer_size)
        self._dropped_query_vector_count = 0
        self._snapshot_available: Optional[bool] = None
        self._analysis_task = None
        self._analysis_running = False
        self._last_report: Optional[DriftReport] = None

    # ── Public API ─────────────────────────────────────────────────────────

    def wrap(self, embed_fn: Callable) -> Callable:
        """
        Wrap any async embedding function. Fully transparent — returns
        the original response unchanged.

        Supports OpenAI/Azure OpenAI client format:
            client.embeddings.create(model=..., input=...)
        """
        @functools.wraps(embed_fn)
        async def wrapped(*args, **kwargs):
            response = await embed_fn(*args, **kwargs)
            if self.scheduler.should_sample():
                for vec in self._extract_vectors(response):
                    if len(self._query_buffer) == self._query_buffer.maxlen:
                        self._dropped_query_vector_count += 1
                    sample = np.asarray(vec, dtype=np.float32).copy()
                    self._query_buffer.append(sample)
                    if self.sampler.cluster_model is not None:
                        self.sampler.add(sample)
            if len(self._query_buffer) >= self.config.analysis_batch_size:
                self._schedule_analysis()
            return response
        return wrapped

    async def create_snapshot(self, vectors: np.ndarray, label: str = "baseline") -> str:
        """
        Create a reference snapshot from your current index vectors.

        Call once at setup, and again after any deliberate re-indexing.
        Never call this automatically in response to detected drift.

        Returns:
            Snapshot ID (8-char hex)
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        snapshot_id = self.snapshot_manager.create(vectors, label)
        self.analyser.projector.fit(vectors)
        self.sampler.fit_clusters(vectors)
        self._snapshot_available = True
        if len(self._query_buffer) >= self.config.analysis_batch_size:
            self._schedule_analysis()
        logger.info(f"Snapshot created: {snapshot_id} ({len(vectors)} vectors, label='{label}')")
        return snapshot_id

    async def force_analysis(self) -> Optional[DriftReport]:
        """Manually analyze one batch, even when it is smaller than analysis_batch_size."""
        return await self._run_analysis(force=True)

    @property
    def last_report(self) -> Optional[DriftReport]:
        return self._last_report

    @property
    def current_sample_size(self) -> int:
        return len(self._query_buffer)

    @property
    def dropped_query_vector_count(self) -> int:
        """Total sampled vectors evicted because the recent-query buffer was full."""
        return self._dropped_query_vector_count

    # ── Internal ───────────────────────────────────────────────────────────

    def _schedule_analysis(self) -> None:
        """Coalesce triggers into at most one pending background analysis task."""
        if self._snapshot_available is False:
            return
        if self._analysis_task is not None and not self._analysis_task.done():
            return
        self._analysis_task = create_task(self._run_analysis())

    async def _run_analysis(self, force: bool = False) -> Optional[DriftReport]:
        # A forced run may flush a partial batch, but it must never overlap another run.
        if self._analysis_running:
            return None

        buffered = len(self._query_buffer)
        if buffered == 0:
            logger.debug("No sampled query vectors are available for analysis.")
            return None
        if buffered < self.config.analysis_batch_size and not force:
            logger.debug(
                f"Buffer too small ({buffered} vectors, "
                f"need {self.config.analysis_batch_size})."
            )
            return None
        if self._snapshot_available is False:
            return None

        batch_size = min(buffered, self.config.analysis_batch_size)
        self._analysis_running = True
        try:
            reference = self.snapshot_manager.load_latest()
            if reference is None:
                self._snapshot_available = False
                logger.warning("No reference snapshot. Call create_snapshot() first.")
                return None
            self._snapshot_available = True

            # Peek first and only remove after successful analysis. A failed analysis
            # therefore remains retryable instead of silently discarding its sample.
            current = np.asarray(list(self._query_buffer)[:batch_size], dtype=np.float32)
            report = self.analyser.analyse(reference, current)
            self.scheduler.update(report.mmd_score)
            for _ in range(batch_size):
                self._query_buffer.popleft()
            self._last_report = report

            logger.info(
                f"[{self.config.index_name}] drift={report.drift_type} "
                f"mmd={report.mmd_score:.4f} p={report.mmd_p_value:.3f} "
                f"coverage_gap={report.coverage_gap_rate:.2%}"
            )
            return report

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return None
        finally:
            self._analysis_running = False

    def _extract_vectors(self, response) -> list[np.ndarray]:
        """Extract embedding vectors from various response formats."""
        # Check numpy before hasattr("data") — np.ndarray has .data (buffer), not embeddings
        if isinstance(response, np.ndarray):
            return [response] if response.ndim == 1 else list(response)
        # OpenAI / Azure OpenAI response format
        if hasattr(response, "data"):
            return [np.array(item.embedding) for item in response.data]
        # Raw list
        if isinstance(response, list):
            if response and isinstance(response[0], (int, float)):
                return [np.array(response)]
            return [np.array(v) for v in response]
        return []
