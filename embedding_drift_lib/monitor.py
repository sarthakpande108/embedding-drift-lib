"""
DriftMonitor — main entry point.

Wraps any async embedding function and runs drift analysis in the background.
No external services required — no Supabase, no alerts, no API keys.
"""

import asyncio
import functools
import logging
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
        self._query_buffer: list[np.ndarray] = []
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
                    self._query_buffer.append(vec)
                    if self.sampler.cluster_model is not None:
                        self.sampler.add(vec)
            if len(self._query_buffer) >= self.config.analysis_batch_size:
                asyncio.create_task(self._run_analysis())
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
        logger.info(f"Snapshot created: {snapshot_id} ({len(vectors)} vectors, label='{label}')")
        return snapshot_id

    async def force_analysis(self) -> Optional[DriftReport]:
        """Manually trigger analysis regardless of buffer size."""
        return await self._run_analysis(force=True)

    @property
    def last_report(self) -> Optional[DriftReport]:
        return self._last_report

    @property
    def current_sample_size(self) -> int:
        return len(self._query_buffer)

    # ── Internal ───────────────────────────────────────────────────────────

    async def _run_analysis(self, force: bool = False) -> Optional[DriftReport]:
        if self._analysis_running and not force:
            return None

        reference = self.snapshot_manager.load_latest()
        if reference is None:
            logger.warning("No reference snapshot. Call create_snapshot() first.")
            return None

        if len(self._query_buffer) < 50:
            logger.debug(f"Buffer too small ({len(self._query_buffer)} vectors, need 50).")
            return None

        self._analysis_running = True
        try:
            current = np.array(self._query_buffer[:self.config.analysis_batch_size])
            self._query_buffer = self._query_buffer[self.config.analysis_batch_size:]

            report = self.analyser.analyse(reference, current)
            self._last_report = report
            self.scheduler.update(report.mmd_score)

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