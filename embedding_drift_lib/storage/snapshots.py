import os
import json
import uuid
import datetime
import numpy as np
from typing import Optional


class SnapshotManager:
    """
    Stores and retrieves reference vector snapshots as local .npy files.

    Rule: only update snapshots when you deliberately re-index.
    Never auto-update in response to detected drift — that erases the signal.
    """

    def __init__(self, snapshot_dir: str = "./snapshots"):
        self.snapshot_dir = snapshot_dir
        os.makedirs(snapshot_dir, exist_ok=True)

    def create(self, vectors: np.ndarray, label: str = "baseline") -> str:
        snapshot_id = uuid.uuid4().hex[:8]
        np.save(self._vec_path(snapshot_id), vectors)
        
        meta = {
            "id":        snapshot_id,
            "label":     label,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "n_vectors": len(vectors),
            "dim":       vectors.shape[1],
            "active":    True,
        }
        
        # Deactivate all existing older snapshots
        for sid in self._list_ids():
            prev = self._load_meta(sid)
            if prev and prev.get("active"):
                prev["active"] = False
                self._save_meta(sid, prev)

        self._save_meta(snapshot_id, meta)
        return snapshot_id

    def load_latest(self) -> Optional[np.ndarray]:
        for sid in sorted(self._list_ids(), reverse=True):
            meta = self._load_meta(sid)
            if meta and meta.get("active"):
                path = self._vec_path(sid)
                if os.path.exists(path):
                    return np.load(path)
        return None

    def load(self, snapshot_id: str) -> Optional[np.ndarray]:
        path = self._vec_path(snapshot_id)
        return np.load(path) if os.path.exists(path) else None

    def list_snapshots(self) -> list[dict]:
        results = []
        for sid in self._list_ids():
            meta = self._load_meta(sid)
            if meta:
                results.append(meta)
        return sorted(results, key=lambda m: m["timestamp"], reverse=True)

    def _vec_path(self, sid: str) -> str:
        return os.path.join(self.snapshot_dir, f"{sid}.npy")

    def _meta_path(self, sid: str) -> str:
        return os.path.join(self.snapshot_dir, f"{sid}.json")

    def _list_ids(self) -> list[str]:
        return [f[:-5] for f in os.listdir(self.snapshot_dir) if f.endswith(".json")]

    def _load_meta(self, sid: str) -> Optional[dict]:
        path = self._meta_path(sid)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def _save_meta(self, sid: str, meta: dict) -> None:
        with open(self._meta_path(sid), "w") as f:
            json.dump(meta, f, indent=2)