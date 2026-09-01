"""Durable, settings-safe checkpoints for per-feature LLM stages."""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

import pandas as pd


def checkpoint_path(out: str | Path) -> Path:
    """Sidecar that identifies the run whose rows live in ``out``."""
    p = Path(out)
    base = p.with_suffix("") if p.suffix else p
    return base.with_name(base.name + ".resume.json")


def _atomic_write_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if path.suffix == ".parquet":
        df.to_parquet(tmp, index=False)
    else:
        df.to_csv(tmp, index=False)
    tmp.replace(path)


def _atomic_write_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


class FeatureCheckpoint:
    """One durable table row per completed feature.

    Resume is intentionally strict: a sidecar signature prevents silently mixing rows
    produced with a different lens, model, prompt configuration, or sampling setup.
    ``fresh=True`` explicitly replaces the old table and sidecar.
    """

    def __init__(self, out: str | Path, signature: dict, *, fresh: bool = False) -> None:
        self.out = Path(out)
        self.sidecar = checkpoint_path(self.out)
        self.signature = signature
        self._lock = Lock()
        self._rows: dict[int, dict] = {}

        if fresh:
            self.out.unlink(missing_ok=True)
            self.sidecar.unlink(missing_ok=True)

        if self.sidecar.exists():
            saved = json.loads(self.sidecar.read_text(encoding="utf-8"))
            saved_signature = saved.get("signature")
            if saved_signature != signature:
                changed = sorted(
                    key for key in set(saved_signature or {}) | set(signature)
                    if (saved_signature or {}).get(key) != signature.get(key))
                detail = ", ".join(changed[:8]) or "unknown settings"
                raise ValueError(
                    f"checkpoint settings differ ({detail}); use --fresh to start over")
        elif self.out.exists():
            raise ValueError(
                f"{self.out} exists without {self.sidecar.name}; use --fresh to replace it, "
                "or move it aside before starting a resumable run")
        else:
            _atomic_write_json({"schema_version": 1, "signature": signature}, self.sidecar)

        if self.out.exists():
            frame = (pd.read_parquet(self.out) if self.out.suffix == ".parquet"
                     else pd.read_csv(self.out))
            if "feature_id" not in frame.columns:
                raise ValueError(f"checkpoint output has no feature_id column: {self.out}")
            ids = frame["feature_id"].astype(int)
            if ids.duplicated().any():
                raise ValueError(f"checkpoint output has duplicate feature_id rows: {self.out}")
            self._rows = {int(row["feature_id"]): row
                          for row in frame.to_dict(orient="records")}

    @property
    def completed_ids(self) -> set[int]:
        return set(self._rows)

    def record(self, row: dict) -> None:
        """Atomically persist one completed feature before more work is awaited."""
        if "feature_id" not in row:
            raise ValueError("checkpoint row has no feature_id")
        with self._lock:
            self._rows[int(row["feature_id"])] = dict(row)
            self._write()

    def merge(self, frame: pd.DataFrame) -> None:
        """Persist final/generated rows such as verifier abstentions."""
        if frame.empty:
            return
        if "feature_id" not in frame.columns:
            raise ValueError("checkpoint frame has no feature_id")
        with self._lock:
            for row in frame.to_dict(orient="records"):
                self._rows[int(row["feature_id"])] = row
            self._write()

    def frame(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame()
        return pd.DataFrame(self._rows.values()).sort_values("feature_id").reset_index(drop=True)

    def _write(self) -> None:
        _atomic_write_frame(self.frame(), self.out)
