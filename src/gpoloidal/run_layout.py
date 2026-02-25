from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
from typing import Any


def _sanitize_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    s = s.strip("._-")
    return s or "run"


@dataclass
class LocalRunLayout:
    base_dir: Path
    experiment_name: str
    experiment_root: Path
    latest_root: Path
    archive_root: Path
    run_label: str
    run_root: Path
    timestamp: str


def prepare_local_run_layout(
    *,
    base_dir: str | Path,
    experiment_name: str,
    run_name: str | None = None,
    timestamp: str | None = None,
) -> LocalRunLayout:
    base = Path(base_dir).resolve()
    exp_name = _sanitize_name(experiment_name)
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    label = ts if not run_name else f"{ts}_{_sanitize_name(run_name)}"

    experiment_root = base / exp_name
    latest_root = experiment_root / "latest"
    archive_root = experiment_root / "archive"
    run_root = archive_root / label

    run_root.mkdir(parents=True, exist_ok=True)
    latest_root.mkdir(parents=True, exist_ok=True)

    return LocalRunLayout(
        base_dir=base,
        experiment_name=exp_name,
        experiment_root=experiment_root,
        latest_root=latest_root,
        archive_root=archive_root,
        run_label=label,
        run_root=run_root,
        timestamp=ts,
    )


def publish_latest_from_archive(layout: LocalRunLayout) -> None:
    """Mirror the current archive run directory into `latest/` (overwrite)."""
    latest = layout.latest_root
    latest.mkdir(parents=True, exist_ok=True)

    for child in list(latest.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    for child in layout.run_root.iterdir():
        dst = latest / child.name
        if child.is_dir():
            shutil.copytree(child, dst)
        else:
            shutil.copy2(child, dst)


def make_run_reference(
    *,
    script: str,
    archive_run_root: str | Path,
    latest_root: str | Path,
    backend_record_root: str | Path,
    run_id: str | None,
    backend_run_record_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a small local reference payload linking human outputs to backend run records."""
    payload: dict[str, Any] = {
        "script": script,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_run_root": str(Path(archive_run_root)),
        "latest_root": str(Path(latest_root)),
        "backend_record_root": str(Path(backend_record_root)),
        "run_id": run_id,
        "backend_run_record_path": str(Path(backend_run_record_path)) if backend_run_record_path else None,
    }
    if extra:
        payload["extra"] = extra
    return payload
