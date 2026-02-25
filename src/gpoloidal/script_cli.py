from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .experiment import default_cache_root, default_record_root

ScriptMode = Literal["dev", "analysis"]
RecordMode = Literal["none", "light", "archive"]


@dataclass(frozen=True)
class RuntimeRoots:
    mode: ScriptMode
    cache_root: Path
    backend_record_root: Path
    output_base_dir: Path


@dataclass(frozen=True)
class RecordModePolicy:
    record_mode: RecordMode
    save_run_record: bool
    strict_traceability: bool
    embed_dependency_manifests: bool
    save_backend_result_artifacts: bool


def add_common_runtime_args(
    parser: argparse.ArgumentParser,
    *,
    include_config: bool = False,
    include_quick: bool = False,
    include_trials_csv_toggle: bool = False,
    output_help: str = "Base directory for analysis_runs-style outputs",
) -> argparse.ArgumentParser:
    if include_config:
        parser.add_argument("--config", type=str, default=None, help="Path to JSON/TOML/YAML config file")
    if include_quick:
        parser.add_argument("--quick", action="store_true", help="Use quick smoke settings")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("dev", "analysis"),
        default="dev",
        help="Default path mode: dev -> <PROJECT_ROOT>/analysis_runs, analysis -> <cwd>/analysis_runs",
    )
    parser.add_argument("--output-dir", type=str, default=None, help=output_help)
    parser.add_argument("--run-name", type=str, default=None, help="Optional suffix for archive run directory")
    parser.add_argument("--backend-record-dir", type=str, default=None, help="Backend run record directory (default: global records path)")
    parser.add_argument(
        "--record-mode",
        type=str,
        choices=("none", "light", "archive"),
        default="light",
        help="none: skip run record, light: save run_*.json only, archive: strict traceability + backend result artifacts",
    )
    # Backward-compatible alias for quick experiments.
    parser.add_argument("--no-run-record", action="store_true", help="Alias for --record-mode none")
    if include_trials_csv_toggle:
        parser.add_argument("--no-trials-csv", action="store_true", help="Do not save trial-level CSV")
    return parser


def parse_known_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args, unknown = parser.parse_known_args()
    if unknown:
        print("[info] ignored unknown args:", unknown)
    return args


def resolve_runtime_roots(
    *,
    mode: str,
    project_root: str | Path,
    backend_experiment_name: str,
    output_dir: str | Path | None = None,
    backend_record_dir: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> RuntimeRoots:
    mode_norm = mode.lower()
    if mode_norm not in {"dev", "analysis"}:
        raise ValueError(f"Unknown mode: {mode!r}")
    project_root = Path(project_root).resolve()
    cache_path = Path(cache_root).expanduser() if cache_root is not None else default_cache_root()
    backend_path = (
        Path(backend_record_dir).expanduser()
        if backend_record_dir is not None
        else default_record_root() / backend_experiment_name
    )
    if output_dir is not None:
        output_base_dir = Path(output_dir).expanduser()
    else:
        output_base_dir = (project_root if mode_norm == "dev" else Path.cwd()) / "analysis_runs"
    return RuntimeRoots(
        mode=mode_norm,  # type: ignore[arg-type]
        cache_root=cache_path,
        backend_record_root=backend_path,
        output_base_dir=output_base_dir,
    )


def resolve_record_mode_policy(*, record_mode: str, no_run_record: bool = False) -> RecordModePolicy:
    if no_run_record:
        mode = "none"
    else:
        mode = record_mode.lower()
    if mode not in {"none", "light", "archive"}:
        raise ValueError(f"Unknown record_mode: {record_mode!r}")

    if mode == "none":
        return RecordModePolicy(
            record_mode="none",
            save_run_record=False,
            strict_traceability=False,
            embed_dependency_manifests=False,
            save_backend_result_artifacts=False,
        )
    if mode == "light":
        return RecordModePolicy(
            record_mode="light",
            save_run_record=True,
            strict_traceability=False,
            embed_dependency_manifests=False,
            save_backend_result_artifacts=False,
        )
    return RecordModePolicy(
        record_mode="archive",
        save_run_record=True,
        strict_traceability=True,
        embed_dependency_manifests=True,
        save_backend_result_artifacts=True,
    )

