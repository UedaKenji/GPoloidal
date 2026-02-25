from __future__ import annotations

"""
Template script for running a reproducible tomography/benchmark workflow
outside the GPoloidal source repository.

How to use:
1) Copy this file into your analysis repository / working directory.
2) Fill the device-specific sections marked with TODO.
3) Run with:
   uv run python external_gpoloidal_analysis_template.py --config config.json

Notes:
- cache is global (LOCALAPPDATA/.../gpoloidal/cache)
- run record is global by default (LOCALAPPDATA/.../gpoloidal/records)
- human-facing outputs are stored locally under analysis_runs/<experiment>/{archive,latest}
"""

# %%
import argparse
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gpoloidal
from gpoloidal.benchmark_utils import (
    apply_flat_dataclass_config,
    load_config_mapping,
    save_json,
)
from gpoloidal.experiment import (
    CallableRef,
    ExperimentRecord,
    FileRef,
    NoiseConfig,
    ObservationMatrixConfig,
    PhantomConfig,
    ProjectStore,
    TomographyConfig,
    default_cache_root,
    default_record_root,
)
from gpoloidal.run_layout import prepare_local_run_layout, publish_latest_from_archive


# %%
@dataclass
class AnalysisConfig:
    experiment_name: str = "my_device_benchmark"
    seed: int = 42
    n_trials: int = 3
    # TODO: add your device-specific parameters here (camera, phantom, etc.)
    # Example:
    # resolution: tuple[int, int] = (64, 64)
    # lnum: int = 1001
    # phantom_name: str = "hollow"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="External GPoloidal analysis template")
    p.add_argument("--config", type=str, default=None, help="JSON/TOML config")
    p.add_argument("--quick", action="store_true", help="Lightweight smoke mode")
    p.add_argument("--run-name", type=str, default=None, help="Suffix for archive directory")
    p.add_argument("--output-dir", type=str, default=None, help="Base dir for analysis_runs")
    p.add_argument("--backend-record-dir", type=str, default=None, help="Global run-record dir override")
    p.add_argument("--no-run-record", action="store_true")
    args, unknown = p.parse_known_args()  # Jupyter/ipykernel-safe
    if unknown:
        print("[info] ignored unknown args:", unknown)
    return args


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {path}")


# %%
# ---- 0) config / paths -------------------------------------------------------
ARGS = parse_args()
CFG = AnalysisConfig()

if ARGS.config:
    cfg_path = Path(ARGS.config).resolve()
    apply_flat_dataclass_config(CFG, load_config_mapping(cfg_path))
else:
    cfg_path = None

if ARGS.quick:
    CFG = replace(CFG, n_trials=1)

# Global reusable cache (script-independent). Cache hits are keyed by config hash.
cache_root = default_cache_root()
backend_record_root = (
    Path(ARGS.backend_record_dir).resolve()
    if ARGS.backend_record_dir
    else default_record_root() / CFG.experiment_name
)
output_base = Path(ARGS.output_dir).resolve() if ARGS.output_dir else Path.cwd() / "analysis_runs"
layout = prepare_local_run_layout(
    base_dir=output_base,
    experiment_name=CFG.experiment_name,
    run_name=ARGS.run_name,
)
output_root = layout.run_root
figure_dir = output_root / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)

save_json(output_root / "config_resolved.json", asdict(CFG))
if cfg_path is not None:
    save_json(output_root / "config_source.json", {"config_path": str(cfg_path)})

print("gpoloidal", gpoloidal.__version__)
print("config:", CFG)
print("cache_root:", cache_root)
print("backend_record_root:", backend_record_root)
print("output_root (archive run):", output_root)
print("output_latest_root:", layout.latest_root)


# %%
# ---- 1) backend store --------------------------------------------------------
store = ProjectStore(cache_root=cache_root, record_root=backend_record_root)


# %%
# ---- 2) TODO: build/load cache-backed artifacts ------------------------------
#
# This section is device-specific and should be implemented by the user.
# Typical pattern:
#
# 1. Create FileRef / CallableRef for reproducibility
# 2. Build ObservationMatrixConfig (and InducingPointConfig if needed)
# 3. Use store.get_or_build_*(...) to reuse cache
#
# Example sketch:
#
# point_file = Path(".../point_temp.npz")
# inducing_cfg = InducingPointConfig(
#     source=FileRef.from_path(point_file),
#     length_sq_function=CallableRef.from_callable(my_length_scale_fn),
# )
# inducing_arrays, ind_rec = store.get_or_build_inducing_points(
#     inducing_cfg,
#     builder=lambda: {...},
# )
# obs_cfg = ObservationMatrixConfig(..., inducing_points=inducing_cfg, ...)
# H, obsmat_rec = store.get_or_build_observation_matrix(
#     obs_cfg,
#     builder=lambda: build_H_somehow(inducing_arrays),
# )
#
# For now, we stop with a clear message.
raise SystemExit(
    "Template ready. Fill section '2) TODO: build/load cache-backed artifacts' "
    "and subsequent analysis steps for your device."
)


# %%
# ---- 3) TODO: run inference / benchmark -------------------------------------
# Implement your workflow here (e.g., GPT_lin_general / GPT_log_general).


# %%
# ---- 4) TODO: save human-facing outputs -------------------------------------
# Save figures/csv/json into output_root / figure_dir


# %%
# ---- 5) optional: save run record (global) ----------------------------------
# Example skeleton:
#
# if not ARGS.no_run_record:
#     record = ExperimentRecord(
#         name=CFG.experiment_name,
#         created_at_utc=datetime.now(timezone.utc).isoformat(),
#         observation_matrix_artifact_id=obsmat_rec.artifact_id,
#         observation_matrix_config=obs_cfg,
#         phantom=PhantomConfig(kind="...", name="..."),
#         noise=NoiseConfig(model="gaussian", level=None, level_definition="...", profile="...", seed=CFG.seed, params={}),
#         tomography=TomographyConfig(model="...", prior_kind="...", length_scale_factor=..., boundary_sigma=..., boundary_value=..., prior_mean=..., normalize=False, obs_noise_level=None),
#         references={"inducing_points_artifact_id": ind_rec.artifact_id},
#         metrics={},
#         outputs={},
#     )
#     run_id = store.save_experiment_record(
#         record,
#         strict_traceability=False,      # or True when manifests/results are retained
#         embed_dependency_manifests=False,
#     )
# else:
#     run_id = None
#
# publish_latest_from_archive(layout)
