from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gpoloidal
import gpoloidal.rt1 as rt1
from gpoloidal.analysis.config import apply_flat_dataclass_config, load_config_mapping, save_json
from gpoloidal.analysis.rt1_plots import (
    plot_loggp_loss_history,
    plot_rt1_reconstruction_panels,
    plot_rt1_truth_and_observation,
)
from gpoloidal.experiment import (
    CallableRef,
    ExperimentRecord,
    ObservationMatrixConfig,
    PhantomConfig,
    ProjectStore,
    TomographyConfig,
)
from gpoloidal.run_layout import make_run_reference, prepare_local_run_layout, publish_latest_from_archive
from gpoloidal.script_cli import add_common_runtime_args, parse_known_args, resolve_record_mode_policy, resolve_runtime_roots
from gpoloidal.scripts.rt1_tomography_single import (
    PROJECT_ROOT,
    SingleTomographyConfig,
    _flatten_single_tomography_config_mapping,
    prepare_single_tomography,
    save_figure,
    solve_single_tomography,
)


@dataclass(frozen=True)
class SweepAxisConfig:
    key: str
    values: tuple[Any, ...]


@dataclass
class SweepConfig:
    experiment_name: str = "rt1_tomography_sweep"
    n_trials: int = 5
    seed: int = 42
    seed_policy: Literal["by_trial", "by_combo_trial"] = "by_trial"
    summary_x_key: str | None = None
    save_case_plots: bool = False
    axes: list[SweepAxisConfig] = field(default_factory=lambda: [
        SweepAxisConfig("tomography.method", ("linGP", "logGP")),
        SweepAxisConfig("noise.snr_rms", (30.0, 10.0, 3.0, 1.0)),
    ])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RT1 tomography sweep runner")
    add_common_runtime_args(p, include_config=True, include_quick=True, include_trials_csv_toggle=True)
    p.add_argument("--save-case-plots", action="store_true", help="Save per-condition plots (trial 1 only)")
    return parse_known_args(p)


def _flatten_sweep_config_mapping(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(data, dict):
        raise TypeError("config root must be a mapping")
    single_data = data.get("single", {k: v for k, v in data.items() if k != "sweep"})
    if not isinstance(single_data, dict):
        raise ValueError("Config section 'single' must be a mapping")
    single_flat = _flatten_single_tomography_config_mapping(single_data)

    sweep = data.get("sweep", {}) or {}
    if not isinstance(sweep, dict):
        raise ValueError("Config section 'sweep' must be a mapping")
    out: dict[str, Any] = {}

    if "experiment" in sweep:
        exp = sweep["experiment"]
        if not isinstance(exp, dict):
            raise ValueError("Config section 'sweep.experiment' must be a mapping")
        if "name" in exp:
            out["experiment_name"] = exp["name"]
        unknown = set(exp) - {"name"}
        if unknown:
            raise ValueError(f"Unknown config keys: {[f'sweep.experiment.{k}' for k in sorted(unknown)]}")

    if "replicate" in sweep:
        rep = sweep["replicate"]
        if not isinstance(rep, dict):
            raise ValueError("Config section 'sweep.replicate' must be a mapping")
        for k, v in rep.items():
            if k not in {"n_trials", "seed", "seed_policy"}:
                raise ValueError(f"Unknown config key: sweep.replicate.{k}")
            if k == "seed_policy" and v not in {"by_trial", "by_combo_trial"}:
                raise ValueError("Config key 'sweep.replicate.seed_policy' must be 'by_trial' or 'by_combo_trial'")
            out[k] = v

    if "report" in sweep:
        rep = sweep["report"]
        if not isinstance(rep, dict):
            raise ValueError("Config section 'sweep.report' must be a mapping")
        for k, v in rep.items():
            if k not in {"summary_x_key", "save_case_plots"}:
                raise ValueError(f"Unknown config key: sweep.report.{k}")
            out[k] = v

    for key in ("experiment_name", "n_trials", "seed", "seed_policy", "summary_x_key", "save_case_plots"):
        if key in sweep:
            if key == "seed_policy" and sweep[key] not in {"by_trial", "by_combo_trial"}:
                raise ValueError("Config key 'sweep.seed_policy' must be 'by_trial' or 'by_combo_trial'")
            out[key] = sweep[key]

    if "axes" in sweep:
        axes = sweep["axes"]
        if not isinstance(axes, list):
            raise ValueError("Config key 'sweep.axes' must be a list")
        norm_axes: list[dict[str, Any]] = []
        for i, a in enumerate(axes):
            if not isinstance(a, dict) or "key" not in a or "values" not in a:
                raise ValueError(f"sweep.axes[{i}] requires mapping with key/values")
            if not isinstance(a["values"], list):
                raise ValueError(f"sweep.axes[{i}].values must be a list")
            norm_axes.append({"key": a["key"], "values": a["values"]})
        out["axes"] = norm_axes

    return single_flat, out


def _apply_sweep_config(cfg: SweepConfig, updates: dict[str, Any]) -> SweepConfig:
    for key, value in updates.items():
        if key == "axes":
            cfg.axes = [SweepAxisConfig(key=str(a["key"]), values=tuple(a["values"])) for a in value]
        else:
            setattr(cfg, key, value)
    return cfg


def _nested_from_dotted(overrides: dict[str, Any]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        cur = root
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value
    return root


def _expand_axis_combinations(axes: list[SweepAxisConfig]) -> list[dict[str, Any]]:
    if not axes:
        return [{}]
    keys = [a.key for a in axes]
    return [dict(zip(keys, vals, strict=False)) for vals in product(*[a.values for a in axes])]


def _case_seed(
    sweep_cfg: SweepConfig,
    *,
    combo_i: int,
    combo: dict[str, Any],
    trial: int,
) -> int:
    if sweep_cfg.seed_policy == "by_trial":
        return int(sweep_cfg.seed + trial)
    if sweep_cfg.seed_policy == "by_combo_trial":
        return int(sweep_cfg.seed + combo_i * 100000 + trial)
    raise ValueError(f"Unknown seed_policy: {sweep_cfg.seed_policy}")


def _slugify_text(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s.strip())
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-") or "case"


def _combo_slug(combo: dict[str, Any], axis_keys: list[str]) -> str:
    if not combo:
        return "base"
    ordered_keys = [k for k in axis_keys if k in combo] + [k for k in combo if k not in axis_keys]
    parts = [f"{k.replace('.', '-')}_{combo[k]}" for k in ordered_keys]
    return _slugify_text("__".join(parts))


def _case_title_text(combo: dict[str, Any], *, trial: int, seed: int) -> str:
    combo_text = ", ".join(f"{k}={v}" for k, v in combo.items()) if combo else "(base)"
    return f"{combo_text}\ntrial={trial + 1}, seed={seed}"


def _forward_prepare_cache_key(cfg: SingleTomographyConfig) -> str:
    # Keys that affect vessel/camera/raytrace/inducing points/observation matrix construction.
    fields = {
        "basis_mode": cfg.basis_mode,
        "observation_matrix_mode": cfg.observation_matrix_mode,
        "point_file": cfg.point_file,
        "grid_r_min": cfg.grid_r_min,
        "grid_r_max": cfg.grid_r_max,
        "grid_r_step": cfg.grid_r_step,
        "grid_z_min": cfg.grid_z_min,
        "grid_z_max": cfg.grid_z_max,
        "grid_z_step": cfg.grid_z_step,
        "grid_inducing_length_scale": cfg.grid_inducing_length_scale,
        "grid_inside_value": cfg.grid_inside_value,
        "grid_boundary_value": cfg.grid_boundary_value,
        "grid_obs_sample_count": cfg.grid_obs_sample_count,
        "vessel_package_resource": cfg.vessel_package_resource,
        "camera_kind": cfg.camera_kind,
        "camera_focal_length": cfg.camera_focal_length,
        "camera_location": list(cfg.camera_location),
        "camera_center_angles": list(cfg.camera_center_angles),
        "camera_sensor_size": list(cfg.camera_sensor_size),
        "camera_rotation": cfg.camera_rotation,
        "resolution": list(cfg.resolution),
        "lnum": cfg.lnum,
        "nreflections": cfg.nreflections,
        "pass_through_first": cfg.pass_through_first,
        "ray_index_for_integral": cfg.ray_index_for_integral,
        "plot_r_min": cfg.plot_r_min,
        "plot_r_max": cfg.plot_r_max,
        "plot_r_num": cfg.plot_r_num,
        "plot_z_min": cfg.plot_z_min,
        "plot_z_max": cfg.plot_z_max,
        "plot_z_num": cfg.plot_z_num,
    }
    return json.dumps(fields, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _solve_case(
    cfg: SingleTomographyConfig,
    *,
    store: ProjectStore,
    prepared_cache: dict[str, dict[str, Any]],
    include_plot_payload: bool = False,
) -> dict[str, Any]:
    key = _forward_prepare_cache_key(cfg)
    prepared = prepared_cache.get(key)
    prepared_cache_hit = prepared is not None
    if prepared is None:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            prepared = prepare_single_tomography(cfg, store=store, show_progress=False)
        prepared_cache[key] = prepared
    solved = solve_single_tomography(cfg, prepared=prepared)
    run = {**prepared, **solved}
    return {
        "metrics": run["metrics"],
        "obs_mode": run["obs_mode"],
        "prior_mode": run["prior_mode"],
        "observation_matrix_config": run["obs_cfg"],
        "observation_matrix_artifact_id": run["obsmat_rec"].artifact_id,
        "inducing_points_artifact_id": run["ind_rec"].artifact_id,
        "prepared_cache_hit": prepared_cache_hit,
        **(
            {
                "plot_payload": {
                    "flux_kernel": run["flux_kernel"],
                    "kernel": run["kernel"],
                    "f_true": run["f_true"],
                    "g_obs": run["g_obs"],
                    "fit": run["fit"],
                    "fit_label": run["fit_label"],
                }
            }
            if include_plot_payload
            else {}
        ),
    }

def _summarize_trials(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    summary = (
        df.groupby(group_cols)
        .agg(
            n_trials=("trial", "count"),
            recon_rmse=("recon_rmse", "mean"),
            recon_rmse_std=("recon_rmse", "std"),
            recon_mae=("recon_mae", "mean"),
            recon_rel_rmse=("recon_rel_rmse", "mean"),
            recon_corr=("recon_corr", "mean"),
            chi2=("chi2", "mean"),
            chi2_std=("chi2", "std"),
            obs_noise_level=("obs_noise_level", "mean"),
            snr_rms_true=("snr_rms_true", "mean"),
        )
        .reset_index()
    )
    if "log_converged" in df.columns:
        extra = (
            df.groupby(group_cols)
            .agg(
                log_nonconverged_frac=("log_converged", lambda s: float(1.0 - np.mean(np.asarray(s, dtype=float)))),
                log_retry_frac=("log_retried", lambda s: float(np.mean(np.asarray(s, dtype=float)))),
                log_iters=("log_iters", "mean"),
            )
            .reset_index()
        )
        summary = summary.merge(extra, on=group_cols, how="left")
    return summary


def _plot_summary(summary_df: pd.DataFrame, *, axis_keys: list[str], x_key: str | None, out_path: Path) -> bool:
    if summary_df.empty:
        return False
    if x_key is None:
        for c in ("noise.snr_rms", "tomography.length_scale_factor", "tomography.uniform_length_scale"):
            if c in summary_df.columns:
                x_key = c
                break
    if x_key is None or x_key not in summary_df.columns:
        return False

    hue_key = None
    for c in axis_keys:
        if c != x_key and c in summary_df.columns:
            hue_key = c
            if c == "tomography.method":
                break

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for ax, metric in zip(axs, ("recon_rmse", "chi2"), strict=False):
        if metric not in summary_df.columns:
            continue
        metric_std = f"{metric}_std"
        use_errorbar = (
            metric_std in summary_df.columns
            and "n_trials" in summary_df.columns
            and np.nanmax(np.asarray(summary_df["n_trials"], dtype=float)) > 1
        )
        if hue_key is None:
            g = summary_df.sort_values(x_key)
            if use_errorbar:
                ax.errorbar(
                    g[x_key],
                    g[metric],
                    yerr=np.asarray(g[metric_std].fillna(0.0), dtype=float),
                    marker="o",
                    capsize=3,
                )
            else:
                ax.plot(g[x_key], g[metric], marker="o")
        else:
            for hv, g in summary_df.groupby(hue_key):
                g = g.sort_values(x_key)
                if use_errorbar:
                    ax.errorbar(
                        g[x_key],
                        g[metric],
                        yerr=np.asarray(g[metric_std].fillna(0.0), dtype=float),
                        marker="o",
                        capsize=3,
                        label=str(hv),
                    )
                else:
                    ax.plot(g[x_key], g[metric], marker="o", label=str(hv))
            ax.legend(fontsize=8)
        ax.set_xlabel(x_key)
        ax.set_ylabel(metric)
        ax.set_title(metric)
        if metric == "chi2":
            ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.3)
        if "snr" in x_key.lower():
            try:
                ax.set_xscale("log")
                ax.invert_xaxis()
            except Exception:
                pass
    save_figure(fig, out_path)
    return True


def main() -> None:
    args = parse_args()
    single_cfg = SingleTomographyConfig()
    sweep_cfg = SweepConfig()

    config_path = (
        (PROJECT_ROOT / args.config).resolve()
        if args.config and not Path(args.config).is_absolute()
        else (Path(args.config).resolve() if args.config else None)
    )
    loaded_raw = None
    if config_path is not None:
        loaded_raw = load_config_mapping(config_path)
        single_updates, sweep_updates = _flatten_sweep_config_mapping(loaded_raw)
        apply_flat_dataclass_config(single_cfg, single_updates)
        _apply_sweep_config(sweep_cfg, sweep_updates)
    
    single_cfg.experiment_name = sweep_cfg.experiment_name
    if args.quick:
        single_cfg = replace(single_cfg, resolution=(48, 48), lnum=min(single_cfg.lnum, 401), grid_obs_sample_count=min(single_cfg.grid_obs_sample_count, 120), max_log_iters=min(single_cfg.max_log_iters, 10))
        sweep_cfg.n_trials = min(sweep_cfg.n_trials, 2)
    if getattr(args, "save_case_plots", False):
        sweep_cfg.save_case_plots = True

    runtime_roots = resolve_runtime_roots(mode=args.mode, project_root=PROJECT_ROOT, backend_experiment_name=sweep_cfg.experiment_name, output_dir=args.output_dir, backend_record_dir=args.backend_record_dir)
    record_policy = resolve_record_mode_policy(record_mode=args.record_mode, no_run_record=args.no_run_record)
    layout = prepare_local_run_layout(base_dir=runtime_roots.output_base_dir, experiment_name=sweep_cfg.experiment_name, run_name=args.run_name)
    output_root = layout.run_root
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    case_figure_dir = figure_dir / "cases"
    if sweep_cfg.save_case_plots:
        case_figure_dir.mkdir(parents=True, exist_ok=True)

    print("gpoloidal", gpoloidal.__version__)
    print("single_base_config:", single_cfg)
    print("sweep_config:", sweep_cfg)
    if config_path is not None:
        print("config_path:", config_path)
    print("mode:", runtime_roots.mode)
    print("record_mode:", record_policy.record_mode)
    print("cache_root:", runtime_roots.cache_root)
    print("backend_record_root:", runtime_roots.backend_record_root)
    print("output_root (archive run):", output_root)
    print("output_latest_root:", layout.latest_root)

    save_json(output_root / "config_resolved_single.json", asdict(single_cfg))
    save_json(output_root / "config_resolved_sweep.json", {**{k: v for k, v in asdict(sweep_cfg).items() if k != "axes"}, "axes": [{"key": a.key, "values": list(a.values)} for a in sweep_cfg.axes]})
    if config_path is not None and loaded_raw is not None:
        save_json(output_root / "config_source.json", {"config_path": str(config_path), "loaded": loaded_raw})

    store = ProjectStore(cache_root=runtime_roots.cache_root, record_root=runtime_roots.backend_record_root)
    axis_keys = [a.key for a in sweep_cfg.axes]
    combos = _expand_axis_combinations(sweep_cfg.axes)
    rows: list[dict[str, Any]] = []
    obsmat_ids: set[str] = set()
    ind_ids: set[str] = set()
    obs_cfgs_by_id: dict[str, ObservationMatrixConfig] = {}
    prepared_cache: dict[str, dict[str, Any]] = {}

    for combo_i, combo in enumerate(combos):
        combo_flat = _flatten_single_tomography_config_mapping(_nested_from_dotted(combo))
        combo_label = ", ".join(f"{k}={v}" for k, v in combo.items()) if combo else "(base)"
        combo_slug = _combo_slug(combo, axis_keys)
        print(f"[sweep combo {combo_i + 1}/{len(combos)}] {combo_label}")
        for trial in range(sweep_cfg.n_trials):
            case_cfg = replace(single_cfg)
            apply_flat_dataclass_config(case_cfg, combo_flat)
            case_cfg.seed = _case_seed(sweep_cfg, combo_i=combo_i, combo=combo, trial=trial)
            save_case_plot_this_trial = bool(sweep_cfg.save_case_plots and trial == 0)
            result = _solve_case(
                case_cfg,
                store=store,
                prepared_cache=prepared_cache,
                include_plot_payload=save_case_plot_this_trial,
            )

            obsmat_ids.add(result["observation_matrix_artifact_id"])
            ind_ids.add(result["inducing_points_artifact_id"])
            obs_cfgs_by_id[result["observation_matrix_artifact_id"]] = result["observation_matrix_config"]

            row = {
                "trial": trial,
                "seed": case_cfg.seed,
                "basis_mode": case_cfg.basis_mode,
                "tomography_method": case_cfg.tomography_method,
                "prior_mode": result["prior_mode"],
                "obs_mode": result["obs_mode"],
                "phantom_name": case_cfg.phantom_name,
                "observation_matrix_artifact_id": result["observation_matrix_artifact_id"],
                "inducing_points_artifact_id": result["inducing_points_artifact_id"],
                **combo,
                **result["metrics"],
            }
            rows.append(row)
            forward_status = "hit" if result.get("prepared_cache_hit", False) else "miss"
            print(
                f"  [trial {trial + 1}/{sweep_cfg.n_trials}] "
                f"seed={case_cfg.seed} forward={forward_status} "
                f"rmse={row.get('recon_rmse'):.4g} chi2={row.get('chi2'):.4g}"
            )
            if save_case_plot_this_trial:
                pp = result["plot_payload"]
                case_title = _case_title_text(combo, trial=trial, seed=case_cfg.seed)

                fig = plot_rt1_truth_and_observation(
                    kernel=pp["kernel"],
                    flux_kernel=pp["flux_kernel"],
                    f_true=pp["f_true"],
                    g_obs=pp["g_obs"],
                    resolution=case_cfg.resolution,
                )
                fig.suptitle(case_title, fontsize=10)
                save_figure(
                    fig,
                    case_figure_dir / f"rt1_truth_and_observation__{combo_slug}.png",
                    dpi=case_cfg.plot_figure_dpi,
                )

                fig, _ = plot_rt1_reconstruction_panels(
                    kernel=pp["kernel"],
                    flux_kernel=pp["flux_kernel"],
                    f_true=pp["f_true"],
                    f_recon=np.asarray(pp["fit"]["f_mean"], dtype=float),
                    f_std=np.asarray(pp["fit"]["f_std"], dtype=float),
                    fit_label=pp["fit_label"],
                    value_vmin=case_cfg.plot_value_vmin,
                    value_vmax=case_cfg.plot_value_vmax,
                    error_percentile=case_cfg.plot_error_percentile,
                )
                fig.suptitle(case_title, fontsize=10)
                save_figure(
                    fig,
                    case_figure_dir / f"rt1_single_reconstruction__{combo_slug}.png",
                    dpi=case_cfg.plot_figure_dpi,
                )

                if case_cfg.tomography_method == "logGP":
                    fig = plot_loggp_loss_history(loss_history=pp["fit"]["loss_history"])
                    fig.suptitle(case_title, fontsize=10)
                    save_figure(
                        fig,
                        case_figure_dir / f"rt1_loggp_loss_history__{combo_slug}.png",
                        dpi=case_cfg.plot_figure_dpi,
                    )

    trials_df = pd.DataFrame.from_records(rows)
    sort_cols = [c for c in axis_keys if c in trials_df.columns] + ["trial"]
    if sort_cols:
        trials_df = trials_df.sort_values(sort_cols).reset_index(drop=True)

    if axis_keys:
        summary_df = _summarize_trials(trials_df, axis_keys)
    else:
        summary_df = _summarize_trials(trials_df.assign(_all="all"), ["_all"]).drop(columns=["_all"])

    trials_csv = output_root / "rt1_tomography_sweep_trials.csv"
    summary_csv = output_root / "rt1_tomography_sweep_summary.csv"
    if not getattr(args, "no_trials_csv", False):
        trials_df.to_csv(trials_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    summary_plot = figure_dir / "rt1_tomography_sweep_summary.png"
    summary_plot_written = _plot_summary(summary_df, axis_keys=axis_keys, x_key=sweep_cfg.summary_x_key, out_path=summary_plot)

    backend_outputs: dict[str, str] = {}
    if record_policy.save_backend_result_artifacts:
        backend_outputs["summary_csv_artifact_id"] = store.save_result_file("rt1_tomography_sweep_summary_csv", summary_csv, kind="table", copy=True).artifact_id
        if not getattr(args, "no_trials_csv", False):
            backend_outputs["trials_csv_artifact_id"] = store.save_result_file("rt1_tomography_sweep_trials_csv", trials_csv, kind="table", copy=True).artifact_id
        if summary_plot_written:
            backend_outputs["summary_plot_artifact_id"] = store.save_result_file("rt1_tomography_sweep_summary_plot", summary_plot, kind="image", copy=True).artifact_id

    run_id = None
    if record_policy.save_run_record:
        single_obsmat_id = next(iter(obsmat_ids)) if len(obsmat_ids) == 1 else None
        record = ExperimentRecord(
            name=sweep_cfg.experiment_name,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            observation_matrix_artifact_id=single_obsmat_id,
            observation_matrix_config=(obs_cfgs_by_id[single_obsmat_id] if single_obsmat_id is not None else None),
            phantom=PhantomConfig(kind="synthetic", name=single_cfg.phantom_name, generator=CallableRef.from_callable(rt1.phantom.get_phantom_function), params={"base_phantom_params": single_cfg.phantom_params}),
            noise=None,
            tomography=TomographyConfig(model="sweep", prior_kind="mixed", extras={"axes": [{"key": a.key, "values": list(a.values)} for a in sweep_cfg.axes], "n_trials": sweep_cfg.n_trials}),
            references={"observation_matrix_artifact_ids": sorted(obsmat_ids), "inducing_points_artifact_ids": sorted(ind_ids), "summary_csv": str(summary_csv), **({} if getattr(args, "no_trials_csv", False) else {"trials_csv": str(trials_csv)})},
            metrics={"n_cases": len(combos), "n_prepared_forwards": len(prepared_cache), "n_trials_total": len(trials_df), "summary_rows": len(summary_df), "best_recon_rmse": (float(summary_df["recon_rmse"].min()) if not summary_df.empty else None)},
            outputs=backend_outputs,
            notes="Sweep-level record; per-case details stored in local CSV.",
        )
        run_id = store.save_experiment_record(record, strict_traceability=record_policy.strict_traceability, embed_dependency_manifests=record_policy.embed_dependency_manifests)

    report = {
        "script": "gpoloidal.scripts.rt1_tomography_sweep",
        "gpoloidal_version": gpoloidal.__version__,
        "config_path": str(config_path) if config_path is not None else None,
        "single_base_config": asdict(single_cfg),
        "sweep_config": {**{k: v for k, v in asdict(sweep_cfg).items() if k != "axes"}, "axes": [{"key": a.key, "values": list(a.values)} for a in sweep_cfg.axes]},
        "runtime": {"mode": runtime_roots.mode, "record_mode": record_policy.record_mode},
        "paths": {
            "output_root": str(output_root), "output_latest_root": str(layout.latest_root), "output_archive_root": str(layout.archive_root),
            "backend_record_root": str(runtime_roots.backend_record_root), "cache_root": str(runtime_roots.cache_root), "summary_csv": str(summary_csv),
            **({} if getattr(args, "no_trials_csv", False) else {"trials_csv": str(trials_csv)}), **({"summary_plot": str(summary_plot)} if summary_plot_written else {}), **({"case_plots_dir": str(case_figure_dir)} if sweep_cfg.save_case_plots else {}), **({"run_record": str(store.run_dir / f"{run_id}.json")} if run_id else {}),
        },
        "artifacts": {"observation_matrix_artifact_ids": sorted(obsmat_ids), "inducing_points_artifact_ids": sorted(ind_ids), "run_id": run_id},
        "summary_preview": summary_df.head(20).to_dict(orient="records"),
    }
    save_json(output_root / "latest_report.json", report)
    save_json(output_root / "latest_paths.json", report["paths"])
    save_json(output_root / "run_ref.json", make_run_reference(script="gpoloidal.scripts.rt1_tomography_sweep", archive_run_root=output_root, latest_root=layout.latest_root, backend_record_root=runtime_roots.backend_record_root, run_id=run_id, backend_run_record_path=(store.run_dir / f"{run_id}.json") if run_id else None, extra={"record_mode": record_policy.record_mode, "seed_policy": sweep_cfg.seed_policy, "save_case_plots": sweep_cfg.save_case_plots, "n_cases": len(combos), "n_prepared_forwards": len(prepared_cache), "n_trials_total": len(trials_df)}))
    publish_latest_from_archive(layout)

    print(json.dumps({"run_id": run_id, "n_cases": len(combos), "n_prepared_forwards": len(prepared_cache), "n_trials_total": len(trials_df), "summary_rows": len(summary_df), "archive_run_root": str(output_root), "latest_root": str(layout.latest_root)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
