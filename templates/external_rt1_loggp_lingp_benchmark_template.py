from __future__ import annotations

"""
External RT1 linGP vs logGP benchmark template.

This template is intended to be copied into an external analysis repository.
It follows the same storage conventions as GPoloidal scripts:
  - cache: global
  - run record: global
  - human-facing outputs: local analysis_runs/<exp>/{archive,latest}

Required edits before first run:
  1) Set `point_file` in config (path to point_temp.npz-like file)
  2) Optionally adjust camera / phantom / benchmark settings
"""

# %%
import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gpoloidal
import gpoloidal.rt1 as rt1
import zray
from gpoloidal.analysis.config import (
    apply_flat_dataclass_config,
    load_config_mapping,
    save_json,
)
from gpoloidal.analysis.noise_sweep import (
    summarize_noise_sweep,
)
from gpoloidal.core.metrics import field_metrics, mean_chi2
from gpoloidal.experiment import (
    CameraConfig,
    CallableRef,
    ExperimentRecord,
    FileRef,
    InducingPointConfig,
    NoiseConfig,
    ObservationMatrixConfig,
    PhantomConfig,
    ProjectStore,
    RaytraceConfig,
    TomographyConfig,
    VesselConfig,
    collect_package_versions,
    default_cache_root,
    default_record_root,
)
from gpoloidal.run_layout import make_run_reference, prepare_local_run_layout, publish_latest_from_archive
from gpoloidal.tomography import GPT_lin_general, GPT_log_general


# %%
@dataclass
class RT1Config:
    # Human-facing name used under analysis_runs/<experiment_name>/...
    experiment_name: str = "external_rt1_loggp_lingp_benchmark"
    # TODO (required): path to inducing-point file generated beforehand.
    # This is the only value you almost certainly need to change first.
    point_file: str = "point_temp.npz"

    # Forward-model (ray integration) settings
    resolution: tuple[int, int] = (200, 200)
    lnum: int = 1001
    nreflections: int = 1

    # Synthetic benchmark settings
    phantom_name: str = "hollow"
    length_scale_factor: float = 2.0
    snr_rms_targets: tuple[float, ...] = (100, 30, 10, 3, 1)
    n_trials: int = 5
    seed: int = 42

    # logGP hyperparameters / optimization settings
    log_prior_mean: float = -3.0
    log_bound_value: float = -5.0
    bound_sig: float = 0.1
    max_log_iters: int = 30
    log_tol: float = 1e-5

    # RT1 camera defaults (same as benchmark script)
    focal_length: float = 0.01
    camera_location: tuple[float, float, float] = (1.2, 0.0, 0.0)
    center_angles: tuple[float, float] = (23, 0)
    sensor_size: tuple[float, float] = (0.0082, 0.0082)
    rotation: float = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="External RT1 linGP/logGP benchmark template")
    p.add_argument("--config", type=str, default=None, help="JSON/TOML/YAML config")
    p.add_argument("--quick", action="store_true", help="n_trials=1, max_log_iters=10")
    p.add_argument("--run-name", type=str, default=None, help="Suffix for archive directory")
    p.add_argument("--output-dir", type=str, default=None, help="Base dir for analysis_runs")
    p.add_argument("--backend-record-dir", type=str, default=None, help="Global run-record dir override")
    p.add_argument("--no-run-record", action="store_true")
    p.add_argument("--no-trials-csv", action="store_true")
    args, unknown = p.parse_known_args()  # Jupyter-safe
    if unknown:
        print("[info] ignored unknown args:", unknown)
    return args


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {path}")


def solve_lingp(H: np.ndarray, g_obs: np.ndarray, obs_noise_std: np.ndarray, K: np.ndarray, mu: np.ndarray) -> dict:
    # Thin wrapper around gpoloidal.tomography.GPT_lin_general
    # We convert absolute noise std vector -> (global level, relative profile).
    model = GPT_lin_general(H=H, Kf_pri=K, muf_pri=mu)
    noise_level = float(np.mean(obs_noise_std))
    noise_profile = np.asarray(obs_noise_std, dtype=float) / (noise_level + 1e-12)
    model.set_obs(g_obs=g_obs, obs_noise_profile=noise_profile, normalize=False, obs_noise_level=noise_level)
    return {"model": model, "f_mean": np.asarray(model.solve(), dtype=float), "f_std": np.asarray(model.f_std, dtype=float)}


def solve_loggp(
    H: np.ndarray,
    g_obs: np.ndarray,
    obs_noise_std: np.ndarray,
    K: np.ndarray,
    mu: np.ndarray,
    *,
    max_iters: int,
    tol: float,
    init: np.ndarray | None = None,
) -> dict:
    # Thin wrapper around gpoloidal.tomography.GPT_log_general
    # The Newton/Laplace loop itself stays in the script so users can inspect/tune it.
    model = GPT_log_general(H=H, Kf_pri=K, muf_pri=mu)
    noise_level = float(np.mean(obs_noise_std))
    noise_profile = np.asarray(obs_noise_std, dtype=float) / (noise_level + 1e-12)
    model.set_obs(g_obs=g_obs, obs_noise_profile=noise_profile, normalize=False, obs_noise_level=noise_level)
    f_latent = np.asarray(mu if init is None else init, dtype=float).copy()
    losses: list[float] = []
    for _ in range(max_iters):
        delta_f, loss = model.update(f_latent)
        f_latent = np.clip(f_latent + delta_f, -12.0, 8.0)
        losses.append(float(loss))
        if loss < tol:
            break
    model.postprocess(f_latent)
    return {
        "model": model,
        "f_latent": f_latent,
        "f_mean": np.asarray(model.expf_mean, dtype=float),
        "f_std": np.asarray(model.expf_std, dtype=float),
        "loss_history": losses,
        "converged": bool(losses and losses[-1] < tol),
    }


# %%
# ---- 0) config / paths -------------------------------------------------------
ARGS = parse_args()
CFG = RT1Config()
if ARGS.config:
    cfg_path = Path(ARGS.config).resolve()
    apply_flat_dataclass_config(CFG, load_config_mapping(cfg_path))
else:
    cfg_path = None
if ARGS.quick:
    CFG = replace(CFG, n_trials=1, max_log_iters=10)

# Storage policy (recommended):
# - cache_root         : global reusable cache (heavy artifacts)
# - backend_record_root: global run records (run_*.json)
# - output_root        : local human-facing output (archive/<timestamp...>)
cache_root = default_cache_root()
backend_record_root = (
    Path(ARGS.backend_record_dir).resolve()
    if ARGS.backend_record_dir
    else default_record_root() / "rt1_loggp_lingp_benchmark"
)
output_base = Path(ARGS.output_dir).resolve() if ARGS.output_dir else Path.cwd() / "analysis_runs"
layout = prepare_local_run_layout(
    base_dir=output_base,
    experiment_name=CFG.experiment_name,
    run_name=ARGS.run_name,
)
# `layout` gives:
# - archive/<timestamp[_name]>/...  (this run, immutable human-facing files)
# - latest/...                      (mirrored view of the newest run)
output_root = layout.run_root
figure_dir = output_root / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)

save_json(output_root / "config_resolved.json", asdict(CFG))
if cfg_path is not None:
    save_json(output_root / "config_source.json", {"config_path": str(cfg_path)})

point_file = Path(CFG.point_file).expanduser()
if not point_file.is_absolute():
    point_file = point_file.resolve()
if not point_file.exists():
    raise FileNotFoundError(
        f"Set RT1Config.point_file to your point_temp.npz path (current: {point_file})"
    )

print("gpoloidal", gpoloidal.__version__)
print("config:", CFG)
print("cache_root:", cache_root)
print("backend_record_root:", backend_record_root)
print("output_root (archive run):", output_root)
print("output_latest_root:", layout.latest_root)
print("point_file:", point_file)


# %%
# ---- 1) ProjectStore + inducing points cache --------------------------------
# ProjectStore is backend-only:
# - cache reuse
# - manifests
# - run record save/load
# It does NOT decide RT1-specific camera/ray settings; that stays in this script.
store = ProjectStore(cache_root=cache_root, record_root=backend_record_root)

inducing_cfg = InducingPointConfig(
    source=FileRef.from_path(point_file, note="external point_temp.npz"),
    length_sq_function=CallableRef.from_callable(rt1.phantom.Length_scale_sq),
    note="RT1 inducing points (external analysis)",
)

inducing_arrays, ind_rec = store.get_or_build_inducing_points(
    inducing_cfg,
    builder=lambda: {
        k: np.asarray(v)
        for k, v in np.load(point_file).items()
        if k in {"r_idc", "z_idc", "r_bd", "z_bd"}
    },
)
# If the same inducing-point config was used before, this loads from global cache.
# (You will see an explicit "[gpoloidal cache hit]" print.)


# %%
# ---- 2) RT1 setup (script-side; device-specific and readable) ----------------
# This section is intentionally "script-local":
# RT1 geometry / camera / ray settings are domain-specific and easier to maintain
# as a readable sequence than as a generic framework abstraction.
k = rt1.Kernel2D_scatter_rt1()
k.load_point(
    r_idc=inducing_arrays["r_idc"],
    z_idc=inducing_arrays["z_idc"],
    r_bd=inducing_arrays["r_bd"],
    z_bd=inducing_arrays["z_bd"],
    length_sq_fuction=rt1.phantom.Length_scale_sq,
    is_plot=False,
)
k.set_grid_interface(
    r_plot=np.linspace(0.05, 1.05, 501),
    z_plot=np.linspace(-0.7, 0.7, 501),
    add_bound=True,
)

camera = zray.measurement.Camera2D_rphiz(
    focal_length=CFG.focal_length,
    location=CFG.camera_location,
    center_angles=CFG.center_angles,
    sensor_size=CFG.sensor_size,
    resolution=CFG.resolution,
    rotation=CFG.rotation,
)
raymodel = zray.Raytracing(k.vessel, camera)
raymodel.main(nreflections=CFG.nreflections, pass_through_first=True)
ray = raymodel.rays[1]  # in-vessel segment for this benchmark setting


# %%
# ---- 3) Observation matrix cache --------------------------------------------
# H is expensive. ProjectStore reuses it when ObservationMatrixConfig hash matches.
obs_cfg = ObservationMatrixConfig(
    method="kernel_weighting",
    lnum=CFG.lnum,
    vessel=VesselConfig(package_resource="gpoloidal.rt1:rt1_simple_frame.json"),
    camera=CameraConfig(
        kind="Camera2D_rphiz",
        params={
            "focal_length": CFG.focal_length,
            "location": list(CFG.camera_location),
            "center_angles": list(CFG.center_angles),
            "sensor_size": list(CFG.sensor_size),
            "resolution": list(CFG.resolution),
            "rotation": CFG.rotation,
        },
    ),
    raytrace=RaytraceConfig(
        nreflections=CFG.nreflections,
        pass_through_first=True,
        ray_index_for_integral=1,
    ),
    inducing_points=inducing_cfg,
    package_versions=collect_package_versions(["gpoloidal", "zray", "numpy", "scipy"]),
)
H, obsmat_rec = store.get_or_build_observation_matrix(
    obs_cfg,
    builder=lambda: np.asarray(k.create_obs_matrix_kernel_weighting(ray=ray, Lnum=CFG.lnum), dtype=float),
)
H = np.asarray(H, dtype=float)


# %%
# ---- 4) Phantom / priors / forward ------------------------------------------
# Phantom and prior settings are "experiment conditions".
# Changing them usually does NOT require recomputing H (unless obs config also changes).
phantom_fn = rt1.phantom.get_phantom_funtion(CFG.phantom_name)
f_true = np.clip(np.asarray(phantom_fn(k.r_idc, k.z_idc), dtype=float), 0.0, None)
g_true = H @ f_true

K_lin_pri, mu_lin_pri = k.set_kernel(
    length_scale_factor=CFG.length_scale_factor,
    is_bound=True,
    bound_value=0.0,
    bound_sig=CFG.bound_sig,
    mean=0.0,
)
K_log_pri, mu_log_pri = k.set_kernel(
    length_scale_factor=CFG.length_scale_factor,
    is_bound=True,
    bound_value=CFG.log_bound_value,
    bound_sig=CFG.bound_sig,
    mean=CFG.log_prior_mean,
)
import json

print(
    json.dumps(
        {
            "nI": int(k.nI),
            "nb": int(k.nb),
            "H_shape": list(H.shape),
            "ray_count": int(len(raymodel.rays)),
            "inducing_artifact": ind_rec.artifact_id,
            "obsmat_artifact": obsmat_rec.artifact_id,
        },
        indent=2,
    )
)


# %%
# ---- 5) Truth / forward plots ------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
axs[0].imshow(k.convert_grid(f_true, boundary=0.0) * k.mask, **k.im_kwargs, cmap="turbo")
k.plt_rt1_flux(ax=axs[0], linewidths=0.7)
axs[0].set_title("Truth (inducing -> grid)")
axs[1].imshow(g_true.reshape(CFG.resolution), origin="lower", cmap="magma")
axs[1].set_title("Forward image g_true")
truth_plot = figure_dir / "rt1_truth_and_forward.png"
save_figure(fig, truth_plot)


# %%
# ---- 6) Noise sweep ----------------------------------------------------------
# The loop below is written sequentially (not overly abstracted) so users can:
# - inspect intermediate values
# - insert custom diagnostics
# - change retry / stopping policy for logGP
rng_master = np.random.default_rng(CFG.seed)
g_mean = float(np.mean(g_true))
g_rms = float(np.sqrt(np.mean(g_true**2)))
rows: list[dict] = []
examples: dict[float, dict] = {}

for snr_rms_target in CFG.snr_rms_targets:
    obs_noise_level = float(g_rms / snr_rms_target)
    obs_noise_std = np.full_like(g_true, obs_noise_level, dtype=float)

    for trial in range(CFG.n_trials):
        seed_i = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(seed_i)
        g_obs = g_true + obs_noise_std * rng.standard_normal(g_true.size)

        lin = solve_lingp(H, g_obs, obs_noise_std, K_lin_pri, mu_lin_pri)
        log = solve_loggp(H, g_obs, obs_noise_std, K_log_pri, mu_log_pri, max_iters=CFG.max_log_iters, tol=CFG.log_tol)
        retried = False
        if not log["converged"]:
            # Practical safeguard:
            # If quick mode / strict iteration limit causes non-convergence,
            # retry with a larger iteration budget to avoid misleading summary plots.
            log = solve_loggp(
                H, g_obs, obs_noise_std, K_log_pri, mu_log_pri,
                max_iters=max(CFG.max_log_iters * 3, 30), tol=CFG.log_tol, init=log["f_latent"]
            )
            retried = True

        row = {
            "snr_rms_target": float(snr_rms_target),
            "trial": trial,
            "seed": seed_i,
            "obs_noise_level": obs_noise_level,
            "snr_mean": float(g_mean / (obs_noise_level + 1e-12)),
            "snr_rms": float(g_rms / (obs_noise_level + 1e-12)),
            "log_iters": len(log["loss_history"]),
            "log_last_loss": float(log["loss_history"][-1]) if log["loss_history"] else np.nan,
            "log_converged": bool(log["converged"]),
            "log_retried": bool(retried),
        }
        row |= field_metrics(lin["f_mean"], f_true, prefix="lin")
        row |= field_metrics(log["f_mean"], f_true, prefix="log")
        row["lin_chi2"] = mean_chi2(H @ lin["f_mean"], g_obs, obs_noise_std)
        row["log_chi2"] = mean_chi2(H @ log["f_mean"], g_obs, obs_noise_std)
        rows.append(row)
        if trial == 0:
            examples[float(snr_rms_target)] = {"g_obs": g_obs, "lin": lin, "log": log}

results_df = pd.DataFrame.from_records(rows).sort_values(
    ["snr_rms_target", "trial"], ascending=[False, True]
).reset_index(drop=True)
summary = summarize_noise_sweep(results_df)
print(summary[["lin_rmse_mean", "log_rmse_mean", "lin_chi2_mean", "log_chi2_mean"]])


# %%
# ---- 7) Save tables + plots --------------------------------------------------
summary_csv = figure_dir / "rt1_noise_sweep_summary.csv"
summary.to_csv(summary_csv)
trials_csv = None
if not ARGS.no_trials_csv:
    trials_csv = figure_dir / "rt1_noise_sweep_trials.csv"
    results_df.to_csv(trials_csv, index=False)

x = summary.index.to_numpy(dtype=float)
fig, axs = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
axs[0].errorbar(x, summary["lin_rmse_mean"], yerr=summary["lin_rmse_std"].fillna(0), marker="o", label="linGP")
axs[0].errorbar(x, summary["log_rmse_mean"], yerr=summary["log_rmse_std"].fillna(0), marker="o", label="logGP")
axs[0].set_xscale("log"); axs[0].invert_xaxis(); axs[0].grid(alpha=0.3); axs[0].legend(); axs[0].set_title("Field RMSE")
axs[1].plot(x, summary["lin_chi2_mean"], marker="o", label="linGP")
axs[1].plot(x, summary["log_chi2_mean"], marker="o", label="logGP")
axs[1].axhline(1.0, color="k", ls="--", lw=1, alpha=0.6)
axs[1].set_xscale("log"); axs[1].invert_xaxis(); axs[1].grid(alpha=0.3); axs[1].legend(fontsize=8); axs[1].set_title("chi2")
ax = axs[2]
ax.plot(x, summary["lin_neg_frac_mean"], marker="o", color="tab:red")
ax.set_xscale("log"); ax.invert_xaxis(); ax.grid(alpha=0.3); ax.set_title("Positivity / convergence")
ax2 = ax.twinx()
ax2.plot(x, summary["log_iters_mean"], marker="s", color="tab:blue")
summary_plot = figure_dir / "rt1_noise_sweep_summary.png"
save_figure(fig, summary_plot)

snr_example = 3.0 if 3.0 in examples else float(sorted(examples.keys())[0])
ex = examples[snr_example]
fig, axs = plt.subplots(2, 3, figsize=(12, 6.4), constrained_layout=True)
panels = [
    (k.convert_grid(f_true, boundary=0.0) * k.mask, "Truth", "turbo", None),
    (k.convert_grid(ex["lin"]["f_mean"], boundary=0.0) * k.mask, f"linGP (SNR_rms={snr_example:g})", "turbo", None),
    (k.convert_grid(ex["log"]["f_mean"], boundary=0.0) * k.mask, f"logGP (SNR_rms={snr_example:g})", "turbo", None),
    (None, "", "gray", None),
    (k.convert_grid(ex["lin"]["f_mean"] - f_true, boundary=0.0) * k.mask, "linGP error", "RdBu_r", 0.4),
    (k.convert_grid(ex["log"]["f_mean"] - f_true, boundary=0.0) * k.mask, "logGP error", "RdBu_r", 0.4),
]
for ax, (im, title, cmap, vmax) in zip(axs.flat, panels):
    if title == "":
        ax.axis("off")
        continue
    if vmax is None:
        ax.imshow(im, **k.im_kwargs, cmap=cmap, vmin=0, vmax=1)
    else:
        ax.imshow(im, **k.im_kwargs, cmap=cmap, vmin=-vmax, vmax=vmax)
    k.plt_rt1_flux(ax=ax, linewidths=0.6)
    ax.set_title(title)
recon_plot = figure_dir / "rt1_reconstruction_panels.png"
save_figure(fig, recon_plot)


# %%
# ---- 8) Global run record (optional) + local reports -------------------------
# Two layers are intentionally separated:
# 1) Global run record (machine-readable / reproducibility)
# 2) Local analysis output (human-facing plots/tables)
run_id = None
if not ARGS.no_run_record:
    record = ExperimentRecord(
        name=CFG.experiment_name,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        observation_matrix_artifact_id=obsmat_rec.artifact_id,
        observation_matrix_config=obs_cfg,
        phantom=PhantomConfig(kind="rt1_synthetic", name=CFG.phantom_name),
        noise=NoiseConfig(
            model="gaussian",
            level=None,
            level_definition="snr_rms_targets",
            profile="flat",
            seed=CFG.seed,
            params={"snr_rms_targets": list(CFG.snr_rms_targets), "n_trials": CFG.n_trials},
        ),
        tomography=TomographyConfig(
            model="linGP_vs_logGP",
            prior_kind="kernel.set_kernel",
            length_scale_factor=CFG.length_scale_factor,
            boundary_sigma=CFG.bound_sig,
            boundary_value=CFG.log_bound_value,
            prior_mean=CFG.log_prior_mean,
            normalize=False,
            obs_noise_level=None,
            max_iters=CFG.max_log_iters,
            tol=CFG.log_tol,
            extras={"lin_bound_value": 0.0},
        ),
        references={
            "inducing_points_artifact_id": ind_rec.artifact_id,
            "point_file": str(point_file),
            "ray_index_for_integral": 1,
        },
        metrics={
            str(snr): {
                "lin_rmse_mean": float(row["lin_rmse_mean"]),
                "log_rmse_mean": float(row["log_rmse_mean"]),
                "lin_chi2_mean": float(row["lin_chi2_mean"]),
                "log_chi2_mean": float(row["log_chi2_mean"]),
                "lin_neg_frac_mean": float(row["lin_neg_frac_mean"]),
                "log_iters_mean": float(row["log_iters_mean"]),
                "log_nonconverged_frac": float(row["log_nonconverged_frac"]),
                "log_retry_frac": float(row["log_retry_frac"]),
            }
            for snr, row in summary.iterrows()
        },
        outputs={},
    )
    run_id = store.save_experiment_record(record, strict_traceability=False, embed_dependency_manifests=False)

report = {
    "script": "external_rt1_loggp_lingp_benchmark_template.py",
    "gpoloidal_version": gpoloidal.__version__,
    "config": asdict(CFG),
    "paths": {
        "output_root": str(output_root),
        "output_latest_root": str(layout.latest_root),
        "output_archive_root": str(layout.archive_root),
        "backend_record_root": str(backend_record_root),
        "cache_root": str(cache_root),
        "run_ref": str(output_root / "run_ref.json"),
        "truth_plot": str(truth_plot),
        "summary_plot": str(summary_plot),
        "summary_csv": str(summary_csv),
        "reconstruction_plot": str(recon_plot),
        **({"trials_csv": str(trials_csv)} if trials_csv else {}),
        **({"run_record": str(store.run_dir / f'{run_id}.json')} if run_id else {}),
    },
}
save_json(output_root / "latest_report.json", report)
save_json(output_root / "latest_paths.json", report["paths"])
save_json(
    output_root / "run_ref.json",
    make_run_reference(
        script="external_rt1_loggp_lingp_benchmark_template.py",
        archive_run_root=output_root,
        latest_root=layout.latest_root,
        backend_record_root=backend_record_root,
        run_id=run_id,
        backend_run_record_path=(store.run_dir / f"{run_id}.json") if run_id else None,
        extra={"observation_matrix_artifact_id": obsmat_rec.artifact_id, "inducing_points_artifact_id": ind_rec.artifact_id},
    ),
)
# Mirror archive/<this run>/... -> latest/... (overwrite latest only)
publish_latest_from_archive(layout)

print(
    {
        "run_id": run_id,
        "archive_run_root": str(output_root),
        "latest_root": str(layout.latest_root),
    }
)

# %%
