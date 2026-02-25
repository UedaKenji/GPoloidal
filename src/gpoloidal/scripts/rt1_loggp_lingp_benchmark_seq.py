from __future__ import annotations

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
import gpoloidal.rt1 as rt1
import zray
from gpoloidal.benchmark_utils import (
    apply_flat_dataclass_config,
    field_metrics,
    load_config_mapping,
    mean_chi2,
    save_json,
    summarize_noise_sweep,
)
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
from gpoloidal.run_layout import prepare_local_run_layout, publish_latest_from_archive
from gpoloidal.tomography import GPT_lin_general, GPT_log_general

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# %%
@dataclass
class BenchmarkConfig:
    point_file: str = "example/rt1tomography/point_temp.npz"
    resolution: tuple[int, int] = (48, 48)
    lnum: int = 1001
    nreflections: int = 1
    phantom_name: str = "hollow"
    length_scale_factor: float = 2.0
    snr_rms_targets: tuple[float, ...] = (100, 30, 10, 3, 1)
    n_trials: int = 5
    seed: int = 42
    log_prior_mean: float = -3.0
    log_bound_value: float = -5.0
    bound_sig: float = 0.1
    max_log_iters: int = 30
    log_tol: float = 1e-5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RT1 benchmark (sequential style): linGP vs logGP")
    p.add_argument("--config", type=str, default=None, help="Path to JSON/TOML config file")
    p.add_argument("--quick", action="store_true", help="n_trials=1, max_log_iters=10")
    p.add_argument("--output-dir", type=str, default=None, help="Base directory for analysis_runs-style outputs")
    p.add_argument("--run-name", type=str, default=None, help="Optional suffix for archive run directory")
    p.add_argument("--backend-record-dir", type=str, default=None)
    p.add_argument("--no-run-record", action="store_true")
    p.add_argument("--no-trials-csv", action="store_true")
    args, unknown = p.parse_known_args()
    if unknown:
        print("[info] ignored unknown args:", unknown)
    return args


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {path}")


def solve_lingp(H: np.ndarray, g_obs: np.ndarray, obs_noise_std: np.ndarray, K: np.ndarray, mu: np.ndarray) -> dict:
    model = GPT_lin_general(H=H, Kf_pri=K, muf_pri=mu)
    noise_level = float(np.mean(obs_noise_std))
    noise_profile = np.asarray(obs_noise_std, dtype=float) / (noise_level + 1e-12)
    model.set_obs(g_obs=g_obs, obs_noise_profile=noise_profile, normalize=False, obs_noise_level=noise_level)
    f_mean = np.asarray(model.solve(), dtype=float)
    return {"model": model, "f_mean": f_mean, "f_std": np.asarray(model.f_std, dtype=float)}


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
    model = GPT_log_general(H=H, Kf_pri=K, muf_pri=mu)
    noise_level = float(np.mean(obs_noise_std))
    noise_profile = np.asarray(obs_noise_std, dtype=float) / (noise_level + 1e-12)
    model.set_obs(g_obs=g_obs, obs_noise_profile=noise_profile, normalize=False, obs_noise_level=noise_level)

    f_latent = np.asarray(mu if init is None else init, dtype=float).copy()
    loss_history: list[float] = []
    for _ in range(max_iters):
        delta_f, loss = model.update(f_latent)
        f_latent = np.clip(f_latent + delta_f, -12.0, 8.0)
        loss_history.append(float(loss))
        if loss < tol:
            break
    model.postprocess(f_latent)
    return {
        "model": model,
        "f_latent": f_latent,
        "f_mean": np.asarray(model.expf_mean, dtype=float),
        "f_std": np.asarray(model.expf_std, dtype=float),
        "loss_history": loss_history,
        "converged": bool(loss_history and loss_history[-1] < tol),
    }


# %%
# ---- 0) CLI / config ---------------------------------------------------------
ARGS = parse_args()
CFG = BenchmarkConfig()
CONFIG_PATH = ((PROJECT_ROOT / ARGS.config).resolve() if ARGS.config and not Path(ARGS.config).is_absolute() else (Path(ARGS.config).resolve() if ARGS.config else None))
if CONFIG_PATH is not None:
    cfg_updates = load_config_mapping(CONFIG_PATH)
    apply_flat_dataclass_config(CFG, cfg_updates)
if ARGS.quick:
    CFG = replace(CFG, n_trials=1, max_log_iters=10)

cache_root = default_cache_root() / "rt1_loggp_lingp_benchmark"
backend_record_root = (
    Path(ARGS.backend_record_dir)
    if ARGS.backend_record_dir
    else default_record_root() / "rt1_loggp_lingp_benchmark"
)
output_base_dir = (
    Path(ARGS.output_dir)
    if ARGS.output_dir
    else Path.cwd() / "analysis_runs"
)
layout = prepare_local_run_layout(
    base_dir=output_base_dir,
    experiment_name="rt1_loggp_lingp_benchmark_seq",
    run_name=ARGS.run_name,
)
output_root = layout.run_root
figure_dir = output_root / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)

print("gpoloidal", gpoloidal.__version__)
print("config:", CFG)
if CONFIG_PATH is not None:
    print("config_path:", CONFIG_PATH)
print("cache_root:", cache_root)
print("backend_record_root:", backend_record_root)
print("output_root (archive run):", output_root)
print("output_latest_root:", layout.latest_root)
save_json(output_root / "config_resolved.json", asdict(CFG))
if CONFIG_PATH is not None:
    save_json(
        output_root / "config_source.json",
        {"config_path": str(CONFIG_PATH), "loaded": load_config_mapping(CONFIG_PATH)},
    )


# %%
# ---- 1) Store / cache-backed inputs -----------------------------------------
store = ProjectStore(cache_root=cache_root, record_root=backend_record_root)
point_file = Path(CFG.point_file)
if not point_file.is_absolute():
    point_file = (PROJECT_ROOT / point_file).resolve()
assert point_file.exists(), f"Missing point file: {point_file}"

inducing_cfg = InducingPointConfig(
    source=FileRef.from_path(point_file, note="example/rt1tomography/create_inducing_point.py output"),
    length_sq_function=CallableRef.from_callable(rt1.phantom.Length_scale_sq),
    note="full inducing points from point_temp.npz",
)

inducing_arrays, ind_rec = store.get_or_build_inducing_points(
    inducing_cfg,
    builder=lambda: {
        k: np.asarray(v)
        for k, v in np.load(point_file).items()
        if k in {"r_idc", "z_idc", "r_bd", "z_bd"}
    },
)

obs_cfg = ObservationMatrixConfig(
    method="kernel_weighting",
    lnum=CFG.lnum,
    vessel=VesselConfig(package_resource="gpoloidal.rt1:rt1_simple_frame.json"),
    camera=CameraConfig(
        kind="Camera2D_rphiz",
        params={
            "focal_length": 0.01,
            "location": [1.2, 0.0, 0.0],
            "center_angles": [23, 0],
            "sensor_size": [0.0082, 0.0082],
            "resolution": list(CFG.resolution),
            "rotation": 0.0,
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


# %%
# ---- 2) Build RT1 kernel / ray model (RT1-specific; intentionally script-side)
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
    focal_length=0.01,
    location=(1.2, 0.0, 0.0),
    center_angles=(23, 0),
    sensor_size=(0.0082, 0.0082),
    resolution=CFG.resolution,
    rotation=0.0,
)
raymodel = zray.Raytracing(k.vessel, camera)
raymodel.main(nreflections=CFG.nreflections, pass_through_first=True)
ray = raymodel.rays[1]


# %%
# ---- 3) Observation matrix cache (same config => cache hit) ------------------
H, obsmat_rec = store.get_or_build_observation_matrix(
    obs_cfg,
    builder=lambda: np.asarray(k.create_obs_matrix_kernel_weighting(ray=ray, Lnum=CFG.lnum), dtype=float),
)
H = np.asarray(H, dtype=float)

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
# ---- 4) Phantom and priors ---------------------------------------------------
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


# %%
# ---- 5) Truth / forward visualization ---------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
axs[0].imshow(k.convert_grid(f_true, boundary=0.0) * k.mask, **k.im_kwargs, cmap="turbo")
k.plt_rt1_flux(ax=axs[0], linewidths=0.7)
axs[0].set_title("Truth (inducing -> grid)")
axs[1].imshow(g_true.reshape(CFG.resolution), origin="lower", cmap="magma")
axs[1].set_title("Forward image g_true")
truth_plot = figure_dir / "rt1_truth_and_forward.png"
save_figure(fig, truth_plot)


# %%
# ---- 6) Noise sweep (sequential loop) ---------------------------------------
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
        log = solve_loggp(
            H, g_obs, obs_noise_std, K_log_pri, mu_log_pri,
            max_iters=CFG.max_log_iters, tol=CFG.log_tol,
        )
        log_retried = False
        if not log["converged"]:
            log = solve_loggp(
                H, g_obs, obs_noise_std, K_log_pri, mu_log_pri,
                max_iters=max(CFG.max_log_iters * 3, 30),
                tol=CFG.log_tol,
                init=log["f_latent"],
            )
            log_retried = True

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
            "log_retried": bool(log_retried),
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
if (summary["log_retry_frac"] > 0).any():
    print("[info] some logGP trials required retry.")
if (summary["log_nonconverged_frac"] > 0).any():
    print("[warn] some logGP trials remained non-converged.")


# %%
# ---- 7) Save tables ----------------------------------------------------------
summary_csv = figure_dir / "rt1_noise_sweep_summary.csv"
summary.to_csv(summary_csv)
trials_csv = None
if not ARGS.no_trials_csv:
    trials_csv = figure_dir / "rt1_noise_sweep_trials.csv"
    results_df.to_csv(trials_csv, index=False)
print("[table]", summary_csv)
if trials_csv is not None:
    print("[table]", trials_csv)


# %%
# ---- 8) Summary plot ---------------------------------------------------------
x = summary.index.to_numpy(dtype=float)
fig, axs = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)

axs[0].errorbar(x, summary["lin_rmse_mean"], yerr=summary["lin_rmse_std"].fillna(0), marker="o", label="linGP")
axs[0].errorbar(x, summary["log_rmse_mean"], yerr=summary["log_rmse_std"].fillna(0), marker="o", label="logGP")
axs[0].set_xscale("log")
axs[0].invert_xaxis()
axs[0].set_title("Field RMSE")
axs[0].set_xlabel("SNR_rms target")
axs[0].grid(alpha=0.3)
axs[0].legend()

axs[1].plot(x, summary["lin_chi2_mean"], marker="o", label="linGP")
axs[1].plot(x, summary["log_chi2_mean"], marker="o", label="logGP")
axs[1].axhline(1.0, color="k", lw=1, ls="--", alpha=0.6)
axs[1].set_xscale("log")
axs[1].invert_xaxis()
axs[1].set_title("chi2")
axs[1].set_xlabel("SNR_rms target")
axs[1].grid(alpha=0.3)
axs[1].legend(fontsize=8)

ax = axs[2]
ax.plot(x, summary["lin_neg_frac_mean"], marker="o", color="tab:red")
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("SNR_rms target")
ax.set_ylabel("lin negative frac", color="tab:red")
ax.tick_params(axis="y", labelcolor="tab:red")
ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.plot(x, summary["log_iters_mean"], marker="s", color="tab:blue")
ax2.set_ylabel("log iterations", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")
ax.set_title("Positivity / convergence")

summary_plot = figure_dir / "rt1_noise_sweep_summary.png"
save_figure(fig, summary_plot)


# %%
# ---- 9) Reconstruction panel -------------------------------------------------
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
# ---- 10) Save record (backend) + human report --------------------------------
run_id = None
if not ARGS.no_run_record:
    record = ExperimentRecord(
        name="rt1_loggp_lingp_benchmark_seq",
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
    run_id = store.save_experiment_record(
        record,
        strict_traceability=False,
        embed_dependency_manifests=False,
    )

report = {
    "script": "gpoloidal.scripts.rt1_loggp_lingp_benchmark_seq",
    "gpoloidal_version": gpoloidal.__version__,
    "config_path": str(CONFIG_PATH) if CONFIG_PATH is not None else None,
    "config": asdict(CFG),
    "paths": {
        "output_root": str(output_root),
        "output_latest_root": str(layout.latest_root),
        "output_archive_root": str(layout.archive_root),
        "backend_record_root": str(backend_record_root),
        "cache_root": str(cache_root),
        "config_resolved": str(output_root / "config_resolved.json"),
        "truth_plot": str(truth_plot),
        "summary_plot": str(summary_plot),
        "summary_csv": str(summary_csv),
        "reconstruction_plot": str(recon_plot),
        **({"trials_csv": str(trials_csv)} if trials_csv else {}),
        **({"config_source": str(output_root / "config_source.json")} if CONFIG_PATH is not None else {}),
        **({"run_record": str(store.run_dir / f'{run_id}.json')} if run_id else {}),
    },
    "summary_metrics": {
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
}
save_json(output_root / "latest_report.json", report)
save_json(output_root / "latest_paths.json", report["paths"])
publish_latest_from_archive(layout)

print(
    json.dumps(
        {
            "run_id": run_id,
            "archive_run_root": str(output_root),
            "latest_root": str(layout.latest_root),
            "summary_rows": int(summary.shape[0]),
        },
        indent=2,
    )
)
