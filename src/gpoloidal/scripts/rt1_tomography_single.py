from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gpoloidal
import gpoloidal.kernel as kernel_mod
import gpoloidal.rt1 as rt1
import zray
from gpoloidal.analysis.config import apply_flat_dataclass_config, load_config_mapping, save_json
from gpoloidal.analysis.rt1_plots import (
    plot_loggp_loss_history,
    plot_rt1_reconstruction_panels,
    plot_rt1_truth_and_observation,
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
)
from gpoloidal.run_layout import make_run_reference, prepare_local_run_layout, publish_latest_from_archive
from gpoloidal.script_cli import add_common_runtime_args, parse_known_args, resolve_record_mode_policy, resolve_runtime_roots
from gpoloidal.tomography import GPT_lin_general, GPT_log_general

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class SingleTomographyConfig:
    experiment_name: str = "rt1_tomography_single"

    # forward / basis
    basis_mode: Literal["inducing", "grid"] = "inducing"
    observation_matrix_mode: Literal["auto", "kernel_weighting", "grid_binning"] = "auto"
    point_file: str = "example/rt1tomography/point_temp.npz"  # used in inducing mode

    # grid basis (used when basis_mode="grid")
    grid_r_min: float = 0.20
    grid_r_max: float = 1.01
    grid_r_step: float = 0.01
    grid_z_min: float = -0.60
    grid_z_max: float = 0.60
    grid_z_step: float = 0.01
    grid_inducing_length_scale: float = 0.015
    grid_inside_value: int = 2
    grid_boundary_value: int = 1
    grid_obs_sample_count: int = 400

    # ray/camera / vessel
    vessel_package_resource: str = "gpoloidal.rt1:rt1_simple_frame.json"
    camera_kind: str = "Camera2D_rphiz"
    camera_focal_length: float = 0.01
    camera_location: tuple[float, float, float] = (1.2, 0.0, 0.0)
    camera_center_angles: tuple[float, float] = (23.0, 0.0)
    camera_sensor_size: tuple[float, float] = (0.0082, 0.0082)
    camera_rotation: float = 0.0
    resolution: tuple[int, int] = (48, 48)
    lnum: int = 1001  # for kernel_weighting
    nreflections: int = 1
    pass_through_first: bool = True
    ray_index_for_integral: int = 1

    # phantom
    phantom_name: str = "hollow"
    phantom_params: dict[str, Any] = field(default_factory=dict)

    # tomography
    tomography_method: Literal["linGP", "logGP"] = "logGP"
    prior_mode: Literal["auto", "gibbs", "uniform_se"] = "auto"
    length_scale_factor: float = 2.0  # for gibbs prior (scatter set_kernel)
    uniform_length_scale: float = 0.05  # for uniform SE prior
    bound_sig: float = 0.1
    lin_bound_value: float = 0.0
    lin_prior_mean: float = 0.0
    log_bound_value: float = -5.0
    log_prior_mean: float = -3.0
    max_log_iters: int = 30
    log_tol: float = 1e-5
    log_latent_clip_min: float = -12.0
    log_latent_clip_max: float = 8.0
    log_retry_multiplier: int = 3
    log_retry_min_iters: int = 30

    # noise (single condition)
    noise_mode: Literal["snr_rms", "absolute"] = "snr_rms"
    snr_rms: float = 3.0
    obs_noise_level: float = 0.05
    seed: int = 42

    # plotting conversion grid
    plot_r_min: float = 0.05
    plot_r_max: float = 1.05
    plot_r_num: int = 501
    plot_z_min: float = -0.7
    plot_z_max: float = 0.7
    plot_z_num: int = 501
    plot_figure_dpi: int = 150
    plot_value_vmin: float = 0.0
    plot_value_vmax: float = 1.0
    plot_error_percentile: float = 99.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RT1 single-condition tomography runner (linGP or logGP)")
    add_common_runtime_args(p, include_config=True, include_quick=True, include_trials_csv_toggle=False)
    return parse_known_args(p)


def _flatten_single_tomography_config_mapping(data: dict[str, Any]) -> dict[str, Any]:
    """Accept a flat mapping and/or a shallow nested mapping and return flat dataclass keys.

    Supported top-level sections:
      - experiment
      - basis
      - grid
      - observation
      - phantom
      - tomography
      - noise
      - plot
    Flat keys remain supported for backward compatibility.
    """
    if not isinstance(data, dict):
        raise TypeError("config root must be a mapping")

    out: dict[str, Any] = {}
    section_names = {"experiment", "basis", "grid", "observation", "phantom", "tomography", "noise", "plot"}

    # Keep flat keys as-is (unknown keys are validated later by apply_flat_dataclass_config).
    for k, v in data.items():
        if k not in section_names:
            out[k] = v

    def require_mapping(section_name: str) -> dict[str, Any]:
        sec = data.get(section_name, None)
        if sec is None:
            return {}
        if not isinstance(sec, dict):
            raise ValueError(f"Config section '{section_name}' must be a mapping")
        return sec

    experiment = require_mapping("experiment")
    for k, v in experiment.items():
        if k == "name":
            out["experiment_name"] = v
        else:
            raise ValueError(f"Unknown config key: experiment.{k}")

    basis = require_mapping("basis")
    for k, v in basis.items():
        if k == "mode":
            out["basis_mode"] = v
        elif k == "point_file":
            out["point_file"] = v
        elif k in {"observation_matrix_mode", "matrix_mode"}:
            out["observation_matrix_mode"] = v
        else:
            raise ValueError(f"Unknown config key: basis.{k}")

    grid = require_mapping("grid")
    for k, v in grid.items():
        mapping = {
            "r_min": "grid_r_min",
            "r_max": "grid_r_max",
            "r_step": "grid_r_step",
            "z_min": "grid_z_min",
            "z_max": "grid_z_max",
            "z_step": "grid_z_step",
            "inducing_length_scale": "grid_inducing_length_scale",
            "inside_value": "grid_inside_value",
            "boundary_value": "grid_boundary_value",
            "obs_sample_count": "grid_obs_sample_count",
        }
        if k in mapping:
            out[mapping[k]] = v
        else:
            raise ValueError(f"Unknown config key: grid.{k}")

    observation = require_mapping("observation")
    for k, v in observation.items():
        mapping = {
            "mode": "observation_matrix_mode",
            "matrix_mode": "observation_matrix_mode",
            "vessel_package_resource": "vessel_package_resource",
            "camera_kind": "camera_kind",
            "camera_focal_length": "camera_focal_length",
            "camera_location": "camera_location",
            "camera_center_angles": "camera_center_angles",
            "camera_sensor_size": "camera_sensor_size",
            "camera_rotation": "camera_rotation",
            "resolution": "resolution",
            "lnum": "lnum",
            "nreflections": "nreflections",
            "pass_through_first": "pass_through_first",
            "ray_index_for_integral": "ray_index_for_integral",
        }
        if k in mapping:
            out[mapping[k]] = v
        else:
            raise ValueError(f"Unknown config key: observation.{k}")

    phantom = require_mapping("phantom")
    for k, v in phantom.items():
        if k == "name":
            out["phantom_name"] = v
        elif k == "params":
            out["phantom_params"] = v
        else:
            raise ValueError(f"Unknown config key: phantom.{k}")

    tomography = require_mapping("tomography")
    for k, v in tomography.items():
        mapping = {
            "method": "tomography_method",
            "prior_mode": "prior_mode",
            "length_scale_factor": "length_scale_factor",
            "uniform_length_scale": "uniform_length_scale",
            "bound_sig": "bound_sig",
            "lin_bound_value": "lin_bound_value",
            "lin_prior_mean": "lin_prior_mean",
            "log_bound_value": "log_bound_value",
            "log_prior_mean": "log_prior_mean",
            "max_log_iters": "max_log_iters",
            "log_tol": "log_tol",
            "log_latent_clip_min": "log_latent_clip_min",
            "log_latent_clip_max": "log_latent_clip_max",
            "log_retry_multiplier": "log_retry_multiplier",
            "log_retry_min_iters": "log_retry_min_iters",
        }
        if k in mapping:
            out[mapping[k]] = v
        else:
            raise ValueError(f"Unknown config key: tomography.{k}")

    noise = require_mapping("noise")
    for k, v in noise.items():
        mapping = {
            "mode": "noise_mode",
            "snr_rms": "snr_rms",
            "obs_noise_level": "obs_noise_level",
            "seed": "seed",
        }
        if k in mapping:
            out[mapping[k]] = v
        else:
            raise ValueError(f"Unknown config key: noise.{k}")

    plot = require_mapping("plot")
    for k, v in plot.items():
        mapping = {
            "r_min": "plot_r_min",
            "r_max": "plot_r_max",
            "r_num": "plot_r_num",
            "z_min": "plot_z_min",
            "z_max": "plot_z_max",
            "z_num": "plot_z_num",
            "figure_dpi": "plot_figure_dpi",
            "value_vmin": "plot_value_vmin",
            "value_vmax": "plot_value_vmax",
            "error_percentile": "plot_error_percentile",
        }
        if k in mapping:
            out[mapping[k]] = v
        else:
            raise ValueError(f"Unknown config key: plot.{k}")

    return out


def save_figure(fig: plt.Figure, path: Path, *, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {path}")


def _resolve_obs_mode(cfg: SingleTomographyConfig) -> str:
    if cfg.observation_matrix_mode != "auto":
        return cfg.observation_matrix_mode
    return "kernel_weighting" if cfg.basis_mode == "inducing" else "grid_binning"


def _resolve_prior_mode(cfg: SingleTomographyConfig) -> str:
    if cfg.prior_mode != "auto":
        return cfg.prior_mode
    return "gibbs" if cfg.basis_mode == "inducing" else "uniform_se"


def _camera_params_from_cfg(cfg: SingleTomographyConfig) -> dict[str, Any]:
    return {
        "focal_length": float(cfg.camera_focal_length),
        "location": [float(v) for v in cfg.camera_location],
        "center_angles": [float(v) for v in cfg.camera_center_angles],
        "sensor_size": [float(v) for v in cfg.camera_sensor_size],
        "resolution": list(cfg.resolution),
        "rotation": float(cfg.camera_rotation),
    }


def solve_lingp(H: np.ndarray, g_obs: np.ndarray, obs_noise_std: np.ndarray, K: np.ndarray, mu: np.ndarray) -> dict[str, Any]:
    model = GPT_lin_general(H=H, Kf_pri=K, muf_pri=mu)
    noise_level = float(np.mean(obs_noise_std))
    noise_profile = np.asarray(obs_noise_std, dtype=float) / (noise_level + 1e-12)
    model.set_obs(g_obs=g_obs, obs_noise_profile=noise_profile, normalize=False, obs_noise_level=noise_level)
    f_mean = np.asarray(model.solve(), dtype=float)
    return {"model": model, "f_mean": f_mean, "f_std": np.asarray(model.f_std, dtype=float), "converged": True}


def solve_loggp(
    H: np.ndarray,
    g_obs: np.ndarray,
    obs_noise_std: np.ndarray,
    K: np.ndarray,
    mu: np.ndarray,
    *,
    max_iters: int,
    tol: float,
    latent_clip_min: float,
    latent_clip_max: float,
    init: np.ndarray | None = None,
) -> dict[str, Any]:
    model = GPT_log_general(H=H, Kf_pri=K, muf_pri=mu)
    noise_level = float(np.mean(obs_noise_std))
    noise_profile = np.asarray(obs_noise_std, dtype=float) / (noise_level + 1e-12)
    model.set_obs(g_obs=g_obs, obs_noise_profile=noise_profile, normalize=False, obs_noise_level=noise_level)

    f_latent = np.asarray(mu if init is None else init, dtype=float).copy()
    loss_history: list[float] = []
    for _ in range(max_iters):
        delta_f, loss = model.update(f_latent)
        f_latent = np.clip(f_latent + delta_f, latent_clip_min, latent_clip_max)
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


def _copy_rt1_plot_attrs(dst: rt1.Kernel2D_scatter_rt1, src: kernel_mod.Kernel2D_scatter) -> None:
    # Required by Kernel2D_scatter_rt1.plt_rt1_flux
    for name in ("r_plot", "z_plot", "im_kwargs", "mask"):
        setattr(dst, name, getattr(src, name))


def _make_grid_axes(cfg: SingleTomographyConfig) -> tuple[np.ndarray, np.ndarray]:
    r_grid = np.arange(cfg.grid_r_min, cfg.grid_r_max, cfg.grid_r_step, dtype=float)
    z_grid = np.arange(cfg.grid_z_min, cfg.grid_z_max, cfg.grid_z_step, dtype=float)
    return r_grid, z_grid


def _load_inducing_points_npz(point_file: Path) -> dict[str, np.ndarray]:
    """Load inducing-point arrays from current or legacy RT-1 .npz layout."""
    with np.load(point_file) as data:
        arrays = {k: np.asarray(v) for k, v in data.items()}

    new_keys = {"r_idc", "z_idc", "r_bd", "z_bd"}
    if new_keys.issubset(arrays):
        return {k: arrays[k] for k in sorted(new_keys)}

    legacy_to_new = {"rI": "r_idc", "zI": "z_idc", "rb": "r_bd", "zb": "z_bd"}
    if set(legacy_to_new).issubset(arrays):
        return {new_k: arrays[old_k] for old_k, new_k in legacy_to_new.items()}

    raise ValueError(
        "Point file must contain inducing-point arrays with keys "
        f"{sorted(new_keys)} or legacy keys {sorted(legacy_to_new)}. "
        f"Got keys={sorted(arrays)}"
    )


def _uniform_prior_for_any_kernel(k, *, length_scale: float, **kwargs):
    # Prefer corrected name if available (Kernel2D_scatter_grid alias), fall back to legacy typo.
    if hasattr(k, "set_uniform_kernel"):
        try:
            return k.set_uniform_kernel(length_scale=length_scale, **kwargs)
        except TypeError:
            pass
    if hasattr(k, "set_unifom_kernel"):
        return k.set_unifom_kernel(length_scale=length_scale, **kwargs)
    raise TypeError("Kernel does not support uniform SE prior setup")


def prepare_single_tomography(
    cfg: SingleTomographyConfig,
    *,
    store: ProjectStore,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Prepare RT1 geometry + raytrace + observation matrix for one condition.

    Reusable for sweeps where only noise/tomography hyperparameters change.
    """
    flux_kernel = rt1.Kernel2D_scatter_rt1()
    obs_mode = _resolve_obs_mode(cfg)
    prior_mode = _resolve_prior_mode(cfg)

    if cfg.basis_mode == "inducing":
        k: kernel_mod.Kernel2D_scatter = flux_kernel
        point_file = Path(cfg.point_file)
        if not point_file.is_absolute():
            point_file = (PROJECT_ROOT / point_file).resolve()
        if not point_file.exists():
            raise FileNotFoundError(f"Missing point file: {point_file}")
        inducing_cfg = InducingPointConfig(
            source=FileRef.from_path(point_file, note=point_file.name),
            length_sq_function=CallableRef.from_callable(rt1.phantom.Length_scale_sq),
            note="inducing points loaded from file",
        )
        inducing_arrays, ind_rec = store.get_or_build_inducing_points(
            inducing_cfg,
            builder=lambda: _load_inducing_points_npz(point_file),
        )
        k.load_point(
            r_idc=inducing_arrays["r_idc"],
            z_idc=inducing_arrays["z_idc"],
            r_bd=inducing_arrays["r_bd"],
            z_bd=inducing_arrays["z_bd"],
            length_sq_fuction=rt1.phantom.Length_scale_sq,
            is_plot=False,
        )
        point_ref_str = str(point_file)
        grid_column_mask = None
        grid_spec_params: dict[str, Any] = {}
        r_grid = z_grid = None
    elif cfg.basis_mode == "grid":
        k = kernel_mod.Kernel2D_scatter_grid(vessel=flux_kernel.vessel)
        r_grid, z_grid = _make_grid_axes(cfg)
        _mask, _ = k.vessel.detect_grid(r_grid=r_grid, z_grid=z_grid, static=True, isnt_print=True)
        fill = k.vessel.fill.copy()
        inside_mask = fill == cfg.grid_inside_value
        grid_length_sq_fn = kernel_mod.Kernel2D_scatter_grid.constant_length_scale_sq_function(
            length_scale=cfg.grid_inducing_length_scale
        )
        r_idc, z_idc, r_bd, z_bd = k.set_inducing_point_from_grid_fill(
            r_grid=r_grid,
            z_grid=z_grid,
            fill=fill,
            length_sq_fuction=grid_length_sq_fn,
            inside_value=cfg.grid_inside_value,
            boundary_value=cfg.grid_boundary_value,
        )
        grid_arrays = {
            "r_idc": np.asarray(r_idc),
            "z_idc": np.asarray(z_idc),
            "r_bd": np.asarray(r_bd),
            "z_bd": np.asarray(z_bd),
        }
        inducing_cfg = InducingPointConfig(
            source=FileRef.from_path(Path(flux_kernel.dict_path), note="rt1 vessel frame used for grid inducing points"),
            length_sq_function=None,
            params={
                "mode": "grid_fill",
                "r_min": cfg.grid_r_min,
                "r_max": cfg.grid_r_max,
                "r_step": cfg.grid_r_step,
                "z_min": cfg.grid_z_min,
                "z_max": cfg.grid_z_max,
                "z_step": cfg.grid_z_step,
                "inside_value": cfg.grid_inside_value,
                "boundary_value": cfg.grid_boundary_value,
                "grid_inducing_length_scale": cfg.grid_inducing_length_scale,
            },
            note="grid-derived inducing points from vessel fill map",
        )
        inducing_arrays, ind_rec = store.get_or_build_inducing_points(inducing_cfg, builder=lambda: grid_arrays)
        k.load_point(
            r_idc=inducing_arrays["r_idc"],
            z_idc=inducing_arrays["z_idc"],
            r_bd=inducing_arrays["r_bd"],
            z_bd=inducing_arrays["z_bd"],
            length_sq_fuction=grid_length_sq_fn,
            is_plot=False,
        )
        point_ref_str = "grid_fill"
        grid_column_mask = inside_mask
        grid_spec_params = {"grid_shape": [int(z_grid.size), int(r_grid.size)]}
    else:
        raise ValueError(f"Unknown basis_mode: {cfg.basis_mode}")

    k.set_grid_interface(
        r_plot=np.linspace(cfg.plot_r_min, cfg.plot_r_max, cfg.plot_r_num),
        z_plot=np.linspace(cfg.plot_z_min, cfg.plot_z_max, cfg.plot_z_num),
        add_bound=True,
    )
    _copy_rt1_plot_attrs(flux_kernel, k)

    if cfg.camera_kind != "Camera2D_rphiz":
        raise ValueError(f"Unsupported camera_kind for this script: {cfg.camera_kind}")
    camera_params = _camera_params_from_cfg(cfg)
    camera = zray.measurement.Camera2D_rphiz(
        focal_length=camera_params["focal_length"],
        location=tuple(camera_params["location"]),
        center_angles=tuple(camera_params["center_angles"]),
        sensor_size=tuple(camera_params["sensor_size"]),
        resolution=tuple(camera_params["resolution"]),
        rotation=camera_params["rotation"],
    )
    raymodel = zray.Raytracing(k.vessel, camera)
    raymodel.main(nreflections=cfg.nreflections, pass_through_first=cfg.pass_through_first)
    if cfg.ray_index_for_integral >= len(raymodel.rays):
        raise IndexError(f"ray_index_for_integral={cfg.ray_index_for_integral} but raymodel has {len(raymodel.rays)} rays")
    ray = raymodel.rays[cfg.ray_index_for_integral]

    obs_cfg = ObservationMatrixConfig(
        method=obs_mode,
        lnum=(cfg.lnum if obs_mode == "kernel_weighting" else cfg.grid_obs_sample_count),
        vessel=VesselConfig(package_resource=cfg.vessel_package_resource),
        camera=CameraConfig(kind=cfg.camera_kind, params=camera_params),
        raytrace=RaytraceConfig(
            nreflections=cfg.nreflections,
            pass_through_first=cfg.pass_through_first,
            ray_index_for_integral=cfg.ray_index_for_integral,
        ),
        inducing_points=inducing_cfg,
        package_versions=collect_package_versions(["gpoloidal", "zray", "numpy", "scipy"]),
    )

    def _build_H():
        if obs_mode == "kernel_weighting":
            return np.asarray(k.create_obs_matrix_kernel_weighting(ray=ray, Lnum=cfg.lnum), dtype=float)
        if obs_mode == "grid_binning":
            if cfg.basis_mode != "grid":
                raise ValueError("grid_binning observation matrix requires basis_mode='grid'")
            if grid_column_mask is None or r_grid is None or z_grid is None:
                raise RuntimeError("grid state missing")
            return np.asarray(
                k.create_obs_matrix_grid_binning(
                    ray,
                    r_grid=r_grid,
                    z_grid=z_grid,
                    sample_count=cfg.grid_obs_sample_count,
                    column_mask=grid_column_mask,
                    sparse_output=False,
                    show_progress=show_progress,
                ),
                dtype=float,
            )
        raise ValueError(f"Unknown observation_matrix_mode: {obs_mode}")

    H, obsmat_rec = store.get_or_build_observation_matrix(obs_cfg, builder=_build_H)
    return {
        "flux_kernel": flux_kernel,
        "kernel": k,
        "ray": ray,
        "raymodel": raymodel,
        "H": np.asarray(H, dtype=float),
        "obs_mode": obs_mode,
        "prior_mode": prior_mode,
        "obs_cfg": obs_cfg,
        "ind_rec": ind_rec,
        "obsmat_rec": obsmat_rec,
        "point_ref_str": point_ref_str,
        "grid_spec_params": grid_spec_params,
        "ray_count": int(len(raymodel.rays)),
    }


def solve_single_tomography(cfg: SingleTomographyConfig, *, prepared: dict[str, Any]) -> dict[str, Any]:
    """Solve one RT1 tomography case on top of prepared geometry/observation."""
    k = prepared["kernel"]
    H = prepared["H"]
    prior_mode = prepared["prior_mode"]
    phantom_fn = rt1.phantom.get_phantom_function(cfg.phantom_name, **cfg.phantom_params)
    f_true = np.clip(np.asarray(phantom_fn(k.r_idc, k.z_idc), dtype=float), 0.0, None)
    g_true = H @ f_true

    if prior_mode == "gibbs":
        if cfg.tomography_method == "linGP":
            K_pri, mu_pri = k.set_kernel(
                length_scale_factor=cfg.length_scale_factor,
                is_bound=True,
                bound_value=cfg.lin_bound_value,
                bound_sig=cfg.bound_sig,
                mean=cfg.lin_prior_mean,
            )
        else:
            K_pri, mu_pri = k.set_kernel(
                length_scale_factor=cfg.length_scale_factor,
                is_bound=True,
                bound_value=cfg.log_bound_value,
                bound_sig=cfg.bound_sig,
                mean=cfg.log_prior_mean,
            )
    elif prior_mode == "uniform_se":
        if cfg.tomography_method == "linGP":
            K_pri, mu_pri = _uniform_prior_for_any_kernel(
                k,
                length_scale=cfg.uniform_length_scale,
                is_bound=True,
                bound_value=cfg.lin_bound_value,
                bound_sig=cfg.bound_sig,
                mean=cfg.lin_prior_mean,
                is_static_kernel=False,
            )
        else:
            K_pri, mu_pri = _uniform_prior_for_any_kernel(
                k,
                length_scale=cfg.uniform_length_scale,
                is_bound=True,
                bound_value=cfg.log_bound_value,
                bound_sig=cfg.bound_sig,
                mean=cfg.log_prior_mean,
                is_static_kernel=False,
            )
    else:
        raise ValueError(f"Unknown prior_mode: {prior_mode}")

    g_rms = float(np.sqrt(np.mean(np.square(g_true))))
    if cfg.noise_mode == "snr_rms":
        obs_noise_level = float(g_rms / cfg.snr_rms)
    elif cfg.noise_mode == "absolute":
        obs_noise_level = float(cfg.obs_noise_level)
    else:
        raise ValueError(f"Unknown noise_mode: {cfg.noise_mode}")
    obs_noise_std = np.full_like(g_true, obs_noise_level, dtype=float)
    rng = np.random.default_rng(cfg.seed)
    g_obs = g_true + obs_noise_std * rng.standard_normal(g_true.size)

    if cfg.tomography_method == "linGP":
        fit = solve_lingp(H, g_obs, obs_noise_std, K_pri, np.asarray(mu_pri, dtype=float))
        fit_label = "linGP"
        log_retry = False
    elif cfg.tomography_method == "logGP":
        fit = solve_loggp(
            H,
            g_obs,
            obs_noise_std,
            K_pri,
            np.asarray(mu_pri, dtype=float),
            max_iters=cfg.max_log_iters,
            tol=cfg.log_tol,
            latent_clip_min=cfg.log_latent_clip_min,
            latent_clip_max=cfg.log_latent_clip_max,
        )
        log_retry = False
        if not fit["converged"]:
            fit = solve_loggp(
                H,
                g_obs,
                obs_noise_std,
                K_pri,
                np.asarray(mu_pri, dtype=float),
                max_iters=max(cfg.max_log_iters * cfg.log_retry_multiplier, cfg.log_retry_min_iters),
                tol=cfg.log_tol,
                latent_clip_min=cfg.log_latent_clip_min,
                latent_clip_max=cfg.log_latent_clip_max,
                init=fit["f_latent"],
            )
            log_retry = True
        fit_label = "logGP"
    else:
        raise ValueError(f"Unknown tomography_method: {cfg.tomography_method}")

    metrics = field_metrics(fit["f_mean"], f_true, prefix="recon")
    metrics["chi2"] = mean_chi2(H @ fit["f_mean"], g_obs, obs_noise_std)
    metrics["snr_rms_true"] = float(g_rms / (obs_noise_level + 1e-12))
    metrics["obs_noise_level"] = float(obs_noise_level)
    metrics["nI"] = int(k.nI)
    metrics["nb"] = int(k.nb)
    metrics["H_rows"] = int(H.shape[0])
    metrics["H_cols"] = int(H.shape[1])
    if cfg.tomography_method == "logGP":
        metrics["log_iters"] = int(len(fit["loss_history"]))
        metrics["log_converged"] = bool(fit["converged"])
        metrics["log_retried"] = bool(log_retry)
        metrics["log_last_loss"] = float(fit["loss_history"][-1]) if fit["loss_history"] else None

    return {
        "f_true": f_true,
        "g_obs": g_obs,
        "fit": fit,
        "fit_label": fit_label,
        "metrics": metrics,
    }


def run_single_tomography(cfg: SingleTomographyConfig, *, store: ProjectStore) -> dict[str, Any]:
    """Run one RT1 tomography condition and return rich intermediate/results for scripts.

    This function intentionally returns internal arrays/objects so higher-level scripts
    (single-run, sweep, notebook-like orchestration) can decide what to save/plot.
    """
    prepared = prepare_single_tomography(cfg, store=store, show_progress=True)
    solved = solve_single_tomography(cfg, prepared=prepared)
    return {**prepared, **solved}

    # Legacy inlined implementation kept below temporarily during refactor cleanup.
    flux_kernel = rt1.Kernel2D_scatter_rt1()
    obs_mode = _resolve_obs_mode(cfg)
    prior_mode = _resolve_prior_mode(cfg)

    if cfg.basis_mode == "inducing":
        k: kernel_mod.Kernel2D_scatter = flux_kernel
        point_file = Path(cfg.point_file)
        if not point_file.is_absolute():
            point_file = (PROJECT_ROOT / point_file).resolve()
        if not point_file.exists():
            raise FileNotFoundError(f"Missing point file: {point_file}")

        inducing_cfg = InducingPointConfig(
            source=FileRef.from_path(point_file, note="point_temp.npz"),
            length_sq_function=CallableRef.from_callable(rt1.phantom.Length_scale_sq),
            note="inducing points loaded from file",
        )
        inducing_arrays, ind_rec = store.get_or_build_inducing_points(
            inducing_cfg,
            builder=lambda: {kk: np.asarray(v) for kk, v in np.load(point_file).items() if kk in {"r_idc", "z_idc", "r_bd", "z_bd"}},
        )
        k.load_point(
            r_idc=inducing_arrays["r_idc"],
            z_idc=inducing_arrays["z_idc"],
            r_bd=inducing_arrays["r_bd"],
            z_bd=inducing_arrays["z_bd"],
            length_sq_fuction=rt1.phantom.Length_scale_sq,
            is_plot=False,
        )
        point_ref_str = str(point_file)
        grid_column_mask = None
        grid_spec_params: dict[str, Any] = {}
    elif cfg.basis_mode == "grid":
        k = kernel_mod.Kernel2D_scatter_grid(vessel=flux_kernel.vessel)
        r_grid, z_grid = _make_grid_axes(cfg)
        _mask, _ = k.vessel.detect_grid(r_grid=r_grid, z_grid=z_grid, static=True, isnt_print=True)
        fill = k.vessel.fill.copy()
        inside_mask = fill == cfg.grid_inside_value

        grid_length_sq_fn = kernel_mod.Kernel2D_scatter_grid.constant_length_scale_sq_function(
            length_scale=cfg.grid_inducing_length_scale
        )
        r_idc, z_idc, r_bd, z_bd = k.set_inducing_point_from_grid_fill(
            r_grid=r_grid,
            z_grid=z_grid,
            fill=fill,
            length_sq_fuction=grid_length_sq_fn,
            inside_value=cfg.grid_inside_value,
            boundary_value=cfg.grid_boundary_value,
        )
        grid_arrays = {
            "r_idc": np.asarray(r_idc),
            "z_idc": np.asarray(z_idc),
            "r_bd": np.asarray(r_bd),
            "z_bd": np.asarray(z_bd),
        }
        inducing_cfg = InducingPointConfig(
            source=FileRef.from_path(Path(flux_kernel.dict_path), note="rt1 vessel frame used for grid inducing points"),
            length_sq_function=None,
            params={
                "mode": "grid_fill",
                "r_min": cfg.grid_r_min,
                "r_max": cfg.grid_r_max,
                "r_step": cfg.grid_r_step,
                "z_min": cfg.grid_z_min,
                "z_max": cfg.grid_z_max,
                "z_step": cfg.grid_z_step,
                "inside_value": cfg.grid_inside_value,
                "boundary_value": cfg.grid_boundary_value,
                "grid_inducing_length_scale": cfg.grid_inducing_length_scale,
            },
            note="grid-derived inducing points from vessel fill map",
        )
        inducing_arrays, ind_rec = store.get_or_build_inducing_points(inducing_cfg, builder=lambda: grid_arrays)
        k.load_point(
            r_idc=inducing_arrays["r_idc"],
            z_idc=inducing_arrays["z_idc"],
            r_bd=inducing_arrays["r_bd"],
            z_bd=inducing_arrays["z_bd"],
            length_sq_fuction=grid_length_sq_fn,
            is_plot=False,
        )
        point_ref_str = "grid_fill"
        grid_column_mask = inside_mask
        grid_spec_params = {"grid_shape": [int(z_grid.size), int(r_grid.size)]}
    else:
        raise ValueError(f"Unknown basis_mode: {cfg.basis_mode}")

    k.set_grid_interface(
        r_plot=np.linspace(cfg.plot_r_min, cfg.plot_r_max, cfg.plot_r_num),
        z_plot=np.linspace(cfg.plot_z_min, cfg.plot_z_max, cfg.plot_z_num),
        add_bound=True,
    )
    _copy_rt1_plot_attrs(flux_kernel, k)

    if cfg.camera_kind != "Camera2D_rphiz":
        raise ValueError(f"Unsupported camera_kind for this script: {cfg.camera_kind}")
    camera_params = _camera_params_from_cfg(cfg)
    camera = zray.measurement.Camera2D_rphiz(
        focal_length=camera_params["focal_length"],
        location=tuple(camera_params["location"]),
        center_angles=tuple(camera_params["center_angles"]),
        sensor_size=tuple(camera_params["sensor_size"]),
        resolution=tuple(camera_params["resolution"]),
        rotation=camera_params["rotation"],
    )
    raymodel = zray.Raytracing(k.vessel, camera)
    raymodel.main(nreflections=cfg.nreflections, pass_through_first=cfg.pass_through_first)
    if cfg.ray_index_for_integral >= len(raymodel.rays):
        raise IndexError(f"ray_index_for_integral={cfg.ray_index_for_integral} but raymodel has {len(raymodel.rays)} rays")
    ray = raymodel.rays[cfg.ray_index_for_integral]

    obs_cfg = ObservationMatrixConfig(
        method=obs_mode,
        lnum=(cfg.lnum if obs_mode == "kernel_weighting" else cfg.grid_obs_sample_count),
        vessel=VesselConfig(package_resource=cfg.vessel_package_resource),
        camera=CameraConfig(kind=cfg.camera_kind, params=camera_params),
        raytrace=RaytraceConfig(
            nreflections=cfg.nreflections,
            pass_through_first=cfg.pass_through_first,
            ray_index_for_integral=cfg.ray_index_for_integral,
        ),
        inducing_points=inducing_cfg,
        package_versions=collect_package_versions(["gpoloidal", "zray", "numpy", "scipy"]),
    )

    def _build_H():
        if obs_mode == "kernel_weighting":
            return np.asarray(k.create_obs_matrix_kernel_weighting(ray=ray, Lnum=cfg.lnum), dtype=float)
        if obs_mode == "grid_binning":
            if cfg.basis_mode != "grid":
                raise ValueError("grid_binning observation matrix requires basis_mode='grid'")
            if grid_column_mask is None:
                raise RuntimeError("grid column mask missing")
            return np.asarray(
                k.create_obs_matrix_grid_binning(
                    ray,
                    r_grid=r_grid,
                    z_grid=z_grid,
                    sample_count=cfg.grid_obs_sample_count,
                    column_mask=grid_column_mask,
                    sparse_output=False,
                    show_progress=True,
                ),
                dtype=float,
            )
        raise ValueError(f"Unknown observation_matrix_mode: {obs_mode}")

    H, obsmat_rec = store.get_or_build_observation_matrix(obs_cfg, builder=_build_H)
    H = np.asarray(H, dtype=float)

    phantom_fn = rt1.phantom.get_phantom_function(cfg.phantom_name, **cfg.phantom_params)
    f_true = np.clip(np.asarray(phantom_fn(k.r_idc, k.z_idc), dtype=float), 0.0, None)
    g_true = H @ f_true

    if prior_mode == "gibbs":
        if cfg.tomography_method == "linGP":
            K_pri, mu_pri = k.set_kernel(
                length_scale_factor=cfg.length_scale_factor,
                is_bound=True,
                bound_value=cfg.lin_bound_value,
                bound_sig=cfg.bound_sig,
                mean=cfg.lin_prior_mean,
            )
        else:
            K_pri, mu_pri = k.set_kernel(
                length_scale_factor=cfg.length_scale_factor,
                is_bound=True,
                bound_value=cfg.log_bound_value,
                bound_sig=cfg.bound_sig,
                mean=cfg.log_prior_mean,
            )
    elif prior_mode == "uniform_se":
        if cfg.tomography_method == "linGP":
            K_pri, mu_pri = _uniform_prior_for_any_kernel(
                k,
                length_scale=cfg.uniform_length_scale,
                is_bound=True,
                bound_value=cfg.lin_bound_value,
                bound_sig=cfg.bound_sig,
                mean=cfg.lin_prior_mean,
                is_static_kernel=False,
            )
        else:
            K_pri, mu_pri = _uniform_prior_for_any_kernel(
                k,
                length_scale=cfg.uniform_length_scale,
                is_bound=True,
                bound_value=cfg.log_bound_value,
                bound_sig=cfg.bound_sig,
                mean=cfg.log_prior_mean,
                is_static_kernel=False,
            )
    else:
        raise ValueError(f"Unknown prior_mode: {prior_mode}")

    g_rms = float(np.sqrt(np.mean(np.square(g_true))))
    if cfg.noise_mode == "snr_rms":
        obs_noise_level = float(g_rms / cfg.snr_rms)
    elif cfg.noise_mode == "absolute":
        obs_noise_level = float(cfg.obs_noise_level)
    else:
        raise ValueError(f"Unknown noise_mode: {cfg.noise_mode}")
    obs_noise_std = np.full_like(g_true, obs_noise_level, dtype=float)

    rng = np.random.default_rng(cfg.seed)
    g_obs = g_true + obs_noise_std * rng.standard_normal(g_true.size)

    if cfg.tomography_method == "linGP":
        fit = solve_lingp(H, g_obs, obs_noise_std, K_pri, np.asarray(mu_pri, dtype=float))
        fit_label = "linGP"
        log_retry = False
    elif cfg.tomography_method == "logGP":
        fit = solve_loggp(
            H,
            g_obs,
            obs_noise_std,
            K_pri,
            np.asarray(mu_pri, dtype=float),
            max_iters=cfg.max_log_iters,
            tol=cfg.log_tol,
            latent_clip_min=cfg.log_latent_clip_min,
            latent_clip_max=cfg.log_latent_clip_max,
        )
        log_retry = False
        if not fit["converged"]:
            fit = solve_loggp(
                H,
                g_obs,
                obs_noise_std,
                K_pri,
                np.asarray(mu_pri, dtype=float),
                max_iters=max(cfg.max_log_iters * cfg.log_retry_multiplier, cfg.log_retry_min_iters),
                tol=cfg.log_tol,
                latent_clip_min=cfg.log_latent_clip_min,
                latent_clip_max=cfg.log_latent_clip_max,
                init=fit["f_latent"],
            )
            log_retry = True
        fit_label = "logGP"
    else:
        raise ValueError(f"Unknown tomography_method: {cfg.tomography_method}")

    metrics = field_metrics(fit["f_mean"], f_true, prefix="recon")
    metrics["chi2"] = mean_chi2(H @ fit["f_mean"], g_obs, obs_noise_std)
    metrics["snr_rms_true"] = float(g_rms / (obs_noise_level + 1e-12))
    metrics["obs_noise_level"] = float(obs_noise_level)
    metrics["nI"] = int(k.nI)
    metrics["nb"] = int(k.nb)
    metrics["H_rows"] = int(H.shape[0])
    metrics["H_cols"] = int(H.shape[1])
    if cfg.tomography_method == "logGP":
        metrics["log_iters"] = int(len(fit["loss_history"]))
        metrics["log_converged"] = bool(fit["converged"])
        metrics["log_retried"] = bool(log_retry)
        metrics["log_last_loss"] = float(fit["loss_history"][-1]) if fit["loss_history"] else None

    return {
        "flux_kernel": flux_kernel,
        "kernel": k,
        "H": H,
        "obs_mode": obs_mode,
        "prior_mode": prior_mode,
        "obs_cfg": obs_cfg,
        "ind_rec": ind_rec,
        "obsmat_rec": obsmat_rec,
        "point_ref_str": point_ref_str,
        "grid_spec_params": grid_spec_params,
        "ray_count": int(len(raymodel.rays)),
        "f_true": f_true,
        "g_obs": g_obs,
        "fit": fit,
        "fit_label": fit_label,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    cfg = SingleTomographyConfig()
    config_path = (
        (PROJECT_ROOT / args.config).resolve()
        if args.config and not Path(args.config).is_absolute()
        else (Path(args.config).resolve() if args.config else None)
    )
    if config_path is not None:
        loaded_cfg = load_config_mapping(config_path)
        apply_flat_dataclass_config(cfg, _flatten_single_tomography_config_mapping(loaded_cfg))
    if args.quick:
        cfg = replace(
            cfg,
            resolution=(48, 48),
            lnum=min(cfg.lnum, 401),
            grid_obs_sample_count=min(cfg.grid_obs_sample_count, 120),
            max_log_iters=min(cfg.max_log_iters, 10),
        )

    runtime_roots = resolve_runtime_roots(
        mode=args.mode,
        project_root=PROJECT_ROOT,
        backend_experiment_name=cfg.experiment_name,
        output_dir=args.output_dir,
        backend_record_dir=args.backend_record_dir,
    )
    record_policy = resolve_record_mode_policy(record_mode=args.record_mode, no_run_record=args.no_run_record)
    layout = prepare_local_run_layout(
        base_dir=runtime_roots.output_base_dir,
        experiment_name=cfg.experiment_name,
        run_name=args.run_name,
    )
    output_root = layout.run_root
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("gpoloidal", gpoloidal.__version__)
    print("config:", cfg)
    if config_path is not None:
        print("config_path:", config_path)
    print("mode:", runtime_roots.mode)
    print("record_mode:", record_policy.record_mode)
    print("cache_root:", runtime_roots.cache_root)
    print("backend_record_root:", runtime_roots.backend_record_root)
    print("output_root (archive run):", output_root)
    print("output_latest_root:", layout.latest_root)

    save_json(output_root / "config_resolved.json", asdict(cfg))
    if config_path is not None:
        save_json(output_root / "config_source.json", {"config_path": str(config_path), "loaded": load_config_mapping(config_path)})

    store = ProjectStore(cache_root=runtime_roots.cache_root, record_root=runtime_roots.backend_record_root)
    run = run_single_tomography(cfg, store=store)
    flux_kernel = run["flux_kernel"]
    k = run["kernel"]
    H = run["H"]
    obs_mode = run["obs_mode"]
    prior_mode = run["prior_mode"]
    obs_cfg = run["obs_cfg"]
    ind_rec = run["ind_rec"]
    obsmat_rec = run["obsmat_rec"]
    point_ref_str = run["point_ref_str"]
    grid_spec_params = run["grid_spec_params"]
    f_true = run["f_true"]
    g_obs = run["g_obs"]
    fit = run["fit"]
    fit_label = run["fit_label"]
    metrics = run["metrics"]
    obs_noise_level = float(metrics["obs_noise_level"])

    print(
        json.dumps(
            {
                "basis_mode": cfg.basis_mode,
                "obs_mode": obs_mode,
                "method": cfg.tomography_method,
                "prior_mode": prior_mode,
                "nI": int(metrics["nI"]),
                "nb": int(metrics["nb"]),
                "H_shape": list(H.shape),
                "ray_count": int(run["ray_count"]),
                "inducing_artifact": ind_rec.artifact_id,
                "obsmat_artifact": obsmat_rec.artifact_id,
            },
            indent=2,
        )
    )

    # Figures
    fig = plot_rt1_truth_and_observation(
        kernel=k,
        flux_kernel=flux_kernel,
        f_true=f_true,
        g_obs=g_obs,
        resolution=cfg.resolution,
    )
    truth_plot = figure_dir / "rt1_truth_and_observation.png"
    save_figure(fig, truth_plot, dpi=cfg.plot_figure_dpi)

    fig, _ = plot_rt1_reconstruction_panels(
        kernel=k,
        flux_kernel=flux_kernel,
        f_true=f_true,
        f_recon=np.asarray(fit["f_mean"], dtype=float),
        f_std=np.asarray(fit["f_std"], dtype=float),
        fit_label=fit_label,
        value_vmin=cfg.plot_value_vmin,
        value_vmax=cfg.plot_value_vmax,
        error_percentile=cfg.plot_error_percentile,
    )
    recon_plot = figure_dir / "rt1_single_reconstruction.png"
    save_figure(fig, recon_plot, dpi=cfg.plot_figure_dpi)

    loss_plot = None
    if cfg.tomography_method == "logGP":
        fig = plot_loggp_loss_history(loss_history=fit["loss_history"])
        loss_plot = figure_dir / "rt1_loggp_loss_history.png"
        save_figure(fig, loss_plot, dpi=cfg.plot_figure_dpi)

    metrics_path = output_root / "metrics.json"
    save_json(metrics_path, metrics)

    backend_outputs: dict[str, str] = {}
    if record_policy.save_backend_result_artifacts:
        backend_outputs["truth_observation_plot_artifact_id"] = store.save_result_file(
            "rt1_truth_and_observation", truth_plot, kind="image", copy=True
        ).artifact_id
        backend_outputs["reconstruction_plot_artifact_id"] = store.save_result_file(
            "rt1_single_reconstruction", recon_plot, kind="image", copy=True
        ).artifact_id
        backend_outputs["metrics_artifact_id"] = store.save_result_file(
            "rt1_single_metrics_json", metrics_path, kind="file", copy=True
        ).artifact_id
        if loss_plot is not None:
            backend_outputs["loss_plot_artifact_id"] = store.save_result_file(
                "rt1_loggp_loss_history", loss_plot, kind="image", copy=True
            ).artifact_id

    run_id = None
    if record_policy.save_run_record:
        noise_params = {"noise_mode": cfg.noise_mode}
        if cfg.noise_mode == "snr_rms":
            noise_params["snr_rms"] = cfg.snr_rms
        else:
            noise_params["obs_noise_level"] = cfg.obs_noise_level

        record = ExperimentRecord(
            name=cfg.experiment_name,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            observation_matrix_artifact_id=obsmat_rec.artifact_id,
            observation_matrix_config=obs_cfg,
            phantom=PhantomConfig(
                kind="synthetic",
                name=cfg.phantom_name,
                generator=CallableRef.from_callable(rt1.phantom.get_phantom_function),
                params={"phantom_params": cfg.phantom_params},
            ),
            noise=NoiseConfig(
                model="gaussian",
                level=float(obs_noise_level),
                level_definition=cfg.noise_mode,
                profile="flat",
                seed=cfg.seed,
                params=noise_params,
            ),
            tomography=TomographyConfig(
                model=cfg.tomography_method,
                prior_kind=prior_mode,
                length_scale_factor=(cfg.length_scale_factor if prior_mode == "gibbs" else None),
                boundary_sigma=cfg.bound_sig,
                boundary_value=(cfg.log_bound_value if cfg.tomography_method == "logGP" else cfg.lin_bound_value),
                prior_mean=(cfg.log_prior_mean if cfg.tomography_method == "logGP" else cfg.lin_prior_mean),
                normalize=False,
                obs_noise_level=float(obs_noise_level),
                max_iters=(cfg.max_log_iters if cfg.tomography_method == "logGP" else None),
                tol=(cfg.log_tol if cfg.tomography_method == "logGP" else None),
                extras={
                    "basis_mode": cfg.basis_mode,
                    "observation_matrix_mode": obs_mode,
                    "uniform_length_scale": cfg.uniform_length_scale,
                },
            ),
            references={
                "inducing_points_artifact_id": ind_rec.artifact_id,
                "point_source": point_ref_str,
                "ray_index_for_integral": cfg.ray_index_for_integral,
                **grid_spec_params,
            },
            metrics=metrics,
            outputs=backend_outputs,
        )
        run_id = store.save_experiment_record(
            record,
            strict_traceability=record_policy.strict_traceability,
            embed_dependency_manifests=record_policy.embed_dependency_manifests,
        )

    report = {
        "script": "gpoloidal.scripts.rt1_tomography_single",
        "gpoloidal_version": gpoloidal.__version__,
        "config_path": str(config_path) if config_path is not None else None,
        "config": asdict(cfg),
        "runtime": {"mode": runtime_roots.mode, "record_mode": record_policy.record_mode},
        "paths": {
            "output_root": str(output_root),
            "output_latest_root": str(layout.latest_root),
            "output_archive_root": str(layout.archive_root),
            "backend_record_root": str(runtime_roots.backend_record_root),
            "cache_root": str(runtime_roots.cache_root),
            "config_resolved": str(output_root / "config_resolved.json"),
            "metrics": str(metrics_path),
            "truth_observation_plot": str(truth_plot),
            "reconstruction_plot": str(recon_plot),
            **({"log_loss_plot": str(loss_plot)} if loss_plot else {}),
            **({"run_record": str(store.run_dir / f'{run_id}.json')} if run_id else {}),
            **({"config_source": str(output_root / 'config_source.json')} if config_path is not None else {}),
        },
        "artifacts": {
            "inducing_points_artifact_id": ind_rec.artifact_id,
            "observation_matrix_artifact_id": obsmat_rec.artifact_id,
            "run_id": run_id,
        },
        "metrics": metrics,
    }
    save_json(output_root / "latest_report.json", report)
    save_json(output_root / "latest_paths.json", report["paths"])
    save_json(
        output_root / "run_ref.json",
        make_run_reference(
            script="gpoloidal.scripts.rt1_tomography_single",
            archive_run_root=output_root,
            latest_root=layout.latest_root,
            backend_record_root=runtime_roots.backend_record_root,
            run_id=run_id,
            backend_run_record_path=(store.run_dir / f"{run_id}.json") if run_id else None,
            extra={
                "observation_matrix_artifact_id": obsmat_rec.artifact_id,
                "inducing_points_artifact_id": ind_rec.artifact_id,
                "record_mode": record_policy.record_mode,
                "basis_mode": cfg.basis_mode,
                "tomography_method": cfg.tomography_method,
            },
        ),
    )
    publish_latest_from_archive(layout)

    print(json.dumps({
        "run_id": run_id,
        "basis_mode": cfg.basis_mode,
        "tomography_method": cfg.tomography_method,
        "obs_mode": obs_mode,
        "archive_run_root": str(output_root),
        "latest_root": str(layout.latest_root),
        "metrics": metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
