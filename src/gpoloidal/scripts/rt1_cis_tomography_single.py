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
import gpoloidal.rt1 as rt1
from gpoloidal.analysis.config import apply_flat_dataclass_config, load_config_mapping, save_json
from gpoloidal.cis.forward import forward_cis_from_etv, forward_emit
from gpoloidal.cis.geometry import coerce_observation_geometry
from gpoloidal.cis.pipeline import CISStepwiseReconstructor
from gpoloidal.cis.types import CISObservedChannels
from gpoloidal.core.metrics import field_metrics, mean_chi2
from gpoloidal.experiment import CallableRef, ExperimentRecord, NoiseConfig, PhantomConfig, ProjectStore, TomographyConfig
from gpoloidal.run_layout import make_run_reference, prepare_local_run_layout, publish_latest_from_archive
from gpoloidal.script_cli import add_common_runtime_args, parse_known_args, resolve_record_mode_policy, resolve_runtime_roots
from gpoloidal.scripts.rt1_tomography_single import (
    PROJECT_ROOT,
    SingleTomographyConfig,
    _flatten_single_tomography_config_mapping,
    _uniform_prior_for_any_kernel,
    prepare_single_tomography,
    save_figure,
)


@dataclass
class RT1CISTomographySingleConfig:
    experiment_name: str = "rt1_cis_tomography_single"

    basis_mode: Literal["inducing", "grid"] = "inducing"
    observation_matrix_mode: Literal["auto", "kernel_weighting", "grid_binning"] = "auto"
    point_file: str = "example/rt1tomography/point_temp.npz"
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
    vessel_package_resource: str = "gpoloidal.rt1:rt1_simple_frame.json"
    camera_kind: str = "Camera2D_rphiz"
    camera_focal_length: float = 0.01
    camera_location: tuple[float, float, float] = (1.2, 0.0, 0.0)
    camera_center_angles: tuple[float, float] = (23.0, 0.0)
    camera_sensor_size: tuple[float, float] = (0.0082, 0.0082)
    camera_rotation: float = 0.0
    resolution: tuple[int, int] = (48, 48)
    lnum: int = 1001
    nreflections: int = 1
    pass_through_first: bool = True
    ray_index_for_integral: int = 1

    phantom_bundle_name: str = "paper_phantom1"
    phantom_emissivity_name: str | None = None
    phantom_temperature_name: str | None = None
    phantom_velocity_name: str | None = None
    phantom_emissivity_params: dict[str, Any] = field(default_factory=dict)
    phantom_temperature_params: dict[str, Any] = field(default_factory=dict)
    phantom_velocity_params: dict[str, Any] = field(default_factory=dict)

    emit_prior_mode: Literal["auto", "gibbs", "uniform_se"] = "auto"
    av_prior_mode: Literal["auto", "gibbs", "uniform_se"] = "uniform_se"
    emit_length_scale_factor: float = 2.0
    emit_uniform_length_scale: float = 0.05
    av_length_scale_factor: float = 2.0
    av_uniform_length_scale: float = 0.05
    emit_bound_sig: float = 0.1
    av_bound_sig: float = 0.1
    emit_log_bound_value: float = -5.0
    emit_log_prior_mean: float = -3.0
    T_bound_value: float = 0.0
    T_prior_mean: float = 0.0
    v_bound_value: float = 0.0
    v_prior_mean: float = 0.0
    emit_max_iters: int = 30
    emit_tol: float = 1e-5
    av_max_iters: int = 50
    av_tol: float = 1e-5
    emit_latent_clip_min: float = -12.0
    emit_latent_clip_max: float = 8.0
    av_latent_clip_min: float = -8.0
    av_latent_clip_max: float = 8.0
    emit_step_size: float = 1.0
    av_step_size: float = 0.5
    av_consider_w2: bool = True
    do_evidence_scan_emit: bool = False
    do_evidence_scan_cis: bool = False

    I0_noise_mode: Literal["snr_rms", "absolute"] = "snr_rms"
    I0_snr_rms: float = 10.0
    I0_obs_noise_level: float = 0.05
    I1_noise_mode: Literal["relative_mean_I0", "absolute"] = "relative_mean_I0"
    I1_relative_to_I0_mean: float = 0.02
    I1_obs_noise_level: float = 0.02
    seed: int = 42

    plot_r_min: float = 0.05
    plot_r_max: float = 1.05
    plot_r_num: int = 501
    plot_z_min: float = -0.7
    plot_z_max: float = 0.7
    plot_z_num: int = 501
    plot_figure_dpi: int = 150


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RT-1 CIS single-condition phantom tomography runner")
    add_common_runtime_args(p, include_config=True, include_quick=True, include_trials_csv_toggle=False)
    return parse_known_args(p)


def _flatten_rt1_cis_config_mapping(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise TypeError("config root must be a mapping")
    out: dict[str, Any] = {}
    section_names = {"experiment", "basis", "grid", "observation", "plot", "phantom", "cis", "noise"}
    for k, v in data.items():
        if k not in section_names:
            out[k] = v

    shared_data = {k: v for k, v in data.items() if k in {"experiment", "basis", "grid", "observation", "plot"}}
    out.update(_flatten_single_tomography_config_mapping(shared_data))

    def require_mapping(section_name: str) -> dict[str, Any]:
        sec = data.get(section_name, None)
        if sec is None:
            return {}
        if not isinstance(sec, dict):
            raise ValueError(f"Config section '{section_name}' must be a mapping")
        return sec

    phantom = require_mapping("phantom")
    for k, v in phantom.items():
        mapping = {
            "bundle_name": "phantom_bundle_name",
            "emissivity_name": "phantom_emissivity_name",
            "temperature_name": "phantom_temperature_name",
            "velocity_name": "phantom_velocity_name",
            "emissivity_params": "phantom_emissivity_params",
            "temperature_params": "phantom_temperature_params",
            "velocity_params": "phantom_velocity_params",
        }
        if k not in mapping:
            raise ValueError(f"Unknown config key: phantom.{k}")
        out[mapping[k]] = v

    cis = require_mapping("cis")
    cis_map = {
        "emit_prior_mode": "emit_prior_mode",
        "av_prior_mode": "av_prior_mode",
        "emit_length_scale_factor": "emit_length_scale_factor",
        "emit_uniform_length_scale": "emit_uniform_length_scale",
        "av_length_scale_factor": "av_length_scale_factor",
        "av_uniform_length_scale": "av_uniform_length_scale",
        "emit_bound_sig": "emit_bound_sig",
        "av_bound_sig": "av_bound_sig",
        "emit_log_bound_value": "emit_log_bound_value",
        "emit_log_prior_mean": "emit_log_prior_mean",
        "T_bound_value": "T_bound_value",
        "T_prior_mean": "T_prior_mean",
        "v_bound_value": "v_bound_value",
        "v_prior_mean": "v_prior_mean",
        "emit_max_iters": "emit_max_iters",
        "emit_tol": "emit_tol",
        "av_max_iters": "av_max_iters",
        "av_tol": "av_tol",
        "emit_latent_clip_min": "emit_latent_clip_min",
        "emit_latent_clip_max": "emit_latent_clip_max",
        "av_latent_clip_min": "av_latent_clip_min",
        "av_latent_clip_max": "av_latent_clip_max",
        "emit_step_size": "emit_step_size",
        "av_step_size": "av_step_size",
        "av_consider_w2": "av_consider_w2",
        "do_evidence_scan_emit": "do_evidence_scan_emit",
        "do_evidence_scan_cis": "do_evidence_scan_cis",
    }
    for k, v in cis.items():
        if k not in cis_map:
            raise ValueError(f"Unknown config key: cis.{k}")
        out[cis_map[k]] = v

    noise = require_mapping("noise")
    noise_map = {
        "I0_mode": "I0_noise_mode",
        "I0_snr_rms": "I0_snr_rms",
        "I0_obs_noise_level": "I0_obs_noise_level",
        "I1_mode": "I1_noise_mode",
        "I1_relative_to_I0_mean": "I1_relative_to_I0_mean",
        "I1_obs_noise_level": "I1_obs_noise_level",
        "seed": "seed",
    }
    for k, v in noise.items():
        if k not in noise_map:
            raise ValueError(f"Unknown config key: noise.{k}")
        out[noise_map[k]] = v
    return out


def _to_single_prepare_cfg(cfg: RT1CISTomographySingleConfig) -> SingleTomographyConfig:
    base = SingleTomographyConfig()
    for name in (
        "experiment_name",
        "basis_mode",
        "observation_matrix_mode",
        "point_file",
        "grid_r_min",
        "grid_r_max",
        "grid_r_step",
        "grid_z_min",
        "grid_z_max",
        "grid_z_step",
        "grid_inducing_length_scale",
        "grid_inside_value",
        "grid_boundary_value",
        "grid_obs_sample_count",
        "vessel_package_resource",
        "camera_kind",
        "camera_focal_length",
        "camera_location",
        "camera_center_angles",
        "camera_sensor_size",
        "camera_rotation",
        "resolution",
        "lnum",
        "nreflections",
        "pass_through_first",
        "ray_index_for_integral",
        "plot_r_min",
        "plot_r_max",
        "plot_r_num",
        "plot_z_min",
        "plot_z_max",
        "plot_z_num",
        "plot_figure_dpi",
    ):
        setattr(base, name, getattr(cfg, name))
    base.tomography_method = "logGP"
    base.prior_mode = "auto"
    return base


def _resolve_emit_prior_mode(cfg: RT1CISTomographySingleConfig) -> str:
    if cfg.emit_prior_mode != "auto":
        return cfg.emit_prior_mode
    return "gibbs" if cfg.basis_mode == "inducing" else "uniform_se"


def _resolve_av_prior_mode(cfg: RT1CISTomographySingleConfig) -> str:
    if cfg.av_prior_mode != "auto":
        return cfg.av_prior_mode
    return "uniform_se"


def _build_prior(k, *, mode: str, length_scale_factor: float, uniform_length_scale: float, bound_sig: float, bound_value: float, mean: float):
    if mode == "gibbs":
        return k.set_kernel(
            length_scale_factor=length_scale_factor,
            is_bound=True,
            bound_value=bound_value,
            bound_sig=bound_sig,
            mean=mean,
        )
    if mode == "uniform_se":
        return _uniform_prior_for_any_kernel(
            k,
            length_scale=uniform_length_scale,
            is_bound=True,
            bound_value=bound_value,
            bound_sig=bound_sig,
            mean=mean,
            is_static_kernel=False,
        )
    raise ValueError(f"Unknown prior mode: {mode}")


def _compute_noise_vectors(cfg: RT1CISTomographySingleConfig, *, I0_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    I0_true = np.asarray(I0_true, dtype=float).reshape(-1)
    I0_rms = float(np.sqrt(np.mean(I0_true**2)))
    if cfg.I0_noise_mode == "snr_rms":
        I0_std_level = float(I0_rms / cfg.I0_snr_rms)
    elif cfg.I0_noise_mode == "absolute":
        I0_std_level = float(cfg.I0_obs_noise_level)
    else:
        raise ValueError(f"Unknown I0_noise_mode: {cfg.I0_noise_mode}")
    sigma_I0 = np.full(I0_true.shape, I0_std_level, dtype=float)

    if cfg.I1_noise_mode == "relative_mean_I0":
        I1_std_level = float(cfg.I1_relative_to_I0_mean * np.mean(np.abs(I0_true)))
    elif cfg.I1_noise_mode == "absolute":
        I1_std_level = float(cfg.I1_obs_noise_level)
    else:
        raise ValueError(f"Unknown I1_noise_mode: {cfg.I1_noise_mode}")
    sigma_I1 = np.full(I0_true.shape, I1_std_level, dtype=float)
    return sigma_I0, sigma_I1


def _imshow_channel(ax: plt.Axes, data: np.ndarray, title: str, cmap: str = "viridis") -> None:
    im = ax.imshow(np.asarray(data, dtype=float), origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_cis_truth_and_observed(*, kernel, flux_kernel, e_true, T_true, v_true, I0_obs, IRe_obs, IIm_obs, im_shape):
    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    for ax, arr, title, cmap, kwargs in [
        (axs[0, 0], e_true, "True emissivity", "plasma", {"vmin": 0}),
        (axs[0, 1], T_true, "True temperature", "plasma", {"vmin": 0}),
        (axs[0, 2], v_true, "True velocity", "seismic", {}),
    ]:
        kernel.plot_mosaic(ax=ax, f=np.asarray(arr, dtype=float), size=0.9, back_ground=0, cmap=cmap, **kwargs)
        flux_kernel.plt_rt1_flux(ax=ax, linewidths=0.4, colors="white")
        ax.grid(False)
    _imshow_channel(axs[1, 0], np.asarray(I0_obs).reshape(im_shape), "Observed I0", cmap="viridis")
    _imshow_channel(axs[1, 1], np.asarray(IRe_obs).reshape(im_shape), "Observed IRe", cmap="RdBu_r")
    _imshow_channel(axs[1, 2], np.asarray(IIm_obs).reshape(im_shape), "Observed IIm", cmap="RdBu_r")
    fig.tight_layout()
    return fig


def _plot_cis_reconstruction(*, kernel, flux_kernel, e_true, T_true, v_true, e_mean, e_std, T_mean, T_std, v_mean, v_std):
    err_cmap = "RdBu_r"
    try:
        from gpoloidal.plot_utils import cmocean_balance as cm_balance
        try:
            matplotlib.colormaps.register(cm_balance, name="cm_balance")
        except ValueError:
            pass
        err_cmap = "cm_balance"
    except Exception:
        err_cmap = "RdBu_r"

    fig, axs = plt.subplots(3, 3, figsize=(14, 12), sharex=True, sharey=True)
    rows = [
        ("Emissivity", e_true, e_mean, e_std, "turbo"),
        ("Temperature", T_true, T_mean, T_std, "plasma"),
        ("Velocity", v_true, v_mean, v_std, "seismic"),
    ]
    for r, (label, truth, mean, std, cmap) in enumerate(rows):
        kernel.plot_mosaic(ax=axs[r, 0], f=np.asarray(mean), size=2, back_ground=0, cmap=cmap)
        kernel.plot_mosaic(ax=axs[r, 1], f=np.asarray(mean) - np.asarray(truth), size=2, back_ground=0, cmap=err_cmap, vmean=0.0)
        kernel.plot_mosaic(ax=axs[r, 2], f=np.asarray(std), size=2, back_ground=0, cmap="viridis", vmin=0)
        for c in range(3):
            flux_kernel.plt_rt1_flux(ax=axs[r, c], linewidths=0.4, colors="white")
            axs[r, c].grid(False)
        axs[r, 0].set_title(f"{label}: mean")
        axs[r, 1].set_title(f"{label}: error")
        axs[r, 2].set_title(f"{label}: std")
    fig.tight_layout()
    return fig


def _plot_loss_histories(*, emit_loss: list[float], av_loss: list[float]):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, loss, title in [(axs[0], emit_loss, "Step1 emit loss"), (axs[1], av_loss, "Step3 av loss")]:
        if loss:
            ax.plot(np.log10(np.maximum(loss, 1e-30)))
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.set_ylabel("log10(loss)")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _write_report_md(*, out_path: Path, cfg: RT1CISTomographySingleConfig, metrics: dict[str, Any], diagnostics: dict[str, Any], paths: dict[str, Any], compare_info: dict[str, Any] | None = None) -> None:
    lines = [
        "# RT-1 CIS Tomography Report",
        "",
        "## 実装概要（追加モジュール・主要クラス）",
        "- `gpoloidal.cis.*`（forward / geometry / pipeline / validation / types）",
        "- `gpoloidal.tomography.GPT_cis_av_general`",
        "- `gpoloidal.rt1.phantom.get_cis_phantom_bundle`",
        "- `gpoloidal.scripts.rt1_cis_tomography_single`",
        "",
        "## 論文 / notebook / rt1kernel 対応関係",
        "- 論文 2410.12424v1 の Step1〜Step4（emit -> a,v -> T,v）に対応",
        "- `test_phantom_data1.ipynb` の phantom と `GPT_log` + `GPT_av` フローを `gpoloidal` 内に移植",
        "- `rt1kernel` の `projection_A2` / `GPT_cis(GPT_av)` を参照しつつ `zray` ベースの Dcos 生成に置換",
        "",
        "## 使用 config の要約",
        f"- bundle: `{cfg.phantom_bundle_name}`",
        f"- basis_mode: `{cfg.basis_mode}`, observation_matrix_mode: `{cfg.observation_matrix_mode}`",
        f"- resolution: `{cfg.resolution}`, lnum: `{cfg.lnum}`",
        f"- emit_prior_mode: `{cfg.emit_prior_mode}`, av_prior_mode: `{cfg.av_prior_mode}`",
        "",
        "## phantom 定義（e/T/v）",
        f"- emissivity: `{cfg.phantom_emissivity_name or 'ring_emissivity'}`",
        f"- temperature: `{cfg.phantom_temperature_name or 'simple_temperature'}`",
        f"- velocity: `{cfg.phantom_velocity_name or 'ring_velocity'}`",
        "",
        "## 収束ログ（Step1/Step3）",
        f"- Step1 emit: iters={diagnostics.get('emit_iters')}, converged={diagnostics.get('emit_converged')}, last_loss={diagnostics.get('emit_last_loss')}",
        f"- Step3 a,v: iters={diagnostics.get('av_iters')}, converged={diagnostics.get('av_converged')}, last_loss={diagnostics.get('av_last_loss')}",
        "",
        "## 主要 metrics（emit/cis mll, chi2, 再投影誤差）",
    ]
    for k in sorted(metrics):
        lines.append(f"- {k}: {metrics[k]}")
    lines += [
        "",
        "## 図一覧とパス",
        f"- truth/observed: `{paths.get('truth_observed_plot')}`",
        f"- reconstruction: `{paths.get('reconstruction_plot')}`",
        f"- loss history: `{paths.get('loss_history_plot')}`",
        "",
        "## rt1kernel 比較結果（実施した場合）",
    ]
    if compare_info:
        for k in sorted(compare_info):
            lines.append(f"- {k}: {compare_info[k]}")
    else:
        lines.append("- 今回は未実施")
    lines += [
        "",
        "## 既知の差分・近似・今後の改善点",
        "- Dcos は `zray` ray midpoint + toroidal unit vector から生成（rt1kernel の解析式 Direction_Cos とは別実装）",
        "- 初版は Phantom 対応のみ、実験データ前処理（raw fringe -> I0/IRe/IIm）は未実装",
        "- 初版の script は kernel_weighting / inducing を前提（grid_binning の実運用検証は未実施）",
        "- evidence scan フラグは config にあるが初版 script では未実装（false 前提）",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = RT1CISTomographySingleConfig()
    config_path = (Path(args.config).resolve() if args.config else None)
    if config_path is not None:
        loaded_cfg = load_config_mapping(config_path)
        apply_flat_dataclass_config(cfg, _flatten_rt1_cis_config_mapping(loaded_cfg))
    if args.quick:
        cfg = replace(
            cfg,
            resolution=(24, 24),
            lnum=min(cfg.lnum, 201),
            emit_max_iters=min(cfg.emit_max_iters, 8),
            av_max_iters=min(cfg.av_max_iters, 12),
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

    save_json(output_root / "config_resolved.json", asdict(cfg))
    if config_path is not None:
        save_json(output_root / "config_source.json", {"config_path": str(config_path), "loaded": load_config_mapping(config_path)})

    store = ProjectStore(cache_root=runtime_roots.cache_root, record_root=runtime_roots.backend_record_root)
    prep_cfg = _to_single_prepare_cfg(cfg)
    prepared = prepare_single_tomography(prep_cfg, store=store, show_progress=True)
    k = prepared["kernel"]
    flux_kernel = prepared["flux_kernel"]
    H = np.asarray(prepared["H"], dtype=float)
    ray = prepared["ray"]
    obs_mode = str(prepared["obs_mode"])
    if cfg.basis_mode != "inducing" or obs_mode != "kernel_weighting":
        raise NotImplementedError(
            "Initial CIS script currently supports basis_mode='inducing' with observation_matrix_mode='kernel_weighting'"
        )

    obs_cfg = prepared["obs_cfg"]
    dcos_obs_cfg = replace(
        obs_cfg,
        method=f"{obs_cfg.method}_dcos",
        extras={
            **dict(getattr(obs_cfg, "extras", {}) or {}),
            "matrix_kind": "directional_cosine",
            "source_observation_matrix_artifact_id": prepared["obsmat_rec"].artifact_id,
        },
    )

    def _build_dcos() -> np.ndarray:
        print("Generating Dcos (directional-cosine matrix)...")
        return np.asarray(k.create_dcos_matrix_kernel_weighting(ray=ray, Lnum=cfg.lnum, H=H), dtype=float)

    Dcos_cached, dcos_cache_rec = store.get_or_build_observation_matrix(
        dcos_obs_cfg,
        builder=_build_dcos,
        storage_format="npy",
    )
    Dcos = np.asarray(Dcos_cached, dtype=float)
    if Dcos.shape != H.shape:
        raise RuntimeError(f"Dcos shape {Dcos.shape} does not match H shape {H.shape}")

    bundle = rt1.phantom.get_cis_phantom_bundle(
        cfg.phantom_bundle_name,
        emissivity_name=cfg.phantom_emissivity_name,
        temperature_name=cfg.phantom_temperature_name,
        velocity_name=cfg.phantom_velocity_name,
        emissivity_params=cfg.phantom_emissivity_params,
        temperature_params=cfg.phantom_temperature_params,
        velocity_params=cfg.phantom_velocity_params,
    )
    rI = np.asarray(k.r_idc, dtype=float)
    zI = np.asarray(k.z_idc, dtype=float)
    e_true = np.clip(np.asarray(bundle.emissivity_fn(rI, zI), dtype=float), 1e-12, None)
    T_true = np.clip(np.asarray(bundle.temperature_fn(rI, zI), dtype=float), 0.0, None)
    v_true = np.asarray(bundle.velocity_fn(rI, zI), dtype=float)
    e_hat_true = np.log(e_true)
    I0_true, IRe_true, IIm_true = forward_cis_from_etv(H, Dcos, e_hat_true, T_true, v_true)

    sigma_I0, sigma_I1 = _compute_noise_vectors(cfg, I0_true=I0_true)
    rng = np.random.default_rng(cfg.seed)
    I0_obs = I0_true + sigma_I0 * rng.standard_normal(I0_true.size)
    IRe_obs = IRe_true + sigma_I1 * rng.standard_normal(IRe_true.size)
    IIm_obs = IIm_true + sigma_I1 * rng.standard_normal(IIm_true.size)

    emit_prior_mode = _resolve_emit_prior_mode(cfg)
    av_prior_mode = _resolve_av_prior_mode(cfg)
    K_e_pri, mu_e_pri = _build_prior(
        k,
        mode=emit_prior_mode,
        length_scale_factor=cfg.emit_length_scale_factor,
        uniform_length_scale=cfg.emit_uniform_length_scale,
        bound_sig=cfg.emit_bound_sig,
        bound_value=cfg.emit_log_bound_value,
        mean=cfg.emit_log_prior_mean,
    )
    K_T_pri, mu_T_pri = _build_prior(
        k,
        mode=av_prior_mode,
        length_scale_factor=cfg.av_length_scale_factor,
        uniform_length_scale=cfg.av_uniform_length_scale,
        bound_sig=cfg.av_bound_sig,
        bound_value=cfg.T_bound_value,
        mean=cfg.T_prior_mean,
    )
    K_v_pri, mu_v_pri = _build_prior(
        k,
        mode=av_prior_mode,
        length_scale_factor=cfg.av_length_scale_factor,
        uniform_length_scale=cfg.av_uniform_length_scale,
        bound_sig=cfg.av_bound_sig,
        bound_value=cfg.v_bound_value,
        mean=cfg.v_prior_mean,
    )

    if cfg.do_evidence_scan_emit or cfg.do_evidence_scan_cis:
        print("[warn] evidence scans are not implemented in the initial CIS script; flags are ignored.")

    im_shape = tuple(getattr(ray.Length, "im_shape", cfg.resolution))
    geometry = coerce_observation_geometry(
        H=H,
        Dcos=Dcos,
        im_shape=im_shape,
        metadata={"obs_mode": obs_mode, "ray_index_for_integral": cfg.ray_index_for_integral},
    )
    obs = CISObservedChannels(I0_obs=I0_obs, IRe_obs=IRe_obs, IIm_obs=IIm_obs, sigma_I0=sigma_I0, sigma_I1=sigma_I1)
    recon = CISStepwiseReconstructor(
        emit_max_iters=cfg.emit_max_iters,
        emit_tol=cfg.emit_tol,
        av_max_iters=cfg.av_max_iters,
        av_tol=cfg.av_tol,
        av_consider_w2=cfg.av_consider_w2,
        latent_clip_emit=(cfg.emit_latent_clip_min, cfg.emit_latent_clip_max),
        latent_clip_av=(cfg.av_latent_clip_min, cfg.av_latent_clip_max),
        step_size_emit=cfg.emit_step_size,
        step_size_av=cfg.av_step_size,
    )
    emit_post, av_post, tv_post = recon.fit(
        geometry=geometry,
        obs=obs,
        K_e_pri=np.asarray(K_e_pri, dtype=float),
        mu_e_pri=np.asarray(mu_e_pri, dtype=float),
        K_T_pri=np.asarray(K_T_pri, dtype=float),
        mu_T_pri=np.asarray(mu_T_pri, dtype=float),
        K_v_pri=np.asarray(K_v_pri, dtype=float),
        mu_v_pri=np.asarray(mu_v_pri, dtype=float),
    )
    diag = recon.diagnostics_as_dict()

    e_mean = np.exp(np.asarray(emit_post.mu_e, dtype=float))
    e_std = np.sqrt(np.exp(2.0 * emit_post.mu_e + emit_post.sig_e**2) * (np.exp(emit_post.sig_e**2) - 1.0))
    I0_pred_emit = forward_emit(H, emit_post.mu_e)
    I0_pred_final, IRe_pred_final, IIm_pred_final = forward_cis_from_etv(H, Dcos, emit_post.mu_e, tv_post.mu_T, tv_post.mu_v)
    diag_corr_Tv = np.divide(
        np.diag(tv_post.K_Tv),
        tv_post.sig_T * tv_post.sig_v,
        out=np.zeros_like(tv_post.sig_T),
        where=(tv_post.sig_T > 0) & (tv_post.sig_v > 0),
    )

    metrics: dict[str, Any] = {}
    metrics.update(field_metrics(e_mean, e_true, prefix="e"))
    metrics.update(field_metrics(tv_post.mu_T, T_true, prefix="T"))
    metrics.update(field_metrics(tv_post.mu_v, v_true, prefix="v"))
    metrics["emit_mll"] = float(emit_post.mll_emit)
    metrics["cis_mll"] = float(av_post.mll_cis)
    metrics["I0_chi2_emit"] = mean_chi2(I0_pred_emit, I0_obs, sigma_I0)
    metrics["I0_chi2_final"] = mean_chi2(I0_pred_final, I0_obs, sigma_I0)
    metrics["IRe_chi2_final"] = mean_chi2(IRe_pred_final, IRe_obs, sigma_I1)
    metrics["IIm_chi2_final"] = mean_chi2(IIm_pred_final, IIm_obs, sigma_I1)
    metrics["sigma_I0_level"] = float(np.mean(sigma_I0))
    metrics["sigma_I1_level"] = float(np.mean(sigma_I1))
    metrics["H_rows"] = int(H.shape[0])
    metrics["H_cols"] = int(H.shape[1])
    metrics["nI"] = int(H.shape[1])
    metrics["Dcos_abs_max"] = float(np.max(np.abs(Dcos)))
    metrics["Dcos_abs_mean"] = float(np.mean(np.abs(Dcos[H > 0]))) if np.any(H > 0) else 0.0
    metrics["diag_corr_Tv_abs_max"] = float(np.max(np.abs(diag_corr_Tv))) if diag_corr_Tv.size else 0.0
    metrics.update(diag)

    arrays_path = output_root / "cis_outputs.npz"
    np.savez(
        arrays_path,
        H=H,
        Dcos=Dcos,
        e_true=e_true,
        T_true=T_true,
        v_true=v_true,
        I0_true=I0_true,
        IRe_true=IRe_true,
        IIm_true=IIm_true,
        I0_obs=I0_obs,
        IRe_obs=IRe_obs,
        IIm_obs=IIm_obs,
        sigma_I0=sigma_I0,
        sigma_I1=sigma_I1,
        e_post_log_mean=emit_post.mu_e,
        e_post_log_std=emit_post.sig_e,
        e_post_mean=e_mean,
        e_post_std=e_std,
        a_post_mean=av_post.mu_a,
        a_post_std=av_post.sig_a,
        v_post_mean=av_post.mu_v,
        v_post_std=av_post.sig_v,
        T_post_mean=tv_post.mu_T,
        T_post_std=tv_post.sig_T,
        diag_corr_Tv=diag_corr_Tv,
        I0_pred_emit=I0_pred_emit,
        I0_pred_final=I0_pred_final,
        IRe_pred_final=IRe_pred_final,
        IIm_pred_final=IIm_pred_final,
    )

    truth_plot = figure_dir / "cis_truth_and_observed.png"
    save_figure(
        _plot_cis_truth_and_observed(
            kernel=k,
            flux_kernel=flux_kernel,
            e_true=e_true,
            T_true=T_true,
            v_true=v_true,
            I0_obs=I0_obs,
            IRe_obs=IRe_obs,
            IIm_obs=IIm_obs,
            im_shape=im_shape,
        ),
        truth_plot,
        dpi=cfg.plot_figure_dpi,
    )
    recon_plot = figure_dir / "cis_reconstruction.png"
    save_figure(
        _plot_cis_reconstruction(
            kernel=k,
            flux_kernel=flux_kernel,
            e_true=e_true,
            T_true=T_true,
            v_true=v_true,
            e_mean=e_mean,
            e_std=e_std,
            T_mean=tv_post.mu_T,
            T_std=tv_post.sig_T,
            v_mean=tv_post.mu_v,
            v_std=tv_post.sig_v,
        ),
        recon_plot,
        dpi=cfg.plot_figure_dpi,
    )
    loss_plot = figure_dir / "cis_loss_history.png"
    save_figure(
        _plot_loss_histories(
            emit_loss=(recon.last_diagnostics.emit_loss_history if recon.last_diagnostics else []),
            av_loss=(recon.last_diagnostics.av_loss_history if recon.last_diagnostics else []),
        ),
        loss_plot,
        dpi=cfg.plot_figure_dpi,
    )

    metrics_path = output_root / "metrics.json"
    save_json(metrics_path, metrics)
    report_md = output_root / "report.md"

    backend_outputs: dict[str, Any] = {}
    dcos_result_artifact_id = None
    if record_policy.save_backend_result_artifacts:
        backend_outputs["arrays_artifact_id"] = store.save_result_file("rt1_cis_outputs_npz", arrays_path, kind="file", copy=True).artifact_id
        backend_outputs["truth_plot_artifact_id"] = store.save_result_file("rt1_cis_truth_observed", truth_plot, kind="image", copy=True).artifact_id
        backend_outputs["reconstruction_plot_artifact_id"] = store.save_result_file("rt1_cis_reconstruction", recon_plot, kind="image", copy=True).artifact_id
        backend_outputs["loss_plot_artifact_id"] = store.save_result_file("rt1_cis_loss_history", loss_plot, kind="image", copy=True).artifact_id
        dcos_result_artifact_id = store.save_result_array("cis_dcos", Dcos).artifact_id
        backend_outputs["dcos_artifact_id"] = dcos_result_artifact_id
        backend_outputs["metrics_artifact_id"] = store.save_result_file("rt1_cis_metrics_json", metrics_path, kind="file", copy=True).artifact_id

    run_id = None
    if record_policy.save_run_record:
        record = ExperimentRecord(
            name=cfg.experiment_name,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            observation_matrix_artifact_id=prepared["obsmat_rec"].artifact_id,
            observation_matrix_config=prepared["obs_cfg"],
            phantom=PhantomConfig(
                kind="synthetic",
                name=cfg.phantom_bundle_name,
                generator=CallableRef.from_callable(rt1.phantom.get_cis_phantom_bundle),
                params={
                    "emissivity_name": cfg.phantom_emissivity_name,
                    "temperature_name": cfg.phantom_temperature_name,
                    "velocity_name": cfg.phantom_velocity_name,
                    "emissivity_params": cfg.phantom_emissivity_params,
                    "temperature_params": cfg.phantom_temperature_params,
                    "velocity_params": cfg.phantom_velocity_params,
                },
            ),
            noise=NoiseConfig(
                model="cis_channels",
                profile="flat",
                seed=cfg.seed,
                params={
                    "I0_mode": cfg.I0_noise_mode,
                    "I0_snr_rms": cfg.I0_snr_rms,
                    "I0_obs_noise_level": cfg.I0_obs_noise_level,
                    "I1_mode": cfg.I1_noise_mode,
                    "I1_relative_to_I0_mean": cfg.I1_relative_to_I0_mean,
                    "I1_obs_noise_level": cfg.I1_obs_noise_level,
                },
            ),
            tomography=TomographyConfig(
                model="cis_loggp_stepwise",
                prior_kind=f"emit:{emit_prior_mode},av:{av_prior_mode}",
                extras={
                    "emit_prior_mode": emit_prior_mode,
                    "av_prior_mode": av_prior_mode,
                    "emit_uniform_length_scale": cfg.emit_uniform_length_scale,
                    "av_uniform_length_scale": cfg.av_uniform_length_scale,
                    "emit_length_scale_factor": cfg.emit_length_scale_factor,
                    "av_length_scale_factor": cfg.av_length_scale_factor,
                    "emit_max_iters": cfg.emit_max_iters,
                    "emit_tol": cfg.emit_tol,
                    "av_max_iters": cfg.av_max_iters,
                    "av_tol": cfg.av_tol,
                    "av_consider_w2": cfg.av_consider_w2,
                },
            ),
            references={
                "inducing_points_artifact_id": prepared["ind_rec"].artifact_id,
                "point_source": prepared["point_ref_str"],
                "ray_index_for_integral": cfg.ray_index_for_integral,
                "dcos_cache_artifact_id": dcos_cache_rec.artifact_id,
                "dcos_result_artifact_id": dcos_result_artifact_id,
                **prepared["grid_spec_params"],
            },
            metrics=metrics,
            outputs=backend_outputs,
        )
        run_id = store.save_experiment_record(
            record,
            strict_traceability=record_policy.strict_traceability,
            embed_dependency_manifests=record_policy.embed_dependency_manifests,
        )

    paths = {
        "output_root": str(output_root),
        "output_latest_root": str(layout.latest_root),
        "output_archive_root": str(layout.archive_root),
        "backend_record_root": str(runtime_roots.backend_record_root),
        "cache_root": str(runtime_roots.cache_root),
        "config_resolved": str(output_root / "config_resolved.json"),
        "arrays_npz": str(arrays_path),
        "metrics": str(metrics_path),
        "truth_observed_plot": str(truth_plot),
        "reconstruction_plot": str(recon_plot),
        "loss_history_plot": str(loss_plot),
        "report_md": str(report_md),
        **({"config_source": str(output_root / "config_source.json")} if config_path is not None else {}),
        **({"run_record": str(store.run_dir / f'{run_id}.json')} if run_id else {}),
    }
    _write_report_md(out_path=report_md, cfg=cfg, metrics=metrics, diagnostics=diag, paths=paths, compare_info=None)

    report_json = {
        "script": "gpoloidal.scripts.rt1_cis_tomography_single",
        "gpoloidal_version": gpoloidal.__version__,
        "config_path": str(config_path) if config_path is not None else None,
        "config": asdict(cfg),
        "runtime": {"mode": runtime_roots.mode, "record_mode": record_policy.record_mode},
        "paths": paths,
        "artifacts": {
            "inducing_points_artifact_id": prepared["ind_rec"].artifact_id,
            "observation_matrix_artifact_id": prepared["obsmat_rec"].artifact_id,
            "dcos_cache_artifact_id": dcos_cache_rec.artifact_id,
            "dcos_result_artifact_id": dcos_result_artifact_id,
            "run_id": run_id,
        },
        "metrics": metrics,
        "diagnostics": diag,
    }
    save_json(output_root / "latest_report.json", report_json)
    save_json(output_root / "latest_paths.json", paths)
    save_json(
        output_root / "run_ref.json",
        make_run_reference(
            script="gpoloidal.scripts.rt1_cis_tomography_single",
            archive_run_root=output_root,
            latest_root=layout.latest_root,
            backend_record_root=runtime_roots.backend_record_root,
            run_id=run_id,
            backend_run_record_path=(store.run_dir / f"{run_id}.json") if run_id else None,
            extra={
                "observation_matrix_artifact_id": prepared["obsmat_rec"].artifact_id,
                "dcos_cache_artifact_id": dcos_cache_rec.artifact_id,
                "inducing_points_artifact_id": prepared["ind_rec"].artifact_id,
                "record_mode": record_policy.record_mode,
                "basis_mode": cfg.basis_mode,
                "obs_mode": obs_mode,
            },
        ),
    )
    publish_latest_from_archive(layout)

    print(json.dumps({"run_id": run_id, "archive_run_root": str(output_root), "latest_root": str(layout.latest_root), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
