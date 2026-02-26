from __future__ import annotations

from pathlib import Path

import pytest

from gpoloidal.script_cli import resolve_record_mode_policy, resolve_runtime_roots
from gpoloidal.scripts.rt1_tomography_single import _flatten_single_tomography_config_mapping
from gpoloidal.scripts.rt1_tomography_sweep import SweepConfig, _case_seed, _flatten_sweep_config_mapping


def test_resolve_record_mode_policy_variants():
    p = resolve_record_mode_policy(record_mode="light", no_run_record=False)
    assert p.save_run_record is True
    assert p.strict_traceability is False
    assert p.record_mode == "light"

    p = resolve_record_mode_policy(record_mode="archive", no_run_record=False)
    assert p.save_run_record is True
    assert p.strict_traceability is True
    assert p.embed_dependency_manifests is True
    assert p.save_backend_result_artifacts is True

    p = resolve_record_mode_policy(record_mode="archive", no_run_record=True)
    assert p.record_mode == "none"
    assert p.save_run_record is False


def test_resolve_runtime_roots_modes(tmp_path: Path):
    project_root = tmp_path / "repo"
    project_root.mkdir()

    dev = resolve_runtime_roots(
        mode="dev",
        project_root=project_root,
        backend_experiment_name="expA",
        output_dir=None,
        backend_record_dir=None,
        cache_root=tmp_path / "cache",
    )
    assert dev.output_base_dir == project_root / "analysis_runs"
    assert dev.backend_record_root.name == "expA"

    custom = resolve_runtime_roots(
        mode="analysis",
        project_root=project_root,
        backend_experiment_name="expA",
        output_dir=tmp_path / "out",
        backend_record_dir=tmp_path / "records",
        cache_root=tmp_path / "cache",
    )
    assert custom.output_base_dir == tmp_path / "out"
    assert custom.backend_record_root == tmp_path / "records"
    assert custom.cache_root == tmp_path / "cache"


def test_flatten_single_tomography_config_mapping_accepts_shallow_sections():
    data = {
        "experiment": {"name": "exp1"},
        "basis": {"mode": "grid", "matrix_mode": "auto"},
        "grid": {"r_min": 0.2, "r_max": 1.01, "r_step": 0.01, "z_min": -0.6, "z_max": 0.6, "z_step": 0.01},
        "observation": {
            "resolution": [48, 48],
            "nreflections": 1,
            "ray_index_for_integral": 1,
            "camera_location": [1.2, 0.0, 0.0],
            "pass_through_first": True,
        },
        "phantom": {"name": "hollow", "params": {"radius": 0.4}},
        "tomography": {"method": "logGP", "prior_mode": "gibbs", "length_scale_factor": 2.0},
        "noise": {"mode": "snr_rms", "snr_rms": 10, "seed": 42},
        "plot": {"r_num": 301, "z_num": 301},
    }
    flat = _flatten_single_tomography_config_mapping(data)
    assert flat["experiment_name"] == "exp1"
    assert flat["basis_mode"] == "grid"
    assert flat["observation_matrix_mode"] == "auto"
    assert flat["grid_r_min"] == 0.2
    assert flat["resolution"] == [48, 48]
    assert flat["camera_location"] == [1.2, 0.0, 0.0]
    assert flat["pass_through_first"] is True
    assert flat["phantom_name"] == "hollow"
    assert flat["phantom_params"] == {"radius": 0.4}
    assert flat["tomography_method"] == "logGP"
    assert flat["noise_mode"] == "snr_rms"
    assert flat["plot_r_num"] == 301


def test_flatten_sweep_config_mapping_accepts_single_and_sweep_sections():
    data = {
        "single": {
            "basis": {"mode": "inducing"},
            "tomography": {"method": "logGP"},
            "noise": {"mode": "snr_rms", "snr_rms": 10.0},
        },
        "sweep": {
            "experiment": {"name": "sweep1"},
            "replicate": {"n_trials": 3, "seed": 123, "seed_policy": "by_trial"},
            "report": {"summary_x_key": "noise.snr_rms", "save_case_plots": True},
            "axes": [
                {"key": "tomography.method", "values": ["linGP", "logGP"]},
                {"key": "noise.snr_rms", "values": [30.0, 10.0, 3.0]},
            ],
        },
    }
    single_flat, sweep_flat = _flatten_sweep_config_mapping(data)
    assert single_flat["basis_mode"] == "inducing"
    assert single_flat["tomography_method"] == "logGP"
    assert sweep_flat["experiment_name"] == "sweep1"
    assert sweep_flat["n_trials"] == 3
    assert sweep_flat["seed"] == 123
    assert sweep_flat["seed_policy"] == "by_trial"
    assert sweep_flat["summary_x_key"] == "noise.snr_rms"
    assert sweep_flat["save_case_plots"] is True
    assert sweep_flat["axes"][0]["key"] == "tomography.method"


def test_flatten_sweep_config_mapping_rejects_unknown_seed_policy():
    data = {
        "single": {"basis": {"mode": "inducing"}},
        "sweep": {"replicate": {"n_trials": 2, "seed": 42, "seed_policy": "bad_policy"}},
    }
    with pytest.raises(ValueError):
        _flatten_sweep_config_mapping(data)


def test_case_seed_by_trial_is_combo_order_independent():
    cfg = SweepConfig(seed=42, seed_policy="by_trial")
    s1 = _case_seed(cfg, combo_i=0, combo={"tomography.method": "linGP"}, trial=1)
    s2 = _case_seed(cfg, combo_i=99, combo={"tomography.method": "logGP"}, trial=1)
    assert s1 == s2 == 43


def test_case_seed_by_combo_trial_keeps_legacy_behavior():
    cfg = SweepConfig(seed=42, seed_policy="by_combo_trial")
    s1 = _case_seed(cfg, combo_i=0, combo={"a": 1}, trial=0)
    s2 = _case_seed(cfg, combo_i=1, combo={"a": 1}, trial=0)
    assert s1 == 42
    assert s2 == 100042
