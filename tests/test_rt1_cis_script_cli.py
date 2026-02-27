from __future__ import annotations

from gpoloidal.scripts.rt1_cis_tomography_single import _flatten_rt1_cis_config_mapping


def test_flatten_rt1_cis_config_mapping_nested_sections():
    cfg = {
        "experiment": {"name": "rt1_cis_test"},
        "basis": {"mode": "inducing", "matrix_mode": "auto", "point_file": "example/rt1tomography/point_temp.npz"},
        "observation": {"resolution": [24, 24], "lnum": 201},
        "phantom": {
            "bundle_name": "paper_phantom1",
            "emissivity_name": "ring_emissivity",
            "temperature_name": "simple_temperature",
            "velocity_name": "ring_velocity",
            "emissivity_params": {"n0": 1.2},
            "temperature_params": {"a": 8.0},
            "velocity_params": {"ring_scale": 12.0},
        },
        "cis": {"emit_prior_mode": "auto", "av_prior_mode": "uniform_se", "emit_max_iters": 10, "av_max_iters": 12},
        "noise": {"I0_mode": "snr_rms", "I0_snr_rms": 8.0, "I1_mode": "relative_mean_I0", "I1_relative_to_I0_mean": 0.02, "seed": 1},
    }
    flat = _flatten_rt1_cis_config_mapping(cfg)
    assert flat["experiment_name"] == "rt1_cis_test"
    assert flat["resolution"] == [24, 24]
    assert flat["lnum"] == 201
    assert flat["phantom_bundle_name"] == "paper_phantom1"
    assert flat["phantom_emissivity_name"] == "ring_emissivity"
    assert flat["emit_max_iters"] == 10
    assert flat["av_max_iters"] == 12
    assert flat["I0_noise_mode"] == "snr_rms"
    assert flat["I1_noise_mode"] == "relative_mean_I0"

