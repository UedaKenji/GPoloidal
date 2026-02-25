from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sps

from gpoloidal.experiment import (
    CallableRef,
    CameraConfig,
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
)


def dummy_length_sq(r, z):
    r = np.asarray(r, dtype=float)
    return np.ones_like(r)


def _make_inducing_config(tmp_path: Path) -> InducingPointConfig:
    src = tmp_path / "raw_points.npz"
    np.savez(
        src,
        r_idc=np.array([0.5, 0.6, 0.7]),
        z_idc=np.array([0.0, 0.1, -0.1]),
        r_bd=np.array([1.0]),
        z_bd=np.array([0.0]),
    )
    return InducingPointConfig(
        source=FileRef.from_path(src),
        stride=2,
        length_sq_function=CallableRef.from_callable(dummy_length_sq),
        note="test",
    )


def _make_obs_config(inducing_cfg: InducingPointConfig) -> ObservationMatrixConfig:
    return ObservationMatrixConfig(
        method="kernel_weighting",
        lnum=41,
        vessel=VesselConfig(package_resource="gpoloidal.rt1:rt1_simple_frame.json"),
        camera=CameraConfig(
            kind="Camera2D_rphiz",
            params={
                "focal_length": 0.01,
                "resolution": [8, 8],
                "location": [1.2, 0.0, 0.0],
            },
        ),
        raytrace=RaytraceConfig(nreflections=1, pass_through_first=True, ray_index_for_integral=1),
        inducing_points=inducing_cfg,
    )


def test_project_store_cache_and_traceability_roundtrip(tmp_path: Path):
    cache_root = tmp_path / "cache"
    record_root = tmp_path / "project" / ".gpoloidal_store"
    store = ProjectStore(cache_root=cache_root, record_root=record_root)

    assert store.layout_mode == "split"
    assert store.obsmat_dir.parent == cache_root
    assert store.results_dir.parent == record_root

    inducing_cfg = _make_inducing_config(tmp_path)
    build_counts = {"indpts": 0, "obsmat": 0}

    def build_inducing():
        build_counts["indpts"] += 1
        return {
            "r_idc": np.array([0.5, 0.7]),
            "z_idc": np.array([0.0, -0.1]),
            "r_bd": np.array([1.0]),
            "z_bd": np.array([0.0]),
        }

    arrays1, ind_rec1 = store.get_or_build_inducing_points(inducing_cfg, builder=build_inducing)
    arrays2, ind_rec2 = store.get_or_build_inducing_points(inducing_cfg, builder=build_inducing)
    assert build_counts["indpts"] == 1
    assert ind_rec1.artifact_id == ind_rec2.artifact_id
    assert np.allclose(arrays1["r_idc"], arrays2["r_idc"])

    obs_cfg = _make_obs_config(inducing_cfg)
    H_ref = np.arange(20, dtype=float).reshape(4, 5)

    def build_obsmat():
        build_counts["obsmat"] += 1
        return H_ref

    H1, ob_rec1 = store.get_or_build_observation_matrix(obs_cfg, builder=build_obsmat)
    H2, ob_rec2 = store.get_or_build_observation_matrix(obs_cfg, builder=build_obsmat)
    assert build_counts["obsmat"] == 1
    assert ob_rec1.artifact_id == ob_rec2.artifact_id
    assert np.allclose(np.asarray(H1), H_ref)
    assert np.allclose(np.asarray(H2), H_ref)

    summary_png = tmp_path / "summary.png"
    summary_png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    outputs = store.save_tomography_outputs(
        mean=np.array([1.0, 2.0]),
        std=np.array([0.1, 0.2]),
        covariance=sps.eye(2, format="csr"),
        summary_image_path=summary_png,
        prefix="unit",
        extra_metadata={"case": "roundtrip"},
    )
    assert {"mean_artifact_id", "std_artifact_id", "covariance_artifact_id", "summary_image_artifact_id"} <= outputs.keys()

    rec = ExperimentRecord(
        name="unit_roundtrip",
        created_at_utc="2026-02-25T00:00:00+00:00",
        observation_matrix_artifact_id=ob_rec1.artifact_id,
        observation_matrix_config=obs_cfg,
        phantom=PhantomConfig(kind="synthetic", name="double"),
        noise=NoiseConfig(model="gaussian", level=0.1, level_definition="obs_noise_level", profile="flat", seed=123),
        tomography=TomographyConfig(
            model="logGP",
            prior_kind="kernel.set_kernel",
            length_scale_factor=1.4,
            boundary_sigma=0.1,
            boundary_value=-5.0,
            prior_mean=-3.0,
            normalize=False,
            obs_noise_level=0.1,
            max_iters=20,
            tol=1e-5,
        ),
        references={"inducing_points_artifact_id": ind_rec1.artifact_id},
        metrics={"rmse": 0.12, "chi2": 1.3},
        outputs=outputs,
    )
    run_id = store.save_experiment_record(rec, strict_traceability=True, embed_dependency_manifests=True)
    loaded = store.load_experiment_record(run_id)

    assert "traceability" in loaded
    trace = loaded["traceability"]
    assert "config_hashes" in trace
    assert "record_hash" in trace
    assert "dependency_manifests" in trace
    assert "observation_matrix" in trace["dependency_manifests"]
    assert "inducing_points" in trace["dependency_manifests"]
    assert "results" in trace["dependency_manifests"]

    # Manifests are split: cache artifacts in cache_root, results/runs in record_root
    assert store._manifest_path(ob_rec1.artifact_id).is_file()
    assert store._manifest_path(ind_rec1.artifact_id).is_file()
    assert str(store._manifest_path(ob_rec1.artifact_id)).startswith(str(cache_root))
    assert str(store._manifest_path(outputs["mean_artifact_id"])).startswith(str(record_root))


def test_strict_traceability_rejects_mismatched_observation_matrix_config(tmp_path: Path):
    store = ProjectStore(cache_root=tmp_path / "cache", record_root=tmp_path / "records")
    inducing_cfg = _make_inducing_config(tmp_path)
    obs_cfg = _make_obs_config(inducing_cfg)

    _, ob_rec = store.get_or_build_observation_matrix(obs_cfg, builder=lambda: np.ones((2, 2)))
    bad_obs_cfg = replace(obs_cfg, lnum=999)

    rec = ExperimentRecord(
        name="bad_trace",
        created_at_utc="2026-02-25T00:00:00+00:00",
        observation_matrix_artifact_id=ob_rec.artifact_id,
        observation_matrix_config=bad_obs_cfg,
        phantom=None,
        noise=None,
        tomography=None,
    )

    with pytest.raises(ValueError):
        store.save_experiment_record(rec, strict_traceability=True)


def test_cache_key_ignores_descriptive_metadata(tmp_path: Path):
    store = ProjectStore(cache_root=tmp_path / "cache", record_root=tmp_path / "records")
    inducing_cfg = _make_inducing_config(tmp_path)
    obs_cfg = _make_obs_config(inducing_cfg)

    inducing_cfg_2 = replace(inducing_cfg, note="different note")
    obs_cfg_2 = replace(
        obs_cfg,
        package_versions={"gpoloidal": "0.2.0", "numpy": "x"},
        extras={"note": "human description only"},
    )

    assert store._build_inducing_points_artifact_id(inducing_cfg) == store._build_inducing_points_artifact_id(inducing_cfg_2)
    assert store._build_matrix_artifact_id(obs_cfg) == store._build_matrix_artifact_id(obs_cfg_2)
