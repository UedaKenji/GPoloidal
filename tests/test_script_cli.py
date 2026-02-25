from __future__ import annotations

from pathlib import Path

from gpoloidal.script_cli import resolve_record_mode_policy, resolve_runtime_roots


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
