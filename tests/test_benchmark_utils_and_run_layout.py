from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from gpoloidal.benchmark_utils import apply_flat_dataclass_config, load_config_mapping, save_json
from gpoloidal.run_layout import make_run_reference, prepare_local_run_layout, publish_latest_from_archive


@dataclass
class _Cfg:
    a: int = 1
    pair: tuple[int, int] = (1, 2)


def test_apply_flat_dataclass_config_validates_and_coerces():
    cfg = _Cfg()
    apply_flat_dataclass_config(cfg, {"a": 3, "pair": [10, 20]})
    assert cfg.a == 3
    assert cfg.pair == (10, 20)

    with pytest.raises(ValueError):
        apply_flat_dataclass_config(cfg, {"unknown": 1})


def test_load_config_mapping_json_toml_yaml(tmp_path: Path):
    j = tmp_path / "cfg.json"
    j.write_text('{"a": 1}', encoding="utf-8")
    assert load_config_mapping(j) == {"a": 1}

    t = tmp_path / "cfg.toml"
    t.write_text("a = 2\n", encoding="utf-8")
    assert load_config_mapping(t) == {"a": 2}

    y = tmp_path / "cfg.yaml"
    y.write_text("a: 3\n", encoding="utf-8")
    try:
        data = load_config_mapping(y)
    except ValueError as e:
        # PyYAML is optional; error should be informative.
        assert "PyYAML" in str(e)
    else:
        assert data == {"a": 3}


def test_local_run_layout_publish_and_run_reference(tmp_path: Path):
    layout = prepare_local_run_layout(
        base_dir=tmp_path / "analysis_runs",
        experiment_name="rt1 benchmark",
        run_name="case A",
        timestamp="20260226_120000",
    )
    # archive run contents
    (layout.run_root / "figures").mkdir(parents=True, exist_ok=True)
    (layout.run_root / "figures" / "plot.txt").write_text("ok", encoding="utf-8")
    save_json(layout.run_root / "latest_report.json", {"x": 1})

    publish_latest_from_archive(layout)

    assert (layout.latest_root / "figures" / "plot.txt").read_text(encoding="utf-8") == "ok"
    assert (layout.latest_root / "latest_report.json").is_file()

    run_ref = make_run_reference(
        script="gpoloidal.scripts.example",
        archive_run_root=layout.run_root,
        latest_root=layout.latest_root,
        backend_record_root=tmp_path / "records",
        run_id="run_abc123",
        backend_run_record_path=tmp_path / "records" / "exp" / "runs" / "run_abc123.json",
        extra={"foo": "bar"},
    )
    assert run_ref["run_id"] == "run_abc123"
    assert run_ref["script"] == "gpoloidal.scripts.example"
    assert run_ref["extra"]["foo"] == "bar"


def test_local_run_layout_avoids_archive_label_collision(tmp_path: Path):
    l1 = prepare_local_run_layout(
        base_dir=tmp_path / "analysis_runs",
        experiment_name="rt1",
        timestamp="20260226_120000",
        run_name="case",
    )
    l2 = prepare_local_run_layout(
        base_dir=tmp_path / "analysis_runs",
        experiment_name="rt1",
        timestamp="20260226_120000",
        run_name="case",
    )
    assert l1.run_root != l2.run_root
    assert l2.run_root.name.startswith(l1.run_label)
