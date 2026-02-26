from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
import importlib
import json


def save_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def load_config_mapping(path: str | Path) -> dict:
    """Load a flat config mapping from JSON/TOML/YAML."""
    path = Path(path)
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(text)
    elif suffix == ".toml":
        import tomllib

        data = tomllib.loads(text)
    elif suffix in {".yaml", ".yml"}:
        if importlib.util.find_spec("yaml") is None:
            raise ValueError(
                f"Unsupported config format: {path} (YAML requires PyYAML; install with `uv add --group dev pyyaml`)."
            )
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported config format: {path} (use .json, .toml, or .yaml)")
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def apply_flat_dataclass_config(instance, updates: dict):
    """Apply a flat mapping to a dataclass instance with validation and tuple coercion."""
    if not is_dataclass(instance):
        raise TypeError("instance must be a dataclass instance")
    if not isinstance(updates, dict):
        raise TypeError("updates must be a mapping")

    valid = {f.name for f in fields(instance)}
    unknown = sorted(set(updates) - valid)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    for key, value in updates.items():
        current = getattr(instance, key)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(instance, key, value)
    return instance

