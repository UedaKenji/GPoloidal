from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
import json
import importlib

import numpy as np
import pandas as pd


def safe_corrcoef(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) < eps or np.std(y) < eps:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def field_metrics(f_est: np.ndarray, f_true: np.ndarray, *, prefix: str) -> dict[str, float]:
    f_est = np.asarray(f_est, dtype=float)
    f_true = np.asarray(f_true, dtype=float)
    diff = f_est - f_true
    return {
        f"{prefix}_rmse": float(np.sqrt(np.mean(diff**2))),
        f"{prefix}_mae": float(np.mean(np.abs(diff))),
        f"{prefix}_rel_rmse": float(np.linalg.norm(diff) / (np.linalg.norm(f_true) + 1e-12)),
        f"{prefix}_corr": safe_corrcoef(f_est, f_true),
        f"{prefix}_neg_frac": float(np.mean(f_est < 0)),
    }


def mean_chi2(g_pred: np.ndarray, g_obs: np.ndarray, obs_noise_std: np.ndarray) -> float:
    g_pred = np.asarray(g_pred, dtype=float)
    g_obs = np.asarray(g_obs, dtype=float)
    obs_noise_std = np.asarray(obs_noise_std, dtype=float)
    return float(np.mean(((g_pred - g_obs) / (obs_noise_std + 1e-12)) ** 2))


def summarize_noise_sweep(results_df: pd.DataFrame) -> pd.DataFrame:
    return (
        results_df.groupby("snr_rms_target")
        .agg(
            n_trials=("trial", "count"),
            lin_rmse_mean=("lin_rmse", "mean"),
            lin_rmse_std=("lin_rmse", "std"),
            log_rmse_mean=("log_rmse", "mean"),
            log_rmse_std=("log_rmse", "std"),
            lin_rel_rmse_mean=("lin_rel_rmse", "mean"),
            log_rel_rmse_mean=("log_rel_rmse", "mean"),
            lin_corr_mean=("lin_corr", "mean"),
            log_corr_mean=("log_corr", "mean"),
            lin_neg_frac_mean=("lin_neg_frac", "mean"),
            lin_chi2_mean=("lin_chi2", "mean"),
            log_chi2_mean=("log_chi2", "mean"),
            log_iters_mean=("log_iters", "mean"),
            log_nonconverged_frac=("log_converged", lambda s: float(1.0 - np.mean(np.asarray(s, dtype=float)))),
            log_retry_frac=("log_retried", lambda s: float(np.mean(np.asarray(s, dtype=float)))),
        )
        .sort_index(ascending=False)
    )


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
    """Apply a flat mapping to a dataclass instance with basic validation.

    - unknown keys -> ValueError
    - list -> tuple coercion when the current field value is a tuple
    """
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
