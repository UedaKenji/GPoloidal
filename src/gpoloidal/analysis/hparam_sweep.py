from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Mapping

import pandas as pd


@dataclass(frozen=True)
class SweepGrid:
    x_name: str
    x_values: tuple[float, ...]
    y_name: str
    y_values: tuple[float, ...]


def run_2d_hparam_sweep(*, grid: SweepGrid, evaluator: Callable[..., Mapping[str, Any]]) -> pd.DataFrame:
    """Run a simple 2D hyperparameter sweep and return a tidy DataFrame."""
    rows: list[dict[str, Any]] = []
    for x, y in product(grid.x_values, grid.y_values):
        rec = {grid.x_name: x, grid.y_name: y}
        rec.update(dict(evaluator(**{grid.x_name: x, grid.y_name: y})))
        rows.append(rec)
    return pd.DataFrame.from_records(rows)


def pivot_metric(df: pd.DataFrame, *, grid: SweepGrid, metric: str) -> pd.DataFrame:
    return df.pivot(index=grid.x_name, columns=grid.y_name, values=metric).sort_index().sort_index(axis=1)

