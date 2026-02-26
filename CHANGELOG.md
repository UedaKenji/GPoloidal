# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-02-26

### Added
- `gpoloidal.core` package group for reusable numerical building blocks (`metrics`, compatibility re-exports for `tomography`, `kernel`, `geometry_matrix`).
- `gpoloidal.analysis` package group for reusable experiment-analysis helpers (`config`, `noise_sweep`, `profiles`, `hparam_sweep`).
- Tests for new `core/analysis` helpers (`tests/test_core_analysis_modules.py`).

### Changed
- `benchmark_utils.py` converted to a backward-compatible shim that re-exports from `gpoloidal.core` / `gpoloidal.analysis`.
- Benchmark scripts and external templates now import reusable helpers from `gpoloidal.core` / `gpoloidal.analysis`.
- Scripts documentation updated to describe the layered module structure (`core`, `analysis`, `rt1`, `experiment`, `scripts`).

## [0.2.0] - 2026-02-25

### Added
- `gpoloidal.experiment` module for traceable experiment/config management.
- Cache/registry support for observation matrices and inducing-point artifacts.
- Run record persistence with traceability metadata and dependency manifest embedding.
- Result artifact saving helpers (arrays, sparse covariance, summary image files).
- `GPT_lin_general` and `GPT_lin_general_2dim_prior` linear-GP tomography solvers.
- Validation and regression tests (`tests/`).
- Validation notebook: `validation_projectstore_tomography.ipynb`.
- Benchmark notebooks for `linGP` vs `logGP` comparison and noise sweep.

### Changed
- `GPT_log_general` API cleanup and naming improvements for observation noise parameters.
- `GPT_log_general_2dim_prior` simplified to wrap the generalized implementation.
- `ProjectStore` default storage layout split into user cache and project-local records.
- `.gitignore` updated to ignore local `.gpoloidal_store/`.

### Fixed
- `Kernel2D_scatter.set_grid_interface(add_bound=True)` length-scale function reference bug.
- `tomography` numerical stability issues in posterior std calculation (`sqrt` clipping).
- `tomography` docstring escape warnings and `log_det` exception handling.
- `postprocess()` state consistency issues in log-GP tomography workflow.

### Removed
- Legacy `GPT_log` class (superseded by `GPT_log_general`).

## [0.1.0] - 2026-02-25

### Added
- Initial packaged `gpoloidal` release with `uv`-friendly packaging metadata.
