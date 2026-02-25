from __future__ import annotations

"""
Compatibility entry point for the RT1 linGP/logGP benchmark.

The sequential implementation (`rt1_loggp_lingp_benchmark_seq.py`) is the maintained
version. This module remains as a stable command target and delegates to it.
"""

import warnings
import runpy


def main() -> None:
    warnings.warn(
        "gpoloidal.scripts.rt1_loggp_lingp_benchmark is maintained as a compatibility wrapper. "
        "Use gpoloidal.scripts.rt1_loggp_lingp_benchmark_seq for the canonical implementation.",
        DeprecationWarning,
        stacklevel=2,
    )
    runpy.run_module("gpoloidal.scripts.rt1_loggp_lingp_benchmark_seq", run_name="__main__")


if __name__ == "__main__":
    main()
