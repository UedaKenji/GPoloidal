"""Backward-compatible shim for legacy benchmark utility imports.

New code should prefer:
  - ``gpoloidal.core.metrics``
  - ``gpoloidal.analysis.config``
  - ``gpoloidal.analysis.noise_sweep``
"""

from .analysis.config import *  # noqa: F401,F403
from .analysis.noise_sweep import *  # noqa: F401,F403
from .core.metrics import *  # noqa: F401,F403

