from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import scipy.sparse as sps


ArrayLikeMatrix = np.ndarray | sps.spmatrix


@dataclass(frozen=True)
class CISObservationGeometry:
    """CIS observation geometry matrices.

    ``Dcos`` corresponds to the directional-cosine matrix (paper ``Theta``).
    """

    H: ArrayLikeMatrix
    Dcos: ArrayLikeMatrix
    mask: np.ndarray | None = None
    im_shape: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CISObservedChannels:
    I0_obs: np.ndarray
    IRe_obs: np.ndarray
    IIm_obs: np.ndarray
    sigma_I0: np.ndarray
    sigma_I1: np.ndarray


@dataclass(frozen=True)
class CISEmitPosterior:
    mu_e: np.ndarray
    K_e: np.ndarray
    sig_e: np.ndarray
    mll_emit: float


@dataclass(frozen=True)
class CISAVPosterior:
    mu_a: np.ndarray
    mu_v: np.ndarray
    K_aa: np.ndarray
    K_av: np.ndarray
    K_va: np.ndarray
    K_vv: np.ndarray
    sig_a: np.ndarray
    sig_v: np.ndarray
    mll_cis: float


@dataclass(frozen=True)
class CISTVPosterior:
    mu_T: np.ndarray
    mu_v: np.ndarray
    K_TT: np.ndarray
    K_Tv: np.ndarray
    K_vT: np.ndarray
    K_vv: np.ndarray
    sig_T: np.ndarray
    sig_v: np.ndarray


@dataclass(frozen=True)
class CISPhantomBundle:
    emissivity_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    temperature_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    velocity_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)

