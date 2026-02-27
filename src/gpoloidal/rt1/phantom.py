from __future__ import annotations

from functools import partial
from typing import Any, Callable

import numpy as np

from . import mag

# Module-level defaults for low-level phantom primitives.
# Use explicit names to avoid accidental collisions in interactive sessions.
DEFAULT_GAUSSIAN_N0 = 2  # 25.99e16*0.8/2
DEFAULT_GAUSSIAN_A = 1.348
DEFAULT_GAUSSIAN_B = 0.5
DEFAULT_GAUSSIAN_RMAX = 0.4577


def _safe_pow_ratio(numerator, denominator, exponent):
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den != 0)
        out = np.power(ratio, exponent)
    out = np.asarray(out, dtype=float)
    out[~np.isfinite(out)] = 0.0
    return out


def _safe_divide(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    shape = np.broadcast(num, den).shape
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = np.divide(num, den, out=np.zeros(shape, dtype=float), where=den != 0)
    out = np.asarray(out, dtype=float)
    out[~np.isfinite(out)] = 0.0
    return out


def gaussian(
    r,
    z,
    n0: float = DEFAULT_GAUSSIAN_N0,
    a: float = DEFAULT_GAUSSIAN_A,
    b: float = DEFAULT_GAUSSIAN_B,
    rmax: float = DEFAULT_GAUSSIAN_RMAX,
    separatrix: bool = True,
):
    psi = mag.psi(r, z, separatrix=separatrix)
    br, bz = mag.bvec(r, z, separatrix=separatrix)
    b_abs = np.sqrt(br**2 + bz**2)
    psi_rmax = mag.psi(rmax, 0, separatrix=separatrix)
    psi0 = mag.psi(1, 0, separatrix=separatrix)
    b0 = mag.b0(r, z, separatrix=separatrix)
    return n0 * np.exp(-a * (psi - psi_rmax) ** 2 / psi0**2) * _safe_pow_ratio(b_abs, b0, -b)


def Length_scale_sq(r, z):
    return 0.0002 / (gaussian(r, z) + 0.05)


def Length_scale(r, z):
    return np.sqrt(Length_scale_sq(r, z))


DEFAULT_PSI_SEPARATRIX = -0.006376568930277712


def sep_factor(r, z):
    psi = mag.psi(r, z, separatrix=True)
    with np.errstate(over="ignore"):
        return 1 / (1 + np.exp(+1000 * (psi - DEFAULT_PSI_SEPARATRIX)))


def func_ring(
    r,
    z,
    n0: float = DEFAULT_GAUSSIAN_N0,
    a: float = DEFAULT_GAUSSIAN_A,
    b: float = DEFAULT_GAUSSIAN_B,
    rmax: float = DEFAULT_GAUSSIAN_RMAX,
    radius: float = 0.5,
    separatrix: bool = True,
):
    psi = mag.psi(r, z, separatrix)
    br, bz = mag.bvec(r, z, separatrix)
    b_abs = np.sqrt(br**2 + bz**2)
    psi_rmax = mag.psi(rmax, 0, separatrix)
    psi0 = mag.psi(1, 0, separatrix)
    b0 = mag.b0(r, z, separatrix)
    b2 = _safe_divide(b_abs, b0)
    if separatrix:
        rs, zs = r.flatten()[np.argmin(b_abs)], z.flatten()[np.argmin(b_abs)]
        fac = (1 - np.exp(-50 * ((r - rs) ** 2 + (z - zs) ** 2)))
    else:
        fac = 1
    q = np.sqrt(((psi - psi_rmax) / psi0) ** 2 + (1 - _safe_divide(b, b2)) ** 2) - radius
    f_cauchy = fac * n0 * a**2 / (q**2 + a**2) * (1 - np.exp(-100 * (r - 1) ** 2))
    return f_cauchy


def _apply_upper_z_clip(f, z, z_clip_max: float | None):
    if z_clip_max is None:
        return f
    z_arr = np.asarray(z, dtype=float)
    return np.where(z_arr > z_clip_max, 0.0, f)


def _two_wall_factor(r, *, c1: float = 200.0, c2: float = 100.0) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return (1 - np.exp(-c1 * (r - 1.0) ** 2)) * (1 - np.exp(-c2 * (r - 1.0) ** 2))


def _upper_sigmoid(z, *, z0: float = 0.4, sharpness: float = 50.0) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(sharpness * (z - z0)))


def phantom_hollow(
    r,
    z,
    *,
    n0: float = 1.0,
    a: float = 0.06,
    b: float = 0.8,
    rmax: float = 0.58,
    radius: float = 0.42,
    separatrix: bool = True,
):
    return func_ring(r=r, z=z, n0=n0, a=a, b=b, rmax=rmax, radius=radius, separatrix=separatrix)


def phantom_single(
    r,
    z,
    *,
    n0: float = 1.0,
    a: float = 6.3,
    b: float = 1.0,
    rmax: float = 0.53,
    separatrix: bool = True,
    apply_sep_factor: bool = True,
    z_clip_max: float | None = 0.48,
):
    f = gaussian(r=r, z=z, n0=n0, a=a, b=b, rmax=rmax, separatrix=separatrix)
    if apply_sep_factor:
        f = f * sep_factor(r, z)
    return _apply_upper_z_clip(f, z, z_clip_max)


def phantom_double(
    r,
    z,
    *,
    # first peak
    n0_1: float = 1.0,
    a_1: float = 15.0,
    b_1: float = 0.65,
    rmax_1: float = 0.65,
    separatrix_1: bool = True,
    # second peak
    weight_2: float = 3.0,
    n0_2: float = 1.0,
    a_2: float = 35.0,
    b_2: float = 2.0,
    rmax_2: float = 0.45,
    separatrix_2: bool = True,
    # post-processing
    apply_sep_factor: bool = True,
    z_clip_max: float | None = 0.48,
):
    f = gaussian(r=r, z=z, n0=n0_1, a=a_1, b=b_1, rmax=rmax_1, separatrix=separatrix_1)
    f = f + weight_2 * gaussian(r=r, z=z, n0=n0_2, a=a_2, b=b_2, rmax=rmax_2, separatrix=separatrix_2)
    if apply_sep_factor:
        f = f * sep_factor(r, z)
    return _apply_upper_z_clip(f, z, z_clip_max)


def phantom_ring_emissivity(
    r,
    z,
    *,
    n0: float = 1.0,
    a: float = 0.06,
    b: float = 0.8,
    rmax: float = 0.58,
    radius: float = 0.42,
    separatrix: bool = True,
):
    """Notebook-compatible ring emissivity phantom (`f_ring2`)."""
    return func_ring(r=r, z=z, n0=n0, a=a, b=b, rmax=rmax, radius=radius, separatrix=separatrix)


def phantom_ring_velocity(
    r,
    z,
    *,
    n0: float = 1.0,
    a: float = 0.06,
    b: float = 0.8,
    rmax: float = 0.58,
    radius: float = 0.42,
    separatrix: bool = True,
    ring_scale: float = 17.0,
    sharpness: float = 50.0,
    z0: float = 0.4,
    z_sharpness: float = 50.0,
    wall_factor: float = 100.0,
    sign: float = -1.0,
):
    """Notebook-compatible ring velocity phantom (`f_ring_v`)."""
    psi = mag.psi(r, z, separatrix)
    br, bz = mag.bvec(r, z, separatrix)
    b_abs = np.sqrt(br**2 + bz**2)
    psi_rmax = mag.psi(rmax, 0, separatrix)
    psi0 = mag.psi(1, 0, separatrix)
    b0 = mag.b0(r, z, separatrix)
    b2 = _safe_divide(b_abs, b0)
    q = np.sqrt(((psi - psi_rmax) / psi0) ** 2 + (1 - _safe_divide(b, b2)) ** 2) - radius
    with np.errstate(over="ignore"):
        upper = 1.0 / (1.0 + np.exp(z_sharpness * (np.asarray(z, dtype=float) - z0)))
    wall = 1.0 - np.exp(-wall_factor * (np.asarray(r, dtype=float) - 1.0) ** 2)
    res = ring_scale * n0 * q * np.exp(-(q**2) * sharpness) * wall * upper
    return sign * res


def phantom_simple_temperature(
    r,
    z,
    *,
    n0: float = 1.0,
    a: float = 10.0,
    b: float = 1.8,
    rmax: float = 0.58,
    separatrix: bool = True,
    wall_factor_1: float = 200.0,
    wall_factor_2: float = 100.0,
    z0: float = 0.4,
    z_sharpness: float = 50.0,
):
    """Notebook-compatible temperature phantom (`f_simple_T`)."""
    f = gaussian(r=r, z=z, n0=n0, a=a, b=b, rmax=rmax, separatrix=separatrix)
    return f * _two_wall_factor(r, c1=wall_factor_1, c2=wall_factor_2) * _upper_sigmoid(z, z0=z0, sharpness=z_sharpness)


_PHANTOM_REGISTRY: dict[str, Callable[..., np.ndarray]] = {}
_PHANTOM_ALIAS_TO_NAME: dict[str, str] = {}


def register_phantom(name: str, fn: Callable[..., np.ndarray], *, aliases: list[str] | tuple[str, ...] = ()) -> None:
    key = name.strip()
    if not key:
        raise ValueError("phantom name must be non-empty")
    _PHANTOM_REGISTRY[key] = fn
    _PHANTOM_ALIAS_TO_NAME[key.lower()] = key
    for alias in aliases:
        _PHANTOM_ALIAS_TO_NAME[str(alias).strip().lower()] = key


def list_phantom_names() -> list[str]:
    return sorted(_PHANTOM_REGISTRY.keys())


register_phantom("hollow", phantom_hollow, aliases=("Hollow",))
register_phantom("single", phantom_single, aliases=("Single", "Single Gaussian"))
register_phantom("double", phantom_double, aliases=("Double", "Double peaked"))
register_phantom("ring_emissivity", phantom_ring_emissivity, aliases=("ring-e", "paper_emissivity", "f_ring2"))
register_phantom("ring_velocity", phantom_ring_velocity, aliases=("ring-v", "paper_velocity", "f_ring_v"))
register_phantom("simple_temperature", phantom_simple_temperature, aliases=("paper_temperature", "f_simple_T"))


def get_phantom_funtion(name: str, **params):
    """Return a phantom callable by name.

    Parameters in ``**params`` override the default values for the selected phantom.
    The defaults preserve the historical behavior of this module.
    """
    try:
        canonical = _PHANTOM_ALIAS_TO_NAME[name.strip().lower()]
    except KeyError as e:
        raise ValueError(f"Unknown phantom function name: {name}. Available={list_phantom_names()}") from e
    return partial(_PHANTOM_REGISTRY[canonical], **params)


def get_phantom_function(name: str, **params):
    """Correctly spelled alias for ``get_phantom_funtion``."""
    return get_phantom_funtion(name, **params)


def get_cis_phantom_bundle(
    name: str = "paper_phantom1",
    *,
    emissivity_name: str | None = None,
    temperature_name: str | None = None,
    velocity_name: str | None = None,
    emissivity_params: dict[str, Any] | None = None,
    temperature_params: dict[str, Any] | None = None,
    velocity_params: dict[str, Any] | None = None,
) -> Any:
    """Return a 3-component phantom bundle for CIS (emissivity / temperature / velocity)."""
    bundle_key = name.strip().lower()
    if bundle_key != "paper_phantom1":
        raise ValueError("Currently supported CIS phantom bundles: ['paper_phantom1']")

    emissivity_name = emissivity_name or "ring_emissivity"
    temperature_name = temperature_name or "simple_temperature"
    velocity_name = velocity_name or "ring_velocity"
    emissivity_params = dict(emissivity_params or {})
    temperature_params = dict(temperature_params or {})
    velocity_params = dict(velocity_params or {})

    # Local import to avoid import-time circular dependency with gpoloidal.cis pipeline imports.
    from ..cis.types import CISPhantomBundle

    return CISPhantomBundle(
        emissivity_fn=get_phantom_function(emissivity_name, **emissivity_params),
        temperature_fn=get_phantom_function(temperature_name, **temperature_params),
        velocity_fn=get_phantom_function(velocity_name, **velocity_params),
        metadata={
            "bundle_name": "paper_phantom1",
            "emissivity_name": emissivity_name,
            "temperature_name": temperature_name,
            "velocity_name": velocity_name,
            "emissivity_params": emissivity_params,
            "temperature_params": temperature_params,
            "velocity_params": velocity_params,
        },
    )
