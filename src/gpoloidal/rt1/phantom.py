from functools import partial

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

def Length_scale_sq(r,z):
    return 0.0002/(gaussian(r,z)+ 0.05)

def Length_scale(r,z):
    return np.sqrt( Length_scale_sq(r,z))



DEFAULT_PSI_SEPARATRIX = -0.006376568930277712
def sep_factor(r,z):
    psi = mag.psi(r,z,separatrix=True)
    with np.errstate(over="ignore"):
        return  1/(1+np.exp(+1000*(psi-DEFAULT_PSI_SEPARATRIX)))

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
    psi = mag.psi(r,z,separatrix)
    br, bz = mag.bvec(r,z,separatrix)
    b_abs = np.sqrt(br**2+bz**2)
    psi_rmax = mag.psi(rmax,0,separatrix)
    psi0 = mag.psi(1,0,separatrix)
    b0 = mag.b0(r,z,separatrix)
    with np.errstate(divide="ignore", invalid="ignore"):
        b2 = np.divide(b_abs, b0, out=np.zeros_like(b_abs, dtype=float), where=np.asarray(b0) != 0)
    b2 = np.asarray(b2, dtype=float)
    b2[~np.isfinite(b2)] = 0.0
    if separatrix:
        rs,zs = r.flatten()[np.argmin(b_abs)],z.flatten()[np.argmin(b_abs)]
        fac = (1-np.exp(-50*((r-rs)**2+(z-zs)**2)))
    else:
        fac = 1


    #f_gaussian = fac*n0 * np.exp(- (np.sqrt(((psi-psi_rmax)/psi0)**2+(1-b/b2)**2)-radius)**2*1/a**2)*(1-np.exp(-100*(r-1)**2))
    f_cauchy =  fac*n0 * a**2/( (np.sqrt(((psi-psi_rmax)/psi0)**2+(1-b/b2)**2)-radius)**2+a**2)*(1-np.exp(-100*(r-1)**2))
    return f_cauchy
    #return n0 *  (np.sqrt(((psi-psi_rmax)/psi0)**2+(1-1/b)**2)-0.5)*a / (a +   (np.sqrt(((psi-psi_rmax)/psi0)**2+(1-1/b)**2)-0.5)**2)*(1-np.exp(-100*(r-1)**2))


#f_ring2_HD =func_ring(r=R_grid,z=Z_grid,n0=1,a=0.06,b=0.8,rmax=0.58,radius=0.42)
#f_ring2 =   func_ring(r=rI,z=zI,    n0=1,a=0.06,b=0.8,rmax=0.58,radius=0.42)

def _apply_upper_z_clip(f, z, z_clip_max: float | None):
    if z_clip_max is None:
        return f
    z_arr = np.asarray(z, dtype=float)
    return np.where(z_arr > z_clip_max, 0.0, f)


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


def get_phantom_funtion(name: str, **params):
    """Return a phantom callable by name.

    Parameters in ``**params`` override the default values for the selected phantom.
    The defaults preserve the historical behavior of this module.
    """
    if name in ['Hollow', 'hollow']:
        return partial(phantom_hollow, **params)

    if name in ['single', 'Single', 'Single Gaussian']:
        return partial(phantom_single, **params)

    if name in ['double', 'Double', 'Double peaked']:
        return partial(phantom_double, **params)

    raise ValueError(f"Unknown phantom function name: {name}.")


def get_phantom_function(name: str, **params):
    """Correctly spelled alias for ``get_phantom_funtion``."""
    return get_phantom_funtion(name, **params)
