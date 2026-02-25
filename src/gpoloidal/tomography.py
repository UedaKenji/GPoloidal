from typing import Optional
import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.sparse as sps

from . import kernel
from .utils import Diag


def Kernel_SE_2dim(x1,x2,y1,y2,l1,l2):
    X1,X2 = np.meshgrid(x1,x2,indexing='ij')
    Y1,Y2 = np.meshgrid(y1,y2,indexing='ij')

    return np.exp(-0.5*((X1-X2)**2/l1**2 + (Y1-Y2)**2/l2**2))


def log_det(A,scale=1):
    try:
        A = scipy.linalg.cholesky(A)
        return np.sum(np.log(np.diag(A)))*2 + A.shape[0]*np.log(scale)
    except scipy.linalg.LinAlgError:
        lam,_ = np.linalg.eigh(A)

        index = lam>0
        return np.sum(np.log(lam[index])) + np.sum(index)*np.log(scale)

def _build_2dim_se_prior_with_boundary(
    r_idc: np.ndarray,
    z_idc: np.ndarray,
    length_scale: float,
    rb: np.ndarray,
    zb: np.ndarray,
    bound_sig: float = 0.0,
    bound_value: float = -5.0,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Condition a 2D SE prior with boundary observations."""
    if (bound_sig < 0) or (bound_sig >= 1):
        raise ValueError('bound_sig must satisfy 0 <= bound_sig < 1')

    k_ii = Kernel_SE_2dim(r_idc, r_idc, z_idc, z_idc, length_scale, length_scale)
    k_ib = Kernel_SE_2dim(r_idc, rb, z_idc, zb, length_scale, length_scale)
    k_bb = Kernel_SE_2dim(rb, rb, zb, zb, length_scale, length_scale)

    factor = 1.0 / (1.0 - bound_sig**2)
    k_bb_scaled = factor * k_bb
    k_bb_inv = np.linalg.inv(k_bb_scaled + eps * np.eye(k_bb_scaled.shape[0]))

    mu_prior = k_ib @ (k_bb_inv @ (bound_value * np.ones(zb.size)))
    k_prior = k_ii - k_ib @ (k_bb_inv @ k_ib.T)
    return k_prior, mu_prior


class GPT_log_general:
    def __init__(self,
        H: np.ndarray,
        Kf_pri: np.ndarray,
        muf_pri: np.ndarray,
        eps: float = 1e-6,
    ) -> None:
        """Laplace-approximate log-GP tomography with externally supplied prior."""
        self.H = H
        self.nI = H.shape[1]
        self.ng = H.shape[0]
        self.regularization = eps

        self.normalized = False
        self.obs_noise_level = 1.0
        self.obs_noise_level_internal = 1.0

        self._set_prior(Kf_pri=Kf_pri, muf_pri=muf_pri)

    def _set_prior(self, Kf_pri: np.ndarray, muf_pri: np.ndarray) -> None:
        self.Kf_pri = 0.5 * (Kf_pri + Kf_pri.T)
        self.muf_pri = np.asarray(muf_pri, dtype=float)
        self.muf_pri_eff = self.muf_pri.copy()

        reg_eye = self.regularization * np.eye(self.Kf_pri.shape[0])
        self.K_inv = np.linalg.inv(self.Kf_pri + reg_eye)
        self.log_det_K = log_det(self.Kf_pri + reg_eye)

    def set_obs(self,
        g_obs: np.ndarray,
        obs_noise_profile: Optional[np.ndarray] = None,
        normalize: bool = False,
        obs_noise_level: float = 1.0,
    ) -> None:
        r"""Set observation model terms.

        Likelihood model:
        $g = H\exp(f) + \epsilon$, $\epsilon \sim N(0, \Sigma_g)$,
        $\Sigma_g = (obs\_noise\_level * Diag(obs\_noise\_profile))^2$.

        Parameters
        ----------
        g_obs : np.ndarray
            Observation vector of shape ``(ng,)``.
        obs_noise_profile : np.ndarray, optional
            Relative per-observation noise standard-deviation profile.
            ``np.ones(ng)`` means homoscedastic noise.
        normalize : bool, optional
            Normalize ``H`` and ``g_obs`` internally for numerical stability.
        obs_noise_level : float, optional
            Global observation-noise level (a scalar hyperparameter, analogous to
            kernel length scale but for the likelihood).
        """
        self.g_obs = np.asarray(g_obs, dtype=float)
        self.normalize(normalize)
        self._set_obs_noise_level(obs_noise_level)

        if obs_noise_profile is None:
            obs_noise_profile = np.ones(self.ng, dtype=float)
        obs_noise_profile = np.asarray(obs_noise_profile, dtype=float)
        self.obs_noise_profile = obs_noise_profile
        self.obs_noise_profile_inv = 1.0 / obs_noise_profile

        Hn = self.H / self.H_scale

        self.Sigi_obs = self.obs_noise_profile_inv * self.g_obs_n
        self.sigiH = sps.diags(self.obs_noise_profile_inv) @ Hn
        self.Hsig2iH = self.sigiH.T @ self.sigiH
        self.log_det_obs_noise_profile = 2 * np.sum(np.log(obs_noise_profile))

    def _set_obs_noise_level(self, obs_noise_level: float) -> None:
        self.obs_noise_level = float(obs_noise_level)
        if self.normalized:
            self.obs_noise_level_internal = self.obs_noise_level / self.g_scale
        else:
            self.obs_noise_level_internal = self.obs_noise_level

    def _to_internal_log_field(self, f: np.ndarray) -> np.ndarray:
        # Public API accepts the log field in the original (physical) scale.
        return np.asarray(f, dtype=float) - self.log_f_scale

    def normalize(self, is_normalize: bool = False) -> None:
        if is_normalize:
            self.H_scale = np.mean(self.H @ np.ones(self.nI))
            self.g_scale = float(np.mean(self.g_obs))
            self.g_obs_n = self.g_obs / self.g_scale

            self.f_scale = self.g_scale / self.H_scale
            self.log_f_scale = float(np.log(self.f_scale))
            self.normalized = True
            self.muf_pri_eff = self.muf_pri - self.log_f_scale
        else:
            self.H_scale = 1.0
            self.g_scale = 1.0
            self.f_scale = 1.0
            self.log_f_scale = 0.0
            self.normalized = False
            self.g_obs_n = self.g_obs
            self.muf_pri_eff = self.muf_pri

        # Keep the effective observation-noise level in sync with normalization.
        self._set_obs_noise_level(getattr(self, 'obs_noise_level', 1.0))

    def update(self,
        f: np.ndarray,
        obs_noise_level: float | None = None,
    ) -> tuple[np.ndarray, float]:
        """One Newton/Laplace update step for the latent log field."""
        if obs_noise_level is not None:
            self._set_obs_noise_level(obs_noise_level)

        f_internal = self._to_internal_log_field(f)
        r_f = f_internal - self.muf_pri_eff
        exp_f = np.exp(f_internal)
        fxf = np.einsum('i,j->ij', exp_f, exp_f)

        SiR = self.sigiH @ exp_f - self.Sigi_obs
        inv_noise_var = 1.0 / self.obs_noise_level_internal**2
        c1 = inv_noise_var * (self.sigiH.T @ SiR) * exp_f
        C1 = inv_noise_var * self.Hsig2iH * fxf

        self.SiR = SiR
        self.r_f = r_f

        Psi_df = -c1 - self.K_inv @ r_f
        Psi_dfdf = -C1 - np.diag(c1) - self.K_inv

        loss = abs(Psi_df).mean()
        delta_f = -np.linalg.solve(Psi_dfdf, Psi_df)
        delta_f = np.clip(delta_f, -3.0, 3.0)
        return delta_f, loss

    def postprocess(self, f: np.ndarray) -> None:
        """Finalize posterior moments after Laplace iterations."""
        f_internal = self._to_internal_log_field(f)
        r_f = f_internal - self.muf_pri_eff
        exp_f = np.exp(f_internal)
        fxf = np.einsum('i,j->ij', exp_f, exp_f)

        SiR = self.sigiH @ exp_f - self.Sigi_obs
        inv_noise_var = 1.0 / self.obs_noise_level_internal**2
        c1 = inv_noise_var * (self.sigiH.T @ SiR) * exp_f
        C1 = inv_noise_var * self.Hsig2iH * fxf

        self.SiR = SiR
        self.r_f = r_f

        Psi_dfdf = -C1 - np.diag(c1) - self.K_inv
        self.Kf_pos_inv = -Psi_dfdf
        self.Kf_pos = np.linalg.inv(self.Kf_pos_inv)
        self.sigf_pos = np.sqrt(np.clip(np.diag(self.Kf_pos), 0.0, None))
        self.f_mean = f_internal + self.log_f_scale

        self.log_det_Kpos = log_det(self.Kf_pos)
        self.loss_g = inv_noise_var * np.dot(self.SiR, self.SiR)
        self.loss_f = self.r_f @ (self.K_inv @ self.r_f)

        self.log_det_obs_noise_cov = (
            self.log_det_obs_noise_profile
            + 2 * np.log(self.obs_noise_level_internal) * self.ng
        )

        self.mll = (
            (-self.loss_g - self.log_det_obs_noise_cov)
            - self.loss_f
            - self.log_det_K
            + self.log_det_Kpos
        )
        self.mll = 0.5 * self.mll - 0.5 * self.ng * np.log(2 * np.pi)

    @property
    def expf_mean(self):
        return np.exp(self.f_mean + 0.5 * self.sigf_pos**2)

    @property
    def expf_median(self):
        return np.exp(self.f_mean)

    @property
    def expf_std(self):
        return np.sqrt(np.exp(2 * self.f_mean + self.sigf_pos**2) * (np.exp(self.sigf_pos**2) - 1))


class GPT_log_general_2dim_prior(GPT_log_general):
    """Convenience wrapper that builds a 2D SE prior from inducing-point coordinates."""

    def __init__(self,
        H: np.ndarray,
        r_idc: np.ndarray,
        z_idc: np.ndarray,
        length_scale,
        eps: float = 1e-6,
    ) -> None:
        self.H = H
        self.rI = np.asarray(r_idc, dtype=float)
        self.zI = np.asarray(z_idc, dtype=float)
        self.length_scale = float(length_scale)
        self.length = self.length_scale  # backward-friendly internal alias
        self.regularization = eps

        self.KII = Kernel_SE_2dim(self.rI, self.rI, self.zI, self.zI, self.length_scale, self.length_scale)
        muf0 = np.zeros(self.rI.size, dtype=float)
        super().__init__(H=H, Kf_pri=self.KII, muf_pri=muf0, eps=eps)

        self.K_pri = self.Kf_pri
        self.f_pri = self.muf_pri

    def set_kernel_and_boundary(self,
        rb: np.ndarray,
        zb: np.ndarray,
        bound_sig: float = 0.0,
        bound_value: float = -5.0,
    ) -> None:
        self.rb = np.asarray(rb, dtype=float)
        self.zb = np.asarray(zb, dtype=float)
        self.nb = self.zb.size

        K_pri, f_pri = _build_2dim_se_prior_with_boundary(
            r_idc=self.rI,
            z_idc=self.zI,
            length_scale=self.length_scale,
            rb=self.rb,
            zb=self.zb,
            bound_sig=bound_sig,
            bound_value=bound_value,
            eps=self.regularization,
        )

        self.K_pri = K_pri
        self.f_pri = f_pri
        self._set_prior(Kf_pri=K_pri, muf_pri=f_pri)

        if hasattr(self, 'g_obs'):
            # Prior mean normalization depends on current scaling state.
            self.normalize(self.normalized)

    def set_kernel_and_boudary(self, *args, **kwargs):
        """Backward-compatible alias for the previous misspelled method name."""
        return self.set_kernel_and_boundary(*args, **kwargs)


class GPT_lin_general:
    def __init__(self,
        H: np.ndarray,
        Kf_pri: np.ndarray,
        muf_pri: np.ndarray,
        eps: float = 1e-6,
    ) -> None:
        """Closed-form linear-GP tomography with externally supplied prior."""
        self.H = H
        self.nI = H.shape[1]
        self.ng = H.shape[0]
        self.regularization = eps

        self.normalized = False
        self.obs_noise_level = 1.0
        self.obs_noise_level_internal = 1.0

        self.Kf_pri = 0.5 * (Kf_pri + Kf_pri.T)
        self.muf_pri = np.asarray(muf_pri, dtype=float)

        # Initialize internal scaling state before building the effective prior.
        self.H_scale = 1.0
        self.g_scale = 1.0
        self.f_scale = 1.0
        self.g_obs_n = None
        self._refresh_internal_prior()

    def _refresh_internal_prior(self) -> None:
        if self.normalized:
            self.muf_pri_internal = self.muf_pri / self.f_scale
            self.Kf_pri_internal = self.Kf_pri / (self.f_scale**2)
        else:
            self.muf_pri_internal = self.muf_pri
            self.Kf_pri_internal = self.Kf_pri

        reg_eye = self.regularization * np.eye(self.nI)
        self.K_inv = np.linalg.inv(self.Kf_pri_internal + reg_eye)
        self.log_det_K = log_det(self.Kf_pri_internal + reg_eye)

    def _set_obs_noise_level(self, obs_noise_level: float) -> None:
        self.obs_noise_level = float(obs_noise_level)
        if self.normalized:
            self.obs_noise_level_internal = self.obs_noise_level / self.g_scale
        else:
            self.obs_noise_level_internal = self.obs_noise_level

    def normalize(self, is_normalize: bool = False) -> None:
        if is_normalize:
            self.H_scale = np.mean(self.H @ np.ones(self.nI))
            self.g_scale = float(np.mean(self.g_obs))
            self.g_obs_n = self.g_obs / self.g_scale
            self.f_scale = self.g_scale / self.H_scale
            self.normalized = True
        else:
            self.H_scale = 1.0
            self.g_scale = 1.0
            self.f_scale = 1.0
            self.g_obs_n = self.g_obs
            self.normalized = False

        self._refresh_internal_prior()
        self._set_obs_noise_level(getattr(self, "obs_noise_level", 1.0))

    def set_obs(self,
        g_obs: np.ndarray,
        obs_noise_profile: Optional[np.ndarray] = None,
        normalize: bool = False,
        obs_noise_level: float = 1.0,
    ) -> None:
        r"""Set observation model terms for ``g = Hf + \epsilon``.

        $\Sigma_g = (obs\_noise\_level * Diag(obs\_noise\_profile))^2$
        """
        self.g_obs = np.asarray(g_obs, dtype=float)
        self.normalize(normalize)
        self._set_obs_noise_level(obs_noise_level)

        if obs_noise_profile is None:
            obs_noise_profile = np.ones(self.ng, dtype=float)
        obs_noise_profile = np.asarray(obs_noise_profile, dtype=float)
        self.obs_noise_profile = obs_noise_profile
        self.obs_noise_profile_inv = 1.0 / obs_noise_profile

        Hn = self.H / self.H_scale
        self.Sigi_obs = self.obs_noise_profile_inv * self.g_obs_n
        self.sigiH = sps.diags(self.obs_noise_profile_inv) @ Hn
        self.Hsig2iH = self.sigiH.T @ self.sigiH
        self.log_det_obs_noise_profile = 2 * np.sum(np.log(obs_noise_profile))

    def solve(self, obs_noise_level: float | None = None) -> np.ndarray:
        """Compute the exact Gaussian posterior for the linear model."""
        if obs_noise_level is not None:
            self._set_obs_noise_level(obs_noise_level)

        inv_noise_var = 1.0 / self.obs_noise_level_internal**2
        rhs = self.K_inv @ self.muf_pri_internal + inv_noise_var * (self.sigiH.T @ self.Sigi_obs)
        self.Kf_pos_inv_internal = self.K_inv + inv_noise_var * self.Hsig2iH
        self.f_mean_internal = np.linalg.solve(self.Kf_pos_inv_internal, rhs)
        self.Kf_pos_internal = np.linalg.inv(self.Kf_pos_inv_internal)

        self.SiR = self.sigiH @ self.f_mean_internal - self.Sigi_obs
        self.r_f = self.f_mean_internal - self.muf_pri_internal

        self.f_mean = self.f_mean_internal * self.f_scale
        self.Kf_pos = self.Kf_pos_internal * (self.f_scale**2)
        self.sigf_pos = np.sqrt(np.clip(np.diag(self.Kf_pos), 0.0, None))

        self.log_det_Kpos = log_det(self.Kf_pos_internal)
        self.loss_g = inv_noise_var * np.dot(self.SiR, self.SiR)
        self.loss_f = self.r_f @ (self.K_inv @ self.r_f)
        self.log_det_obs_noise_cov = (
            self.log_det_obs_noise_profile
            + 2 * np.log(self.obs_noise_level_internal) * self.ng
        )
        self.mll = (
            (-self.loss_g - self.log_det_obs_noise_cov)
            - self.loss_f
            - self.log_det_K
            + self.log_det_Kpos
        )
        self.mll = 0.5 * self.mll - 0.5 * self.ng * np.log(2 * np.pi)
        return self.f_mean

    def postprocess(self, obs_noise_level: float | None = None) -> np.ndarray:
        """Alias of :meth:`solve` for API similarity with ``GPT_log_general``."""
        return self.solve(obs_noise_level=obs_noise_level)

    @property
    def f_std(self):
        return self.sigf_pos


class GPT_lin_general_2dim_prior(GPT_lin_general):
    """Convenience wrapper that builds a 2D SE prior from inducing-point coordinates."""

    def __init__(self,
        H: np.ndarray,
        r_idc: np.ndarray,
        z_idc: np.ndarray,
        length_scale,
        eps: float = 1e-6,
    ) -> None:
        self.H = H
        self.rI = np.asarray(r_idc, dtype=float)
        self.zI = np.asarray(z_idc, dtype=float)
        self.length_scale = float(length_scale)
        self.length = self.length_scale
        self.regularization = eps

        self.KII = Kernel_SE_2dim(self.rI, self.rI, self.zI, self.zI, self.length_scale, self.length_scale)
        muf0 = np.zeros(self.rI.size, dtype=float)
        super().__init__(H=H, Kf_pri=self.KII, muf_pri=muf0, eps=eps)

        self.K_pri = self.Kf_pri
        self.f_pri = self.muf_pri

    def set_kernel_and_boundary(self,
        rb: np.ndarray,
        zb: np.ndarray,
        bound_sig: float = 0.0,
        bound_value: float = 0.0,
    ) -> None:
        self.rb = np.asarray(rb, dtype=float)
        self.zb = np.asarray(zb, dtype=float)
        self.nb = self.zb.size

        K_pri, f_pri = _build_2dim_se_prior_with_boundary(
            r_idc=self.rI,
            z_idc=self.zI,
            length_scale=self.length_scale,
            rb=self.rb,
            zb=self.zb,
            bound_sig=bound_sig,
            bound_value=bound_value,
            eps=self.regularization,
        )
        self.K_pri = K_pri
        self.f_pri = f_pri
        self.Kf_pri = 0.5 * (K_pri + K_pri.T)
        self.muf_pri = np.asarray(f_pri, dtype=float)
        self._refresh_internal_prior()
        if hasattr(self, "g_obs"):
            self.normalize(self.normalized)

    def set_kernel_and_boudary(self, *args, **kwargs):
        """Backward-compatible alias for the previous misspelled method name."""
        return self.set_kernel_and_boundary(*args, **kwargs)
