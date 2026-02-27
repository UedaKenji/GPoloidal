from typing import Optional
import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.sparse as sps


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


def _symmetrize(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    return 0.5 * (A + A.T)


def build_a_prior_from_emit_posterior(
    *,
    mu_e: np.ndarray,
    K_e: np.ndarray,
    mu_T_pri: np.ndarray,
    K_T_pri: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Step-2 helper of CIS tomography: build the local-amplitude prior.

    With ``a = e - T`` and independent Gaussian priors/posteriors,
    ``a | I0 ~ N(mu_e - mu_T_pri, K_e + K_T_pri)``.
    """
    mu_e = np.asarray(mu_e, dtype=float).reshape(-1)
    mu_T_pri = np.asarray(mu_T_pri, dtype=float).reshape(-1)
    K_e = _symmetrize(np.asarray(K_e, dtype=float))
    K_T_pri = _symmetrize(np.asarray(K_T_pri, dtype=float))
    if mu_e.shape != mu_T_pri.shape:
        raise ValueError("mu_e and mu_T_pri must have the same shape")
    if K_e.shape != K_T_pri.shape:
        raise ValueError("K_e and K_T_pri must have the same shape")
    if K_e.shape[0] != mu_e.size:
        raise ValueError("Covariance shape must match mean vector size")
    mu_a_pri = mu_e - mu_T_pri
    K_a_pri = _symmetrize(K_e + K_T_pri)
    return mu_a_pri, K_a_pri


class GPT_cis_av_general:
    """Laplace/Gauss-Newton solver for CIS local amplitude ``a`` and velocity ``v``.

    The forward model corresponds to the real/imaginary CIS channels:
    ``IRe = sum_j H_ij exp(a_j) cos(Dcos_ij v_j)``
    ``IIm = sum_j H_ij exp(a_j) sin(Dcos_ij v_j)``

    ``Dcos`` is the directional-cosine matrix (paper ``Theta`` counterpart).
    """

    def __init__(
        self,
        H: np.ndarray,
        Dcos: np.ndarray,
        *,
        eps: float = 1e-6,
    ) -> None:
        H = np.asarray(H, dtype=float)
        Dcos = np.asarray(Dcos, dtype=float)
        if H.shape != Dcos.shape:
            raise ValueError(f"H and Dcos must have the same shape, got {H.shape} vs {Dcos.shape}")
        self.H = H
        self.Dcos = np.clip(Dcos, -1.0, 1.0)
        self.ng, self.nI = H.shape
        self.regularization = float(eps)
        self.obs_noise_level = 1.0
        self.obs_noise_level_internal = 1.0

    def set_kernel(
        self,
        *,
        Ka_pri: np.ndarray,
        Kv_pri: np.ndarray,
        mua_pri: np.ndarray,
        muv_pri: np.ndarray,
        eps: float | None = None,
    ) -> None:
        eps = self.regularization if eps is None else float(eps)
        Ka_pri = _symmetrize(np.asarray(Ka_pri, dtype=float))
        Kv_pri = _symmetrize(np.asarray(Kv_pri, dtype=float))
        mua_pri = np.asarray(mua_pri, dtype=float).reshape(-1)
        muv_pri = np.asarray(muv_pri, dtype=float).reshape(-1)
        if Ka_pri.shape != (self.nI, self.nI) or Kv_pri.shape != (self.nI, self.nI):
            raise ValueError("Prior covariance shapes must be (nI, nI)")
        if mua_pri.size != self.nI or muv_pri.size != self.nI:
            raise ValueError("Prior mean vectors must have length nI")

        self.Ka_pri = Ka_pri
        self.Kv_pri = Kv_pri
        self.mua_pri = mua_pri
        self.muv_pri = muv_pri

        self.Ka_pri_eff = _symmetrize(Ka_pri + eps * np.eye(self.nI))
        self.Kv_pri_eff = _symmetrize(Kv_pri + eps * np.eye(self.nI))
        self.Ka_inv = _symmetrize(np.linalg.inv(self.Ka_pri_eff))
        self.Kv_inv = _symmetrize(np.linalg.inv(self.Kv_pri_eff))
        self.log_det_K = log_det(self.Ka_pri_eff) + log_det(self.Kv_pri_eff)

        self.K_f_inv = np.zeros((2 * self.nI, 2 * self.nI), dtype=float)
        self.K_f_inv[: self.nI, : self.nI] = self.Ka_inv
        self.K_f_inv[self.nI :, self.nI :] = self.Kv_inv

    def _set_obs_noise_level(self, obs_noise_level: float) -> None:
        self.obs_noise_level = float(obs_noise_level)
        self.obs_noise_level_internal = self.obs_noise_level

    def set_obs(
        self,
        *,
        IRe_obs: np.ndarray,
        IIm_obs: np.ndarray,
        obs_noise_profile: np.ndarray | None = None,
        obs_noise_level: float = 1.0,
    ) -> None:
        IRe_obs = np.asarray(IRe_obs, dtype=float).reshape(-1)
        IIm_obs = np.asarray(IIm_obs, dtype=float).reshape(-1)
        if IRe_obs.size != self.ng or IIm_obs.size != self.ng:
            raise ValueError("IRe_obs and IIm_obs must have length ng")

        if obs_noise_profile is None:
            obs_noise_profile = np.ones(self.ng, dtype=float)
        obs_noise_profile = np.asarray(obs_noise_profile, dtype=float).reshape(-1)
        if obs_noise_profile.size != self.ng:
            raise ValueError("obs_noise_profile must have length ng")
        if np.any(obs_noise_profile <= 0):
            raise ValueError("obs_noise_profile must be strictly positive")

        self.IRe_obs = IRe_obs
        self.IIm_obs = IIm_obs
        self.obs_noise_profile = obs_noise_profile
        self.obs_noise_profile_inv = 1.0 / obs_noise_profile
        self._set_obs_noise_level(obs_noise_level)

        self.sigiH = self.obs_noise_profile_inv[:, None] * self.H
        self.sigiObs_re = self.obs_noise_profile_inv * self.IRe_obs
        self.sigiObs_im = self.obs_noise_profile_inv * self.IIm_obs
        self.sigiObs_stack = np.hstack((self.sigiObs_re, self.sigiObs_im))
        # Stacked channels share the same per-pixel profile.
        self.log_det_obs_noise_profile = 4.0 * np.sum(np.log(obs_noise_profile))

    def _compute_terms(self, a: np.ndarray, v: np.ndarray) -> dict[str, np.ndarray | float]:
        a = np.asarray(a, dtype=float).reshape(-1)
        v = np.asarray(v, dtype=float).reshape(-1)
        if a.size != self.nI or v.size != self.nI:
            raise ValueError("a and v must have length nI")

        r_a = a - self.mua_pri
        r_v = v - self.muv_pri

        exp_a = np.exp(a)
        phase = self.Dcos * v[None, :]
        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)

        base = self.sigiH * exp_a[None, :]
        Rc = base * cos_phase
        Rs = base * sin_phase
        Ac = Rc.sum(axis=1)
        As = Rs.sum(axis=1)

        inv_noise = 1.0 / (self.obs_noise_level_internal**2)
        # resA is the standardized residual scaled by one extra sigma factor so that
        # J^T @ resA matches the old rt1kernel formulation exactly.
        resA = (1.0 / self.obs_noise_level_internal) * (
            np.hstack((Ac, As)) - self.sigiObs_stack
        )

        Rc_D = Rc * self.Dcos
        Rs_D = Rs * self.Dcos
        J = np.vstack(
            (
                np.hstack((Rc, -Rs_D)),
                np.hstack((Rs, +Rc_D)),
            )
        )
        J *= 1.0 / self.obs_noise_level_internal

        prior_grad = np.hstack((self.Ka_inv @ r_a, self.Kv_inv @ r_v))
        grad = -J.T @ resA - prior_grad
        W1 = J.T @ J
        H_gn = -W1 - self.K_f_inv

        Rc_D2 = Rc_D * self.Dcos
        Rs_D2 = Rs_D * self.Dcos
        d_aa = np.hstack((Rc.T, Rs.T)) @ resA
        d_vv = np.hstack((-Rc_D2.T, -Rs_D2.T)) @ resA
        d_av = np.hstack((-Rs_D.T, Rc_D.T)) @ resA
        W2 = np.zeros((2 * self.nI, 2 * self.nI), dtype=float)
        W2[: self.nI, : self.nI] = np.diag(d_aa)
        W2[self.nI :, self.nI :] = np.diag(d_vv)
        W2[: self.nI, self.nI :] = np.diag(d_av)
        W2[self.nI :, : self.nI] = np.diag(d_av)

        loss_g = float(np.dot(resA, resA))
        loss_f = float(r_a @ (self.Ka_inv @ r_a) + r_v @ (self.Kv_inv @ r_v))

        return {
            "a": a,
            "v": v,
            "r_a": r_a,
            "r_v": r_v,
            "exp_a": exp_a,
            "phase": phase,
            "Rc": Rc,
            "Rs": Rs,
            "Ac": Ac,
            "As": As,
            "resA": resA,
            "J": J,
            "grad": grad,
            "H_gn": H_gn,
            "W2": W2,
            "loss_g": loss_g,
            "loss_f": loss_f,
            "inv_noise": inv_noise,
        }

    def log_posterior(self, a: np.ndarray, v: np.ndarray) -> float:
        terms = self._compute_terms(a, v)
        return -0.5 * (float(terms["loss_g"]) + float(terms["loss_f"]))

    def gradient_hessian(
        self,
        a: np.ndarray,
        v: np.ndarray,
        *,
        consider_w2: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        terms = self._compute_terms(a, v)
        H_gn = np.asarray(terms["H_gn"], dtype=float)
        if consider_w2:
            H_exact = H_gn - np.asarray(terms["W2"], dtype=float)
        else:
            H_exact = H_gn
        return np.asarray(terms["grad"], dtype=float), _symmetrize(H_exact)

    def update(
        self,
        a: np.ndarray,
        v: np.ndarray,
        *,
        obs_noise_level: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if obs_noise_level is not None:
            self._set_obs_noise_level(obs_noise_level)

        terms = self._compute_terms(a, v)
        grad = np.asarray(terms["grad"], dtype=float)
        H_gn = np.asarray(terms["H_gn"], dtype=float)
        loss = float(np.mean(np.abs(grad)))
        delta = -np.linalg.solve(H_gn, grad)
        delta = np.clip(delta, -5.0, 5.0)

        self._last_terms = terms
        self._last_h_gn = H_gn
        self.a_latest = np.asarray(a, dtype=float).copy()
        self.v_latest = np.asarray(v, dtype=float).copy()
        return delta[: self.nI], delta[self.nI :], loss

    def postprocess(
        self,
        a: np.ndarray,
        v: np.ndarray,
        *,
        consider_w2: bool = True,
    ) -> None:
        terms = self._compute_terms(a, v)
        H_gn = np.asarray(terms["H_gn"], dtype=float)
        W2 = np.asarray(terms["W2"], dtype=float)
        H_exact = H_gn - W2 if consider_w2 else H_gn
        self.Kf_pos_inv = _symmetrize(-H_exact)
        self.Kf_pos = np.linalg.inv(self.Kf_pos_inv)
        self.Kf_pos = _symmetrize(self.Kf_pos)

        self.a_mean = np.asarray(a, dtype=float).copy()
        self.v_mean = np.asarray(v, dtype=float).copy()
        self.K_aa_pos = self.Kf_pos[: self.nI, : self.nI]
        self.K_av_pos = self.Kf_pos[: self.nI, self.nI :]
        self.K_va_pos = self.Kf_pos[self.nI :, : self.nI]
        self.K_vv_pos = self.Kf_pos[self.nI :, self.nI :]
        # Backward-friendly aliases.
        self.K_a_pos = self.K_aa_pos
        self.K_v_pos = self.K_vv_pos
        self.sig_a_pos = np.sqrt(np.clip(np.diag(self.K_aa_pos), 0.0, None))
        self.sig_v_pos = np.sqrt(np.clip(np.diag(self.K_vv_pos), 0.0, None))

        self.r_a = np.asarray(terms["r_a"], dtype=float)
        self.r_v = np.asarray(terms["r_v"], dtype=float)
        self.resA = np.asarray(terms["resA"], dtype=float)
        self.Rc = np.asarray(terms["Rc"], dtype=float)
        self.Rs = np.asarray(terms["Rs"], dtype=float)
        self.Ac = np.asarray(terms["Ac"], dtype=float)
        self.As = np.asarray(terms["As"], dtype=float)
        self.loss_g = float(terms["loss_g"])
        self.loss_f = float(terms["loss_f"])

        self.log_det_Kpos = log_det(self.Kf_pos)
        self.log_det_obs_noise_cov = (
            self.log_det_obs_noise_profile
            + 2.0 * np.log(self.obs_noise_level_internal) * (2 * self.ng)
        )
        self.mll = (
            (-self.loss_g - self.log_det_obs_noise_cov)
            - self.loss_f
            - self.log_det_K
            + self.log_det_Kpos
        )
        self.mll = 0.5 * self.mll - 0.5 * (2 * self.ng) * np.log(2 * np.pi)


def recover_tv_posterior_from_emit_and_av(
    *,
    mu_e: np.ndarray,
    K_e: np.ndarray,
    mu_T_pri: np.ndarray,
    K_T_pri: np.ndarray,
    mu_a: np.ndarray,
    mu_v: np.ndarray,
    K_aa: np.ndarray,
    K_av: np.ndarray,
    K_vv: np.ndarray,
):
    """Step-4 helper of CIS tomography: recover ``(T, v)`` posterior from ``e`` and ``(a,v)``."""
    mu_e = np.asarray(mu_e, dtype=float).reshape(-1)
    mu_T_pri = np.asarray(mu_T_pri, dtype=float).reshape(-1)
    mu_a = np.asarray(mu_a, dtype=float).reshape(-1)
    mu_v = np.asarray(mu_v, dtype=float).reshape(-1)
    K_e = _symmetrize(np.asarray(K_e, dtype=float))
    K_T_pri = _symmetrize(np.asarray(K_T_pri, dtype=float))
    K_aa = _symmetrize(np.asarray(K_aa, dtype=float))
    K_av = np.asarray(K_av, dtype=float)
    K_vv = _symmetrize(np.asarray(K_vv, dtype=float))

    if not (mu_e.shape == mu_T_pri.shape == mu_a.shape == mu_v.shape):
        if not (mu_e.shape == mu_T_pri.shape == mu_a.shape):
            raise ValueError("mu_e, mu_T_pri, mu_a must have the same shape")
    nI = mu_e.size
    if mu_v.size != nI:
        raise ValueError("mu_v must have the same length as mu_e")
    for name, K in (("K_e", K_e), ("K_T_pri", K_T_pri), ("K_aa", K_aa), ("K_vv", K_vv)):
        if K.shape != (nI, nI):
            raise ValueError(f"{name} must have shape {(nI, nI)}")
    if K_av.shape != (nI, nI):
        raise ValueError(f"K_av must have shape {(nI, nI)}")

    _, K_a_pri = build_a_prior_from_emit_posterior(mu_e=mu_e, K_e=K_e, mu_T_pri=mu_T_pri, K_T_pri=K_T_pri)
    K_a_pri_inv = _symmetrize(np.linalg.inv(K_a_pri))

    mu_T = mu_T_pri + K_T_pri @ (K_a_pri_inv @ (mu_e - mu_T_pri - mu_a))
    K_TT = K_T_pri @ (K_a_pri_inv @ K_e @ K_a_pri_inv) @ K_T_pri
    K_TT = K_TT + K_T_pri @ (K_a_pri_inv @ K_aa @ K_a_pri_inv) @ K_T_pri
    K_TT = _symmetrize(K_TT)
    K_Tv = -(K_T_pri @ (K_a_pri_inv @ K_av))
    K_vT = K_Tv.T
    K_vv = _symmetrize(K_vv)
    sig_T = np.sqrt(np.clip(np.diag(K_TT), 0.0, None))
    sig_v = np.sqrt(np.clip(np.diag(K_vv), 0.0, None))

    # Local import to avoid circular dependency during module import.
    from .cis.types import CISTVPosterior

    return CISTVPosterior(
        mu_T=np.asarray(mu_T, dtype=float),
        mu_v=np.asarray(mu_v, dtype=float),
        K_TT=K_TT,
        K_Tv=np.asarray(K_Tv, dtype=float),
        K_vT=np.asarray(K_vT, dtype=float),
        K_vv=K_vv,
        sig_T=sig_T,
        sig_v=sig_v,
    )
