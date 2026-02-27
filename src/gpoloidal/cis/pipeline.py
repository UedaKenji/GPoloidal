from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..tomography import (
    GPT_cis_av_general,
    GPT_log_general,
    build_a_prior_from_emit_posterior,
    recover_tv_posterior_from_emit_and_av,
)
from .types import CISAVPosterior, CISEmitPosterior, CISObservedChannels, CISObservationGeometry, CISTVPosterior


@dataclass
class CISFitDiagnostics:
    emit_loss_history: list[float]
    emit_iters: int
    emit_converged: bool
    av_loss_history: list[float]
    av_iters: int
    av_converged: bool
    emit_model: GPT_log_general
    av_model: GPT_cis_av_general
    emit_latent: np.ndarray
    av_a: np.ndarray
    av_v: np.ndarray


def _noise_profile_and_level(sig: np.ndarray) -> tuple[np.ndarray, float]:
    sig = np.asarray(sig, dtype=float).reshape(-1)
    if sig.size == 0:
        raise ValueError("noise vector must be non-empty")
    if np.any(sig <= 0):
        raise ValueError("noise standard deviations must be positive")
    level = float(np.mean(sig))
    if level <= 0:
        raise ValueError("noise level must be positive")
    profile = sig / level
    return profile, level


class CISStepwiseReconstructor:
    """Stepwise CIS tomography (emit -> a,v -> T,v) for phantom and future pipelines."""

    def __init__(
        self,
        *,
        emit_max_iters: int = 50,
        emit_tol: float = 1e-5,
        av_max_iters: int = 50,
        av_tol: float = 1e-5,
        av_consider_w2: bool = True,
        latent_clip_emit: tuple[float, float] = (-12.0, 8.0),
        latent_clip_av: tuple[float, float] = (-8.0, 8.0),
        step_size_emit: float = 1.0,
        step_size_av: float = 0.5,
    ) -> None:
        self.emit_max_iters = int(emit_max_iters)
        self.emit_tol = float(emit_tol)
        self.av_max_iters = int(av_max_iters)
        self.av_tol = float(av_tol)
        self.av_consider_w2 = bool(av_consider_w2)
        self.latent_clip_emit = tuple(float(v) for v in latent_clip_emit)
        self.latent_clip_av = tuple(float(v) for v in latent_clip_av)
        self.step_size_emit = float(step_size_emit)
        self.step_size_av = float(step_size_av)
        self.last_diagnostics: CISFitDiagnostics | None = None

    def fit(
        self,
        *,
        geometry: CISObservationGeometry,
        obs: CISObservedChannels,
        K_e_pri: np.ndarray,
        mu_e_pri: np.ndarray,
        K_T_pri: np.ndarray,
        mu_T_pri: np.ndarray,
        K_v_pri: np.ndarray,
        mu_v_pri: np.ndarray,
        emit_init: np.ndarray | None = None,
        a_init: np.ndarray | None = None,
        v_init: np.ndarray | None = None,
    ) -> tuple[CISEmitPosterior, CISAVPosterior, CISTVPosterior]:
        H = geometry.H
        Dcos = geometry.Dcos
        if H.shape != Dcos.shape:
            raise ValueError(f"H and Dcos must have identical shape, got {H.shape} vs {Dcos.shape}")
        nI = H.shape[1]

        sigma_I0 = np.asarray(obs.sigma_I0, dtype=float).reshape(-1)
        sigma_I1 = np.asarray(obs.sigma_I1, dtype=float).reshape(-1)
        if sigma_I0.size != H.shape[0]:
            raise ValueError("sigma_I0 length must equal number of rows of H")
        if sigma_I1.size != H.shape[0]:
            raise ValueError("sigma_I1 length must equal number of rows of H")

        emit_profile, emit_level = _noise_profile_and_level(sigma_I0)
        emit_model = GPT_log_general(H=np.asarray(H, dtype=float), Kf_pri=np.asarray(K_e_pri, dtype=float), muf_pri=np.asarray(mu_e_pri, dtype=float))
        emit_model.set_obs(
            g_obs=np.asarray(obs.I0_obs, dtype=float).reshape(-1),
            obs_noise_profile=emit_profile,
            normalize=False,
            obs_noise_level=emit_level,
        )
        f = np.asarray(emit_init, dtype=float).copy() if emit_init is not None else np.asarray(mu_e_pri, dtype=float).copy()
        emit_loss_history: list[float] = []
        emit_converged = False
        for _ in range(self.emit_max_iters):
            df, loss = emit_model.update(f)
            emit_loss_history.append(float(loss))
            f = np.clip(f + self.step_size_emit * df, self.latent_clip_emit[0], self.latent_clip_emit[1])
            if loss <= self.emit_tol:
                emit_converged = True
                break
        emit_model.postprocess(f)
        emit_post = CISEmitPosterior(
            mu_e=np.asarray(emit_model.f_mean, dtype=float),
            K_e=np.asarray(emit_model.Kf_pos, dtype=float),
            sig_e=np.asarray(emit_model.sigf_pos, dtype=float),
            mll_emit=float(emit_model.mll),
        )

        mu_a_pri, K_a_pri = build_a_prior_from_emit_posterior(
            mu_e=np.asarray(emit_post.mu_e, dtype=float),
            K_e=np.asarray(emit_post.K_e, dtype=float),
            mu_T_pri=np.asarray(mu_T_pri, dtype=float),
            K_T_pri=np.asarray(K_T_pri, dtype=float),
        )

        cis_profile, cis_level = _noise_profile_and_level(sigma_I1)
        av_model = GPT_cis_av_general(
            H=np.asarray(H, dtype=float),
            Dcos=np.asarray(Dcos, dtype=float),
        )
        av_model.set_kernel(
            Ka_pri=np.asarray(K_a_pri, dtype=float),
            Kv_pri=np.asarray(K_v_pri, dtype=float),
            mua_pri=np.asarray(mu_a_pri, dtype=float),
            muv_pri=np.asarray(mu_v_pri, dtype=float),
        )
        av_model.set_obs(
            IRe_obs=np.asarray(obs.IRe_obs, dtype=float).reshape(-1),
            IIm_obs=np.asarray(obs.IIm_obs, dtype=float).reshape(-1),
            obs_noise_profile=cis_profile,
            obs_noise_level=cis_level,
        )

        a = np.asarray(a_init, dtype=float).copy() if a_init is not None else np.asarray(mu_a_pri, dtype=float).copy()
        v = np.asarray(v_init, dtype=float).copy() if v_init is not None else np.asarray(mu_v_pri, dtype=float).copy()
        if a.size != nI or v.size != nI:
            raise ValueError("Initial a/v vectors must match basis size")

        av_loss_history: list[float] = []
        av_converged = False
        for _ in range(self.av_max_iters):
            da, dv, loss = av_model.update(a, v)
            av_loss_history.append(float(loss))
            a = np.clip(a + self.step_size_av * da, self.latent_clip_av[0], self.latent_clip_av[1])
            v = np.clip(v + self.step_size_av * dv, self.latent_clip_av[0], self.latent_clip_av[1])
            if loss <= self.av_tol:
                av_converged = True
                break
        av_model.postprocess(a, v, consider_w2=self.av_consider_w2)
        av_post = CISAVPosterior(
            mu_a=np.asarray(av_model.a_mean, dtype=float),
            mu_v=np.asarray(av_model.v_mean, dtype=float),
            K_aa=np.asarray(av_model.K_aa_pos, dtype=float),
            K_av=np.asarray(av_model.K_av_pos, dtype=float),
            K_va=np.asarray(av_model.K_va_pos, dtype=float),
            K_vv=np.asarray(av_model.K_vv_pos, dtype=float),
            sig_a=np.asarray(av_model.sig_a_pos, dtype=float),
            sig_v=np.asarray(av_model.sig_v_pos, dtype=float),
            mll_cis=float(av_model.mll),
        )

        tv_post = recover_tv_posterior_from_emit_and_av(
            mu_e=np.asarray(emit_post.mu_e, dtype=float),
            K_e=np.asarray(emit_post.K_e, dtype=float),
            mu_T_pri=np.asarray(mu_T_pri, dtype=float),
            K_T_pri=np.asarray(K_T_pri, dtype=float),
            mu_a=np.asarray(av_post.mu_a, dtype=float),
            mu_v=np.asarray(av_post.mu_v, dtype=float),
            K_aa=np.asarray(av_post.K_aa, dtype=float),
            K_av=np.asarray(av_post.K_av, dtype=float),
            K_vv=np.asarray(av_post.K_vv, dtype=float),
        )

        self.last_diagnostics = CISFitDiagnostics(
            emit_loss_history=emit_loss_history,
            emit_iters=len(emit_loss_history),
            emit_converged=bool(emit_converged or (emit_loss_history and emit_loss_history[-1] <= self.emit_tol)),
            av_loss_history=av_loss_history,
            av_iters=len(av_loss_history),
            av_converged=bool(av_converged or (av_loss_history and av_loss_history[-1] <= self.av_tol)),
            emit_model=emit_model,
            av_model=av_model,
            emit_latent=f,
            av_a=a,
            av_v=v,
        )
        return emit_post, av_post, tv_post

    def diagnostics_as_dict(self) -> dict[str, Any]:
        d = self.last_diagnostics
        if d is None:
            return {}
        return {
            "emit_iters": d.emit_iters,
            "emit_converged": d.emit_converged,
            "emit_last_loss": (float(d.emit_loss_history[-1]) if d.emit_loss_history else None),
            "av_iters": d.av_iters,
            "av_converged": d.av_converged,
            "av_last_loss": (float(d.av_loss_history[-1]) if d.av_loss_history else None),
        }
