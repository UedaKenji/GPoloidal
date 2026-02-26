from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_noise_sweep(results_df: pd.DataFrame) -> pd.DataFrame:
    return (
        results_df.groupby("snr_rms_target")
        .agg(
            n_trials=("trial", "count"),
            lin_rmse_mean=("lin_rmse", "mean"),
            lin_rmse_std=("lin_rmse", "std"),
            log_rmse_mean=("log_rmse", "mean"),
            log_rmse_std=("log_rmse", "std"),
            lin_rel_rmse_mean=("lin_rel_rmse", "mean"),
            log_rel_rmse_mean=("log_rel_rmse", "mean"),
            lin_corr_mean=("lin_corr", "mean"),
            log_corr_mean=("log_corr", "mean"),
            lin_neg_frac_mean=("lin_neg_frac", "mean"),
            lin_chi2_mean=("lin_chi2", "mean"),
            log_chi2_mean=("log_chi2", "mean"),
            log_iters_mean=("log_iters", "mean"),
            log_nonconverged_frac=("log_converged", lambda s: float(1.0 - np.mean(np.asarray(s, dtype=float)))),
            log_retry_frac=("log_retried", lambda s: float(np.mean(np.asarray(s, dtype=float)))),
        )
        .sort_index(ascending=False)
    )

