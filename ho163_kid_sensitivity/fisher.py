from __future__ import annotations

import numpy as np

from .configs import SimulationConfig
from .model import expected_counts


def _model_derivatives(
    config: SimulationConfig, live_time_years: float, reference_mnu2_ev2: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    mu0, model = expected_counts(config, live_time_years, mnu2_ev2=reference_mnu2_ev2)
    mu0 = np.maximum(mu0, 1e-12)

    dm = 1e-3
    mu_plus, _ = expected_counts(config, live_time_years, mnu2_ev2=reference_mnu2_ev2 + dm)
    mu_minus, _ = expected_counts(config, live_time_years, mnu2_ev2=reference_mnu2_ev2 - dm)
    d_mnu2 = (mu_plus - mu_minus) / (2.0 * dm)

    d_norm = mu0
    d_flat = np.ones_like(mu0)

    saved_tau = config.tau_eff_us
    step_us = max(0.01, 0.05 * max(saved_tau, 0.1))
    config.tau_eff_us = saved_tau + step_us
    mu_tau_plus, _ = expected_counts(config, live_time_years, mnu2_ev2=reference_mnu2_ev2)
    config.tau_eff_us = max(0.0, saved_tau - step_us)
    mu_tau_minus, _ = expected_counts(config, live_time_years, mnu2_ev2=reference_mnu2_ev2)
    config.tau_eff_us = saved_tau
    d_tau = (mu_tau_plus - mu_tau_minus) / (2.0 * step_us)
    return mu0, np.vstack([d_mnu2, d_norm, d_flat, d_tau]), np.array([0.0, 1e-18, 1e-18, 1e-18]), model


def fisher_mnu_sensitivity(config: SimulationConfig, live_time_years: float) -> dict:
    """Estimate mass resolution with a compact Fisher model.

    Parameters are mnu2, normalization, flat background, and pileup fraction.
    Q-value is fixed in this GUI estimator. This makes the result a planning
    estimate, not a final sensitivity claim.
    """
    mu0, derivs, regularizer, model = _model_derivatives(
        config, live_time_years, reference_mnu2_ev2=config.mnu2_ev2
    )

    fisher = derivs @ ((derivs / mu0).T)
    fisher += np.diag(regularizer)
    cov = np.linalg.pinv(fisher, rcond=1e-12)
    sigma_mnu2 = float(np.sqrt(max(cov[0, 0], 0.0)))
    mnu90_ev = float(np.sqrt(max(1.64 * sigma_mnu2, 0.0)))
    return {
        "sigma_mnu2_ev2": sigma_mnu2,
        "mnu90_ev": mnu90_ev,
        "fisher": fisher,
        "model": model,
    }


def estimate_mnu_from_counts(
    counts: np.ndarray,
    config: SimulationConfig,
    live_time_years: float,
    reference_mnu2_ev2: float = 0.0,
) -> dict:
    """Linearized Poisson best-fit mnu estimate from accumulated counts."""
    if live_time_years <= 0.0 or float(np.sum(counts)) <= 0.0:
        return {
            "mnu2_hat_ev2": float("nan"),
            "mnu_hat_mev": float("nan"),
            "sigma_mnu2_ev2": float("nan"),
            "sigma_mnu_mev": float("nan"),
        }

    mu0, derivs, regularizer, _ = _model_derivatives(config, live_time_years, reference_mnu2_ev2)
    fisher = derivs @ ((derivs / mu0).T)
    fisher += np.diag(regularizer)
    cov = np.linalg.pinv(fisher, rcond=1e-12)
    residual = np.asarray(counts, dtype=float) - mu0
    rhs = derivs @ (residual / mu0)
    delta = cov @ rhs
    mnu2_hat = float(reference_mnu2_ev2 + delta[0])
    sigma_mnu2 = float(np.sqrt(max(cov[0, 0], 0.0)))
    mnu_hat_mev = float(np.sqrt(max(mnu2_hat, 0.0)) * 1000.0)
    sigma_mnu_mev = (
        float(1000.0 * sigma_mnu2 / (2.0 * np.sqrt(mnu2_hat)))
        if mnu2_hat > 0.0
        else float("nan")
    )
    return {
        "mnu2_hat_ev2": mnu2_hat,
        "mnu_hat_mev": mnu_hat_mev,
        "sigma_mnu2_ev2": sigma_mnu2,
        "sigma_mnu_mev": sigma_mnu_mev,
    }
