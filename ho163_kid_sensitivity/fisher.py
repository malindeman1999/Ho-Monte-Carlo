from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy.optimize import minimize

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
    deriv_list = [d_mnu2, d_norm]
    regularizer = [0.0, 1e-18]
    if getattr(config, "fit_use_flat_offset", True):
        deriv_list.append(np.ones_like(mu0))
        regularizer.append(1e-18)
    return mu0, np.vstack(deriv_list), np.array(regularizer, dtype=float), model


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
    reference_mnu2_ev2: float | None = None,
    fit_method: str | None = None,
    endpoint_weight: float | None = None,
) -> dict:
    """Linearized Poisson best-fit mnu estimate from accumulated counts."""
    if live_time_years <= 0.0 or float(np.sum(counts)) <= 0.0:
        n_par = 3 if getattr(config, "fit_use_flat_offset", True) else 2
        return {
            "mnu2_hat_ev2": float("nan"),
            "mnu_hat_mev": float("nan"),
            "sigma_mnu2_ev2": float("nan"),
            "sigma_mnu_mev": float("nan"),
            "mu_fit": np.asarray(counts, dtype=float) * np.nan,
            "delta_hat": np.full(n_par, np.nan),
            "fit_params": {},
        }

    method = (fit_method or getattr(config, "fit_method", "linearized")).lower()
    # Backward-compat alias: robust_mle now means nonlinear full-model Poisson.
    if method == "poisson":
        method = "robust_mle"
    if reference_mnu2_ev2 is None:
        reference_mnu2_ev2 = float(config.mnu2_ev2)
    endpoint_w = float(
        endpoint_weight
        if endpoint_weight is not None
        else getattr(config, "endpoint_weight", 1.0)
    )
    endpoint_w = max(1.0, endpoint_w)

    mu0, derivs, regularizer, model = _model_derivatives(config, live_time_years, reference_mnu2_ev2)
    weights = np.ones_like(mu0, dtype=float)
    if endpoint_w > 1.0:
        edge = float(model["bin_edges_ev"][-1])
        mask = np.asarray(model["bin_centers_ev"], dtype=float) >= (edge - 1.0)
        weights[mask] = endpoint_w

    fisher = derivs @ ((derivs * (weights / mu0)).T)
    fisher += np.diag(regularizer)
    cov = np.linalg.pinv(fisher, rcond=1e-12)

    residual = np.asarray(counts, dtype=float) - mu0
    rhs = derivs @ (weights * residual / mu0)
    delta_linear = cov @ rhs
    delta = delta_linear

    if method in {"basis_mle"}:
        design = derivs.T
        data = np.asarray(counts, dtype=float)

        def nll_and_grad(beta: np.ndarray) -> tuple[float, np.ndarray]:
            mu = np.clip(mu0 + design @ beta, 1e-12, None)
            value = float(np.sum(weights * (mu - data * np.log(mu))))
            grad = design.T @ (weights * (1.0 - data / mu))
            return value, np.asarray(grad, dtype=float)

        def objective(beta: np.ndarray) -> float:
            val, _ = nll_and_grad(beta)
            return val

        def gradient(beta: np.ndarray) -> np.ndarray:
            _, grad = nll_and_grad(beta)
            return grad

        start = np.asarray(delta_linear, dtype=float)
        try:
            opt = minimize(
                objective,
                start,
                jac=gradient,
                method="L-BFGS-B",
                options={"maxiter": 200, "ftol": 1e-10},
            )
            if opt.success and np.all(np.isfinite(opt.x)):
                delta = np.asarray(opt.x, dtype=float)
        except Exception:
            delta = delta_linear
    elif method in {"robust_mle", "mle", "robust", "poisson_full"}:
        data = np.asarray(counts, dtype=float)
        use_flat = bool(getattr(config, "fit_use_flat_offset", True))
        mu_sum = max(float(mu0.sum()), 1e-12)
        norm0 = max(float(data.sum()) / mu_sum, 1e-6)
        if use_flat:
            flat0 = max(float(np.percentile(data, 10)) * 0.1, 1e-9)
            nuisance_start = np.array([np.log(norm0), np.log(flat0)], dtype=float)
            nuisance_bounds = [(-8.0, 8.0), (-30.0, 20.0)]
        else:
            nuisance_start = np.array([np.log(norm0)], dtype=float)
            nuisance_bounds = [(-8.0, 8.0)]

        def fit_nuisance(mnu2_value: float, start: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
            mu_base, _ = expected_counts(config, live_time_years, mnu2_ev2=float(mnu2_value))

            def nll_nuisance(x: np.ndarray) -> float:
                norm = float(np.exp(x[0]))
                flat_bin = float(np.exp(x[1])) if use_flat else 0.0
                mu = np.clip(norm * mu_base + flat_bin, 1e-12, None)
                return float(np.sum(weights * (mu - data * np.log(mu))))

            best_x = np.asarray(start, dtype=float)
            try:
                opt = minimize(
                    nll_nuisance,
                    best_x,
                    method="L-BFGS-B",
                    bounds=nuisance_bounds,
                    options={"maxiter": 250, "ftol": 1e-11},
                )
                if opt.success and np.all(np.isfinite(opt.x)):
                    best_x = np.asarray(opt.x, dtype=float)
                    best_val = float(opt.fun)
                else:
                    best_val = float(nll_nuisance(best_x))
            except Exception:
                best_val = float(nll_nuisance(best_x))
            return best_val, best_x, mu_base

        span = 0.05
        coarse = np.linspace(max(0.0, reference_mnu2_ev2 - span), reference_mnu2_ev2 + span, 31)
        best = (np.inf, float(reference_mnu2_ev2), nuisance_start.copy(), mu0.copy())
        current_start = nuisance_start.copy()
        for mnu2_c in coarse:
            val, x_hat, mu_base_c = fit_nuisance(float(mnu2_c), current_start)
            current_start = x_hat
            if val < best[0]:
                best = (val, float(mnu2_c), x_hat.copy(), mu_base_c.copy())

        mnu2_center = best[1]
        refine = np.linspace(max(0.0, mnu2_center - 0.01), mnu2_center + 0.01, 41)
        current_start = best[2].copy()
        for mnu2_r in refine:
            val, x_hat, mu_base_r = fit_nuisance(float(mnu2_r), current_start)
            current_start = x_hat
            if val < best[0]:
                best = (val, float(mnu2_r), x_hat.copy(), mu_base_r.copy())

        mnu2_nl = float(best[1])
        norm_nl = float(np.exp(best[2][0]))
        flat_nl = float(np.exp(best[2][1])) if use_flat else 0.0
        mu_base_fit = best[3]
        mu_fit = np.clip(norm_nl * mu_base_fit + flat_nl, 1e-12, None)
        mnu2_hat = float(mnu2_nl)
        # Keep sigma estimate from linearized Fisher as a local uncertainty proxy.
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
            "mu_fit": mu_fit,
            "delta_hat": np.asarray(
                [mnu2_hat - reference_mnu2_ev2, np.log(norm_nl)]
                + ([np.log(flat_nl)] if use_flat else []),
                dtype=float,
            ),
            "fit_params": {
                "method": method,
                "mnu2_ev2": mnu2_hat,
                "norm": float(norm_nl),
                "flat_bin": float(flat_nl),
            },
        }

    mnu2_hat = float(reference_mnu2_ev2 + delta[0])
    mu_fit = np.clip(mu0 + derivs.T @ delta, 1e-12, None)
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
        "mu_fit": mu_fit,
        "delta_hat": np.asarray(delta, dtype=float),
        "fit_params": {
            "method": method,
            "mnu2_ev2": mnu2_hat,
        },
    }
