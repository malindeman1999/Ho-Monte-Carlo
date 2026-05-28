from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .configs import SimulationConfig
from .fisher import estimate_mnu_from_counts
from .model import expected_counts
from .state import RunState


def _seed_from_parts(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
    return int.from_bytes(digest, "little")


def _base_seed(config: SimulationConfig) -> int:
    return 0 if config.rng_seed is None else int(config.rng_seed)


def _live_time_key(live_time_years: float) -> int:
    return int(round(float(live_time_years) * 1_000_000_000_000))


def new_state(config: SimulationConfig) -> RunState:
    mu, model = expected_counts(config, 0.0)
    return RunState(
        config=config,
        live_time_years=0.0,
        counts=np.zeros_like(mu),
        bin_edges_ev=model["bin_edges_ev"],
        sensitivity_history=[],
    )


def rng_from_state(state: RunState) -> np.random.Generator:
    rng = np.random.default_rng(state.config.rng_seed)
    if state.rng_state:
        rng.bit_generator.state = state.rng_state
    return rng


def _signed_mnu_mev(mnu2_ev2: float) -> float:
    return float(np.sign(mnu2_ev2) * np.sqrt(abs(mnu2_ev2)) * 1000.0)


def _physical_mnu_mev(mnu2_ev2: float) -> float:
    return float(np.sqrt(max(mnu2_ev2, 0.0)) * 1000.0)


def _fit_pseudoexperiment(
    counts: np.ndarray, config: SimulationConfig, live_time_years: float
) -> tuple[float, float, float]:
    estimate = estimate_mnu_from_counts(
        counts,
        config,
        live_time_years,
        fit_method=config.fit_method,
        endpoint_weight=config.endpoint_weight,
    )
    mnu2_hat = float(estimate["mnu2_hat_ev2"])
    return mnu2_hat, _signed_mnu_mev(mnu2_hat), _physical_mnu_mev(mnu2_hat)


def _parallel_fit_summary(
    config: SimulationConfig, live_time_years: float, rng: np.random.Generator
) -> tuple[float, float, float, float, float, float, int]:
    n_runs = max(1, int(config.parallel_fit_runs))
    cumulative_mu, _ = expected_counts(config, live_time_years)
    ensemble_rng = np.random.default_rng(
        _seed_from_parts("fit-ensemble", _base_seed(config), _live_time_key(live_time_years))
    )
    pseudo_counts = ensemble_rng.poisson(cumulative_mu, size=(n_runs, cumulative_mu.size))
    max_workers = min(n_runs, max(1, (os.cpu_count() or 2) // 2), 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        fit_pairs = np.asarray(
            list(
                executor.map(
                    lambda sample: _fit_pseudoexperiment(
                        sample, config, live_time_years
                    ),
                    pseudo_counts,
                )
            ),
            dtype=float,
        )
    if fit_pairs.ndim != 2 or fit_pairs.shape[1] != 3:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            n_runs,
        )
    mnu2_fits = fit_pairs[:, 0]
    signed_fits = fit_pairs[:, 1]
    physical_fits = fit_pairs[:, 2]
    finite = signed_fits[np.isfinite(signed_fits)]
    if finite.size == 0:
        signed_mean = float("nan")
        signed_std = float("nan")
        signed_rms_error = float("nan")
    else:
        signed_mean = float(np.mean(finite))
        signed_std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        signed_error = finite - config.true_mnu_mev
        signed_rms_error = float(np.sqrt(np.mean(signed_error**2)))
    physical_finite = physical_fits[np.isfinite(physical_fits)]
    mnu2_finite = mnu2_fits[np.isfinite(mnu2_fits)]
    if mnu2_finite.size == 0:
        boundary_fraction = float("nan")
    else:
        boundary_fraction = float(np.mean(mnu2_finite <= 0.0))
    if physical_finite.size == 0:
        physical_mean = float("nan")
        physical_std = float("nan")
        physical_rms_error = float("nan")
    else:
        physical_mean = float(np.mean(physical_finite))
        physical_std = float(np.std(physical_finite, ddof=1)) if physical_finite.size > 1 else 0.0
        physical_error = physical_finite - config.true_mnu_mev
        physical_rms_error = float(np.sqrt(np.mean(physical_error**2)))
    return (
        physical_mean,
        physical_std,
        physical_rms_error,
        physical_mean,
        physical_rms_error,
        boundary_fraction,
        n_runs,
    )


def compute_chunk_update(
    config: SimulationConfig,
    live_time_years: float,
    counts: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, dict, dict]:
    years = config.chunk_days / 365.25
    new_live_time_years = min(live_time_years + years, config.live_time_years_target)
    if np.isclose(new_live_time_years, config.live_time_years_target, rtol=0.0, atol=1e-12):
        new_live_time_years = float(config.live_time_years_target)
    cumulative_mu, model = expected_counts(config, new_live_time_years)
    data_rng = np.random.default_rng(
        _seed_from_parts("data", _base_seed(config), _live_time_key(new_live_time_years))
    )
    new_counts = data_rng.poisson(cumulative_mu).astype(float)
    estimate = estimate_mnu_from_counts(
        new_counts,
        config,
        new_live_time_years,
        fit_method=config.fit_method,
        endpoint_weight=config.endpoint_weight,
    )
    (
        fit_mean_mev,
        fit_std_mev,
        fit_signed_rms_error_mev,
        fit_physical_mean_mev,
        fit_physical_rms_error_mev,
        fit_boundary_fraction,
        fit_runs,
    ) = _parallel_fit_summary(
        config, new_live_time_years, rng
    )
    entry = {
        "live_time_years": new_live_time_years,
        "total_counts": float(new_counts.sum()),
        "mnu2_hat_ev2": estimate["mnu2_hat_ev2"],
        "mnu_hat_mev": estimate["mnu_hat_mev"],
        "sigma_mnu2_ev2": estimate["sigma_mnu2_ev2"],
        "sigma_mnu_hat_mev": estimate["sigma_mnu_mev"],
        "fit_ensemble_mean_mev": fit_mean_mev,
        "fit_ensemble_std_mev": fit_std_mev,
        "fit_ensemble_rms_error_mev": fit_signed_rms_error_mev,
        "fit_ensemble_signed_rms_error_mev": fit_signed_rms_error_mev,
        "fit_ensemble_physical_mean_mev": fit_physical_mean_mev,
        "fit_ensemble_physical_rms_error_mev": fit_physical_rms_error_mev,
        "fit_ensemble_boundary_fraction": fit_boundary_fraction,
        "fit_ensemble_runs": fit_runs,
        "pileup_fraction": model["pileup_fraction"],
    }
    return new_counts, new_live_time_years, entry, rng.bit_generator.state


def simulate_chunk(state: RunState, rng: np.random.Generator) -> dict:
    counts, live_time_years, entry, rng_state = compute_chunk_update(
        state.config,
        state.live_time_years,
        state.counts,
        rng,
    )
    state.counts = counts
    state.live_time_years = live_time_years
    state.sensitivity_history.append(entry)
    state.rng_state = rng_state
    return entry
