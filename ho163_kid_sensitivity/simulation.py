from __future__ import annotations

import numpy as np

from .configs import SimulationConfig
from .fisher import estimate_mnu_from_counts, fisher_mnu_sensitivity
from .model import expected_counts
from .state import RunState


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


def simulate_chunk(state: RunState, rng: np.random.Generator) -> dict:
    years = state.config.chunk_days / 365.25
    mu, model = expected_counts(state.config, years)
    draw = rng.poisson(mu)
    state.counts += draw
    state.live_time_years += years
    fisher = fisher_mnu_sensitivity(state.config, max(state.live_time_years, years))
    estimate = estimate_mnu_from_counts(
        state.counts,
        state.config,
        state.live_time_years,
        fit_method=state.config.fit_method,
        endpoint_weight=state.config.endpoint_weight,
    )
    entry = {
        "live_time_years": state.live_time_years,
        "total_counts": float(state.counts.sum()),
        "mnu90_ev": fisher["mnu90_ev"],
        "sigma_mnu2_ev2": fisher["sigma_mnu2_ev2"],
        "mnu2_hat_ev2": estimate["mnu2_hat_ev2"],
        "mnu_hat_mev": estimate["mnu_hat_mev"],
        "sigma_mnu_hat_mev": estimate["sigma_mnu_mev"],
        "pileup_fraction": model["pileup_fraction"],
    }
    state.sensitivity_history.append(entry)
    state.rng_state = rng.bit_generator.state
    return entry
