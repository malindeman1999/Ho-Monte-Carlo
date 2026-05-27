from __future__ import annotations

import numpy as np

from .configs import SECONDS_PER_YEAR, SimulationConfig
from .backend import gpu_status
from .response import bin_density, gaussian_convolve_density, pileup_density
from .spectra import ho163_spectrum, make_energy_grid, normalize_density


def build_model(config: SimulationConfig, mnu2_ev2: float | None = None) -> dict:
    if config.use_gpu:
        status = gpu_status()
        # This installation currently has CPU-only PyTorch. The branch is kept
        # explicit so users see a truthful fallback instead of a silent no-op.
        if status["available"]:
            from .gpu_model import build_model_gpu

            return build_model_gpu(config, mnu2_ev2=mnu2_ev2)

    q = config.q_ec_ev
    energy = make_energy_grid(q, config.n_grid)
    single = ho163_spectrum(
        energy,
        q,
        config.mnu2_ev2 if mnu2_ev2 is None else mnu2_ev2,
        config.atomic_lines,
    )
    pp_energy, pp = pileup_density(energy, single)
    pp_on_grid = np.interp(energy, pp_energy, pp, left=0.0, right=0.0)

    f_pp = np.clip(config.pileup_fraction, 0.0, 0.25)
    mixed = (1.0 - f_pp) * single + f_pp * pp_on_grid

    if config.background_per_ev_year > 0.0:
        bg = np.ones_like(energy)
        bg = normalize_density(energy, bg)
        # Convert requested background density into an approximate fraction
        # relative to one year of Ho events, keeping this as a nuisance-scale
        # planning term rather than a physical background model.
        n_ho_year = config.total_rate_hz * SECONDS_PER_YEAR
        bg_counts = config.background_per_ev_year * (energy[-1] - energy[0])
        bg_frac = min(0.5, bg_counts / max(n_ho_year, 1.0))
        mixed = (1.0 - bg_frac) * mixed + bg_frac * bg

    measured = gaussian_convolve_density(energy, normalize_density(energy, mixed), config.energy_fwhm_ev)

    low = max(0.0, q + config.fit_low_offset_ev)
    high = min(energy[-1], q + config.fit_high_offset_ev)
    bin_edges = np.linspace(low, high, config.n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_probs = bin_density(energy, measured, bin_edges)

    return {
        "energy_ev": energy,
        "single_density": single,
        "pileup_density": pp_on_grid,
        "passing_pileup_density": f_pp * pp_on_grid,
        "measured_density": measured,
        "bin_edges_ev": bin_edges,
        "bin_centers_ev": bin_centers,
        "bin_probabilities": bin_probs,
        "pileup_fraction": f_pp,
    }


def expected_counts(config: SimulationConfig, live_time_years: float, mnu2_ev2: float | None = None) -> tuple[np.ndarray, dict]:
    model = build_model(config, mnu2_ev2=mnu2_ev2)
    n_events = config.total_rate_hz * live_time_years * SECONDS_PER_YEAR
    return model["bin_probabilities"] * n_events, model
