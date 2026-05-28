from __future__ import annotations

from collections import OrderedDict
import threading

import numpy as np

from .configs import SECONDS_PER_YEAR, SimulationConfig
from .backend import gpu_status
from .response import bin_density_interpolated, gaussian_convolve_density, pileup_density
from .spectra import ho163_spectrum, make_energy_grid, normalize_density


_MODEL_CACHE_MAX = 16
_MODEL_CACHE: OrderedDict[tuple, dict] = OrderedDict()
_MODEL_CACHE_LOCK = threading.RLock()


def _atomic_lines_key(lines: list[dict]) -> tuple:
    return tuple(
        (
            str(line["label"]),
            float(line["energy_ev"]),
            float(line["width_ev"]),
            float(line["strength"]),
        )
        for line in lines
    )


def _model_cache_key(config: SimulationConfig, mnu2_ev2: float | None) -> tuple:
    return (
        float(config.q_ec_ev),
        float(config.mnu2_ev2 if mnu2_ev2 is None else mnu2_ev2),
        float(config.energy_fwhm_ev),
        int(config.n_detectors),
        float(config.activity_bq),
        float(config.tau_eff_us),
        float(config.background_per_ev_year),
        int(config.n_grid),
        int(config.n_bins),
        bool(config.use_gpu),
        float(config.fit_low_offset_ev),
        float(config.fit_high_offset_ev),
        _atomic_lines_key(config.atomic_lines),
    )


def build_model(config: SimulationConfig, mnu2_ev2: float | None = None) -> dict:
    cache_key = _model_cache_key(config, mnu2_ev2)
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            _MODEL_CACHE.move_to_end(cache_key)
            return cached

    model = _build_model_uncached(config, mnu2_ev2=mnu2_ev2)

    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[cache_key] = model
        _MODEL_CACHE.move_to_end(cache_key)
        while len(_MODEL_CACHE) > _MODEL_CACHE_MAX:
            _MODEL_CACHE.popitem(last=False)
    return model


def _build_model_uncached(config: SimulationConfig, mnu2_ev2: float | None = None) -> dict:
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
    bin_probs = bin_density_interpolated(energy, measured, bin_edges)

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
