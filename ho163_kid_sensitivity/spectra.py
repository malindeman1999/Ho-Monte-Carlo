from __future__ import annotations

import numpy as np

from .configs import AtomicLine, DEFAULT_LINES


def make_energy_grid(q_ec_ev: float, n_grid: int) -> np.ndarray:
    return np.linspace(0.0, 2.0 * q_ec_ev + 50.0, n_grid)


def normalize_density(energy_ev: np.ndarray, density: np.ndarray) -> np.ndarray:
    density = np.clip(np.asarray(density, dtype=float), 0.0, None)
    area = np.trapezoid(density, energy_ev)
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("Cannot normalize a non-positive spectrum.")
    return density / area


def lorentzian(energy_ev: np.ndarray, center_ev: float, width_ev: float) -> np.ndarray:
    half = 0.5 * width_ev
    return (half / np.pi) / ((energy_ev - center_ev) ** 2 + half**2)


def coerce_lines(lines: list[dict] | None = None) -> tuple[AtomicLine, ...]:
    if lines is None:
        return DEFAULT_LINES
    return tuple(AtomicLine(**line) for line in lines)


def ho163_spectrum(
    energy_ev: np.ndarray,
    q_ec_ev: float,
    mnu2_ev2: float = 0.0,
    lines: list[dict] | None = None,
) -> np.ndarray:
    """Return normalized single-event calorimetric Ho-163 spectrum.

    The phase-space factor is parameterized in m_nu^2. Negative values are
    allowed for Fisher derivatives but the square-root argument is clipped at
    zero where the spectrum is outside the physical endpoint.
    """
    atomic = np.zeros_like(energy_ev, dtype=float)
    for line in coerce_lines(lines):
        atomic += line.strength * lorentzian(energy_ev, line.energy_ev, line.width_ev)

    enu = q_ec_ev - energy_ev
    root_arg = np.maximum(enu**2 - mnu2_ev2, 0.0)
    phase = np.where(enu > 0.0, enu * np.sqrt(root_arg), 0.0)
    return normalize_density(energy_ev, atomic * phase)
