from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import fftconvolve

from .spectra import normalize_density


def convolve_densities(a: np.ndarray, b: np.ndarray, d_e: float) -> np.ndarray:
    return fftconvolve(a, b, mode="full") * d_e


def pileup_density(energy_ev: np.ndarray, single_density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d_e = energy_ev[1] - energy_ev[0]
    conv = convolve_densities(single_density, single_density, d_e)
    conv_energy = np.arange(conv.size) * d_e
    return conv_energy, normalize_density(conv_energy, conv)


def gaussian_kernel(d_e: float, sigma_ev: float, nsigma: float = 8.0) -> np.ndarray:
    if sigma_ev <= 0.0:
        return np.array([1.0])
    radius = max(1, int(np.ceil(nsigma * sigma_ev / d_e)))
    x = np.arange(-radius, radius + 1) * d_e
    kernel = np.exp(-0.5 * (x / sigma_ev) ** 2)
    kernel /= kernel.sum()
    return kernel


def gaussian_convolve_density(
    energy_ev: np.ndarray, density: np.ndarray, fwhm_ev: float
) -> np.ndarray:
    d_e = energy_ev[1] - energy_ev[0]
    kernel = gaussian_kernel(d_e, fwhm_ev / 2.355)
    smoothed = fftconvolve(density, kernel, mode="same")
    return normalize_density(energy_ev, smoothed)


def bin_density(
    energy_ev: np.ndarray, density: np.ndarray, bin_edges_ev: np.ndarray
) -> np.ndarray:
    cumulative = np.zeros_like(energy_ev)
    cumulative[1:] = np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(energy_ev))
    edge_cdf = np.interp(bin_edges_ev, energy_ev, cumulative, left=0.0, right=cumulative[-1])
    return np.diff(edge_cdf)


def bin_density_interpolated(
    energy_ev: np.ndarray,
    density: np.ndarray,
    bin_edges_ev: np.ndarray,
    samples_per_bin: int = 4,
    min_points: int = 4097,
) -> np.ndarray:
    """Integrate a smooth local interpolation of a density over bin edges.

    This avoids making fine fit-window histograms depend on the coarse global
    spectrum-grid spacing. It interpolates only inside the requested bin window;
    it does not add physical information beyond the sampled model.
    """
    bin_edges_ev = np.asarray(bin_edges_ev, dtype=float)
    if bin_edges_ev.size < 2:
        return np.array([], dtype=float)
    n_local = max(int(min_points), int(samples_per_bin) * (bin_edges_ev.size - 1) + 1)
    local_energy = np.linspace(float(bin_edges_ev[0]), float(bin_edges_ev[-1]), n_local)
    interpolator = PchipInterpolator(energy_ev, density, extrapolate=False)
    local_density = np.asarray(interpolator(local_energy), dtype=float)
    local_density = np.clip(np.nan_to_num(local_density, nan=0.0), 0.0, None)
    return bin_density(local_energy, local_density, bin_edges_ev)
