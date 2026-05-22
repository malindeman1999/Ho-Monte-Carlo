from __future__ import annotations

import math
import numpy as np

from .backend import gpu_status
from .configs import SimulationConfig


def can_use_gpu() -> bool:
    return bool(gpu_status()["available"])


def build_model_gpu(config: SimulationConfig, mnu2_ev2: float | None = None) -> dict:
    """Build the binned model with torch on CUDA when available."""
    if not can_use_gpu():
        raise RuntimeError(gpu_status()["reason"])

    import torch  # pragma: no cover

    device = torch.device("cuda")
    dtype = torch.float64
    q = float(config.q_ec_ev)
    energy = torch.linspace(0.0, 2.0 * q + 50.0, int(config.n_grid), device=device, dtype=dtype)
    d_e = energy[1] - energy[0]

    def trapz(y, x):
        return torch.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))

    def normalize(y, x):
        y = torch.clamp(y, min=0.0)
        area = trapz(y, x)
        return y / torch.clamp(area, min=torch.finfo(dtype).tiny)

    atomic = torch.zeros_like(energy)
    for line in config.atomic_lines:
        half = 0.5 * float(line["width_ev"])
        atomic += float(line["strength"]) * (half / math.pi) / (
            (energy - float(line["energy_ev"])) ** 2 + half**2
        )

    mnu2 = float(config.mnu2_ev2 if mnu2_ev2 is None else mnu2_ev2)
    enu = q - energy
    phase = torch.where(enu > 0.0, enu * torch.sqrt(torch.clamp(enu**2 - mnu2, min=0.0)), 0.0)
    single = normalize(atomic * phase, energy)

    def fft_convolve(a, b):
        out_len = a.numel() + b.numel() - 1
        n_fft = 1 << (out_len - 1).bit_length()
        fa = torch.fft.rfft(a, n=n_fft)
        fb = torch.fft.rfft(b, n=n_fft)
        return torch.fft.irfft(fa * fb, n=n_fft)[:out_len] * d_e

    pp = fft_convolve(single, single)
    pp_energy = torch.arange(pp.numel(), device=device, dtype=dtype) * d_e
    pp = normalize(pp, pp_energy)
    idx = torch.clamp((energy / d_e).round().long(), 0, pp.numel() - 1)
    pp_on_grid = pp[idx]

    f_pp = min(max(config.pileup_fraction, 0.0), 0.25)
    mixed = normalize((1.0 - f_pp) * single + f_pp * pp_on_grid, energy)

    sigma = float(config.energy_fwhm_ev) / 2.355
    if sigma > 0.0:
        radius = max(1, int(math.ceil(8.0 * sigma / float(d_e.detach().cpu()))))
        gx = torch.arange(-radius, radius + 1, device=device, dtype=dtype) * d_e
        kernel = torch.exp(-0.5 * (gx / sigma) ** 2)
        kernel = kernel / torch.sum(kernel)
        full = fft_convolve(mixed, kernel)
        start = (kernel.numel() - 1) // 2
        measured = normalize(full[start : start + mixed.numel()], energy)
    else:
        measured = mixed

    low = max(0.0, q + float(config.fit_low_offset_ev))
    high = min(float(energy[-1].detach().cpu()), q + float(config.fit_high_offset_ev))
    bin_edges = torch.linspace(low, high, int(config.n_bins) + 1, device=device, dtype=dtype)
    cumulative = torch.zeros_like(energy)
    cumulative[1:] = torch.cumsum(0.5 * (measured[1:] + measured[:-1]) * (energy[1:] - energy[:-1]), dim=0)
    positions = torch.searchsorted(energy, bin_edges).clamp(1, energy.numel() - 1)
    x0 = energy[positions - 1]
    x1 = energy[positions]
    y0 = cumulative[positions - 1]
    y1 = cumulative[positions]
    edge_cdf = y0 + (bin_edges - x0) * (y1 - y0) / torch.clamp(x1 - x0, min=torch.finfo(dtype).eps)
    probs = torch.diff(edge_cdf)
    probs = probs / torch.clamp(torch.sum(probs), min=torch.finfo(dtype).tiny)

    edges_np = to_numpy(bin_edges)
    return {
        "energy_ev": to_numpy(energy),
        "single_density": to_numpy(single),
        "pileup_density": to_numpy(pp_on_grid),
        "measured_density": to_numpy(measured),
        "bin_edges_ev": edges_np,
        "bin_centers_ev": 0.5 * (edges_np[:-1] + edges_np[1:]),
        "bin_probabilities": to_numpy(probs),
        "pileup_fraction": f_pp,
    }


def to_numpy(array) -> np.ndarray:
    try:
        return array.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(array)
