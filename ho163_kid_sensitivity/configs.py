from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


SECONDS_PER_YEAR = 365.25 * 24 * 3600


@dataclass
class AtomicLine:
    label: str
    energy_ev: float
    width_ev: float
    strength: float


# Simplified line set for development and trade studies. Values are deliberately
# compact rather than publication-grade atomic inputs.
DEFAULT_LINES = (
    AtomicLine("M1", 2047.0, 13.2, 1.00),
    AtomicLine("M2", 1842.0, 6.0, 0.05),
    AtomicLine("N1", 414.2, 5.4, 0.24),
    AtomicLine("N2", 333.5, 5.3, 0.012),
    AtomicLine("O1", 49.9, 3.0, 0.032),
    AtomicLine("O2", 26.3, 3.0, 0.0015),
)


@dataclass
class SimulationConfig:
    q_ec_ev: float = 2860.0
    mnu2_ev2: float = 0.100**2
    energy_fwhm_ev: float = 0.2
    n_detectors: int = 100_000
    activity_bq: float = 30.0
    tau_eff_us: float = 0.001
    background_per_ev_year: float = 0.0
    live_time_years_target: float = 1000.0
    chunk_days: float = 100.0
    n_grid: int = 32768
    n_bins: int = 50
    use_gpu: bool = False
    fit_low_offset_ev: float = -1.0
    fit_high_offset_ev: float = 0.0
    fit_method: str = "linearized"
    endpoint_weight: float = 10.0
    fit_use_flat_offset: bool = False
    rng_seed: int | None = 12345
    run_name: str = "kid_ho163_run"
    output_dir: str = "runs"
    atomic_lines: list[dict] = field(
        default_factory=lambda: [asdict(line) for line in DEFAULT_LINES]
    )

    @property
    def tau_eff_s(self) -> float:
        return self.tau_eff_us * 1e-6

    @property
    def pileup_fraction(self) -> float:
        return self.activity_bq * self.tau_eff_s

    @property
    def total_rate_hz(self) -> float:
        return self.n_detectors * self.activity_bq

    @property
    def true_mnu_mev(self) -> float:
        return (max(self.mnu2_ev2, 0.0) ** 0.5) * 1000.0

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    def as_json_dict(self) -> dict:
        return asdict(self)


def config_from_dict(data: dict) -> SimulationConfig:
    known = {field.name for field in SimulationConfig.__dataclass_fields__.values()}
    return SimulationConfig(**{key: value for key, value in data.items() if key in known})
