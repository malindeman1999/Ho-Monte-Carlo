from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .configs import SimulationConfig, config_from_dict


@dataclass
class RunState:
    config: SimulationConfig
    live_time_years: float
    counts: np.ndarray
    bin_edges_ev: np.ndarray
    sensitivity_history: list[dict]
    rng_state: dict | None = None


def run_stem(config: SimulationConfig) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in config.run_name)
    return config.output_path / safe_name


def save_state(state: RunState) -> tuple[Path, Path]:
    state.config.output_path.mkdir(parents=True, exist_ok=True)
    stem = run_stem(state.config)
    npz_path = stem.with_suffix(".npz")
    json_path = stem.with_suffix(".json")
    np.savez_compressed(npz_path, counts=state.counts, bin_edges_ev=state.bin_edges_ev)
    meta = {
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "config": state.config.as_json_dict(),
        "live_time_years": state.live_time_years,
        "sensitivity_history": state.sensitivity_history,
        "rng_state": state.rng_state,
        "npz_file": npz_path.name,
    }
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return json_path, npz_path


def load_state(json_path: str | Path) -> RunState:
    json_path = Path(json_path)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    config = config_from_dict(meta["config"])
    npz_path = json_path.with_name(meta.get("npz_file", json_path.with_suffix(".npz").name))
    arr = np.load(npz_path)
    return RunState(
        config=config,
        live_time_years=float(meta["live_time_years"]),
        counts=arr["counts"],
        bin_edges_ev=arr["bin_edges_ev"],
        sensitivity_history=list(meta.get("sensitivity_history", [])),
        rng_state=meta.get("rng_state"),
    )
