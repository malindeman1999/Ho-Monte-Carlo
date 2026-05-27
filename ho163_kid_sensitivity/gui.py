from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

from .backend import backend_label, gpu_status
from .configs import SECONDS_PER_YEAR, SimulationConfig
from .fisher import estimate_mnu_from_counts, fisher_mnu_sensitivity
from .model import expected_counts
from .response import bin_density, gaussian_convolve_density
from .simulation import new_state, rng_from_state, simulate_chunk
from .state import RunState, load_state, save_state


class SimulatorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Ho-163 KID/TKID Sensitivity Simulator")
        self.geometry("1280x800")
        self.queue: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.state_lock = threading.Lock()
        self.state: RunState | None = None
        self.spectrum_preview_config: SimulationConfig | None = None
        self.entries: dict[str, tk.Variable] = {}
        self._build_ui()
        self.after(150, self._poll_queue)

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=10)
        left.grid(row=0, column=0, sticky="ns")
        left.columnconfigure(1, weight=1)

        fields = [
            ("run_name", "Run name", tk.StringVar(value="kid_ho163_run")),
            ("q_ec_ev", "Q EC (eV)", tk.DoubleVar(value=2860.0)),
            ("mnu_mev", "True m_nu (meV)", tk.DoubleVar(value=100.0)),
            ("energy_fwhm_ev", "FWHM (eV)", tk.DoubleVar(value=0.2)),
            ("n_detectors", "Detectors", tk.IntVar(value=100000)),
            ("activity_bq", "Activity/pixel (Bq)", tk.DoubleVar(value=30.0)),
            ("tau_eff_us", "Tau eff (us)", tk.DoubleVar(value=0.001)),
            ("live_time_years_target", "Target years", tk.DoubleVar(value=1000.0)),
            ("chunk_days", "Chunk days", tk.DoubleVar(value=100.0)),
            ("n_grid", "Energy grid", tk.IntVar(value=32768)),
            ("n_bins", "Histogram bins", tk.IntVar(value=50)),
            ("fit_low_offset_ev", "Fit low Q+ (eV)", tk.DoubleVar(value=-1.0)),
            ("fit_high_offset_ev", "Fit high Q+ (eV)", tk.DoubleVar(value=0.0)),
            ("endpoint_weight", "Endpoint weight", tk.DoubleVar(value=10.0)),
            ("rng_seed", "RNG seed", tk.IntVar(value=12345)),
        ]
        for row, (key, label, var) in enumerate(fields):
            ttk.Label(left, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(left, textvariable=var, width=18).grid(row=row, column=1, sticky="ew", pady=2)
            self.entries[key] = var

        self.entries["use_gpu"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Use GPU if available", variable=self.entries["use_gpu"]).grid(
            row=len(fields), column=0, columnspan=2, sticky="w", pady=(8, 2)
        )
        method_row = len(fields) + 1
        ttk.Label(left, text="Fit method").grid(row=method_row, column=0, sticky="w", pady=2)
        self.entries["fit_method"] = tk.StringVar(value="linearized")
        ttk.Combobox(
            left,
            textvariable=self.entries["fit_method"],
            values=("linearized", "robust_mle", "basis_mle"),
            state="readonly",
            width=16,
        ).grid(row=method_row, column=1, sticky="ew", pady=2)
        offset_row = method_row + 1
        self.entries["fit_use_flat_offset"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            left,
            text="Use flat offset nuisance",
            variable=self.entries["fit_use_flat_offset"],
        ).grid(row=offset_row, column=0, columnspan=2, sticky="w", pady=(2, 0))

        buttons = ttk.Frame(left)
        buttons.grid(row=len(fields) + 3, column=0, columnspan=2, sticky="ew", pady=10)
        for idx, (text, command) in enumerate(
            [
                ("New", self.new_run),
                ("Start", self.start),
                ("Pause", self.pause),
                ("Save", self.save),
                ("Load", self.load),
            ]
        ):
            ttk.Button(buttons, text=text, command=command).grid(row=0, column=idx, padx=2)
        ttk.Button(buttons, text="Update spectra", command=self.update_spectra).grid(
            row=1, column=0, columnspan=5, sticky="ew", padx=2, pady=(6, 0)
        )

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(left, textvariable=self.status, wraplength=280).grid(
            row=len(fields) + 4, column=0, columnspan=2, sticky="ew", pady=6
        )

        gpu_row = len(fields) + 5
        gpu_frame = ttk.Frame(left)
        gpu_frame.grid(row=gpu_row, column=0, columnspan=2, sticky="w", pady=(2, 6))
        ttk.Label(gpu_frame, text="GPU status:").grid(row=0, column=0, sticky="w")
        self.gpu_light = tk.Canvas(gpu_frame, width=14, height=14, highlightthickness=0)
        self.gpu_light.grid(row=0, column=1, padx=(8, 4))
        self.gpu_light_id = self.gpu_light.create_oval(2, 2, 12, 12, fill="#888888", outline="#555555")
        self.gpu_state_label = tk.StringVar(value="unknown")
        ttk.Label(gpu_frame, textvariable=self.gpu_state_label).grid(row=0, column=2, sticky="w")

        self.metrics = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.metrics, justify="left", wraplength=280).grid(
            row=len(fields) + 6, column=0, columnspan=2, sticky="ew", pady=6
        )

        right = ttk.Frame(self, padding=(0, 10, 10, 10))
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax_full = self.fig.add_subplot(411)
        self.ax_hist = self.fig.add_subplot(412)
        self.ax_fit = self.fig.add_subplot(413)
        self.ax_sens = self.fig.add_subplot(414)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.new_run()

    def config_from_ui(self) -> SimulationConfig:
        return SimulationConfig(
            run_name=str(self.entries["run_name"].get()),
            q_ec_ev=float(self.entries["q_ec_ev"].get()),
            mnu2_ev2=(float(self.entries["mnu_mev"].get()) / 1000.0) ** 2,
            energy_fwhm_ev=float(self.entries["energy_fwhm_ev"].get()),
            n_detectors=int(self.entries["n_detectors"].get()),
            activity_bq=float(self.entries["activity_bq"].get()),
            tau_eff_us=float(self.entries["tau_eff_us"].get()),
            live_time_years_target=float(self.entries["live_time_years_target"].get()),
            chunk_days=float(self.entries["chunk_days"].get()),
            n_grid=int(self.entries["n_grid"].get()),
            n_bins=int(self.entries["n_bins"].get()),
            fit_low_offset_ev=float(self.entries["fit_low_offset_ev"].get()),
            fit_high_offset_ev=float(self.entries["fit_high_offset_ev"].get()),
            fit_method=str(self.entries["fit_method"].get()),
            endpoint_weight=float(self.entries["endpoint_weight"].get()),
            fit_use_flat_offset=bool(self.entries["fit_use_flat_offset"].get()),
            rng_seed=int(self.entries["rng_seed"].get()),
            use_gpu=bool(self.entries["use_gpu"].get()),
        )

    def push_config_to_ui(self, config: SimulationConfig) -> None:
        for key, var in self.entries.items():
            if key == "mnu_mev":
                var.set(config.true_mnu_mev)
                continue
            if hasattr(config, key):
                var.set(getattr(config, key))

    def new_run(self) -> None:
        try:
            config = self.config_from_ui()
            self.state = new_state(config)
            self.spectrum_preview_config = None
            result = fisher_mnu_sensitivity(config, max(config.chunk_days / 365.25, 1e-6))
            self.status.set(f"New run ready. Backend: {backend_label(config.use_gpu)}")
            self._update_metrics(result["mnu90_ev"])
            self._redraw(refresh_spectrum=True)
        except Exception as exc:
            messagebox.showerror("New run failed", str(exc))

    def update_spectra(self) -> None:
        try:
            config = self.config_from_ui()
            running = bool(self.worker and self.worker.is_alive())
            empty_run = self.state is None or (
                self.state.counts.sum() == 0.0 and not self.state.sensitivity_history
            )
            if not running and empty_run:
                self.state = new_state(config)
                self.spectrum_preview_config = None
                result = fisher_mnu_sensitivity(config, max(config.chunk_days / 365.25, 1e-6))
                self.status.set("Spectra updated. These parameters will be used when the run starts.")
                self._update_metrics(result["mnu90_ev"])
            else:
                self.spectrum_preview_config = config
                self.status.set(
                    "Spectrum preview updated. Active run parameters are unchanged; use New to apply inputs to a run."
                )
            self._redraw(refresh_spectrum=True)
        except Exception as exc:
            messagebox.showerror("Spectrum update failed", str(exc))

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        if self.state is None:
            self.new_run()
        self.spectrum_preview_config = None
        self._redraw(refresh_spectrum=True)
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._run_worker, daemon=True)
        self.worker.start()
        self.status.set("Running...")

    def pause(self) -> None:
        self.stop_event.set()
        self.status.set("Pause requested. Current chunk will finish first.")

    def save(self) -> None:
        if self.state is None:
            return
        try:
            json_path, _ = save_state(self.state)
            self.status.set(f"Saved {json_path}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def load(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=str(Path("runs").resolve()),
            filetypes=[("Run metadata", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.state = load_state(path)
            self.spectrum_preview_config = None
            self.push_config_to_ui(self.state.config)
            self.status.set(f"Loaded {path}")
            self._update_metrics()
            self._redraw(refresh_spectrum=True)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))

    def _run_worker(self) -> None:
        assert self.state is not None
        rng = rng_from_state(self.state)
        try:
            while not self.stop_event.is_set():
                with self.state_lock:
                    if self.state.live_time_years >= self.state.config.live_time_years_target:
                        break
                    entry = simulate_chunk(self.state, rng)
                    save_state(self.state)
                self.queue.put(("progress", entry))
            self.queue.put(("stopped", None))
        except Exception as exc:
            self.queue.put(("error", str(exc)))

    def _poll_queue(self) -> None:
        latest_progress = None
        stopped = False
        error = None
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "progress":
                    latest_progress = payload
                elif kind == "stopped":
                    stopped = True
                elif kind == "error":
                    error = payload
        except queue.Empty:
            pass
        if latest_progress is not None:
            self._update_metrics(latest_progress["mnu90_ev"])
            self._redraw()
        if stopped:
            self.status.set("Stopped. Progress has been saved.")
        if error is not None:
            self.status.set("Error.")
            messagebox.showerror("Simulation error", error)
        self.after(150, self._poll_queue)

    def _state_snapshot(self) -> tuple[SimulationConfig, float, np.ndarray, np.ndarray, list[dict]]:
        assert self.state is not None
        with self.state_lock:
            return (
                self.state.config,
                float(self.state.live_time_years),
                self.state.counts.copy(),
                self.state.bin_edges_ev.copy(),
                [dict(row) for row in self.state.sensitivity_history],
            )

    def _update_metrics(self, mnu90: float | None = None) -> None:
        if self.state is None:
            return
        cfg, live_time_years, counts, _, sensitivity_history = self._state_snapshot()
        self._update_gpu_indicator(cfg)
        if mnu90 is None and sensitivity_history:
            mnu90 = sensitivity_history[-1]["mnu90_ev"]
        mnu = "pending" if mnu90 is None else f"{1000.0 * mnu90:.1f} meV"
        estimate = None
        if sensitivity_history:
            estimate = sensitivity_history[-1]
        elif counts.sum() > 0.0:
            estimate = estimate_mnu_from_counts(
                counts,
                cfg,
                live_time_years,
                fit_method=cfg.fit_method,
                endpoint_weight=cfg.endpoint_weight,
            )
        best = "pending"
        best_mnu2 = "pending"
        if estimate and np.isfinite(estimate.get("mnu_hat_mev", np.nan)):
            err = estimate.get("sigma_mnu_hat_mev", np.nan)
            best = f"{estimate['mnu_hat_mev']:.1f} meV"
            if np.isfinite(err):
                best += f" +/- {err:.1f} meV"
        if estimate and np.isfinite(estimate.get("mnu2_hat_ev2", np.nan)):
            best_mnu2 = f"{estimate['mnu2_hat_ev2']:+.4g} eV^2"
            sig2 = estimate.get("sigma_mnu2_ev2", np.nan)
            if np.isfinite(sig2):
                best_mnu2 += f" +/- {sig2:.3g}"
        power_txt = "pending"
        crossing_txt = "pending"
        if len(sensitivity_history) >= 3:
            fit = self._fit_power_law_crossing(
                np.array([row["live_time_years"] for row in sensitivity_history], dtype=float),
                np.array([1000.0 * row["mnu90_ev"] for row in sensitivity_history], dtype=float),
                cfg.true_mnu_mev,
            )
            if fit is not None:
                power_txt = f"m90 = {fit['a_mev']:.3g} * t^{fit['b']:.3g}"
                crossing_txt = fit["crossing_text"]
        self.metrics.set(
            "\n".join(
                [
                    f"Live time: {live_time_years:.4g} yr",
                    f"Counts: {counts.sum():.4g}",
                    f"True m_nu: {cfg.true_mnu_mev:.1f} meV",
                    f"Fit m_nu^2: {best_mnu2}",
                    f"Physical m_nu: {best}",
                    f"f_pp: {cfg.pileup_fraction:.3g}",
                    f"Rate: {cfg.total_rate_hz:.3g} /s",
                    f"90% sensitivity: {mnu}",
                    f"Power law fit: {power_txt}",
                    f"Crossing @ true m_nu: {crossing_txt}",
                    f"Backend: {backend_label(cfg.use_gpu)}",
                ]
            )
        )

    def _fit_power_law_crossing(
        self, years: np.ndarray, m90_mev: np.ndarray, target_mev: float
    ) -> dict | None:
        mask = np.isfinite(years) & np.isfinite(m90_mev) & (years > 0.0) & (m90_mev > 0.0)
        x = years[mask]
        y = m90_mev[mask]
        if x.size < 3:
            return None
        lx = np.log(x + 1e-300)
        ly = np.log(y + 1e-300)
        b0, ln_a0 = np.polyfit(lx, ly, 1)
        a0 = float(np.exp(ln_a0))

        def power_law(t, a, b):
            return a * np.power(t, b)

        try:
            (a_fit, b_fit), _ = curve_fit(
                power_law,
                x,
                y,
                p0=(max(a0, 1e-12), b0),
                bounds=([1e-12, -5.0], [1e6, 2.0]),
                maxfev=20000,
            )
            a = float(a_fit)
            b = float(b_fit)
        except Exception:
            a = a0
            b = float(b0)

        out = {"a_mev": a, "b": float(b), "crossing_years": np.nan, "crossing_text": "no crossing"}
        if target_mev <= 0.0 or not np.isfinite(target_mev):
            out["crossing_text"] = "invalid target mass"
            return out
        if b >= 0.0:
            out["crossing_text"] = "no crossing (slope >= 0)"
            return out
        # Compute crossing in log-space to avoid overflow when |1/b| is large.
        if abs(b) < 1e-12:
            out["crossing_text"] = "no crossing (|slope| too small)"
            return out
        log_ratio = np.log(target_mev) - np.log(max(a, 1e-300))
        log_t_cross = log_ratio / b
        max_log = np.log(np.finfo(float).max)
        if not np.isfinite(log_t_cross) or log_t_cross > max_log:
            out["crossing_text"] = "no physical crossing"
            return out
        if log_t_cross < -max_log:
            out["crossing_text"] = "no physical crossing"
            return out
        t_cross = float(np.exp(log_t_cross))
        if not np.isfinite(t_cross) or t_cross <= 0.0:
            out["crossing_text"] = "no physical crossing"
            return out
        out["crossing_years"] = float(t_cross)
        if t_cross <= np.max(x):
            out["crossing_text"] = f"{t_cross:.3g} yr (already crossed)"
        else:
            out["crossing_text"] = f"{t_cross:.3g} yr"
        return out

    def _update_gpu_indicator(self, cfg: SimulationConfig) -> None:
        if not cfg.use_gpu:
            self.gpu_light.itemconfig(self.gpu_light_id, fill="#b9b9b9", outline="#666666")
            self.gpu_state_label.set("CPU mode")
            return
        status = gpu_status()
        if status["available"]:
            self.gpu_light.itemconfig(self.gpu_light_id, fill="#21c45a", outline="#0e7a34")
            self.gpu_state_label.set("GPU active")
        else:
            self.gpu_light.itemconfig(self.gpu_light_id, fill="#f0b429", outline="#946200")
            self.gpu_state_label.set("GPU requested, CPU fallback")

    def _expected_histogram_components(
        self, config: SimulationConfig, bin_edges_ev: np.ndarray, live_time_years: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        total, model = expected_counts(config, live_time_years)
        energy = model["energy_ev"]
        f_pp = model["pileup_fraction"]
        n_events = config.total_rate_hz * live_time_years * SECONDS_PER_YEAR
        single_density = gaussian_convolve_density(
            energy, model["single_density"], config.energy_fwhm_ev
        )
        pileup_density = gaussian_convolve_density(
            energy, model["pileup_density"], config.energy_fwhm_ev
        )
        single = (1.0 - f_pp) * n_events * bin_density(
            energy, single_density, bin_edges_ev
        )
        passing_pileup = f_pp * n_events * bin_density(
            energy, pileup_density, bin_edges_ev
        )
        return total, single, passing_pileup

    def _redraw(self, refresh_spectrum: bool = False) -> None:
        if self.state is None:
            return
        cfg, live_time_years, counts, bin_edges_ev, sensitivity_history = self._state_snapshot()
        self.ax_hist.clear()
        self.ax_fit.clear()
        self.ax_sens.clear()
        if refresh_spectrum or not self.ax_full.lines:
            self.ax_full.clear()
            spectrum_config = self.spectrum_preview_config or cfg
            _, full_model = expected_counts(
                spectrum_config,
                max(live_time_years, spectrum_config.chunk_days / 365.25, 1e-6),
            )
            energy = full_model["energy_ev"]
            self.ax_full.plot(
                energy,
                full_model["single_density"],
                color="#244f7a",
                lw=1.0,
                label="Single Ho",
            )
            self.ax_full.plot(
                energy,
                full_model["measured_density"],
                color="#2f6f73",
                lw=1.1,
                label="Measured mix",
            )
            self.ax_full.plot(
                energy,
                full_model["pileup_density"],
                color="#db8b2b",
                lw=1.1,
                label="Normalized pileup curve",
            )
            self.ax_full.axvspan(
                full_model["bin_edges_ev"][0],
                full_model["bin_edges_ev"][-1],
                color="#5a4a8f",
                alpha=0.12,
                label="Fit/display window",
            )
            self.ax_full.plot(
                energy,
                full_model["passing_pileup_density"],
                color="#cf3320",
                lw=1.6,
                ls="--",
                zorder=10,
                label="Pileup passing filter",
            )
            self.ax_full.set_yscale("log")
            self.ax_full.set_xlim(0.0, energy[-1])
            plotted_densities = np.concatenate(
                [
                    full_model["single_density"],
                    full_model["measured_density"],
                    full_model["pileup_density"],
                    full_model["passing_pileup_density"],
                ]
            )
            positive_densities = plotted_densities[
                np.isfinite(plotted_densities) & (plotted_densities > 0.0)
            ]
            if positive_densities.size:
                passing_pileup = full_model["passing_pileup_density"]
                passing_positive = passing_pileup[
                    np.isfinite(passing_pileup) & (passing_pileup > 0.0)
                ]
                lower_limit = 1e-12
                if passing_positive.size:
                    lower_limit = min(lower_limit, 0.1 * passing_positive.max())
                self.ax_full.set_ylim(
                    bottom=max(lower_limit, np.finfo(float).tiny),
                    top=2.0 * positive_densities.max(),
                )
            self.ax_full.set_ylabel("Density / eV")
            self.ax_full.set_xlabel("Measured energy (eV)")
            if self.spectrum_preview_config is not None:
                self.ax_full.set_title("Spectrum preview from edited inputs", fontsize=10)
            self.ax_full.legend(loc="upper right", ncols=2, fontsize=8)
            self.ax_full.grid(True, alpha=0.2)

        edges = bin_edges_ev
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
        self.ax_hist.bar(centers, counts, width=widths, color="#2f6f73", alpha=0.75, label="Accumulated")
        display_years = max(
            live_time_years,
            cfg.chunk_days / 365.25,
            1e-6,
        )
        expected_total, expected_single, expected_pileup = self._expected_histogram_components(
            cfg, edges, display_years
        )
        if live_time_years > 0.0:
            expected_label = "Expected cumulative total"
            single_label = "Expected cumulative Single Ho"
            pileup_label = "Expected cumulative pileup passing filter"
        else:
            expected_label = "Expected total per first chunk"
            single_label = "Expected Single Ho per first chunk"
            pileup_label = "Expected pileup passing filter per first chunk"
        self.ax_hist.plot(centers, expected_total, color="#a33f2a", lw=1.4, label=expected_label)
        self.ax_hist.plot(
            centers,
            expected_single,
            color="#244f7a",
            lw=1.1,
            label=single_label,
        )
        self.ax_hist.plot(
            centers,
            expected_pileup,
            color="#cf3320",
            lw=1.4,
            ls="--",
            zorder=10,
            label=pileup_label,
        )
        self.ax_hist.set_yscale("log")
        plotted_counts = np.concatenate(
            [counts, expected_total, expected_single, expected_pileup]
        )
        finite_counts = plotted_counts[np.isfinite(plotted_counts)]
        upper_count = max(0.5, float(finite_counts.max())) if finite_counts.size else 0.5
        self.ax_hist.set_ylim(0.5, max(1.0, 1.05 * upper_count))
        self.ax_hist.set_ylabel("Counts/bin")
        self.ax_hist.set_xlabel("Measured energy (eV)")
        self.ax_hist.legend(loc="upper right")
        self.ax_hist.grid(True, alpha=0.2)

        fit_centers = centers
        fit_widths = widths
        self.ax_fit.bar(
            fit_centers,
            counts,
            width=fit_widths,
            color="#7393b3",
            alpha=0.65,
            label="Fit bins (data)",
        )
        fit_estimate = estimate_mnu_from_counts(
            counts,
            cfg,
            live_time_years,
            fit_method=cfg.fit_method,
            endpoint_weight=cfg.endpoint_weight,
        )
        fit_mnu2 = fit_estimate.get("mnu2_hat_ev2", np.nan)
        fit_curve = np.asarray(fit_estimate.get("mu_fit", np.array([])), dtype=float)
        if fit_curve.size == counts.size and np.any(np.isfinite(fit_curve)):
            self.ax_fit.plot(
                fit_centers,
                fit_curve,
                color="#1f5a24",
                lw=1.0,
                marker="o",
                ms=2.5,
                label=f"Best fit bins (full model, m_nu^2={fit_mnu2:+.3g} eV^2)",
            )
            fit_params = fit_estimate.get("fit_params", {})
            if (
                fit_params.get("method") in {"robust_mle", "mle", "robust", "poisson_full"}
                and np.isfinite(fit_params.get("mnu2_ev2", np.nan))
                and np.isfinite(fit_params.get("norm", np.nan))
                and np.isfinite(fit_params.get("flat_bin", np.nan))
            ):
                _, smooth_model = expected_counts(
                    cfg,
                    display_years,
                    mnu2_ev2=float(fit_params["mnu2_ev2"]),
                )
                n_events = cfg.total_rate_hz * display_years * SECONDS_PER_YEAR
                bin_w = float(np.mean(fit_widths)) if fit_widths.size else 1.0
                smooth_counts = (
                    float(fit_params["norm"]) * smooth_model["measured_density"] * n_events * bin_w
                    + float(fit_params["flat_bin"])
                )
                x_smooth = smooth_model["energy_ev"]
                x_mask = (x_smooth >= edges[0]) & (x_smooth <= edges[-1])
                self.ax_fit.plot(
                    x_smooth[x_mask],
                    smooth_counts[x_mask],
                    color="#7a9626",
                    lw=1.25,
                    alpha=0.95,
                    label="Smooth full-model fit",
                )
        else:
            fit_curve, _ = expected_counts(
                cfg,
                max(cfg.chunk_days / 365.25, 1e-6),
                mnu2_ev2=cfg.mnu2_ev2,
            )
            self.ax_fit.plot(
                fit_centers,
                fit_curve,
                color="#1f5a24",
                lw=1.2,
                ls=":",
                label="Reference curve (no fit yet)",
            )
        self.ax_fit.set_yscale("log")
        fit_positive = np.concatenate([counts, fit_curve])
        fit_positive = fit_positive[np.isfinite(fit_positive) & (fit_positive > 0.0)]
        if fit_positive.size:
            self.ax_fit.set_ylim(0.5, max(1.0, 1.05 * float(fit_positive.max())))
        self.ax_fit.set_ylabel("Counts/bin")
        self.ax_fit.set_xlabel("Measured energy (eV)")
        self.ax_fit.legend(loc="upper right")
        self.ax_fit.grid(True, alpha=0.2)

        if sensitivity_history:
            years = np.array([row["live_time_years"] for row in sensitivity_history], dtype=float)
            meV = np.array([1000.0 * row["mnu90_ev"] for row in sensitivity_history], dtype=float)
            self.ax_sens.plot(years, meV, marker="o", ms=3, color="#5a4a8f", label="90% sensitivity")
            if years.size >= 3:
                fit = self._fit_power_law_crossing(years, meV, cfg.true_mnu_mev)
                if fit is not None:
                    t_fit = np.geomspace(max(np.min(years[years > 0.0]), 1e-8), np.max(years), 200)
                    y_fit = fit["a_mev"] * np.power(t_fit, fit["b"])
                    self.ax_sens.plot(t_fit, y_fit, color="#1f9f6e", lw=1.4, label="Power-law fit")
            fit_years = [row["live_time_years"] for row in sensitivity_history if np.isfinite(row.get("mnu2_hat_ev2", np.nan))]
            fit_mev = [
                np.sign(row["mnu2_hat_ev2"]) * np.sqrt(abs(row["mnu2_hat_ev2"])) * 1000.0
                for row in sensitivity_history
                if np.isfinite(row.get("mnu2_hat_ev2", np.nan))
            ]
            if fit_years:
                self.ax_sens.plot(fit_years, fit_mev, marker="s", ms=3, color="#a33f2a", label="Signed fit sqrt(|m_nu^2|)")
                self.ax_sens.axhline(cfg.true_mnu_mev, color="#555555", lw=1.0, ls=":", label="True m_nu")
                self.ax_sens.axhline(0.0, color="#888888", lw=0.8)
            self.ax_sens.legend(loc="best")
        self.ax_sens.set_ylabel("m_nu scale (meV)")
        self.ax_sens.set_xlabel("Live time (yr)")
        self.ax_sens.grid(True, alpha=0.2)
        self.fig.tight_layout()
        self.canvas.draw_idle()


def main() -> None:
    app = SimulatorApp()
    app.mainloop()
