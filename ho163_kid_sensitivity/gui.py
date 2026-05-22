from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .backend import backend_label, gpu_status
from .configs import SimulationConfig
from .fisher import estimate_mnu_from_counts, fisher_mnu_sensitivity
from .model import expected_counts
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
        self.state: RunState | None = None
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
            ("mnu_mev", "True m_nu (meV)", tk.DoubleVar(value=75.0)),
            ("energy_fwhm_ev", "FWHM (eV)", tk.DoubleVar(value=0.2)),
            ("n_detectors", "Detectors", tk.IntVar(value=100000)),
            ("activity_bq", "Activity/pixel (Bq)", tk.DoubleVar(value=300.0)),
            ("tau_eff_us", "Tau eff (us)", tk.DoubleVar(value=0.1)),
            ("live_time_years_target", "Target years", tk.DoubleVar(value=1.0)),
            ("chunk_days", "Chunk days", tk.DoubleVar(value=1.0)),
            ("n_grid", "Energy grid", tk.IntVar(value=32768)),
            ("n_bins", "Histogram bins", tk.IntVar(value=500)),
            ("fit_low_offset_ev", "Fit low Q+ (eV)", tk.DoubleVar(value=-300.0)),
            ("fit_high_offset_ev", "Fit high Q+ (eV)", tk.DoubleVar(value=50.0)),
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

        buttons = ttk.Frame(left)
        buttons.grid(row=len(fields) + 1, column=0, columnspan=2, sticky="ew", pady=10)
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

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(left, textvariable=self.status, wraplength=280).grid(
            row=len(fields) + 2, column=0, columnspan=2, sticky="ew", pady=6
        )

        gpu_row = len(fields) + 3
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
            row=len(fields) + 4, column=0, columnspan=2, sticky="ew", pady=6
        )

        right = ttk.Frame(self, padding=(0, 10, 10, 10))
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax_full = self.fig.add_subplot(311)
        self.ax_hist = self.fig.add_subplot(312)
        self.ax_sens = self.fig.add_subplot(313)
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
            result = fisher_mnu_sensitivity(config, max(config.chunk_days / 365.25, 1e-6))
            self.status.set(f"New run ready. Backend: {backend_label(config.use_gpu)}")
            self._update_metrics(result["mnu90_ev"])
            self._redraw()
        except Exception as exc:
            messagebox.showerror("New run failed", str(exc))

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        if self.state is None:
            self.new_run()
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
            self.push_config_to_ui(self.state.config)
            self.status.set(f"Loaded {path}")
            self._update_metrics()
            self._redraw()
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))

    def _run_worker(self) -> None:
        assert self.state is not None
        rng = rng_from_state(self.state)
        try:
            while not self.stop_event.is_set():
                if self.state.live_time_years >= self.state.config.live_time_years_target:
                    break
                entry = simulate_chunk(self.state, rng)
                save_state(self.state)
                self.queue.put(("progress", entry))
            self.queue.put(("stopped", None))
        except Exception as exc:
            self.queue.put(("error", str(exc)))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "progress":
                    self._update_metrics(payload["mnu90_ev"])
                    self._redraw()
                elif kind == "stopped":
                    self.status.set("Stopped. Progress has been saved.")
                elif kind == "error":
                    self.status.set("Error.")
                    messagebox.showerror("Simulation error", payload)
        except queue.Empty:
            pass
        self.after(150, self._poll_queue)

    def _update_metrics(self, mnu90: float | None = None) -> None:
        if self.state is None:
            return
        cfg = self.state.config
        self._update_gpu_indicator(cfg)
        if mnu90 is None and self.state.sensitivity_history:
            mnu90 = self.state.sensitivity_history[-1]["mnu90_ev"]
        mnu = "pending" if mnu90 is None else f"{1000.0 * mnu90:.1f} meV"
        estimate = None
        if self.state.sensitivity_history:
            estimate = self.state.sensitivity_history[-1]
        elif self.state.counts.sum() > 0.0:
            estimate = estimate_mnu_from_counts(self.state.counts, cfg, self.state.live_time_years)
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
        self.metrics.set(
            "\n".join(
                [
                    f"Live time: {self.state.live_time_years:.4g} yr",
                    f"Counts: {self.state.counts.sum():.4g}",
                    f"True m_nu: {cfg.true_mnu_mev:.1f} meV",
                    f"Fit m_nu^2: {best_mnu2}",
                    f"Physical m_nu: {best}",
                    f"f_pp: {cfg.pileup_fraction:.3g}",
                    f"Rate: {cfg.total_rate_hz:.3g} /s",
                    f"90% sensitivity: {mnu}",
                    f"Backend: {backend_label(cfg.use_gpu)}",
                ]
            )
        )

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

    def _redraw(self) -> None:
        if self.state is None:
            return
        self.ax_full.clear()
        self.ax_hist.clear()
        self.ax_sens.clear()
        _, full_model = expected_counts(self.state.config, max(self.state.live_time_years, self.state.config.chunk_days / 365.25, 1e-6))
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
            full_model["pileup_density"] * max(full_model["pileup_fraction"], 1e-30),
            color="#a33f2a",
            lw=0.9,
            alpha=0.85,
            label="Pileup contribution",
        )
        self.ax_full.axvspan(
            self.state.bin_edges_ev[0],
            self.state.bin_edges_ev[-1],
            color="#5a4a8f",
            alpha=0.12,
            label="Fit/display window",
        )
        self.ax_full.set_yscale("log")
        self.ax_full.set_xlim(0.0, energy[-1])
        self.ax_full.set_ylim(bottom=1e-12)
        self.ax_full.set_ylabel("Density / eV")
        self.ax_full.set_xlabel("Measured energy (eV)")
        self.ax_full.legend(loc="upper right", ncols=2, fontsize=8)
        self.ax_full.grid(True, alpha=0.2)

        edges = self.state.bin_edges_ev
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
        counts = self.state.counts
        self.ax_hist.bar(centers, counts, width=widths, color="#2f6f73", alpha=0.75, label="Accumulated")
        if counts.sum() == 0.0:
            mu, _ = expected_counts(self.state.config, max(self.state.config.chunk_days / 365.25, 1e-6))
            self.ax_hist.plot(centers, mu, color="#a33f2a", lw=1.4, label="Expected per first chunk")
        self.ax_hist.set_yscale("log")
        self.ax_hist.set_ylabel("Counts/bin")
        self.ax_hist.set_xlabel("Measured energy (eV)")
        self.ax_hist.legend(loc="upper right")
        self.ax_hist.grid(True, alpha=0.2)

        hist = self.state.sensitivity_history
        if hist:
            years = [row["live_time_years"] for row in hist]
            meV = [1000.0 * row["mnu90_ev"] for row in hist]
            self.ax_sens.plot(years, meV, marker="o", ms=3, color="#5a4a8f", label="90% sensitivity")
            fit_years = [row["live_time_years"] for row in hist if np.isfinite(row.get("mnu2_hat_ev2", np.nan))]
            fit_mev = [
                np.sign(row["mnu2_hat_ev2"]) * np.sqrt(abs(row["mnu2_hat_ev2"])) * 1000.0
                for row in hist
                if np.isfinite(row.get("mnu2_hat_ev2", np.nan))
            ]
            if fit_years:
                self.ax_sens.plot(fit_years, fit_mev, marker="s", ms=3, color="#a33f2a", label="Signed fit sqrt(|m_nu^2|)")
                self.ax_sens.axhline(self.state.config.true_mnu_mev, color="#555555", lw=1.0, ls=":", label="True m_nu")
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
