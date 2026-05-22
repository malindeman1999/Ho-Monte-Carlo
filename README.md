# Ho-163 KID/TKID Sensitivity Simulator

This workspace contains a runnable GUI simulator for a HOLMES-style
calorimetric Ho-163 neutrino-mass measurement with a large KID/TKID detector
array. It implements the corrected Phase-1 plan from the Holmes notes:

- binned Ho-163 electron-capture spectrum;
- Gaussian energy response;
- unresolved pileup from self-convolution;
- Poisson accumulation in resumable chunks;
- live histogram updates;
- Fisher-matrix neutrino-mass sensitivity estimates.

Run the GUI:

```powershell
python run_gui.py
```

Open the local wiki documentation:

```powershell
start docs\index.html
```

Saved runs are written under `runs/` as `.json` metadata plus `.npz` count
arrays. Use **Load Run** in the GUI to continue accumulating statistics.
