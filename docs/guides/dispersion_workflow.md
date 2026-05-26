# Surface-Wave Dispersion Workflow

SeisGo's `dispersion` module extracts group/phase velocity dispersion images from
seismic waveform gathers and computes synthetic dispersion curves via forward modeling.

## Overview

```
Waveform gather (traces × samples)
           │
           ▼
  get_dispersion_image()   ← phase-shift method (Park et al. 1998)
           │
           ▼
  Dispersion image (period × velocity)
           │
           ▼
  Pick dispersion curve manually or automatically
           │
           ▼
  forward_solver()         ← synthetic dispersion from Vs model
```

---

## Extracting a dispersion image

### Phase-shift method

```python
from seisgo import dispersion
import numpy as np

# g: 2-D array, shape (n_traces, n_samples)
# t: time vector (s), may be two-sided (negative and positive lags)
# d: distance vector (km), one entry per trace

dimage, velocities, periods = dispersion.get_dispersion_image(
    g,
    t,
    d,
    pmin=5,    # minimum period (s)
    pmax=30,   # maximum period (s)
    vmin=2.0,  # minimum phase velocity (km/s)
    vmax=5.0,  # maximum phase velocity (km/s)
    dp=1,      # period increment (s)
    dv=0.05,   # velocity increment (km/s)
    plot=True,
)
```

`dimage` shape depends on the time-series polarity:

- **Two-sided** (`t` spans negative to positive): returns array of shape `(2, n_periods, n_velocities)`
  with index 0 = negative lag, index 1 = positive lag.
- **One-sided**: returns `(n_periods, n_velocities)`.

### Picking the best velocity per period

```python
dimage, velocities, periods, best_v = dispersion.get_dispersion_image(
    g, t, d,
    pmin=5, pmax=30, vmin=2.0, vmax=5.0,
    get_best_v=True,
    plot=True,
)
```

---

## Narrowband filtering

Generate a suite of narrowband-filtered waveforms (useful for visual group-velocity analysis):

```python
from seisgo import dispersion

filtered_waveforms, period_vector = dispersion.narrowband_waveforms(
    trace,          # 1-D array
    dt=0.05,        # sampling interval (s)
    pmin=5,
    pmax=30,
    dp=1,
    pscale="ln",    # "ln" (linear) or "nln" (logarithmic)
    extend=10,      # extra periods for filter roll-off
)
```

### CWT-based dispersion waveforms

```python
filtered_cwt, freq_vector = dispersion.get_dispersion_waveforms_cwt(
    trace,
    dt=0.05,
    fmin=0.03,
    fmax=0.2,
    dj=1/12,
    wvn="morlet",
)
```

---

## Forward modeling (synthetic dispersion)

Compute Rayleigh or Love wave dispersion curves for a layered Vs model using `surf96`:

```python
from seisgo import dispersion
import numpy as np

vs         = np.array([1.8, 2.5, 3.2, 3.8])   # Vs per layer (km/s)
thickness  = np.array([5.0, 10.0, 15.0, 0.0])  # layer thickness (km); 0 = half-space
periods    = np.arange(5, 35, 1)                # target periods (s)

group_vel = dispersion.forward_solver(
    vs,
    periods,
    thickness,
    wave_type="rayleigh",
    mode=1,
    velocity_type="group",
)
```

`forward_solver` internally maps Vs → Vp (using a Vp/Vs ratio of √3) and
density (using an empirical Birch's law approximation), then calls `surf96`.

---

## Energy computation options

The `energy_type` parameter in `get_dispersion_image` controls how peak energy is measured
in each velocity–period cell:

| Option | Description |
|--------|-------------|
| `power_sum` | Sum of squared amplitudes in the travel-time window (default) |
| `envelope` | Maximum of the Hilbert envelope |

---

## Plotting tips

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.imshow(
    dimage.T,
    extent=[periods[0], periods[-1], velocities[0], velocities[-1]],
    origin="lower",
    aspect="auto",
    cmap="jet",
)
plt.colorbar(label="Normalized energy")
plt.xlabel("Period (s)")
plt.ylabel("Phase velocity (km/s)")
plt.show()
```

---

## See also

- [Dispersion API](../api/dispersion.rst)
- [Simulation API](../api/simulation.rst) — 1-D finite-difference acoustic modeling
