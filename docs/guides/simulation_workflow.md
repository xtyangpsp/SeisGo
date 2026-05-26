# Seismic Wave Simulation Workflow

SeisGo's `simulation` module provides a 1-D finite-difference acoustic wave solver and
a layered velocity model builder, useful for synthetic testing and method validation.

---

## Building a layered velocity model

`build_vmodel` creates a fine-grid, linearly increasing velocity model with optional
per-layer perturbations:

```python
from seisgo import simulation
import numpy as np

z, v, rho = simulation.build_vmodel(
    zmax=500,          # maximum depth / model length (m or km)
    dz=1.0,            # grid spacing
    nlayer=5,          # number of velocity layers
    vmin=1500,         # minimum velocity (units consistent with dz)
    vmax=3500,         # maximum velocity
    rhomin=1800,       # minimum density
    rhomax=2500,       # maximum density
    zmin=0,            # starting depth (default 0)
    layer_dv=None,     # optional per-layer fractional perturbation
)
```

### Adding velocity anomalies

`layer_dv` is an array of fractional velocity perturbations applied multiplicatively to
each layer:

```python
# Make layer 3 (0-indexed) 10% slower
layer_dv = np.zeros(5)
layer_dv[2] = -0.10

z, v, rho = simulation.build_vmodel(
    zmax=500, dz=1.0, nlayer=5,
    vmin=1500, vmax=3500,
    rhomin=1800, rhomax=2500,
    layer_dv=layer_dv,
)
```

---

## Running a 1-D finite-difference simulation

`fd1d_dx4dt4` solves the 1-D acoustic wave equation with:

- **4th-order spatial accuracy** O(Δx⁴)
- **4th-order temporal accuracy** O(Δt⁴) via the Adams–Bashforth scheme

```python
from seisgo import simulation
import numpy as np

# Model setup
x = np.arange(0, 500, 1.0)    # spatial grid (m)
vmodel = v                      # from build_vmodel above
rho_model = rho

tout, seis = simulation.fd1d_dx4dt4(
    x=x,
    dt=1e-4,           # time step (s) — must satisfy CFL condition
    tmax=0.3,          # total simulation time (s)
    vmodel=vmodel,
    rho=rho_model,
    xsrc=50,           # source grid index
    xrcv=250,          # receiver grid index
    stf_freq=30,       # source central frequency (Hz) for Ricker wavelet
    stf_shift=None,    # time shift (default: 3/stf_freq)
    stf_type="ricker", # "ricker" or "gaussian"
    t_interval=1,      # output time sub-sampling factor
)
```

### CFL stability condition

The solver automatically checks that `dt` satisfies the Courant–Friedrichs–Lewy condition:

$$dt \leq 0.5 \cdot \frac{\Delta x_{\max}}{v_{\max}}$$

A `ValueError` is raised if the provided `dt` is too large.

### Source time functions

| `stf_type` | `stf_freq` meaning |
|------------|--------------------|
| `"ricker"` | Central frequency (Hz) |
| `"gaussian"` | Width parameter σ (s) |

Both wavelets are available from `seisgo.utils`:

```python
from seisgo import utils

t_ricker, w_ricker   = utils.ricker(dt=1e-4, f=30, t0=0.05)
t_gauss,  w_gauss    = utils.gaussian(dt=1e-4, width=0.02, shift=0.05)
```

---

## Plotting the synthetic seismogram

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 3))
plt.plot(tout, seis, "k", lw=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Pressure amplitude")
plt.title("Synthetic seismogram — 1-D FD simulation")
plt.tight_layout()
plt.show()
```

---

## Full example: synthetic CorrData for method testing

```python
from seisgo import simulation, utils
import numpy as np

# Build model
x = np.arange(0, 1000, 2.0)
z, v, rho = simulation.build_vmodel(1000, 2.0, 6, 1500, 3200, 1800, 2600)

# Simulate with source at one end, receiver at mid-point
tout, seis = simulation.fd1d_dx4dt4(
    x, dt=2e-4, tmax=0.5, vmodel=v, rho=rho,
    xsrc=10, xrcv=250, stf_freq=20, stf_type="ricker",
)

print(f"Simulated {len(tout)} time samples, dt={tout[1]-tout[0]:.2e} s")
```

---

## See also

- [Simulation API](../api/simulation.rst)
- [Dispersion Workflow](dispersion_workflow.md) — use synthetic gathers for dispersion testing
- [Utils API](../api/utils.rst) — `ricker`, `gaussian`, and other signal utilities
