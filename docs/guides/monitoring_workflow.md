# Seismic Monitoring (dv/v) Workflow

SeisGo's `monitoring` module measures relative seismic velocity changes (dv/v) from
ambient-noise cross-correlation functions using stretching or wavelet-based methods.

## Overview

Velocity changes are detected by comparing individual cross-correlation windows (the
*current* state) against a *reference* stack. A positive dv/v indicates the medium
has sped up (e.g., from increased effective stress); a negative value indicates slowing.

```
CorrData (time series of xcorr windows)
          │
          ▼
     get_dvv()
          │
    ┌─────┴──────┐
    │  reference │  ← linear/pws/robust stack of all windows (or user-supplied)
    └─────┬──────┘
          │  stretch or wavelet transform each window vs. reference
          ▼
     DvvData object
```

---

## Basic usage

```python
from seisgo import monitoring

dvvdata = monitoring.get_dvv(
    corrdata,           # CorrData object
    freq=[0.1, 1.0],    # frequency band for dv/v measurement (Hz)
    win_len=50.0,       # coda window length (s)
    method="wts",       # "wts" (wavelet) or "ts" (stretching)
    dvmax=0.05,         # search range: ±5%
    vmin=1.0,           # minimum group velocity to set start of coda window
    offset=1.0,         # extra offset from main arrival (s)
    normalize=True,
)
```

**Available methods** (`helpers.dvv_methods()`):

| Method | Description |
|--------|-------------|
| `wts` | Wavelet-transform stretching — frequency-dependent dv/v |
| `ts` | Time-domain stretching — single broadband dv/v value |

---

## Reference trace options

By default the reference is computed by linearly stacking all windows in `corrdata`. You
can supply a custom reference:

```python
import numpy as np
my_ref = np.mean(corrdata.data, axis=0)

dvvdata = monitoring.get_dvv(corrdata, freq=[0.1, 1.0], win_len=50,
                              ref=my_ref)
```

---

## Temporal resolution (sub-stacking)

The `resolution` parameter controls temporal resampling before measurement. For example,
to measure dv/v on daily sub-stacks from hourly correlations:

```python
dvvdata = monitoring.get_dvv(
    corrdata,
    freq=[0.1, 1.0],
    win_len=50.0,
    resolution=86400,      # 1 day in seconds
    stack_method="linear",
)
```

---

## Multiple measurement windows

Pass lists to `win_len` and `offset` to measure at several coda windows simultaneously:

```python
dvvdata = monitoring.get_dvv(
    corrdata,
    freq=[0.1, 1.0],
    win_len=[30.0, 60.0],
    offset=[1.0, 1.0],
)
```

---

## Whitening before measurement

Apply spectral whitening to reduce contamination from strong spectral peaks:

```python
dvvdata = monitoring.get_dvv(
    corrdata,
    freq=[0.1, 1.0],
    win_len=50.0,
    whiten="rma",
    whiten_smooth=20,
    whiten_pad=100,
)
```

---

## Plotting and saving

```python
# Plot the dv/v time series
dvvdata.plot()

# Save to ASDF
dvvdata.to_asdf("dvv_TA.A25A_TA.B25A.h5")

# Save to pickle
monitoring.get_dvv(..., save=True, outdir="./dvv_output", format="pickle")
```

---

## Parallel processing

For large datasets with many correlation pairs, enable multi-processing:

```python
dvvdata = monitoring.get_dvv(corrdata, freq=[0.1, 1.0], win_len=50,
                              nproc=8)
```

---

## See also

- [Monitoring API](../api/monitoring.rst)
- [Stacking API](../api/stacking.rst)
- [Data Formats](data_formats.md)
