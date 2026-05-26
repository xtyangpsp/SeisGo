# Ambient Noise Cross-Correlation Workflow

SeisGo provides a complete ambient noise cross-correlation (ANCC) pipeline.  The diagram below
shows the data flow:

```
Raw waveforms (ASDF)
        │
        ▼
  assemble_raw()          ← orientation correction, metadata
        │
        ▼
  compute_fft()           ← windowing, normalization, FFT
        │
        ▼
  do_xcorr() / correlate()  ← pairwise cross-correlation
        │
        ▼
  CorrData (ASDF / pickle)
        │
        ▼
  stack()                 ← linear / pws / robust / cluster …
        │
        ▼
  Stacked CorrData
```

---

## Step 1: Assemble raw data

Read and orient-correct raw waveforms from an ASDF file:

```python
import pyasdf
from seisgo import noise

ds = pyasdf.ASDFDataSet("raw_2020_001.h5", mode="r")

# Returns a list of dicts, each containing 'data' (list of Traces) and 'inv'
raw_all = noise.assemble_raw(ds, sta="TA.A25A", correct_orientation=True)
```

`assemble_raw` automatically rotates channels 1/2 → N/E when orientation information is
available in the StationXML inventory.

---

## Step 2: Compute FFTs

```python
from seisgo import noise

fft_objects = noise.assemble_fft(
    raw_all,
    win_len=3600,
    step=1800,
    freqmin=0.05,
    freqmax=2.0,
    time_norm="one_bit",   # temporal normalization
    freq_norm="rma",        # spectral normalization
    smooth=20,
)
```

**Temporal normalization options** (`helpers.xcorr_norm_methods(mode="t")`):

- `rma` — running mean absolute value
- `one_bit` — sign-only normalization
- `ftn` — frequency-time normalization

**Spectral normalization options** (`helpers.xcorr_norm_methods(mode="f")`):

- `rma` — running mean in frequency domain
- `phase_only` — retain phase, discard amplitude

---

## Step 3: Cross-correlate

```python
from seisgo import noise

# Single pair
corrdata = noise.correlate(fft1, fft2, maxlag=500, method="xcorr")

# Batch cross-correlation across all pairs in a dataset
noise.do_xcorr(
    ds,
    outdir="./xcorr_output",
    maxlag=500,
    method="xcorr",
    output_structure="station-pair",
)
```

**Cross-correlation methods:**

- `xcorr` — standard normalized cross-correlation
- `deconv` — spectral deconvolution
- `coherency` — phase coherency

---

## Step 4: Stack correlations

```python
from seisgo import stacking

# corrdata.data is shape (n_windows, n_samples)
linear = stacking.stack(corrdata.data, method="linear")
pws    = stacking.stack(corrdata.data, method="pws", par={"p": 2})
robust = stacking.stack(corrdata.data, method="robust", par={"maxstep": 10})
```

Or stack directly on a `CorrData` object:

```python
stacked = corrdata.stack(method="pws")  # returns a new CorrData with one row
```

---

## Step 5: Quality control with SNR

```python
from seisgo import utils
import numpy as np

snr = utils.get_snr(
    corrdata.data,
    t=np.linspace(-corrdata.lag, corrdata.lag, corrdata.data.shape[1]),
    dist=corrdata.dist,
    vmin=1.5,
    vmax=4.5,
    side="a",
)
print("SNR (neg, pos):", snr)
```

---

## Memory estimation

Before running large jobs, estimate memory consumption:

```python
from seisgo import noise

mem_gb = noise.cc_memory(
    inc_hours=1,
    sps=20,
    nsta=50,
    ncomp=3,
    cc_len=3600,
    cc_step=1800,
)
print(f"Estimated memory: {mem_gb:.2f} GB")
```

---

## Saving and loading results

```python
# Save
corrdata.to_asdf("xcorr_TA.A25A_TA.B25A.h5")

# Load
from seisgo import noise
loaded = noise.extract_corrdata("xcorr_TA.A25A_TA.B25A.h5")
```

---

## See also

- [Stacking API](../api/stacking.rst)
- [Noise API](../api/noise.rst)
- [Data Formats](data_formats.md)
