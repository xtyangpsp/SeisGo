# Quick Start

This page walks through the core SeisGo workflow from raw waveform data to cross-correlation
stacks in a few steps.

## 1. Download seismic data

Use the `downloaders` module (or ObsPy's FDSN client directly) to fetch waveforms and save
them in ASDF format.

```python
from seisgo import downloaders

# Example: download one day of BHZ data from USArray
downloaders.download(
    starttime="2020-01-01",
    endtime="2020-01-02",
    network="TA",
    station="*",
    channel="BHZ",
    outdir="./raw_data",
)
```

## 2. Compute FFTs

Prepare normalized frequency-domain data using `noise.compute_fft`:

```python
import obspy
from seisgo import noise

tr = obspy.read("./raw_data/TA.A25A..BHZ.mseed")[0]

fftdata = noise.compute_fft(
    tr,
    win_len=3600,       # 1-hour windows
    step=1800,          # 50% overlap
    freqmin=0.05,
    freqmax=2.0,
    time_norm="one_bit",
    freq_norm="rma",
    smooth=20,
)
print(fftdata)
```

## 3. Cross-correlate station pairs

```python
from seisgo.types import CorrData
from seisgo import noise

# Assuming fft1 and fft2 are FFTData objects for two stations
corrdata = noise.correlate(fft1, fft2, maxlag=500)
print(corrdata)
```

## 4. Stack correlations

```python
from seisgo import stacking

# Linear stack
linear_stack = stacking.stack(corrdata.data, method="linear")

# Phase-weighted stack
pws_stack = stacking.stack(corrdata.data, method="pws", par={"p": 2})
```

## 5. Measure seismic velocity changes (dv/v)

```python
from seisgo import monitoring

dvvdata = monitoring.get_dvv(
    corrdata,
    freq=[0.1, 1.0],
    win_len=50.0,
    method="wts",
)
dvvdata.plot()
```

## 6. Extract surface-wave dispersion

```python
from seisgo import dispersion

dimage, velocities, periods = dispersion.get_dispersion_image(
    gather,          # 2-D array: traces × samples
    t,               # time vector
    distances,       # offset vector (km)
    pmin=5, pmax=30,
    vmin=2.0, vmax=5.0,
    plot=True,
)
```

## Next steps

- See the [Ambient Noise Workflow](ambient_noise_workflow.md) for a complete end-to-end example.
- See [Data Formats](data_formats.md) for details on ASDF, CorrData, FFTData, and DvvData.
- Explore the [API Reference](../api/noise.rst) for all available functions and parameters.
