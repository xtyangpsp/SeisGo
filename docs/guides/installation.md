# Installation

## Requirements

SeisGo requires Python 3.8 or later. The following packages are core dependencies:

- `numpy`, `scipy`, `matplotlib`
- `obspy` — seismic data I/O and processing
- `pyasdf` — ASDF file format support
- `pandas` — tabular data handling
- `tslearn` — time-series machine learning (clustering)
- `minisom` — self-organizing maps
- `kneed` — knee/elbow detection
- `pycwt` — continuous wavelet transform
- `pysurf96` — surface-wave forward modeling
- `utm` — UTM coordinate conversion
- `cartopy` or `pygmt` — map plotting (anisotropy module)
- `plotly` — interactive maps (clustering module)
- `numba` — JIT acceleration (utils module)
- `netCDF4` — NetCDF data I/O
- `shapely` — geometric operations
- `stockwell` — Stockwell transform (stacking)

## Installing with pip

```bash
pip install seisgo
```

## Installing from source

```bash
git clone https://github.com/xtyangpsp/SeisGo.git
cd SeisGo
pip install -e .
```

## Conda environment (recommended)

A reproducible environment using conda:

```bash
conda create -n seisgo python=3.10
conda activate seisgo

# Core scientific stack
conda install numpy scipy matplotlib pandas

# ObsPy and seismology tools
conda install -c conda-forge obspy pyasdf

# Map libraries (pick one or both)
conda install -c conda-forge cartopy
conda install -c conda-forge pygmt

# Remaining dependencies
pip install tslearn minisom kneed pycwt pysurf96 utm stockwell
pip install seisgo
```

## Verifying the installation

```python
import seisgo
from seisgo import noise, stacking, monitoring, utils
print("SeisGo installed successfully.")
```
