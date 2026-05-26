# Azimuthal Anisotropy Workflow (BANX)

SeisGo implements the **BANX** method (Beamforming Azimuthal ANisotropy using noise
Cross-correlations) for measuring crustal azimuthal anisotropy from ambient noise data.

> **Reference:** Castellanos et al. (2020), *Science Advances*, 6(28).
> https://doi.org/10.1126/sciadv.abb0476

---

## Conceptual overview

```
Station array (receiver cluster around reference site)
          │
          ▼
  Load ZZ xcorr data from each source → receiver-cluster pair
          │
          ▼
  SNR quality control + signal windowing
          │
          ▼
  Slowness beamforming  (grid search in Ux, Uy space)
          │
          ▼
  Extract best phase velocity + back-azimuth per source
          │
          ▼
  Fit cos(2θ) anisotropy model → strength (RHO) + fast direction (THETA)
```

The method treats the dense receiver cluster as a mini-array, steers incoming plane waves
across all back-azimuths, and recovers the azimuthal dependence of phase velocity.

---

## Preparing input data

### Station dictionary

```python
# Keys are station names; values are [lat, lon]
stationdict = {
    "TA.A25A": [45.12, -120.34],
    "TA.A26A": [45.20, -120.15],
    "TA.B25A": [45.55, -120.40],
    # ... all network stations
}
```

### Cross-correlation data layout

`do_BANX` expects ASDF files (`.h5`) named `source_receiver*.h5` inside `datadir`.
Each file must contain stacked or raw cross-correlations accessible via
`noise.extract_corrdata()`.

---

## Running BANX

```python
from seisgo import anisotropy

beam_results, ani_params = anisotropy.do_BANX(
    stationdict_all=stationdict,
    reference_site="TA.A25A",      # the receiver cluster center
    period_band=[10, 20],          # analysis period band (s)
    reference_velocity=3.2,        # reference phase velocity (km/s)
    datadir="./xcorr_output",
    outdir_root="./banx_results",
    sampling_rate=10,              # target sampling rate (sps)
    min_stations=10,
    min_snr=5,
    cc_comp="ZZ",
    doublesided=True,
    show_fig=False,
    plot_moveout=True,
    plot_beampower=True,
    plot_station_result=True,
    map_engine="cartopy",          # or "gmt"
    verbose=False,
)
```

### Return values

`beam_results` — `pandas.DataFrame` with one row per source, columns:

| Column | Description |
|--------|-------------|
| `lat`, `lon` | Cluster centre coordinates |
| `velocity` | Best-fit phase velocity (km/s) |
| `baz` | Back-azimuth from cluster to source (°) |
| `sharpness` | Beam sharpness (max/median power ratio) |
| `power` | Peak beam power |
| `radius` | Receiver cluster radius (km) |
| `num` | Number of traces used |

`ani_params` — `pandas.DataFrame` (one row) with anisotropy parameters:

| Column | Description |
|--------|-------------|
| `A0` | Isotropic velocity (km/s) |
| `A1`, `A2` | Anisotropy coefficients |
| `RHO` | Anisotropy magnitude (%) |
| `THETA` | Fast-axis direction (° from N) |
| `lat`, `lon` | Effective measurement location |

---

## Geometric helper functions

```python
from seisgo import anisotropy

# Array aperture and centre
radius, center = anisotropy.get_ArrayAttributes(lat_array, lon_array)

# Include UTM coordinates
radius, center, utm_info = anisotropy.get_ArrayAttributes(
    lat_array, lon_array, get_utm=True
)
```

---

## Anisotropy model

The azimuthal velocity variation is modelled as:

$$v(\theta) = A_0 + A_1 \cos(2\theta) + A_2 \sin(2\theta)$$

```python
from seisgo import anisotropy
import numpy as np

azimuths = np.linspace(0, 360, 180)
v_model  = anisotropy.compute_anisotropy(azimuths, a=3.2, b=0.05, c=0.03)
```

---

## Key parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `min_stations` | 10 | Minimum receivers in cluster |
| `min_snr` | 5 | SNR threshold for trace inclusion |
| `velocity_perturbation` | 0.4 | ±40% search around reference velocity |
| `max_slowness` | 0.5 s/km | Beamforming grid half-width |
| `slowness_step` | 0.005 s/km | Beamforming grid resolution |
| `azimuth_step` | 6° | Back-azimuth bin width for QC |
| `min_good_bazbin` | 5 | Min. bins with ≥`min_baz_measurements` sources |
| `min_beam_sharpness` | 0 | Sharpness QC threshold (0 = disabled) |

---

## Output files

All output is written under `outdir_root/<period_min>_<period_max>/`:

- `<reference_site>_beam.csv` — filtered beam results for the reference site
- `<reference_site>_anisotropy.csv` — fitted anisotropy parameters
- `figures/<reference_site>/` — moveout, beam power, station maps, result plots (PDF)

---

## See also

- [Anisotropy API](../api/anisotropy.rst)
- [Noise API](../api/noise.rst)
