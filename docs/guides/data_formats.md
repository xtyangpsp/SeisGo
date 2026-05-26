# Data Formats

SeisGo uses a small set of structured data objects and file formats throughout its pipeline.

## File formats

### ASDF (`.h5`)

The primary on-disk format is **ASDF** (Adaptable Seismic Data Format), a self-describing HDF5
container that stores waveforms alongside station metadata (StationXML). SeisGo reads and writes
ASDF files via [pyasdf](https://seismicdata.github.io/pyasdf/).

ASDF files are used for:

- Raw waveform archives
- FFT intermediate products
- Cross-correlation output
- dv/v monitoring output

### Pickle (`.pk`)

Lightweight serialization for intermediate results and clustering outputs. Useful for quick
experimentation but not recommended for long-term archival.

### CSV (`.csv`)

Plain text output for anisotropy measurements (`beamforming results`, `anisotropy parameters`)
and cluster map DataFrames.

---

## In-memory data types (`seisgo.types`)

### `FFTData`

Holds the frequency-domain representation of one station–component time series, ready for
cross-correlation.

Key attributes:

| Attribute | Description |
|-----------|-------------|
| `data` | 2-D complex array `(n_windows, n_freqs)` |
| `dt` | Sampling interval (s) |
| `std` | Per-window standard deviation |
| `time` | Window start times (UTCDateTime) |
| `freq` | Frequency vector (Hz) |
| `win_len` | Analysis window length (s) |
| `step` | Window step / overlap (s) |
| `net`, `sta`, `loc`, `chan` | SEED channel identifiers |

### `CorrData`

Stores a time-lag cross-correlation function (or autocorrelation) between a station pair over
multiple time windows.

Key attributes:

| Attribute | Description |
|-----------|-------------|
| `data` | 2-D array `(n_windows, n_lags)` |
| `dt` | Sampling interval (s) |
| `lag` | Maximum lag (s) |
| `dist` | Inter-station distance (km) |
| `cc_comp` | Component pair, e.g. `"ZZ"`, `"RR"` |
| `side` | `"a"` (both), `"n"` (negative), `"p"` (positive) |
| `sta` | `[source_name, receiver_name]` |
| `time` | Window timestamps |
| `misc` | Dictionary of auxiliary metadata |

Useful methods:

```python
corrdata.stack(method="linear")      # collapse to a single stack
corrdata.filter(fmin, fmax)          # bandpass filter in-place
corrdata.plot()                      # quick visualization
corrdata.to_asdf(outfile)            # save to ASDF
```

### `DvvData`

Stores dv/v measurements derived from a `CorrData` object.

Key attributes:

| Attribute | Description |
|-----------|-------------|
| `dvv` | dv/v values array |
| `err` | Measurement error / uncertainty |
| `cc` | Cross-correlation coefficient of each measurement |
| `time` | Time vector of measurements |
| `freq` | Frequency bands |
| `method` | Method used (`"wts"` or `"ts"`) |
| `cc_comp` | Component pair |

Useful methods:

```python
dvvdata.plot()
dvvdata.to_asdf(outfile)
```

---

## Xcorr output structure options

When saving cross-correlations with `noise.do_xcorr`, the subdirectory layout is controlled by
the `output_structure` parameter. Available options (from `helpers.xcorr_output_structure()`):

| Option | Short | Layout |
|--------|-------|--------|
| `raw` | `r` | By time chunk, all pairs together |
| `source` | `s` | Subfolder per virtual source |
| `station-pair` | `sp` | Subfolder per station pair |
| `station-component-pair` | `scp` | Nested source / component folders |

---

## Stacking methods

Available methods (from `helpers.stack_methods()`):

`linear`, `pws`, `tf-pws`, `robust`, `acf`, `nroot`, `selective`, `cluster`

See [Stacking API](../api/stacking.rst) for parameter details.

## Cross-correlation methods

Available methods (from `helpers.xcorr_methods()`):

`xcorr`, `deconv`, `coherency`
