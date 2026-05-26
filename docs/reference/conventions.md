# Conventions and Notation

This page documents the coordinate systems, unit conventions, and naming standards
used consistently across SeisGo.

---

## Units

| Quantity | Unit |
|----------|------|
| Distance | km |
| Depth | km |
| Time / lag | s |
| Frequency | Hz |
| Period | s |
| Velocity | km/s |
| Slowness | s/km |
| Azimuth / bearing | degrees clockwise from North |
| Anisotropy strength (RHO) | % |
| Fast-axis direction (THETA) | degrees from North |
| Density | kg/m³ (simulation module uses consistent units with velocity) |

---

## Coordinate conventions

- **Latitude** is always given before **longitude** in function arguments and data arrays,
  consistent with the `(lat, lon)` convention.
- **UTM** conversion uses the `utm` package; the zone is inferred automatically from
  the coordinate extent.
- **Azimuth** (``az``) is measured clockwise from North at the source toward the receiver.
- **Back-azimuth** (``baz``) is measured clockwise from North at the receiver back toward
  the source.

---

## SEED channel naming

SeisGo follows standard SEED naming:

```
NET.STA.LOC.CHAN   e.g.  TA.A25A..BHZ
```

Component codes follow the SEED convention:

| Code | Description |
|------|-------------|
| `Z` | Vertical |
| `N` | North |
| `E` | East |
| `R` | Radial (rotated) |
| `T` | Transverse (rotated) |
| `1`, `2` | Horizontal (non-standard orientation; auto-rotated to N/E) |

---

## Cross-correlation lag conventions

The lag axis of a `CorrData` object spans ``[-lag, +lag]`` seconds.

- **Negative lag**: energy arrives at the *receiver* before the *source* in the causal direction.
  Conceptually, this represents waves propagating from receiver to source.
- **Positive lag**: the conventional causal direction, source to receiver.

The `side` attribute records which part of the lag axis is stored:

| `side` | Meaning |
|--------|---------|
| `"a"` | Both sides stored; zero lag at `data[:, len//2]` |
| `"n"` | Negative side only |
| `"p"` | Positive side only |
| `"o"` | One-sided (sign unspecified) |
| `"u"` | Not applicable (e.g., autocorrelations stored without sign) |

---

## Time conventions

- All absolute times use **UTC** via `obspy.UTCDateTime`.
- Relative times (e.g., simulation time, lag axis) are in **seconds**.

---

## File naming patterns

### Raw waveform ASDF files

Typically named by network or time chunk, e.g.:

```
TA_2020_001.h5
```

### Cross-correlation ASDF files

When `output_structure = "station-pair"`:

```
xcorr_output/
  TA.A25A_TA.B25A/
    TA.A25A_TA.B25A_2020_001.h5
```

### dv/v output files

```
dvv_TA.A25A_TA.B25A_ZZ.h5
```

### Anisotropy output (CSV)

```
banx_results/10_20/
  TA.A25A_beam.csv
  TA.A25A_anisotropy.csv
```

### Clustering output (pickle)

```
kmean_clusters_k5_results.pk
som_clusters_k3x3_results.pk
```

---

## Velocity model array layout

3-D velocity models follow the axis ordering `(n_depth, n_lat, n_lon)`:

```python
# Access velocity at depth index d, latitude index i, longitude index j:
v_point = vmodel[d, i, j]
```
