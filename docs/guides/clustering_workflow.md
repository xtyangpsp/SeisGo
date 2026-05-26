# Velocity Model Clustering Workflow

SeisGo's `clustering` module groups depth-velocity profiles from 3-D seismic velocity
models into coherent spatial clusters, revealing structural provinces or tectonic domains.
Two algorithms are supported: **k-means** (via `tslearn`) and
**self-organizing maps / SOM** (via `minisom`).

---

## Overview

```
3-D velocity model  (lat × lon × depth)
          │
          ▼
  Extract 1-D depth profiles at each grid node
          │
          ▼
  Optional: interpolate to uniform depth grid (dz)
          │
          ▼
  K-means  ──or──  SOM
          │
          ▼
  Cluster labels  +  Cluster-mean profiles
          │
          ▼
  Map view (plotly)  +  Profile plot (matplotlib)
```

---

## K-means clustering

### Automatic cluster-count selection

If you do not know the optimal number of clusters, SeisGo uses the **knee/elbow method**
on the within-cluster sum of distances:

```python
from seisgo import clustering
import numpy as np

# ts: tslearn time-series dataset (from to_time_series_dataset)
nbest, distortions = clustering.vpcluster_evaluate_kmean(
    ts,
    nrange=np.arange(2, 15),
    smooth=True,
    smooth_n=3,
    plot=True,
)
print("Best cluster count:", nbest)
```

### Running k-means

```python
from seisgo import clustering

outdict = clustering.vpcluster_kmean(
    lat,          # 1-D latitude array
    lon,          # 1-D longitude array
    dep,          # 1-D depth array (km)
    vmodel,       # 3-D velocity array (n_depth × n_lat × n_lon)
    ncluster=5,   # set None to auto-detect via elbow method
    spacing=1,    # spatial sub-sampling stride
    zrange=[0, 60],   # depth range to consider (km)
    dz=2.0,           # depth interpolation interval (km); None = use model grid
    metric="euclidean",
    max_iter_barycenter=100,
    random_state=0,
    plot=True,
    savefig=True,
    figbase="my_model",
    save=True,
    source="tomography",
    tag="Vs",
)
```

When `save=False`, the function returns `outdict` directly instead of writing to disk.

### Output dictionary keys

| Key | Contents |
|-----|----------|
| `method` | `"k-means"` |
| `source` | User-supplied source label |
| `tag` | User-supplied variable label |
| `depth` | Depth vector used for clustering |
| `model` | Fitted `TimeSeriesKMeans` object |
| `pred` | List of arrays — profiles assigned to each cluster |
| `para` | Dict of all algorithm parameters |
| `cluster_map` | `pandas.DataFrame` with columns `lat`, `lon`, `cluster` |

---

## SOM clustering

Self-organizing maps do not require specifying the number of clusters in advance; instead
you specify a 2-D grid size `(som_x, som_y)` and the number of clusters is `som_x × som_y`
(reduced to only populated nodes).

```python
from seisgo import clustering

outdict = clustering.vpcluster_som(
    lat,
    lon,
    depth,        # 1-D depth vector
    v,            # 3-D velocity array (n_depth × n_lat × n_lon)
    grid_size=None,     # None = auto-size as ceil(√√N)²
    spacing=1,
    niteration=50000,
    sigma=0.3,          # neighbourhood radius
    rate=0.1,           # learning rate
    plot=True,
    savefig=True,
    figbase="som_model",
    save=True,
)
```

When `grid_size` is `None`, SeisGo estimates it as
`ceil(√(√N))` in each dimension, where *N* is the number of profile samples.

---

## Choosing between k-means and SOM

| | K-means | SOM |
|---|---------|-----|
| Cluster count | Explicit (or auto via elbow) | Implicit (grid size) |
| Cluster shape | Spherical | Topologically ordered |
| Speed | Fast | Slower (iterative) |
| Interpretability | Direct centroid profiles | Continuous map of profiles |
| Best for | Well-separated structural domains | Smooth gradients and transitions |

---

## Visualisations produced

Both functions generate two figures when `plot=True`:

1. **Profile plot** — all depth-velocity profiles coloured by cluster, with the
   cluster centroid (k-means) or cluster mean (SOM) overlaid in red.
2. **Map view** — interactive `plotly` scatter-map of cluster labels at each grid node,
   rendered on a USGS imagery basemap.

Figures are saved as PNG when `savefig=True`:

- `<figbase>_clusters_k<N>.png`
- `<figbase>_clustermap_k<N>.png`

---

## Loading saved results

```python
import pickle

with open("my_model_clusters_k5_results.pk", "rb") as f:
    outdict = pickle.load(f)

cluster_map = outdict["cluster_map"]   # DataFrame: lat, lon, cluster
depth       = outdict["depth"]
profiles    = outdict["pred"]          # list of arrays, one per cluster
```

---

## See also

- [Clustering API](../api/clustering.rst)
- [Utils API](../api/utils.rst) — `box_smooth` and other helpers used internally
