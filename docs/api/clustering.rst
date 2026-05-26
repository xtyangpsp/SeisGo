.. _api-clustering:

seisgo.clustering
=================

The ``clustering`` module groups 1-D depth-velocity profiles from 3-D seismic tomography
models into coherent spatial clusters using k-means or self-organizing maps (SOM).

.. automodule:: seisgo.clustering
   :members:
   :undoc-members:
   :show-inheritance:

----

Function Summary
----------------

.. autosummary::
   :nosignatures:

   seisgo.clustering.vpcluster_kmean
   seisgo.clustering.vpcluster_evaluate_kmean
   seisgo.clustering.vpcluster_som

----

``vpcluster_kmean`` Parameter Reference
-----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``lat``
     - *required*
     - 1-D array of latitudes for the model grid.
   * - ``lon``
     - *required*
     - 1-D array of longitudes for the model grid.
   * - ``dep``
     - *required*
     - 1-D depth array (km).
   * - ``vmodel``
     - *required*
     - 3-D velocity array, shape ``(n_depth, n_lat, n_lon)``.
   * - ``ncluster``
     - ``None``
     - Number of clusters. If ``None``, automatically determined via the elbow method.
   * - ``nrange``
     - ``None``
     - Range of cluster counts to evaluate when ``ncluster=None``. Default: 2–20.
   * - ``spacing``
     - ``1``
     - Spatial sub-sampling stride (every Nth lat/lon point).
   * - ``zrange``
     - ``None``
     - ``[z_min, z_max]`` depth range to use. Default: full model range.
   * - ``dz``
     - ``None``
     - Depth interpolation interval (km). ``None`` = use model depth grid as-is.
   * - ``metric``
     - ``"euclidean"``
     - Distance metric for k-means. Passed to ``TimeSeriesKMeans``.
   * - ``max_iter_barycenter``
     - ``100``
     - Maximum DBA (DTW Barycenter Averaging) iterations.
   * - ``random_state``
     - ``0``
     - Random seed for reproducibility.
   * - ``njob``
     - ``1``
     - Number of parallel jobs.
   * - ``plot``
     - ``True``
     - Plot cluster profiles and map.
   * - ``savefig``
     - ``True``
     - Save figures to PNG.
   * - ``figbase``
     - ``"kmean"``
     - Base name for output figure and pickle files.
   * - ``save``
     - ``True``
     - Save results to a pickle file. If ``False``, returns ``outdict``.
   * - ``evaluate_smooth``
     - ``False``
     - Smooth the distortion curve before knee detection.
   * - ``evaluate_plot``
     - ``True``
     - Plot the elbow curve when auto-detecting cluster count.

----

``vpcluster_som`` Parameter Reference
---------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``lat``, ``lon``, ``depth``, ``v``
     - *required*
     - Same as ``vpcluster_kmean`` (note: ``depth`` not ``dep``; ``v`` not ``vmodel``).
   * - ``grid_size``
     - ``None``
     - ``[som_x, som_y]`` SOM grid dimensions. ``None`` = auto: ``ceil(√(√N))²``.
   * - ``spacing``
     - ``1``
     - Spatial sub-sampling stride.
   * - ``niteration``
     - ``50000``
     - Number of SOM training iterations.
   * - ``sigma``
     - ``0.3``
     - Initial neighbourhood radius.
   * - ``rate``
     - ``0.1``
     - Initial learning rate.
   * - ``plot``, ``savefig``, ``figbase``, ``save``
     - same as k-means
     - Same behaviour as ``vpcluster_kmean``.

----

Output dictionary structure
----------------------------

Both functions return (or save) a dictionary with the following keys:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Description
   * - ``method``
     - ``"k-means"`` or ``"som"``
   * - ``source``
     - User-supplied source label string
   * - ``tag``
     - User-supplied variable label string
   * - ``depth``
     - 1-D depth vector used for clustering
   * - ``model``
     - Fitted ``TimeSeriesKMeans`` or ``MiniSom`` object
   * - ``pred``
     - List of length ``n_clusters``; each element is an array of profiles in that cluster
   * - ``para``
     - Dictionary of algorithm parameters
   * - ``cluster_map``
     - ``pandas.DataFrame`` with columns ``lat``, ``lon``, ``cluster``

----

Elbow / Knee detection
-----------------------

:func:`vpcluster_evaluate_kmean` fits k-means for each value in ``nrange`` and uses the
``kneed`` library to locate the knee of the distortion (within-cluster sum of distances) curve.

.. code-block:: python

   from seisgo import clustering
   import numpy as np
   from tslearn.utils import to_time_series_dataset

   ts = to_time_series_dataset(all_profiles)
   nbest, distortions = clustering.vpcluster_evaluate_kmean(
       ts,
       nrange=np.arange(2, 15),
       smooth=True,
       plot=True,
   )
   print("Recommended cluster count:", nbest)
