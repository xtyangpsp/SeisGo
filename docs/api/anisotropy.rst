.. _api-anisotropy:

seisgo.anisotropy
=================

The ``anisotropy`` module implements the **BANX** method for measuring seismic azimuthal
anisotropy from ambient-noise cross-correlations.

.. note::
   The BANX implementation was adapted from original MATLAB code by Jorge C. Castellanos-Martinez.
   Please cite the following when using this module:

   Castellanos, J. C., Perry-Houts, J., Clayton, R. W., Kim, Y., Stanciu, A. C., Niday, B.,
   & Humphreys, E. (2020). Seismic anisotropy reveals crustal flow driven by mantle vertical
   loading in the Pacific NW. *Science Advances*, 6(28). https://doi.org/10.1126/sciadv.abb0476

.. automodule:: seisgo.anisotropy
   :members:
   :undoc-members:
   :show-inheritance:

----

Function Summary
----------------

.. autosummary::
   :nosignatures:

   seisgo.anisotropy.do_BANX
   seisgo.anisotropy.get_ArrayAttributes
   seisgo.anisotropy.compute_anisotropy

----

``do_BANX`` Parameter Reference
---------------------------------

Geometry and station selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``stationdict_all``
     - *required*
     - ``dict`` mapping station name → ``[lat, lon]`` for all stations.
   * - ``reference_site``
     - *required*
     - Name of the reference station (receiver-cluster centre).
   * - ``period_band``
     - *required*
     - ``[T_min, T_max]`` — analysis period band (s).
   * - ``reference_velocity``
     - *required*
     - Reference phase velocity (km/s) for the period band.
   * - ``datadir``
     - *required*
     - Directory containing ASDF cross-correlation files.
   * - ``min_stations``
     - ``10``
     - Minimum stations in the receiver cluster.
   * - ``min_radius_scaling``
     - ``1``
     - Minimum cluster radius = scaling × (T_min × v_ref).
   * - ``max_radius_scaling``
     - ``1.5``
     - Maximum cluster radius = scaling × (T_max × v_ref).
   * - ``min_distance_scaling``
     - ``2.5``
     - Minimum source distance = scaling × (T_max × v_ref).

Data processing
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``sampling_rate``
     - ``None``
     - Target sampling rate (sps). Must be an integer ratio of the input rate.
   * - ``min_snr``
     - ``5``
     - Minimum SNR for trace inclusion.
   * - ``signal_window_velocity``
     - ``None``
     - Group velocity for signal-window prediction. Default: 80% of ``reference_velocity``.
   * - ``signal_extent_scaling``
     - ``3``
     - Signal window half-width = scaling × T_max.
   * - ``taper_length_scaling``
     - ``5``
     - Taper length = scaling × T_max.
   * - ``doublesided``
     - ``True``
     - Cross-correlation data contains both negative and positive lags.
   * - ``cc_comp``
     - ``"ZZ"``
     - Cross-correlation component pair.

Beamforming
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``max_slowness``
     - ``0.5``
     - Maximum slowness (s/km) in the beamforming search grid.
   * - ``slowness_step``
     - ``0.005``
     - Slowness grid spacing (s/km).
   * - ``velocity_perturbation``
     - ``0.4``
     - Fraction of ``reference_velocity`` defining the allowed velocity range.

Quality control
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``azimuth_step``
     - ``6``
     - Back-azimuth bin width (°) for QC histogram.
   * - ``min_baz_measurements``
     - ``3``
     - Minimum measurements per azimuth bin to count as a "good" bin.
   * - ``min_good_bazbin``
     - ``5``
     - Minimum number of good azimuth bins required for curve fitting.
   * - ``min_beam_sharpness``
     - ``0``
     - Minimum beam-sharpness ratio. ``0`` disables this filter.

Plotting
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``show_fig``
     - ``True``
     - Display figures interactively.
   * - ``plot_moveout``
     - ``True``
     - Plot waveform moveout for each source–cluster pair.
   * - ``plot_clustermap``
     - ``True``
     - Plot station map for each source–cluster pair.
   * - ``map_engine``
     - ``"cartopy"``
     - Map backend: ``"cartopy"`` or ``"gmt"``.
   * - ``map_region``
     - ``None``
     - ``[lon_min, lon_max, lat_min, lat_max]``. Auto-computed if ``None``.
   * - ``receiver_box``
     - ``None``
     - Draw a bounding box: ``[lon_min, lon_max, lat_min, lat_max]``.
   * - ``plot_beampower``
     - ``True``
     - Plot the slowness-space beam power image.
   * - ``plot_station_result``
     - ``True``
     - Plot velocity vs. BAZ scatter and fitted anisotropy curve.

----

Anisotropy model
-----------------

The velocity–azimuth relationship modelled by :func:`compute_anisotropy`:

.. math::

   v(\theta) = A_0 + A_1 \cos(2\theta) + A_2 \sin(2\theta)

- :math:`A_0` — isotropic (background) velocity
- :math:`A_1, A_2` — anisotropy coefficients
- **Strength**: :math:`\rho = \sqrt{A_1^2 + A_2^2} / A_0 \times 100\%`
- **Fast direction**: :math:`\Theta = \frac{1}{2} \arctan(A_2 / A_1)` (°, from N)

Reference: Smith & Dahlen (1973), *JGR*, 78, 3321–3333.
