.. _api-monitoring:

seisgo.monitoring
=================

The ``monitoring`` module measures relative seismic velocity changes (dv/v) from
ambient-noise cross-correlations using time-domain stretching or wavelet-based methods.

.. automodule:: seisgo.monitoring
   :members:
   :undoc-members:
   :show-inheritance:

----

Function Summary
----------------

.. autosummary::
   :nosignatures:

   seisgo.monitoring.get_dvv
   seisgo.monitoring.measure_dvv_wts
   seisgo.monitoring.measure_dvv_ts

----

``get_dvv`` Parameter Reference
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``corrdata``
     - *required*
     - :class:`seisgo.types.CorrData` object containing the cross-correlation time series.
   * - ``freq``
     - *required*
     - ``[fmin, fmax]`` — frequency band for measurement (Hz).
   * - ``win_len``
     - *required*
     - Coda window length in seconds. Accepts a scalar or list for multiple windows.
   * - ``ref``
     - ``None``
     - Reference trace. If ``None``, computed by stacking all windows with ``stack_method``.
   * - ``stack_method``
     - ``"linear"``
     - Stacking method to compute the reference. See :mod:`seisgo.stacking`.
   * - ``offset``
     - ``1.0``
     - Lag offset (s) from the main arrival (xcorr) or from zero (autocorr).
       Accepts a scalar or list aligned with ``win_len``.
   * - ``resolution``
     - ``None``
     - Temporal sub-stack resolution (s). Resamples the data before measuring dv/v.
   * - ``vmin``
     - ``1.0``
     - Minimum group velocity (km/s) used to set the start of the coda window.
   * - ``normalize``
     - ``True``
     - Normalize traces before measuring.
   * - ``whiten``
     - ``"no"``
     - Spectral whitening: ``"no"``, ``"rma"``, or ``"phase_only"``.
   * - ``method``
     - ``"wts"``
     - dv/v method: ``"wts"`` (wavelet stretching) or ``"ts"`` (time-domain stretching).
   * - ``dvmax``
     - ``0.05``
     - Maximum search range for dv/v (5% by default).
   * - ``subfreq``
     - ``True``
     - If ``True`` (``wts`` only), return frequency-dependent dv/v.
   * - ``nproc``
     - ``None``
     - Number of parallel processes. ``None`` = serial.
   * - ``save``
     - ``False``
     - If ``True``, write :class:`seisgo.types.DvvData` to ``outdir``.
   * - ``format``
     - ``None``
     - Output format: ``"asdf"`` or ``"pickle"``. Auto-detected from ``outfile`` extension.

----

dv/v Methods
------------

Wavelet-transform stretching (``"wts"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default method. Decomposes each coda window into the time–frequency domain, applies
a stretching operator at each scale, and finds the stretch factor that maximises
cross-correlation with the reference. Returns a dv/v value per frequency band.

Credit: adapted from Congcong Yuan (Harvard), originally by Chengxin Jiang and Marine Denolle.

Time-domain stretching (``"ts"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Applies a uniform time-axis stretch to the full coda window and maximises the
cross-correlation coefficient against the reference. Returns a single broadband dv/v value.
Set ``subfreq=False`` automatically when ``method="ts"``.

----

Output: ``DvvData``
-------------------

:func:`get_dvv` returns a :class:`seisgo.types.DvvData` object (or saves it to disk).

Key attributes:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Description
   * - ``dvv``
     - dv/v array, shape ``(n_windows,)`` or ``(n_windows, n_freqs)``
   * - ``err``
     - Measurement uncertainty array (same shape as ``dvv``)
   * - ``cc``
     - Cross-correlation coefficient of each measurement
   * - ``time``
     - Timestamps of each measurement window
   * - ``freq``
     - Frequency vector (for ``wts``) or scalar band
   * - ``method``
     - Method used (``"wts"`` or ``"ts"``)
   * - ``cc_comp``
     - Component pair (e.g. ``"ZZ"``)

.. code-block:: python

   dvvdata.plot()
   dvvdata.to_asdf("dvv_output.h5")
