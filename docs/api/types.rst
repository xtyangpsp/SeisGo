.. _api-types:

seisgo.types
============

The ``types`` module defines the core data container classes used throughout SeisGo:
:class:`FFTData`, :class:`CorrData`, and :class:`DvvData`.

.. automodule:: seisgo.types
   :members:
   :undoc-members:
   :show-inheritance:

----

Class Summary
-------------

.. autosummary::
   :nosignatures:

   seisgo.types.FFTData
   seisgo.types.CorrData
   seisgo.types.DvvData

----

``FFTData``
-----------

Frequency-domain representation of one station–component time series, with all
processing (windowing, normalization, FFT) already applied.

**Constructor**

.. code-block:: python

   from seisgo.types import FFTData

   fftdata = FFTData(
       trace,           # obspy.Trace
       win_len=3600,
       step=1800,
       stainv=None,     # obspy.Inventory or None
       freqmin=0.05,
       freqmax=2.0,
       time_norm="one_bit",
       freq_norm="rma",
       smooth=20,
       smooth_spec=None,
       taper_frac=0.05,
       df=None,
   )

**Key attributes**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Description
   * - ``data``
     - Complex 2-D array, shape ``(n_windows, n_freqs)``
   * - ``std``
     - Per-window standard deviation, shape ``(n_windows,)``
   * - ``time``
     - Window start times (UTC)
   * - ``freq``
     - Frequency vector (Hz)
   * - ``dt``
     - Sampling interval (s)
   * - ``win_len``
     - Analysis window length (s)
   * - ``step``
     - Window step / overlap (s)
   * - ``net``, ``sta``, ``loc``, ``chan``
     - SEED channel identifiers
   * - ``lat``, ``lon``, ``ele``
     - Station coordinates

----

``CorrData``
------------

Time-lag cross-correlation (or auto-correlation) between a station pair over multiple
time windows.

**Constructor**

:class:`CorrData` is normally created by :func:`seisgo.noise.correlate` or loaded with
:func:`seisgo.noise.extract_corrdata`. Direct construction is also supported.

**Key attributes**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Description
   * - ``data``
     - 2-D float array, shape ``(n_windows, n_lags)``
   * - ``dt``
     - Sampling interval (s)
   * - ``lag``
     - Maximum lag (s)
   * - ``dist``
     - Inter-station distance (km)
   * - ``az``, ``baz``
     - Forward and back azimuths (°)
   * - ``cc_comp``
     - Component pair string, e.g. ``"ZZ"``, ``"RR"``, ``"TT"``
   * - ``side``
     - ``"a"`` both, ``"n"`` negative, ``"p"`` positive, ``"o"`` one-sided
   * - ``sta``
     - ``[source_name, receiver_name]``
   * - ``net``
     - ``[source_net, receiver_net]``
   * - ``time``
     - Window timestamps
   * - ``misc``
     - Auxiliary metadata dictionary

**Key methods**

.. code-block:: python

   # Bandpass filter in-place
   corrdata.filter(fmin=0.05, fmax=2.0, corners=4, zerophase=True)

   # Collapse to a single stacked trace (returns new CorrData)
   stacked = corrdata.stack(method="pws")

   # Stack in-place (overwrites data)
   corrdata.stack(method="linear", overwrite=True)

   # Save to ASDF
   corrdata.to_asdf("output.h5")

   # Quick plot
   corrdata.plot()

----

``DvvData``
-----------

Stores dv/v measurements produced by :func:`seisgo.monitoring.get_dvv`.

**Key attributes**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Description
   * - ``dvv``
     - dv/v values, shape ``(n_windows,)`` or ``(n_windows, n_freqs)``
   * - ``err``
     - Measurement uncertainty (same shape as ``dvv``)
   * - ``cc``
     - Waveform cross-correlation coefficient per measurement
   * - ``time``
     - Timestamp of each measurement window
   * - ``freq``
     - Frequency vector (``wts``) or scalar band (``ts``)
   * - ``method``
     - ``"wts"`` or ``"ts"``
   * - ``cc_comp``
     - Component pair
   * - ``sta``
     - Station pair names

**Key methods**

.. code-block:: python

   dvvdata.plot()
   dvvdata.to_asdf("dvv_output.h5")
