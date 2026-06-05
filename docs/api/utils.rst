.. _api-utils:

seisgo.utils
============

The ``utils`` module provides a broad collection of signal processing, coordinate
transformation, SNR estimation, and I/O utility functions used throughout SeisGo.

.. automodule:: seisgo.utils
   :members:
   :undoc-members:
   :show-inheritance:

----

Function Summary
----------------

Signal processing
~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   seisgo.utils.rms
   seisgo.utils.get_snr
   seisgo.utils.demean
   seisgo.utils.detrend
   seisgo.utils.box_smooth
   seisgo.utils.gpr_smooth
   seisgo.utils.ricker
   seisgo.utils.gaussian
   seisgo.utils.taper

Coordinate utilities
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   seisgo.utils.cart2compass
   seisgo.utils.get_filelist

----

Key Functions
-------------

``rms``
~~~~~~~

.. code-block:: python

   from seisgo import utils
   noise_rms = utils.rms(noise_trace)   # √(mean(d²))

``get_snr``
~~~~~~~~~~~

Compute the signal-to-noise ratio using physics-based signal and noise windows:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``d``
     - *required*
     - 1-D or 2-D data array.
   * - ``t``
     - *required*
     - Time vector (s).
   * - ``dist``
     - *required*
     - Source–receiver distance (km).
   * - ``vmin``, ``vmax``
     - *required*
     - Velocity range defining the signal window.
   * - ``extend``
     - ``0``
     - Extend the signal window by this many seconds.
   * - ``offset``
     - ``20``
     - Gap (s) between the signal and noise windows.
   * - ``axis``
     - ``1``
     - Trace axis for 2-D arrays.
   * - ``db``
     - ``False``
     - Return SNR in decibels.
   * - ``getwindow``
     - ``False``
     - Also return the window index arrays.
   * - ``side``
     - ``"a"``
     - ``"a"`` (both lags), ``"n"`` (negative), or ``"p"`` (positive).
   * - ``shorten_noise``
     - ``False``
     - Allow noise window to be shorter than signal window if needed.

.. code-block:: python

   snr = utils.get_snr(corrdata.data, t, dist=50.0, vmin=1.5, vmax=4.5)
   # snr[i] = [snr_negative, snr_positive] for each trace i

``ricker`` and ``gaussian``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate standard source wavelets:

.. code-block:: python

   t_r, w_r = utils.ricker(dt=0.001, f=30, t0=0.05)
   t_g, w_g = utils.gaussian(dt=0.001, width=0.02, shift=0.05)

The Gaussian wavelet follows the FWANT convention:

.. math::

   g(t) = \frac{\exp[-(t-t_0)^2/a^2]}{\sqrt{\pi}\,a}

where *a* is the width parameter (σ) and *t₀* is the time shift.

``cart2compass``
~~~~~~~~~~~~~~~~

Convert Cartesian slowness components to compass bearing and slowness magnitude:

.. code-block:: python

   bearing_deg, slowness = utils.cart2compass(Ux, Uy)

``box_smooth``
~~~~~~~~~~~~~~

Apply a simple box-car (moving-average) smoother to a 1-D array:

.. code-block:: python

   smoothed = utils.box_smooth(distortion_curve, n=3)
