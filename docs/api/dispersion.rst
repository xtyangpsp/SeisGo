.. _api-dispersion:

seisgo.dispersion
=================

The ``dispersion`` module extracts group/phase velocity dispersion images from seismic
waveform gathers and computes synthetic dispersion curves via forward modeling.

.. note::
   This module is actively developed. Some functions may be extended in future releases.

.. automodule:: seisgo.dispersion
   :members:
   :undoc-members:
   :show-inheritance:

----

Function Summary
----------------

.. autosummary::
   :nosignatures:

   seisgo.dispersion.get_dispersion_image
   seisgo.dispersion.narrowband_waveforms
   seisgo.dispersion.get_dispersion_waveforms_cwt
   seisgo.dispersion.forward_solver

----

``get_dispersion_image`` Parameter Reference
---------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``g``
     - *required*
     - 2-D waveform gather, shape ``(n_traces, n_samples)``.
   * - ``t``
     - *required*
     - Time vector (s). May be two-sided (negative to positive) or one-sided.
   * - ``d``
     - *required*
     - Distance vector (km), one value per trace row.
   * - ``pmin``, ``pmax``
     - *required*
     - Period range (s).
   * - ``vmin``, ``vmax``
     - *required*
     - Velocity search range (km/s).
   * - ``dp``
     - ``1``
     - Period increment (s).
   * - ``dv``
     - ``0.1``
     - Velocity grid spacing (km/s).
   * - ``window``
     - ``1``
     - Number of wavelengths for the travel-time extraction window.
       Can be a 2-element list ``[min, max]`` for period-dependent window size.
   * - ``pscale``
     - ``"ln"``
     - Period sampling: ``"ln"`` (linear) or ``"nln"`` (logarithmic).
   * - ``pband_extend``
     - ``5``
     - Number of extra period steps used when constructing narrowband filters.
   * - ``min_trace``
     - ``5``
     - Minimum number of traces satisfying the far-field criterion.
   * - ``min_wavelength``
     - ``1.5``
     - Minimum source–receiver distance expressed in wavelengths.
   * - ``energy_type``
     - ``"power_sum"``
     - Energy measure: ``"power_sum"`` or ``"envelope"``.
   * - ``get_best_v``
     - ``False``
     - If ``True``, also return the peak-energy velocity per period.
   * - ``plot``
     - ``False``
     - Display dispersion image immediately.

Return values
~~~~~~~~~~~~~

When the time vector is **two-sided** (``side = "a"``):

.. code-block:: python

   dimage, vout, pout = get_dispersion_image(...)
   # dimage.shape == (2, n_periods, n_velocities)
   # index 0 = negative lag, index 1 = positive lag

When ``get_best_v=True``, a fourth return value ``best_v`` is appended.

----

``forward_solver`` Reference
-----------------------------

Wraps ``surf96`` (from ``pysurf96``) to compute Rayleigh or Love wave group/phase
velocity dispersion curves for a layered Earth model.

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``vs``
     - *required*
     - Vs for each layer (km/s), from surface to half-space.
   * - ``periods``
     - *required*
     - Target periods (s).
   * - ``thickness``
     - *required*
     - Layer thicknesses (km). The last element should be ``0`` (half-space).
   * - ``wave_type``
     - ``"rayleigh"``
     - ``"rayleigh"`` or ``"love"``.
   * - ``mode``
     - ``1``
     - Mode number (1 = fundamental mode).
   * - ``velocity_type``
     - ``"group"``
     - ``"group"`` or ``"phase"``.

Internal mapping (Vs → Vp and density):

- **Vp/Vs** = √3
- **Density** via Birch's law: ρ = 0.32·Vp + 0.77

.. code-block:: python

   from seisgo import dispersion
   import numpy as np

   vs        = np.array([2.0, 3.0, 3.8, 4.2])
   thickness = np.array([8.0, 12.0, 20.0, 0.0])
   periods   = np.arange(5, 40, 2)

   gv = dispersion.forward_solver(vs, periods, thickness,
                                  wave_type="rayleigh", velocity_type="group")
