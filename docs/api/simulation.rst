.. _api-simulation:

seisgo.simulation
=================

The ``simulation`` module provides a 1-D finite-difference acoustic wave solver and
a layered velocity model builder for synthetic testing and method validation.

.. automodule:: seisgo.simulation
   :members:
   :undoc-members:
   :show-inheritance:

----

Function Summary
----------------

.. autosummary::
   :nosignatures:

   seisgo.simulation.fd1d_dx4dt4
   seisgo.simulation.build_vmodel

----

``fd1d_dx4dt4`` — 1-D FD Acoustic Solver
------------------------------------------

Solves the first-order acoustic wave equation on a 1-D spatial grid using:

- **Spatial accuracy**: O(Δx⁴) — 4th-order staggered-grid finite differences
- **Temporal accuracy**: O(Δt⁴) — Adams–Bashforth multi-step time integration

.. note::
   Based on the scheme described in:

   Bohlen, T., & Wittkamp, F. (2016). Three-dimensional viscoelastic time-domain
   finite-difference seismic modelling using the staggered Adams-Bashforth time integrator.
   *Geophysical Journal International*, 204(3), 1781–1788.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``x``
     - *required*
     - 1-D spatial grid vector (m or km, consistent with ``vmodel``).
   * - ``dt``
     - *required*
     - Time step (s). Must satisfy CFL condition: ``dt ≤ 0.5 · Δx_max / v_max``.
   * - ``tmax``
     - *required*
     - Total simulation duration (s).
   * - ``vmodel``
     - *required*
     - 1-D velocity model, same length as ``x``.
   * - ``rho``
     - *required*
     - 1-D density model, same length as ``x``.
   * - ``xsrc``
     - *required*
     - Source grid index (integer).
   * - ``xrcv``
     - *required*
     - Receiver grid index (integer).
   * - ``stf_freq``
     - ``1``
     - Source time function frequency parameter: central frequency (Hz) for Ricker;
       width σ (s) for Gaussian.
   * - ``stf_shift``
     - ``None``
     - Source onset time (s). Default: ``3 / stf_freq`` for Ricker.
   * - ``stf_type``
     - ``"ricker"``
     - Source wavelet: ``"ricker"`` or ``"gaussian"``.
   * - ``t_interval``
     - ``1``
     - Output time sub-sampling factor. ``1`` = full resolution.

Returns: ``(tout, seisout)``

- ``tout`` — time vector (s) starting from 0, accounting for source shift
- ``seisout`` — pressure seismogram at the receiver

CFL check
~~~~~~~~~

The solver raises ``ValueError`` if the provided ``dt`` exceeds the CFL limit:

.. math::

   dt_{\max} = 0.5 \cdot \frac{\Delta x_{\max}}{v_{\max}}

----

``build_vmodel`` — Layered Velocity Model Builder
---------------------------------------------------

Constructs a fine-grid 1-D layered velocity model with linearly increasing velocity
between ``vmin`` and ``vmax`` across ``nlayer`` layers.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``zmax``
     - *required*
     - Maximum model depth/length.
   * - ``dz``
     - *required*
     - Fine-grid spacing.
   * - ``nlayer``
     - *required*
     - Number of velocity layers.
   * - ``vmin``, ``vmax``
     - *required*
     - Velocity range (surface to bottom).
   * - ``rhomin``, ``rhomax``
     - *required*
     - Density range.
   * - ``zmin``
     - ``0``
     - Starting depth.
   * - ``layer_dv``
     - ``None``
     - Array of fractional velocity perturbations per layer, e.g. ``[-0.1, 0, 0.05, …]``.

Returns: ``(z, v, rho)`` — depth grid, velocity, and density arrays.

.. code-block:: python

   from seisgo import simulation
   import numpy as np

   # 500-point model, 5 layers, with layer 2 perturbed -10%
   layer_dv = np.zeros(5)
   layer_dv[1] = -0.10

   z, v, rho = simulation.build_vmodel(
       zmax=500, dz=1.0, nlayer=5,
       vmin=1500, vmax=3500,
       rhomin=1800, rhomax=2500,
       layer_dv=layer_dv,
   )
