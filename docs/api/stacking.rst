.. _api-stacking:

seisgo.stacking
===============

The ``stacking`` module provides a unified interface to multiple waveform stacking
algorithms, from simple linear averaging to advanced phase-weighted and cluster-based methods.

.. automodule:: seisgo.stacking
   :members:
   :undoc-members:
   :show-inheritance:

----

Function Summary
----------------

.. autosummary::
   :nosignatures:

   seisgo.stacking.stack
   seisgo.stacking.seisstack
   seisgo.stacking.robust
   seisgo.stacking.adaptive_filter
   seisgo.stacking.pws
   seisgo.stacking.tfpws
   seisgo.stacking.tfpws_dost
   seisgo.stacking.nroot
   seisgo.stacking.selective
   seisgo.stacking.clusterstack

----

Stacking Methods Reference
--------------------------

All methods are accessible via the unified :func:`seisgo.stacking.stack` wrapper.

.. list-table::
   :header-rows: 1
   :widths: 20 55 25

   * - Method name
     - Description
     - Key parameter(s)
   * - ``linear``
     - Arithmetic mean along the stack axis.
     - ``axis``
   * - ``pws``
     - Phase-weighted stack. Applies a phase coherence weight in the time domain.
     - ``p`` (phase power, default 2)
   * - ``tf-pws``
     - Time-frequency phase-weighted stack using the Stockwell transform.
     - ``p``
   * - ``tfpws-dost``
     - TF-PWS using the discrete orthonormal Stockwell transform (DOST).
     - ``p``
   * - ``robust``
     - Iterative robust stack (Pavlis & Vernon 2010). Downweights incoherent traces.
     - ``epsilon``, ``maxstep``, ``win``, ``ref``
   * - ``acf``
     - Adaptive covariance filter (Nakata et al. 2015). Enhances coherent signal.
     - ``g``
   * - ``nroot``
     - N-th root stack. Enhances coherence at the cost of amplitude fidelity.
     - ``p``
   * - ``selective``
     - Correlation-based selective stack. Only includes traces above a coherence threshold.
     - ``cc_min``, ``epsilon``, ``maxstep``, ``win``, ``ref``
   * - ``cluster``
     - Cluster-based stack. Groups traces by similarity and returns the dominant cluster.
     - ``h``, ``win``, ``normalize``

----

Usage Examples
--------------

Unified wrapper
~~~~~~~~~~~~~~~

.. code-block:: python

   from seisgo import stacking
   import numpy as np

   # d: 2-D array, shape (n_traces, n_samples)
   linear = stacking.stack(d, method="linear")
   pws    = stacking.stack(d, method="pws",    par={"p": 2})
   robust = stacking.stack(d, method="robust", par={"epsilon": 1e-5, "maxstep": 10})

Selective stack with custom reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   ref = np.median(d, axis=0)
   sel = stacking.stack(d, method="selective",
                        par={"cc_min": 0.6, "ref": ref, "win": [100, 400]})

Returning stacking statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   newstack, weights, n_iter = stacking.robust(d, stat=True)
