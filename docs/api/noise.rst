.. _api-noise:

seisgo.noise
============

The ``noise`` module is the core of SeisGo's ambient noise cross-correlation pipeline.
It handles FFT computation, data assembly, cross-correlation, and I/O of correlation results.

.. automodule:: seisgo.noise
   :members:
   :undoc-members:
   :show-inheritance:

----

Function Summary
----------------

Data assembly
~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   seisgo.noise.assemble_raw
   seisgo.noise.assemble_fft

FFT / preprocessing
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   seisgo.noise.compute_fft
   seisgo.noise.cc_memory

Cross-correlation
~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   seisgo.noise.correlate
   seisgo.noise.do_xcorr

I/O
~~~

.. autosummary::
   :nosignatures:

   seisgo.noise.extract_corrdata
   seisgo.noise.save_corrdata

----

Key Notes
---------

Temporal normalization
~~~~~~~~~~~~~~~~~~~~~~

Controlled by the ``time_norm`` parameter in :func:`seisgo.noise.compute_fft`.
Available methods (from :func:`seisgo.helpers.xcorr_norm_methods`):

- ``"no"`` — no normalization
- ``"one_bit"`` — sign-only clipping
- ``"rma"`` — running-mean absolute-value normalization
- ``"ftn"`` — frequency–time normalization

Spectral normalization
~~~~~~~~~~~~~~~~~~~~~~

Controlled by ``freq_norm``:

- ``"no"`` — no normalization
- ``"rma"`` — running mean in frequency domain
- ``"phase_only"`` — unit-amplitude spectral whitening

Cross-correlation methods
~~~~~~~~~~~~~~~~~~~~~~~~~

Controlled by the ``method`` parameter in :func:`seisgo.noise.correlate`
and :func:`seisgo.noise.do_xcorr`:

- ``"xcorr"`` — normalized cross-correlation
- ``"deconv"`` — spectral deconvolution
- ``"coherency"`` — phase coherency

Output structure options
~~~~~~~~~~~~~~~~~~~~~~~~

See :func:`seisgo.helpers.xcorr_output_structure` for the full list of directory
layout options (``raw``, ``source``, ``station-pair``, ``station-component-pair``).
