.. _api-helpers:

seisgo.helpers
==============

The ``helpers`` module provides self-describing accessor functions that return the lists
of valid option strings accepted by other SeisGo functions. Use these instead of
hard-coding strings, so your code stays compatible as SeisGo evolves.

.. automodule:: seisgo.helpers
   :members:
   :undoc-members:
   :show-inheritance:

----

Function Summary
----------------

.. autosummary::
   :nosignatures:

   seisgo.helpers.xcorr_methods
   seisgo.helpers.stack_methods
   seisgo.helpers.dvv_methods
   seisgo.helpers.wavelet_labels
   seisgo.helpers.xcorr_norm_methods
   seisgo.helpers.xcorr_output_structure
   seisgo.helpers.xcorr_sides
   seisgo.helpers.outdatafile_formats
   seisgo.helpers.datafile_extension

----

Reference Tables
----------------

Cross-correlation methods
~~~~~~~~~~~~~~~~~~~~~~~~~

From :func:`seisgo.helpers.xcorr_methods`:

.. code-block:: python

   ["xcorr", "deconv", "coherency"]

Stacking methods
~~~~~~~~~~~~~~~~

From :func:`seisgo.helpers.stack_methods`:

.. code-block:: python

   ["linear", "pws", "tf-pws", "robust", "acf", "nroot", "selective", "cluster"]

dv/v methods
~~~~~~~~~~~~

From :func:`seisgo.helpers.dvv_methods`:

.. code-block:: python

   ["wts", "ts"]

Wavelet labels
~~~~~~~~~~~~~~

From :func:`seisgo.helpers.wavelet_labels`:

.. code-block:: python

   ["gaussian", "ricker"]

Temporal normalization methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From :func:`seisgo.helpers.xcorr_norm_methods(mode="t")`:

.. code-block:: python

   ["rma", "one_bit", "ftn"]

Spectral normalization methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From :func:`seisgo.helpers.xcorr_norm_methods(mode="f")`:

.. code-block:: python

   ["rma", "phase_only"]

xcorr output structures
~~~~~~~~~~~~~~~~~~~~~~~

From :func:`seisgo.helpers.xcorr_output_structure`:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Name
     - Short code
     - Layout
   * - ``raw``
     - ``r``
     - By time chunk; all pairs in one file
   * - ``source``
     - ``s``
     - Subdirectory per virtual source
   * - ``station-pair``
     - ``sp``
     - Subdirectory per station pair
   * - ``station-component-pair``
     - ``scp``
     - Nested source / component subdirectories

xcorr lag sides
~~~~~~~~~~~~~~~

From :func:`seisgo.helpers.xcorr_sides`:

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Code
     - Meaning
   * - ``a``
     - Both negative and positive lags joined
   * - ``n``
     - Negative lag only
   * - ``p``
     - Positive lag only
   * - ``o``
     - One-sided (sign unknown)
   * - ``u``
     - Not applicable

File formats and extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

From :func:`seisgo.helpers.outdatafile_formats` and :func:`seisgo.helpers.datafile_extension`:

.. code-block:: python

   formats    = ["asdf", "pickle"]
   extensions = ["h5", "pk"]
