API reference
=============

All the objects below are importable from the top-level ``healpix_convert``
package.

Conversion
----------

:py:func:`~healpix_convert.create_healpix_dataset` runs the whole conversion in
one call. The other functions run it in separate stages, e.g., to convert the
data in parallel jobs (see :doc:`tutorials/quickstart`).

.. note::

   :py:func:`~healpix_convert.create_healpix_dataset` is meant for quick
   conversions of small inputs and for test cases. For production conversions
   of large datasets, prefer the staged functions below and run them with a
   workflow orchestration tool (Prefect, Airflow, Argo Workflows, etc.), which
   handles resource management, task failures and retries. The multi-stage API
   is designed for defining the tasks of such a workflow.

.. autosummary::
   :toctree: generated

   healpix_convert.create_healpix_dataset
   healpix_convert.create_staging_cache
   healpix_convert.prepare_healpix_dataset
   healpix_convert.convert_group_to_healpix

Settings
--------

.. autosummary::
   :toctree: generated

   healpix_convert.ConvertSettings
   healpix_convert.get_settings

Staging cache
-------------

.. autosummary::
   :toctree: generated

   healpix_convert.ConvertStagingCache
