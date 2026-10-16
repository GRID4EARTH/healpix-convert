API reference
=============

All the objects below are importable from the top-level ``healpix_convert``
package.

Conversion
----------

:py:func:`~healpix_convert.create_healpix_dataset` runs the whole conversion in
one call. The other functions run it in separate stages, e.g., to convert the
data in parallel jobs (see :doc:`tutorials/quickstart`).

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
