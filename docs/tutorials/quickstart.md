# Quickstart

This page walks through a conversion with `healpix-convert`, from the
conversion settings to a HEALPix Zarr dataset. The code below is not run when
the documentation is built: replace the paths with your own datasets.

## Overview

A conversion runs in six steps:

1. **Settings** describe how each input Zarr group is resampled onto HEALPix
   ({py:class}`~healpix_convert.ConvertSettings`).
2. The **staging cache** holds what is parsed from the input datasets and the
   settings, so that this work is done only once
   ({py:func}`~healpix_convert.create_staging_cache`).
3. **Prepare** the output Zarr dataset: all groups, arrays and metadata are
   written, but the arrays along the HEALPix cell dimension are still empty
   ({py:func}`~healpix_convert.prepare_healpix_dataset`).
4. **Convert** the data of each group: resample it onto HEALPix and write it in
   the output arrays, possibly chunk by chunk in several parallel jobs
   ({py:func}`~healpix_convert.convert_group_to_healpix`).
5. **Generate the pyramids**: aggregate the converted data onto the coarser
   HEALPix refinement levels of the multiscale groups that have pyramid
   settings, whose groups and metadata the third step has already created
   ({py:func}`~healpix_convert.generate_pyramids`, see
   {doc}`../user-guide/pyramids`). It does nothing without such settings.
6. **Finalize** the output dataset: consolidate its Zarr metadata
   ({py:func}`~healpix_convert.finalize_healpix_dataset`, see
   {doc}`../user-guide/metadata`).

{py:func}`~healpix_convert.create_healpix_dataset` runs all these steps in a
single call:

```python
import healpix_convert as hc

hc.create_healpix_dataset(["input.zarr"], settings, "output.zarr")
```

:::{note}
This all-at-once function is meant for quick conversions of small inputs and
for test cases. Avoid it for production conversions of large datasets: run the
steps below as the tasks of a workflow orchestration tool instead (e.g.,
[Prefect](https://www.prefect.io), [Airflow](https://airflow.apache.org) or
[Argo Workflows](https://argoproj.github.io/workflows/)). The multi-stage API
is designed for that, and the orchestrator handles resource management, task
failures and retries.
:::

The rest of this page goes through the steps one by one.

## Input datasets

The input is one or more Zarr datasets (e.g., EOPF products) with the same
hierarchy of groups:

- the root group has a `stac_discovery` attribute (a STAC item), whose
  `geometry` gives the spatial extent of the dataset. Its `proj:*` properties
  (STAC projection extension) give the CRS, WGS84 if there are none, and for a
  projected CRS the extent in projected coordinates (`proj:bbox`);
- the groups to convert to HEALPix are detected from their spatial metadata:
  either STAC projection attributes on the data variables and `x` / `y`
  dimensions, or CF longitude / latitude coordinates (identified by their
  `standard_name`).

## 1. Conversion settings

Predefined settings are available for the supported products:

```python
settings = hc.get_settings("era5")  # or "cams-eac4", "climatedt-sfc", "dem-glo30"
```

You can also create the settings from a dictionary, here for a single input
group `measurements/land_cover` (the settings used in the
[ESA CCI land cover tutorial](notebooks/esacci_land_cover_converter.ipynb)):

```python
settings = hc.ConvertSettings.from_dict(
    {
        "group_settings": {
            "measurements/land_cover": {
                # output HEALPix grid
                "healpix": {
                    "refinement_level": 15,
                    "indexing_scheme": "nested",
                    "ellipsoid": {"name": "wgs84"},
                },
                # output chunks: one chunk per HEALPix cell at level 8
                "chunk": {
                    "method": "healpix_cell_dense",
                    "healpix": {
                        "refinement_level": 8,
                        "indexing_scheme": "nested",
                        "ellipsoid": {"name": "wgs84"},
                    },
                    "chunk_buffer_width": 1000.0,  # metres
                },
                # resampling method
                "resampler": {"name": "nearest"},
            }
        }
    }
)
```

The settings are validated when they are created. Every function below also
accepts the settings as a plain dictionary. See {doc}`../user-guide/settings`
for all the options.

## 2. Staging cache

```python
input_paths = ["input-1.zarr", "input-2.zarr"]

cache = hc.create_staging_cache(input_paths, settings)
```

The cache holds the opened input datasets, their spatial information, the
groups to convert and the spatial extent of the output dataset: by default the
union of the extents of the input datasets. Pass `output_extent` (a
{py:class}`shapely.Polygon` or a GeoJSON-like dictionary with longitude /
latitude coordinates) to set it explicitly, and `groups` to convert only some
groups of the input datasets.

## 3. Prepare the output dataset

```python
output_path = "output.zarr"

hc.prepare_healpix_dataset(cache, settings, output_path)
```

This creates the output Zarr dataset with all its groups, arrays and metadata
(see {doc}`../user-guide/metadata`) and returns it as an
{py:class}`xarray.DataTree`. With `dry_run=True`, the dataset is only created
in memory, which is useful to check the output structure before converting
any data.

## 4. Convert the data

Convert each group with {py:func}`~healpix_convert.convert_group_to_healpix`.
The group is given by its path in the input datasets:

```python
hc.convert_group_to_healpix(
    "measurements/land_cover",
    cache=cache,
    settings=settings,
    output_path=output_path,
)
```

### Convert in parallel jobs

With the `healpix_cell_dense` and `healpix_cell_sparse` chunk methods, each
chunk of the output arrays along the HEALPix cell dimension can be converted
independently. `chunk_indices` selects the chunks converted by one call, so
the conversion can be split between several jobs (e.g., the steps of a
workflow running on a cluster), all writing to the same output dataset.

The number of chunks is the number of chunks of the group's cell ids array in
the prepared output dataset (the output group has the path of the input group,
except for the children of a multiscale group, see
{doc}`../user-guide/settings`):

```python
import zarr

n_chunks = zarr.open_array(output_path, path="measurements/land_cover/cell_ids").nchunks
```

The cache and the settings can be passed to the jobs as JSON strings. When a
cache is read back, the input datasets are opened again from their paths:

```python
cache_json = cache.to_json()
settings_json = settings.to_json()
```

Then, in job `job_index` out of `n_jobs`:

```python
import healpix_convert as hc

cache = hc.ConvertStagingCache.from_json(cache_json)
settings = hc.ConvertSettings.from_json(settings_json)

hc.convert_group_to_healpix(
    "measurements/land_cover",
    cache=cache,
    settings=settings,
    output_path=output_path,
    chunk_indices=range(job_index, n_chunks, n_jobs),
)
```

Within one job, the chunks can also be converted in parallel on a Dask
cluster: see {doc}`../user-guide/dask`.

## 5. Generate the pyramids

```python
hc.generate_pyramids(cache, settings, output_path)
```

This aggregates the converted data onto the coarser HEALPix refinement levels
of the multiscale groups that have pyramid settings, and does nothing without
such settings. See {doc}`../user-guide/pyramids`.

## 6. Finalize the output dataset

```python
hc.finalize_healpix_dataset(cache, settings, output_path)
```

This consolidates the Zarr metadata of the output dataset, which the previous
stages leave alone: consolidated metadata is a copy, stale as soon as a group
or an array is written next to it. The `metadata.consolidate` settings select
the groups it is written for (see {doc}`../user-guide/metadata`).

Run it last, once every group of the output dataset has been written, i.e.
after the parallel conversion jobs are done.

## Open the output dataset

```python
import xarray as xr

ds = xr.open_dataset(output_path, group="measurements/land_cover", engine="zarr")
```

The data variables have a `cells` dimension (the HEALPix cell dimension, last
if the variable has other dimensions) and a `cell_ids` coordinate: these are
the default names, set in the `healpix` group settings. See
{doc}`../user-guide/metadata` for the metadata describing the HEALPix grid.
