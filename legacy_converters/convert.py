"""Main conversion functions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import PurePath
from typing import Any

import dask
import distributed
import shapely
import structlog
import xarray as xr
import zarr.api.synchronous as zarr

from legacy_converters.cache import create_staging_cache
from legacy_converters.core.conversion_models import ConvertStagingCache
from legacy_converters.core.healpix_conventions import DGGSZarrConvention
from legacy_converters.core.multiscales_conventions import MultiscalesZarrConvention
from legacy_converters.healpix_converters import (
    HealpixGroupConverter,
    init_converter,
)
from legacy_converters.settings.common import (
    ConvertSettings,
    validate_convert_settings,
)
from legacy_converters.stac_operations import _extract_stac_metadata, _merge_stac_items

log = structlog.get_logger()


def _convert_group_to_healpix(
    converter: HealpixGroupConverter,
    *,
    chunk_indices: Sequence[int] | None = None,
    resources: dict | None = None,
    load_input_data: bool = False,
):
    # TODO: remove
    if not converter.spatial_info.is_projected:
        return

    log.info(f"••• converting group {converter.output_path} to HEALPix...")

    log.info(
        f"resampling data on HEALPix using {converter.settings.resampler.name} method"
    )
    log.info(f"chunking output data using {converter.settings.chunk.method!r} method")

    try:
        client = distributed.Client.current()
    except ValueError:
        client = None

    n_chunks = int(math.ceil(converter.cell_dim_size / converter.cell_dim_chunk_size))

    if chunk_indices is None:
        chunk_indices = list(range(n_chunks))

    if any(idx < 0 or idx > n_chunks - 1 for idx in chunk_indices):
        raise ValueError("out of bounds chunk indices")

    if client is not None:
        log.info(
            f"••• converting {len(chunk_indices)} chunks in parallel using dask (distributed)..."
        )

        if resources is None:
            resources = {}

        if load_input_data:
            log.info("persist input dataset(s) on distributed memory with dask")
            converter.load_datasets(client=client)

        # use a wrapper func as passing _convert_one_chunk kwargs to client.map doesn't work
        # (dask/distributed does not like pickling `group_datasets`)
        func = lambda idx: converter.convert(idx)
        with dask.config.set({"optimization.fuse.active": False}):
            futures = client.map(func, chunk_indices, resources=resources)
        client.gather(futures)

    else:
        log.info(
            f"••• converting {len(chunk_indices)} chunks serially (it may take a while)..."
        )

        if load_input_data:
            log.info("pre-load input datasets in memory!")
            converter.load_datasets()

        for chunk_index in chunk_indices:
            converter.convert(chunk_index)

    log.info(f"••• finished converting {len(chunk_indices)} chunks.")


def _create_and_process_output(
    cache: ConvertStagingCache,
    settings: ConvertSettings,
    output_path: str,
    *,
    output_storage_options: dict[str, Any] | None = None,
    convert_data: bool = True,
    dry_run: bool = False,
) -> xr.DataTree:
    if dry_run:
        convert_data = False

    # --- create output Zarr dataset
    # TODO: check for any existing output Zarr dataset
    # TODO: write root metadata (STAC / STAC discovery attributes, etc.)
    log.info(f"••• creating {output_path} Zarr dataset...")

    if dry_run:
        root_group = zarr.create_group(store=None)
    else:
        root_group = zarr.open_group(
            output_path, mode="w", storage_options=output_storage_options
        )

    output_store = root_group.store

    log.info("••• propagating stac metadata...")
    stac_metadata = _extract_stac_metadata(cache.input_datatrees)
    merged_stac_metadata = _merge_stac_items(stac_metadata, cache.output_spatial)

    root_group.attrs["stac_discovery"] = merged_stac_metadata.model_dump()
    log.info("••• finished propagating stac metadata.")

    # --- process groups
    log.info("••• processing groups...")

    dt0 = next(iter(cache.input_datatrees.values()))

    for path in dt0.groups:
        path = PurePath(path.lstrip("/"))

        if path in cache.output_groups.excluded_groups:
            continue
        if str(path) == ".":
            # root path (skip it here if attributes are written above)
            continue

        group_path_rel = str(cache.output_groups.io_path_map[path])

        if path in cache.output_groups.multiscale_groups:
            log.info(f"••• writing group '{group_path_rel}' to zarr")
            multiscales_obj = cache.output_groups.multiscale_groups[path]
            zarr.create_group(
                output_store,
                path=group_path_rel,
                attributes={
                    "zarr_conventions": [
                        MultiscalesZarrConvention().model_dump(),
                        DGGSZarrConvention().model_dump(),
                    ],
                    "multiscales": multiscales_obj.model_dump(),
                },
            )
        elif path in cache.input_spatial_groups:
            log.info(f"••• writing group '{group_path_rel}' to zarr")
            converter = init_converter(
                path, cache=cache, settings=settings, output_store=output_store
            )
            if convert_data:
                _convert_group_to_healpix(converter)
        elif not dt0[str(path)].has_data:
            zarr.create_group(output_store, path=group_path_rel)
        else:
            # non-spatial group (skip for now)
            log.warning(
                f"••• skip writing non-spatial group '{group_path_rel}' to zarr "
                "(not yet supported)"
            )

    # --- consolidate Zarr metadata
    log.info("••• consolidating output Zarr metadata...")
    zarr.consolidate_metadata(output_store)
    # TODO: consolidate metadata for each group/node

    log.info("••• done!")

    if dry_run:
        dt_out = xr.open_datatree(output_store, engine="zarr", chunks="auto")  # type: ignore
    else:
        dt_out = xr.open_datatree(
            output_path,
            storage_options=output_storage_options,
            engine="zarr",
            chunks="auto",
        )

    return dt_out


def prepare_healpix_dataset(
    cache: ConvertStagingCache,
    settings: ConvertSettings,
    output_path: str,
    *,
    output_storage_options: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> xr.DataTree:
    """Create a new Zarr dataset without any data resampled on the HEALPix grid (yet).

    All output groups and arrays are created along with their metadata. Arrays
    having a HEALPix cell dimension are still empty.

    Useful for multi-stage conversion workflows:

    - Use :py:func:`create_staging_cache` first to create the conversion
      staging cache.
    - Use :py:func:`convert_group_to_healpix` later (maybe multiple times)
      to resample data on HEALPix and write it to the output dataset
      (fill arrays with data values).

    Parameters
    ----------
    cache : :py:class:`ConvertStagingCache`
        A data cache containing information extracted from the input dataset(s)
        and/or pre-computed for the output dataset, as returned by
        :py:func:`create_staging_cache`.
    settings : dict or object
        Healpix conversion settings.
    output_path : str
        Path to the HEALPix output Zarr dataset.
    output_storage_options : dict, optional
        Storage options passed to Zarr for the dataset creation
        (useful for, e.g., writing on S3 object store).
    dry_run : bool, optional
        If True, only display what would have been done. The output
        dataset also won't be written to a persistent storage but
        instead will be created in-memory. No data conversion will
        happen (`convert_data` set to False).
        Default: False.

    Returns
    -------
    :py:class:`xarray.DataTree`
        The output Zarr dataset with HEALPix metadata and empty data, returned
        as an Xarray DataTree object.

    See Also
    --------
    create_staging_cache
    convert_group_to_healpix

    """
    log.info("••• reading conversion settings...")
    settings = validate_convert_settings(settings)

    return _create_and_process_output(
        cache,
        settings,
        output_path,
        output_storage_options=output_storage_options,
        convert_data=False,
        dry_run=dry_run,
    )


def convert_group_to_healpix(
    group: str,
    *,
    cache: ConvertStagingCache,
    settings: ConvertSettings,
    output_path: str,
    chunk_indices: Sequence[int] | None = None,
    output_storage_options: dict[str, Any] | None = None,
    resources: dict | None = None,
    load_input_data: bool = False,
):
    """Convert data in one group to HEALPix and write the output resampled data to
    the output zarr dataset.

    Parameters
    ----------
    group : str
        Path to the input Zarr group to convert to HEALPix.
    cache : :py:class:`ConvertStagingCache`
        A data cache containing information extracted from the input dataset(s)
        and/or pre-computed for the output dataset, as returned by
        :py:func:`create_staging_cache`.
    settings : dict or object
        Healpix conversion settings.
    output_path : str
        Path to the HEALPix output Zarr dataset.
    chunk_indices : iterable of ints, optional
        Resample and write data only for the given chunk indices. If None (default),
        all chunks will be processed and written.
    output_storage_options : dict, optional
        Storage options passed to Zarr for the dataset creation
        (useful for, e.g., writing on S3 object store).
    load_input_data : bool, optional
        If True, pre-load all input datasets into memory or presist them
        in distributed memory when using dask (default: False).
        This could speed-up the conversion process but this could also
        consume a lot of memory! Use it carefully.

    See Also
    --------
    create_staging_cache
    prepare_healpix_dataset

    """
    log.info("••• reading conversion settings...")
    settings = validate_convert_settings(settings)

    group_path = PurePath(group.lstrip("/"))

    root_group = zarr.open_group(
        output_path, mode="a", storage_options=output_storage_options
    )
    output_store = root_group.store

    converter = init_converter(
        group_path, cache=cache, settings=settings, output_store=output_store
    )
    _convert_group_to_healpix(
        converter,
        chunk_indices=chunk_indices,
        resources=resources,
        load_input_data=load_input_data,
    )


def create_healpix_dataset(
    input_paths: Sequence[str],
    settings: dict | ConvertSettings,
    output_path: str,
    *,
    groups: str | Sequence[str] | None = None,
    output_extent: dict | shapely.Polygon | None = None,
    output_storage_options: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> xr.DataTree:
    """Create a new Zarr dataset with data resampled on the HEALPix grid.

    Parameters
    ----------
    input_paths : list of paths
        Path(s) to the input Zarr dataset(s) to convert to HEALPix.
    settings : dict or object
        Healpix conversion settings.
    output_path : str
        Path to the HEALPix output Zarr dataset.
    groups : str or list, optional
        Groups (paths) in the input datasets to process. It can be
        either a string (single group or unix-like pattern) or a
        list of groups. By default, all groups found in input datasets
        are processed.
    output_extent : dict or :py:class:`shapely.Polygon`, optional
        Spatial extent of the output HEALPix dataset, given as a
        (GEOJSON-like or object) polygon with lat-lon coordinates.
        If not given (default), the spatial extent will be the union
        of the spatial extents of the input datasets (single polygon).
    output_storage_options : dict, optional
        Storage options passed to Zarr for the dataset creation
        (useful for, e.g., writing on S3 object store).
    dry_run : bool, optional
        If True, only display what would have been done. The output
        dataset also won't be written to a persistent storage but
        instead will be created in-memory. No data conversion will
        happen (`convert_data` set to False).
        Default: False.

    Returns
    -------
    :py:class:`xarray.DataTree`
        The output Zarr dataset with HEALPix data and/or metadata, returned
        as an Xarray DataTree object.

    """
    log.info(f"••• start converting {len(input_paths)} input datasets into HEALPix...")

    log.info("••• reading conversion settings...")
    settings = validate_convert_settings(settings)

    cache = create_staging_cache(
        input_paths, settings, groups=groups, output_extent=output_extent
    )

    return _create_and_process_output(
        cache,
        settings,
        output_path,
        output_storage_options=output_storage_options,
        convert_data=True,
        dry_run=dry_run,
    )
