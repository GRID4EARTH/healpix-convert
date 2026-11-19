"""Binning of 1-D samples (profiles, along-track points, point observations) onto
HEALPix cells.

The inputs of this module are *samples*: a dataset with one dimension that runs
over the samples (the "sample dimension", e.g. the along-track dimension of a
satellite curtain), a longitude and a latitude per sample, and any number of
variables that have this dimension, possibly together with other dimensions
(e.g. a vertical axis). The output is a dataset on a sparse list of the HEALPix
cells that hold at least one sample.

Three functions, which are independent of any product:

- :py:func:`bin_samples` puts each sample into the cell that contains it, counts
  the samples of each cell (``sample_count``) and reduces the variables of
  several samples in one cell with a method chosen per variable;
- :py:func:`split_by_parent` cuts the result into one dataset per cell of a
  coarser level (the "tiles");
- :py:func:`write_cell_dataset` writes one such dataset as a Zarr store with the
  same DGGS / CF metadata as the stores of the converters.

Nothing is resampled: a sample is never spread over several cells, so the
counts of the cells add up to the number of valid input samples.

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

from healpix_convert.core.healpix_conventions import Healpix, WGS84Ellipsoid
from healpix_convert.core.metadata import (
    CF_GRID_MAPPING_VARIABLE,
    get_cf_cell_id_attrs,
    get_cf_data_variable_attrs,
    get_cf_grid_mapping_attrs,
    get_healpix_group_attrs,
    write_root_conventions,
)
from healpix_convert.settings.common import MetadataSettings

#: Reduction methods of :py:func:`bin_samples`, and the special value that leaves
#: a variable out.
METHODS = ("mean", "first", "last", "mode", "min", "max", "sum", "rms")
DROP = "drop"

#: Name of the cell dimension and of the cell ids coordinate (as in the
#: converters).
CELL_DIM = "cells"
CELL_IDS = "cell_ids"
SAMPLE_COUNT = "sample_count"


def _cell_ids(lon: np.ndarray, lat: np.ndarray, level: int, ellipsoid: str):
    import healpix_geo

    return healpix_geo.nested.lonlat_to_healpix(
        np.ascontiguousarray(lon, dtype="float64"),
        np.ascontiguousarray(lat, dtype="float64"),
        level,
        ellipsoid=ellipsoid,
    )


def _reduce(
    name: str,
    method: str,
    sorted_values: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Reduce ``sorted_values`` (samples first, any other shape) per group.

    The samples are sorted by cell (stably, so the samples of a cell stay in
    input order); ``starts`` and ``counts`` are the first position and the size
    of each group. Returns an array with one row per group.
    """
    values = sorted_values
    kind = values.dtype.kind

    if method in ("first", "last"):
        index = starts if method == "first" else starts + counts - 1
        return sorted_values[index]

    if kind in "OUSV":
        raise ValueError(
            f"variable {name!r} (dtype {values.dtype}) can only be reduced with "
            f"'first' or 'last', not {method!r}"
        )

    if kind in "Mm":
        if method not in ("min", "max"):
            raise ValueError(
                f"variable {name!r} (dtype {values.dtype}) can only be reduced "
                f"with 'first', 'last', 'min' or 'max', not {method!r}"
            )
        as_int = sorted_values.view("int64")
        nat = np.iinfo("int64").min  # NaT: the smallest int64, left out of the reduction
        top = np.iinfo("int64").max
        if method == "min":
            out = np.minimum.reduceat(np.where(as_int == nat, top, as_int), starts, axis=0)
            out = np.where(out == top, nat, out)
        else:
            out = np.maximum.reduceat(as_int, starts, axis=0)
        return out.view(values.dtype)

    flat = sorted_values.reshape(sorted_values.shape[0], -1)

    if method == "mode":
        out = _mode(flat, starts, counts)
        return out.reshape((starts.size,) + values.shape[1:])

    if method in ("min", "max"):
        func = np.fmin if method == "min" else np.fmax
        out = func.reduceat(flat, starts, axis=0)
        return out.reshape((starts.size,) + values.shape[1:]).astype(values.dtype)

    # sum, mean, rms: NaN-aware, in float64, NaN when no valid value in a cell
    as_float = flat.astype("float64")
    valid = np.isfinite(as_float)
    if method == "rms":
        as_float = as_float * as_float
    total = np.add.reduceat(np.where(valid, as_float, 0.0), starts, axis=0)
    n_valid = np.add.reduceat(valid.astype("int64"), starts, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        if method == "sum":
            out = np.where(n_valid > 0, total, np.nan)
        elif method == "mean":
            out = total / n_valid
        else:
            out = np.sqrt(total / n_valid)
    out = out.reshape((starts.size,) + values.shape[1:])
    return out.astype(values.dtype) if kind == "f" else out


def _mode(flat: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Most frequent value of each group and column (the first one met on a tie).

    NaN is never counted, unless a group holds nothing else (then NaN is kept).
    """
    out = flat[starts].copy()
    for group in np.nonzero(counts > 1)[0]:
        block = flat[starts[group] : starts[group] + counts[group]]
        k, n_columns = block.shape
        if k <= 64:
            eq = block[:, None, :] == block[None, :, :]  # NaN != NaN: never counted
            frequency = eq.sum(axis=1)  # (k, n_columns)
            best = frequency.argmax(axis=0)
            picked = block[best, np.arange(n_columns)]
            all_nan = frequency.max(axis=0) == 0
            out[group] = np.where(all_nan, block[0], picked)
        else:
            for column in range(n_columns):
                values = block[:, column]
                values = values[~np.isnan(values)] if values.dtype.kind == "f" else values
                if values.size == 0:
                    out[group, column] = block[0, column]
                    continue
                unique, first_index, frequency = np.unique(
                    values, return_index=True, return_counts=True
                )
                top = np.nonzero(frequency == frequency.max())[0]
                out[group, column] = unique[top[np.argmin(first_index[top])]]
    return out


def bin_samples(
    ds: xr.Dataset,
    *,
    sample_dim: str,
    level: int,
    methods: Mapping[str, str],
    lon: str = "longitude",
    lat: str = "latitude",
    default_method: str | None = None,
    dtypes: Mapping[str, Any] | None = None,
    ellipsoid: str = "WGS84",
    count_name: str = SAMPLE_COUNT,
    cell_dim: str = CELL_DIM,
    cell_ids_name: str = CELL_IDS,
) -> xr.Dataset:
    """Bin the samples of ``ds`` into the HEALPix cells of ``level`` (nested).

    Each sample goes into the one cell that contains its longitude and
    latitude. The output holds the cells that contain at least one valid
    sample, sorted by cell id, on the dimension ``cell_dim`` with the
    coordinate ``cell_ids_name`` (as the converters name them).

    Parameters
    ----------
    ds : xarray.Dataset
        The samples. Longitude and latitude are variables of the sample
        dimension (1-D).
    sample_dim : str
        Name of the dimension that runs over the samples.
    level : int
        HEALPix refinement level of the cells.
    methods : mapping of str to str
        How to reduce each variable that has ``sample_dim`` when a cell holds
        several samples: ``"mean"`` (NaN-aware), ``"first"``, ``"last"``,
        ``"mode"`` (most frequent value, the first one on a tie), ``"min"``,
        ``"max"``, ``"sum"``, ``"rms"`` (NaN-aware), or ``"drop"`` to leave the
        variable out. ``"first"`` and ``"last"`` follow the input order of the
        samples; they work for any dtype, the others for numbers (``"min"`` and
        ``"max"`` also for times). Floating point values keep their dtype,
        integer and other values reduced by ``"mean"``, ``"sum"`` or
        ``"rms"`` come out as float64.
    lon, lat : str
        Names of the longitude and latitude variables, in degrees. They are
        left out of the output unless ``methods`` names them.
    default_method : str, optional
        Method for the variables of the sample dimension that ``methods`` does
        not name. Without it such a variable is an error, so that no variable is
        lost or reduced by accident.
    dtypes : mapping of str to dtype, optional
        Output dtype of a variable, e.g. ``"float32"`` for an integer variable
        reduced by ``"mean"`` (the reduced values are cast to it).
    ellipsoid : str
        Ellipsoid of the cells (``"WGS84"`` or ``"sphere"``).
    count_name : str
        Name of the variable that holds the number of samples of each cell.
    cell_dim, cell_ids_name : str
        Names of the cell dimension and of the cell ids coordinate.

    Returns
    -------
    xarray.Dataset
        Each variable of the sample dimension is reduced to one value per cell:
        its sample dimension is replaced by ``cell_dim``, in place (other
        dimensions are carried through as they are). Variables without the sample
        dimension are copied unchanged. ``count_name`` (int32) holds the number of
        samples of each cell. The attributes of the dataset and of the variables
        are kept; ``attrs["n_samples_input"]`` and ``attrs["n_samples_binned"]``
        count the samples read and the samples with a valid position (the sum of
        ``count_name``), so that none is lost silently.

    Raises
    ------
    ValueError
        For an unknown method, a variable without a method (and no
        ``default_method``), a method that does not fit a dtype, a method for a
        variable that ``ds`` does not have, or longitudes / latitudes that are not
        1-D on the sample dimension.

    """
    methods = dict(methods)
    for name, method in {**methods, "default_method": default_method}.items():
        if method is not None and method != DROP and method not in METHODS:
            raise ValueError(
                f"unknown method {method!r} for {name!r}: use one of {METHODS} or {DROP!r}"
            )
    if default_method == DROP:
        raise ValueError("default_method must not be 'drop': name what to drop in methods")
    unknown = sorted(set(methods) - set(ds.variables))
    if unknown:
        raise ValueError(f"methods name variables that the dataset does not have: {unknown}")
    for name in (lon, lat):
        if name not in ds.variables or ds[name].dims != (sample_dim,):
            raise ValueError(f"{name!r} must be a variable of the single dimension {sample_dim!r}")

    longitude = ds[lon].values.astype("float64")
    latitude = ds[lat].values.astype("float64")
    n_input = longitude.size
    valid = (
        np.isfinite(longitude)
        & np.isfinite(latitude)
        & (np.abs(latitude) <= 90.0)
        & (np.abs(longitude) <= 360.0)
    )
    keep = np.nonzero(valid)[0]
    cells = _cell_ids(longitude[keep], latitude[keep], level, ellipsoid)

    # stable: the samples of a cell stay in input order, "first" is the first sample
    order = np.argsort(cells, kind="stable")
    sorted_cells = cells[order]
    unique, starts, counts = np.unique(sorted_cells, return_index=True, return_counts=True)
    order_in_ds = keep[order]  # positions in the input, sorted by cell

    out: dict[str, xr.Variable] = {}
    skipped = {lon, lat}
    for name, var in ds.variables.items():
        name = str(name)
        if sample_dim not in var.dims:
            continue
        if name in methods:
            method = methods[name]
        elif name in skipped:
            continue
        elif name == sample_dim:  # an index coordinate of the samples
            continue
        elif default_method is not None:
            method = default_method
        else:
            raise ValueError(
                f"variable {name!r} has dimension {sample_dim!r} and no method: "
                f"add it to methods (or {DROP!r}), or give default_method"
            )
        if method == DROP:
            continue
        axis = var.dims.index(sample_dim)
        data = np.moveaxis(var.values, axis, 0)
        # reduce on the valid samples only, sorted by cell
        reduced = _reduce(name, method, data[order_in_ds], starts, counts)
        dims = list(var.dims)
        dims[axis] = cell_dim
        reduced = np.moveaxis(reduced, 0, axis)
        if dtypes and name in dtypes:
            reduced = reduced.astype(dtypes[name])
        out[name] = xr.Variable(tuple(dims), reduced, attrs=dict(var.attrs))

    result = xr.Dataset(out, attrs=dict(ds.attrs))
    result = result.assign_coords({cell_ids_name: (cell_dim, unique.astype("uint64"))})
    result[count_name] = xr.Variable(
        (cell_dim,),
        counts.astype("int32"),
        attrs={"long_name": "number of input samples in the cell", "units": "1"},
    )

    # what has no sample dimension: carried through
    for name, var in ds.variables.items():
        if sample_dim in var.dims or name in result.variables:
            continue
        if name in methods and methods[name] == DROP:
            continue
        result[str(name)] = var

    result.attrs["n_samples_input"] = int(n_input)
    result.attrs["n_samples_binned"] = int(counts.sum())
    return result


def split_by_parent(
    ds: xr.Dataset,
    *,
    level: int,
    parent_level: int,
    cell_dim: str = CELL_DIM,
    cell_ids_name: str = CELL_IDS,
) -> dict[int, xr.Dataset]:
    """Cut a dataset on sparse nested cells into one dataset per parent cell.

    Parameters
    ----------
    ds : xarray.Dataset
        A dataset on ``cell_dim`` with the nested cell ids of ``level`` (e.g. the
        result of :py:func:`bin_samples`).
    level : int
        Refinement level of the cell ids.
    parent_level : int
        Refinement level of the parents (``<= level``).

    Returns
    -------
    dict
        Parent cell id -> the dataset of the cells inside it, in increasing order
        of the parent id. Variables without the cell dimension are copied into
        each dataset. The cells of the datasets are the cells of ``ds``, each in
        exactly one of them.

    """
    if not 0 <= parent_level <= level:
        raise ValueError(f"parent_level must be between 0 and level={level}, got {parent_level}")
    cell_ids = ds[cell_ids_name].values.astype("uint64")
    parents = cell_ids >> np.uint64(2 * (level - parent_level))
    out: dict[int, xr.Dataset] = {}
    for parent in np.unique(parents):
        positions = np.nonzero(parents == parent)[0]
        out[int(parent)] = ds.isel({cell_dim: positions})
    return out


def write_cell_dataset(
    ds: xr.Dataset,
    store: Any,
    *,
    level: int,
    group: str | None = None,
    ellipsoid: str = "WGS84",
    metadata: MetadataSettings | None = None,
    root_attrs: Mapping[str, Any] | None = None,
    cell_dim: str = CELL_DIM,
    cell_ids_name: str = CELL_IDS,
    cell_axis: str = "last",
    chunks: int | None = None,
    consolidate: bool = True,
    storage_options: Mapping[str, Any] | None = None,
) -> None:
    """Write a dataset on sparse nested cells as a Zarr (v3) store with the DGGS
    metadata of the converters.

    The group gets the ``dggs`` attribute (and the ``zarr_conventions`` entry)
    that :py:class:`~healpix_convert.settings.common.MetadataSettings` selects, the
    cell ids coordinate and the data variables get the CF attributes it selects
    (``standard_name``, ``grid_mapping``), and the root group the CF
    ``Conventions`` attribute. The cell ids are listed in the ``cell_ids``
    coordinate (the compression of the ids is ``"none"``).

    Parameters
    ----------
    ds : xarray.Dataset
        A dataset on ``cell_dim`` with the coordinate ``cell_ids_name``.
    store : path, URL or Zarr store
        Where to write (an existing store is replaced).
    level : int
        Refinement level of the cell ids.
    group : str, optional
        Group of the store to write the dataset into (the root group if None).
    ellipsoid : str
        ``"WGS84"`` or ``"sphere"`` (the latter declares no ellipsoid).
    metadata : MetadataSettings, optional
        The conventions to declare (all of the defaults if None).
    root_attrs : mapping, optional
        Attributes for the root group (e.g. provenance).
    cell_axis : {"last", "first"}
        Where the cell dimension goes in the arrays: ``"last"`` (the default) as
        in the stores of the converters, the other dimensions first;
        ``"first"`` keeps the order of the dataset.
    chunks : int, optional
        Chunk size along the cell dimension (all the cells in one chunk if None).
    consolidate : bool
        Write consolidated metadata (root and group).
    storage_options : mapping, optional
        fsspec options for a remote ``store`` URL (e.g. ``s3://``).

    """
    import zarr

    if cell_axis not in ("last", "first"):
        raise ValueError(f"cell_axis must be 'last' or 'first', got {cell_axis!r}")
    metadata = metadata if metadata is not None else MetadataSettings()
    grid: dict[str, Any] = {}
    if ellipsoid.upper() == "WGS84":
        grid["ellipsoid"] = WGS84Ellipsoid()
    healpix = Healpix(
        refinement_level=level,
        indexing_scheme="nested",
        spatial_dimension=cell_dim,
        coordinate=cell_ids_name,
        compression="none",
        **grid,
    )

    ds = ds.copy()
    if cell_axis == "last":
        ds = ds.transpose(..., cell_dim)
    cf_data = get_cf_data_variable_attrs(healpix, metadata)
    cf_data.pop("coordinates", None)  # xarray writes it
    for name, var in ds.data_vars.items():
        if cell_dim in var.dims:
            var.attrs.update(cf_data)
    ds[cell_ids_name].attrs.update(get_cf_cell_id_attrs(metadata))
    grid_mapping = get_cf_grid_mapping_attrs(healpix, metadata)
    if grid_mapping is not None:
        ds[CF_GRID_MAPPING_VARIABLE] = xr.DataArray(np.int8(0), attrs=grid_mapping)
    ds.attrs.update(get_healpix_group_attrs(healpix, metadata))

    encoding = {}
    if chunks is not None:
        for name, var in ds.variables.items():
            if cell_dim in var.dims:
                encoding[str(name)] = {
                    "chunks": tuple(min(chunks, s) if d == cell_dim else s for d, s in zip(var.dims, var.shape))
                }
    ds.to_zarr(
        store,
        group=group,
        mode="w",
        zarr_format=3,
        consolidated=False,
        encoding=encoding or None,
        storage_options=dict(storage_options) if storage_options else None,
    )

    root = zarr.open_group(store, mode="r+", storage_options=dict(storage_options) if storage_options else None)
    write_root_conventions(root, metadata)
    if root_attrs:
        root.attrs.update(dict(root_attrs))
    if consolidate:
        if group:
            zarr.consolidate_metadata(root.store, path=group)
        zarr.consolidate_metadata(root.store)


def count_samples(datasets: Sequence[xr.Dataset], count_name: str = SAMPLE_COUNT) -> int:
    """Total of the ``count_name`` variable over several datasets (e.g. the tiles)."""
    return int(sum(int(d[count_name].sum()) for d in datasets))
