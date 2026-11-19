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


def valid_positions(lon, lat) -> np.ndarray:
    """Which samples have a usable position: finite longitude and latitude, ``|lat| <= 90`` and
    ``|lon| <= 360`` (degrees). The one rule of :py:func:`bin_samples`; use it to predict which
    samples it will bin."""
    lon = np.asarray(lon, dtype="float64")
    lat = np.asarray(lat, dtype="float64")
    return np.isfinite(lon) & np.isfinite(lat) & (np.abs(lat) <= 90.0) & (np.abs(lon) <= 360.0)


def _pick(flat: np.ndarray, ok: np.ndarray, starts: np.ndarray, which: str):
    """First / last row of each group whose value is ``ok`` (per column); ``(values, found)``."""
    n, n_columns = flat.shape
    index = np.arange(n)[:, None]
    if which == "first":
        position = np.minimum.reduceat(np.where(ok, index, n), starts, axis=0)
        found = position < n
    else:
        position = np.maximum.reduceat(np.where(ok, index, -1), starts, axis=0)
        found = position >= 0
    out = flat[np.where(found, position, 0), np.arange(n_columns)[None, :]]
    return out, found


def _check_fill(name: str, fill, dtype) -> None:
    """The fill value must be a value of ``dtype`` (a clear error, not a raw OverflowError)."""
    dtype = np.dtype(dtype)
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        try:
            ok = int(fill) == fill and info.min <= int(fill) <= info.max
        except (TypeError, ValueError, OverflowError):
            ok = False
        if not ok:
            raise ValueError(
                f"variable {name!r}: the fill value {fill!r} is not a value of {dtype} [{info.min}, {info.max}]"
            )
    elif dtype.kind == "b" and fill not in (0, 1):
        raise ValueError(f"variable {name!r}: the fill value {fill!r} of a bool variable must be 0 or 1")


def _cast(name: str, values: np.ndarray, dtype, fill) -> np.ndarray:
    """Cast reduced values to ``dtype``.

    To an integer: the fill value must fit ``dtype``; a float is rounded, NaN becomes ``fill``;
    Inf, a value out of the range of the type, and a valid value that equals ``fill`` are errors
    (never wrapped, never confused with the fill); an integer out of the range is an error.
    From an integer to a float: the fill value (a cell without a valid sample) becomes NaN.
    """
    dtype = np.dtype(dtype)
    kind = values.dtype.kind
    if dtype.kind in "iu":
        if fill is not None:
            _check_fill(name, fill, dtype)
        info = np.iinfo(dtype)
        if kind == "f":
            nan = np.isnan(values)
            if nan.any() and fill is None:
                raise ValueError(
                    f"variable {name!r}: NaN cannot be cast to {dtype}: give its fill value in fill_values"
                )
            rounded = np.where(nan, 0, np.rint(np.where(np.isinf(values), np.nan, values)))
            # the bounds as floats: 2**(bits-1) / 2**bits are exact, float(info.max) is not
            upper = 2.0 ** (info.bits - 1) if dtype.kind == "i" else 2.0**info.bits
            lower = -(2.0 ** (info.bits - 1)) if dtype.kind == "i" else 0.0
            bad = np.isinf(values) | (~nan & ((rounded < lower) | (rounded >= upper)))
            if bad.any():
                raise ValueError(
                    f"variable {name!r}: {int(bad.sum())} value(s) are infinite or out of the range "
                    f"of {dtype} [{info.min}, {info.max}] after rounding"
                )
            # integer domain from here on: the fill is written as an integer, never through a float
            out = rounded.astype(dtype)
            if fill is not None:
                fill_int = dtype.type(int(fill))
                if (out[~nan] == fill_int).any():
                    raise ValueError(
                        f"variable {name!r}: a valid value equals the fill value {fill!r} after the cast to {dtype}"
                    )
                out[nan] = fill_int
            return out
        if kind in "iub" and values.size:
            low, high = int(values.min()), int(values.max())
            if low < info.min or high > info.max:
                raise ValueError(
                    f"variable {name!r}: values [{low}, {high}] are out of the range of {dtype} [{info.min}, {info.max}]"
                )
        return values.astype(dtype)
    if dtype.kind == "f" and kind in "iu" and fill is not None:
        out = values.astype(dtype)
        out[values == fill] = np.nan
        return out
    return values.astype(dtype)


def _reduce(
    name: str,
    method: str,
    sorted_values: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    ok: np.ndarray | None = None,
    fill=None,
) -> np.ndarray:
    """Reduce ``sorted_values`` (samples first, any other shape) per group.

    ``ok`` (same shape, integer values only) marks the valid samples; the others are left out, and
    ``fill`` is the result of a group without a valid sample. Integers are reduced as integers,
    except by mean, sum and rms (float64).

    The samples are sorted by cell (stably, so the samples of a cell stay in
    input order); ``starts`` and ``counts`` are the first position and the size
    of each group. Returns an array with one row per group.
    """
    values = sorted_values
    kind = values.dtype.kind

    if ok is not None and kind in "iu":
        if method in ("mean", "sum", "rms"):
            sorted_values = values = np.where(ok, values.astype("float64"), np.nan)
            kind = "f"
        else:
            flat = values.reshape(values.shape[0], -1)
            okf = ok.reshape(flat.shape)
            if method in ("first", "last"):
                out, found = _pick(flat, okf, starts, method)
            elif method == "mode":
                out, found = _mode(flat, okf, starts, counts, fill), np.ones((starts.size, flat.shape[1]), bool)
            else:
                info = np.iinfo(values.dtype)
                if method == "min":
                    out = np.minimum.reduceat(np.where(okf, flat, info.max), starts, axis=0)
                else:
                    out = np.maximum.reduceat(np.where(okf, flat, info.min), starts, axis=0)
                found = np.add.reduceat(okf.astype("int64"), starts, axis=0) > 0
            out = np.where(found, out, fill).astype(values.dtype)
            return out.reshape((starts.size,) + values.shape[1:])

    if method in ("first", "last"):
        if kind in "fMm":
            # the first / last VALID sample (not NaN / NaT) of each cell and column
            flat = sorted_values.reshape(sorted_values.shape[0], -1)
            ok = ~np.isnan(flat) if kind == "f" else ~np.isnat(flat)
            out, found = _pick(flat, ok, starts, method)
            missing = np.nan if kind == "f" else np.array("NaT", dtype=values.dtype)
            out = np.where(found, out, missing).astype(values.dtype)
            return out.reshape((starts.size,) + values.shape[1:])
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
        out = _mode(flat, ~np.isnan(flat) if kind == "f" else np.ones(flat.shape, bool), starts, counts, np.nan)
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


def _mode(flat: np.ndarray, ok: np.ndarray, starts: np.ndarray, counts: np.ndarray, missing) -> np.ndarray:
    """Most frequent valid value of each group and column (the first one met on a tie);
    ``missing`` where a group has no valid value. ``ok`` marks the valid values."""
    out = flat[starts].copy()
    first_bad = ~ok[starts]
    if first_bad.any():
        out[first_bad] = missing
    for group in np.nonzero(counts > 1)[0]:
        sl = slice(starts[group], starts[group] + counts[group])
        block, valid = flat[sl], ok[sl]
        k, n_columns = block.shape
        if k <= 64:
            eq = (block[:, None, :] == block[None, :, :]) & valid[:, None, :] & valid[None, :, :]
            frequency = eq.sum(axis=1)  # (k, n_columns); 0 for an invalid sample
            best = frequency.argmax(axis=0)
            picked = block[best, np.arange(n_columns)]
            none = frequency.max(axis=0) == 0
            out[group] = np.where(none, missing, picked) if none.any() else picked
        else:
            for column in range(n_columns):
                values = block[valid[:, column], column]
                if values.size == 0:
                    out[group, column] = missing
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
    fill_values: Mapping[str, Any] | None = None,
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
        samples and take the first / last VALID sample (not NaN, not NaT, not the fill
        value) of each cell and column, like the other methods; they work for any dtype, the others for numbers (``"min"`` and
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
    fill_values : mapping of str to number, optional
        Input fill value of a variable (e.g. -127 of an int8 variable): samples equal to it
        are left out of the reduction like NaN. A float keeps its dtype (NaN where nothing is
        valid). With ``"first"``, ``"last"``, ``"mode"``, ``"min"`` and ``"max"`` an integer is
        reduced as an integer (exact for int64 / uint64) and keeps its dtype, with the fill
        value where a cell has no valid sample; an integer output gets ``encoding["_FillValue"]``
        (``write_cell_dataset`` writes it as the CF ``_FillValue`` and as the Zarr fill value);
        a bool variable is reduced as 0 / 1 and returned as bool (a cell without a valid sample
        holds the fill value); with ``"mean"``, ``"sum"`` and ``"rms"`` it is float64 (NaN
        there). Also the value that a float reduction cast by ``dtypes`` to an integer gives
        for NaN (such a cast rounds; a NaN without a fill value, an Inf or a value out of the
        range of the integer type is an error).
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
    keep = np.nonzero(valid_positions(longitude, latitude))[0]
    cells = _cell_ids(longitude[keep], latitude[keep], level, ellipsoid)

    # stable: the samples of a cell stay in input order, "first" is the first sample
    order = np.argsort(cells, kind="stable")
    sorted_cells = cells[order]
    unique, starts, counts = np.unique(sorted_cells, return_index=True, return_counts=True)
    order_in_ds = keep[order]  # positions in the input, sorted by cell
    fill_values = dict(fill_values or {})

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
        fill = fill_values.get(name)
        was_bool = fill is not None and data.dtype.kind == "b"
        if fill is not None:
            _check_fill(name, fill, data.dtype)
        if was_bool:  # reduced as 0 / 1 integers, returned as bool
            data, fill = data.astype("u1"), int(fill)
        sorted_data = data[order_in_ds]  # the valid samples, sorted by cell
        ok = None
        if fill is not None and data.dtype.kind == "f":
            # fill samples are left out like NaN, in the input float dtype
            sorted_data = sorted_data.copy()
            sorted_data[sorted_data == fill] = np.nan
        elif fill is not None and data.dtype.kind in "iu":
            ok = sorted_data != fill
        if starts.size:
            reduced = _reduce(name, method, sorted_data, starts, counts, ok, fill)
        else:  # no valid sample: the reduction of one dummy sample, cut to no cell
            dummy = np.zeros((1,) + data.shape[1:], dtype=data.dtype)
            reduced = _reduce(name, method, dummy, np.array([0]), np.array([1]), None if ok is None else np.ones(dummy.shape, bool), fill)[:0]
        if was_bool and reduced.dtype == np.dtype("u1"):
            reduced = reduced.astype(bool)
        dims = list(var.dims)
        dims[axis] = cell_dim
        reduced = np.moveaxis(reduced, 0, axis)
        if dtypes and name in dtypes:
            reduced = _cast(name, reduced, dtypes[name], None if was_bool else fill)
        out[name] = xr.Variable(tuple(dims), reduced, attrs=dict(var.attrs))
        if fill is not None and reduced.dtype.kind in "iu":
            # the fill value of the integer output (a CF ``_FillValue``: write_cell_dataset also sets
            # the Zarr array fill_value to it), so that generic readers mask it
            out[name].encoding["_FillValue"] = fill

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

    if chunks is not None and chunks < 1:
        raise ValueError(f"chunks must be a positive number of cells, got {chunks}")
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

    for name, var in ds.variables.items():
        enc = dict(var.encoding)
        if "_FillValue" in var.attrs:  # an attribute would clash with the encoding: move it
            enc.setdefault("_FillValue", var.attrs["_FillValue"])
            var.attrs = {k: v for k, v in var.attrs.items() if k != "_FillValue"}
        if "_FillValue" in enc and enc["_FillValue"] is not None and var.dtype.kind in "iu":
            enc["fill_value"] = enc["_FillValue"]  # the Zarr array fill_value agrees with the CF attribute
        if chunks is not None and cell_dim in var.dims:
            enc["chunks"] = tuple(
                max(1, min(chunks, s)) if d == cell_dim else max(1, s) for d, s in zip(var.dims, var.shape)
            )
        var.encoding = enc
    ds.to_zarr(
        store,
        group=group,
        mode="w",
        zarr_format=3,
        consolidated=False,
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
