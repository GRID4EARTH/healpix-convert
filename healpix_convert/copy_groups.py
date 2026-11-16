"""Copy of the non-spatial input groups (``copy_groups`` setting)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePath
from typing import Any

import numpy as np
import structlog
import xarray as xr

from healpix_convert.core.utils import matches_any, retry_store_conflicts
from healpix_convert.settings.common import CopyGroupSettings

log = structlog.get_logger()

#: attributes that say how the stored values are read: they must be the same in
#: every input for a group to be copied once
READING_ATTRS = ("scale_factor", "add_offset", "_FillValue", "missing_value", "units", "calendar")


def _raw_group(input_path: Any, path: PurePath, settings: CopyGroupSettings) -> xr.Dataset:
    """The group of an input dataset as stored: no CF decoding (no masking, no
    scaling, no time decoding), so that writing it back gives the same values."""
    try:
        ds = xr.open_dataset(
            input_path,
            group=path.as_posix(),
            engine="zarr",
            decode_cf=False,
            chunks=None,
            create_default_indexes=False,
        )
    except Exception as err:
        raise ValueError(
            f"copied group {path.as_posix()!r}: cannot read it from the input "
            f"{input_path!r} ({err})"
        ) from err
    names = [str(k) for k in ds.variables if matches_any(str(k), settings.drop_variables)]
    ds = ds.drop_vars(names) if names else ds
    for name, var in ds.variables.items():
        if var.dtype.kind in "OSUV":
            raise ValueError(
                f"copied group {path.as_posix()!r}: variable {name!r} has the data "
                f"type {var.dtype} (strings and other non-numeric types are not "
                "supported): leave it out with 'drop_variables'"
            )
    return ds.load()


def _equal(x: Any, y: Any) -> bool:
    """Equal values or arrays, NaN equal to NaN."""
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        return False
    nan = x.dtype.kind in "fc" and y.dtype.kind in "fc"
    return bool(np.array_equal(x, y, equal_nan=nan))


def _same(a: xr.Variable, b: xr.Variable) -> bool:
    if a.dims != b.dims or a.dtype != b.dtype or a.shape != b.shape:
        return False
    for key in READING_ATTRS:
        if (key in a.attrs) != (key in b.attrs):
            return False
        if key in a.attrs and not _equal(a.attrs[key], b.attrs[key]):
            return False
    return _equal(a.values, b.values)


def _differing_variables(ds: xr.Dataset, other: xr.Dataset) -> list[str]:
    names = set(map(str, ds.variables)) ^ set(map(str, other.variables))
    for name in set(map(str, ds.variables)) & set(map(str, other.variables)):
        if not _same(ds[name].variable, other[name].variable):
            names.add(name)
    return sorted(names)


def copy_group_dataset(
    input_paths: Sequence[Any],
    path: PurePath,
    settings: CopyGroupSettings,
) -> xr.Dataset:
    """Return the dataset of the input group ``path`` to copy to the output
    dataset: the group of the first input as stored (no CF decoding), without
    the dropped variables, ready to be written back unchanged.

    Raise a ValueError if the group is not the same in every input (variables,
    dimensions, data types, stored values or the attributes that say how they
    are read, :py:data:`READING_ATTRS`; the other attributes are taken from the
    first input), or if it holds a non-numeric variable.
    """
    ds = _raw_group(input_paths[0], path, settings)

    for index, input_path in enumerate(input_paths[1:], start=1):
        other = _raw_group(input_path, path, settings)
        differing = _differing_variables(ds, other)
        if differing:
            raise ValueError(
                f"copied group {path.as_posix()!r} differs between the input "
                f"datasets 0 and {index} (variables {differing}): it can only be "
                "copied if it is the same in every input (leave the differing "
                "variables out with 'drop_variables')"
            )

    # the fill value goes back to where xarray writes it from; the packing and
    # missing_value attributes stay attributes: the stored values are written
    # as they are
    for var in ds.variables.values():
        fill = var.attrs.pop("_FillValue", None)
        var.encoding = {"dtype": var.dtype, "_FillValue": fill}

    return ds


def write_copy_group(
    output_store: Any,
    group_path: str,
    input_paths: Sequence[Any],
    path: PurePath,
    settings: CopyGroupSettings,
) -> None:
    """Write the input group ``path`` unchanged to ``group_path`` of the output
    store (see :py:func:`copy_group_dataset`)."""
    ds = copy_group_dataset(input_paths, path, settings)
    log.info(
        f"••• copying non-spatial group '{group_path}' to zarr "
        f"({len(ds.variables)} variables)"
    )
    retry_store_conflicts(ds.to_zarr)(
        output_store,
        group=group_path,
        mode="a",
        consolidated=False,
        zarr_format=3,
    )
