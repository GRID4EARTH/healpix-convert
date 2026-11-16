"""Copy of the non-spatial input groups (``copy_groups`` setting).

The group is copied array by array with Zarr, without xarray: the stored values,
data types, fill values and attributes of the input arrays are written as they
are, with no CF decoding or encoding in between (which would replace NaN by a
fill value, change booleans or datetimes, or round 64-bit integers).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePath
from typing import Any

import numpy as np
import structlog
import zarr

from healpix_convert.core.utils import matches_any, retry_store_conflicts
from healpix_convert.settings.common import CopyGroupSettings

log = structlog.get_logger()

#: the attribute holding the dimension names of a Zarr v2 array (xarray convention)
V2_DIMENSIONS_ATTR = "_ARRAY_DIMENSIONS"


class _Table:
    """A copied array: its stored values and what is needed to write it back."""

    def __init__(self, array: zarr.Array) -> None:
        self.dtype = array.dtype
        self.shape = array.shape
        self.fill_value = array.fill_value
        self.zarr_format = array.metadata.zarr_format
        attrs = dict(array.attrs)
        if self.zarr_format == 2:
            self.dims = tuple(attrs.pop(V2_DIMENSIONS_ATTR, ()))
        else:
            self.dims = tuple(array.metadata.dimension_names or ())
        self.attrs = attrs
        self.values = np.asarray(array[...])

    def output_attrs(self) -> dict[str, Any]:
        """Attributes of the output (Zarr v3) array.

        A Zarr v2 array marks its missing values with its ``fill_value``, which
        xarray reads as ``_FillValue``; in Zarr v3 xarray reads ``_FillValue``
        from the attributes. The fill value of a v2 input is written there, the
        way xarray writes it, so that the output is read as the input.
        """
        attrs = dict(self.attrs)
        # a boolean has no value left to mark a missing one (xarray would read the
        # array as objects): its fill value is not written as _FillValue
        if (
            self.zarr_format == 2
            and self.fill_value is not None
            and self.dtype.kind != "b"
            and "_FillValue" not in attrs
        ):
            from xarray.backends.zarr import FillValueCoder

            attrs["_FillValue"] = FillValueCoder.encode(self.fill_value, self.dtype)
        return attrs


def _values_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    if a.dtype.kind in "mM":  # NaT equal to NaT
        return bool(np.array_equal(a.view("i8"), b.view("i8")))
    return bool(np.array_equal(a, b))


def _attrs_equal(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    for key in a:
        x, y = np.asarray(a[key]), np.asarray(b[key])
        if x.shape != y.shape:
            return False
        nan = x.dtype.kind in "fc" and y.dtype.kind in "fc"
        if not np.array_equal(x, y, equal_nan=nan):
            return False
    return True


def _same(a: _Table, b: _Table) -> bool:
    return (
        a.dims == b.dims
        and _values_equal(a.values, b.values)
        and _attrs_equal(a.attrs, b.attrs)
        and _values_equal(np.asarray(a.fill_value), np.asarray(b.fill_value))
    )


def _read_group(
    input_path: Any, path: PurePath, settings: CopyGroupSettings
) -> tuple[dict[str, Any], dict[str, _Table]]:
    """The attributes and the arrays (as stored) of a group of an input dataset,
    without the dropped variables."""
    try:
        group = zarr.open_group(input_path, mode="r")[path.as_posix()]
    except Exception as err:
        raise ValueError(
            f"copied group {path.as_posix()!r}: cannot read it from the input "
            f"{input_path!r} ({err})"
        ) from err
    tables = {}
    for name, array in sorted(group.arrays(), key=lambda item: item[0]):
        if matches_any(name, settings.drop_variables):
            continue
        if array.dtype.kind in "OSUVT":
            raise ValueError(
                f"copied group {path.as_posix()!r}: variable {name!r} has the data "
                f"type {array.dtype} (strings and other non-numeric types are not "
                "supported): leave it out with 'drop_variables'"
            )
        tables[name] = _Table(array)
    return dict(group.attrs), tables


def copy_group_tables(
    input_paths: Sequence[Any],
    path: PurePath,
    settings: CopyGroupSettings,
) -> tuple[dict[str, Any], dict[str, _Table]]:
    """Return the attributes and the arrays of the input group ``path`` to copy:
    those of the first input, without the dropped variables.

    Raise a ValueError if the group is not the same in every input: the same
    arrays, with the same dimensions, data types, stored values, fill values and
    attributes (the attributes of the group itself are taken from the first
    input), or if it holds a non-numeric array.
    """
    attrs, tables = _read_group(input_paths[0], path, settings)

    for index, input_path in enumerate(input_paths[1:], start=1):
        _, other = _read_group(input_path, path, settings)
        differing = set(tables) ^ set(other)
        differing |= {n for n in set(tables) & set(other) if not _same(tables[n], other[n])}
        if differing:
            raise ValueError(
                f"copied group {path.as_posix()!r} differs between the input "
                f"datasets 0 and {index} (variables {sorted(differing)}): it can "
                "only be copied if it is the same in every input (leave the "
                "differing variables out with 'drop_variables')"
            )

    return attrs, tables


def _write(output_store: Any, group_path: str, attrs: dict, tables: dict[str, _Table]) -> None:
    root = zarr.open_group(output_store, mode="a")
    group = root.require_group(group_path)
    group.attrs.update(attrs)
    for name, table in tables.items():
        array = group.create_array(
            name,
            shape=table.shape,
            dtype=table.dtype,
            # one chunk (at least one element along each dimension)
            chunks=tuple(max(1, n) for n in table.shape),
            fill_value=table.fill_value,
            dimension_names=table.dims or None,
            attributes=table.output_attrs(),
            overwrite=True,
        )
        if table.values.size:
            array[...] = table.values


def write_copy_group(
    output_store: Any,
    group_path: str,
    input_paths: Sequence[Any],
    path: PurePath,
    settings: CopyGroupSettings,
) -> None:
    """Write the input group ``path`` unchanged to ``group_path`` of the output
    store (see :py:func:`copy_group_tables`)."""
    attrs, tables = copy_group_tables(input_paths, path, settings)
    log.info(
        f"••• copying non-spatial group '{group_path}' to zarr ({len(tables)} arrays)"
    )
    retry_store_conflicts(_write)(output_store, group_path, attrs, tables)
