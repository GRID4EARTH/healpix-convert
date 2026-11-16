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

    def __init__(self, array: zarr.Array, where: str) -> None:
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
        if self.zarr_format == 2 and self.dtype.kind in "mM" and self.fill_value is not None:
            # xarray reads such a fill value as a missing value, and it cannot be
            # written as a _FillValue attribute: the copy would not read the same
            raise ValueError(
                f"{where}: a Zarr v2 {self.dtype} array with a fill value "
                f"({self.fill_value!r}) cannot be copied: leave it out with "
                "'drop_variables'"
            )
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


def _attrs_equal(a: Any, b: Any) -> bool:
    """Equal attribute values (JSON-like: mappings, lists, numbers, strings,
    arrays), NaN equal to NaN."""
    if isinstance(a, dict) or isinstance(b, dict):
        return (
            isinstance(a, dict)
            and isinstance(b, dict)
            and set(a) == set(b)
            and all(_attrs_equal(a[k], b[k]) for k in a)
        )
    if isinstance(a, (list, tuple, np.ndarray)) or isinstance(b, (list, tuple, np.ndarray)):
        if not (isinstance(a, (list, tuple, np.ndarray)) and isinstance(b, (list, tuple, np.ndarray))):
            return False
        return len(a) == len(b) and all(_attrs_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        return bool(a == b or (np.isnan(a) and np.isnan(b)))
    return type(a) is type(b) and bool(a == b) if isinstance(a, str) or isinstance(b, str) else bool(a == b)


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
        tables[name] = _Table(array, f"copied group {path.as_posix()!r}: variable {name!r}")
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


def read_copy_groups(
    input_paths: Sequence[Any],
    copy_groups: dict[PurePath, CopyGroupSettings],
    input_groups: Sequence[str],
) -> dict[PurePath, tuple[dict[str, Any], dict[str, _Table]]]:
    """Read and check every copied group that the inputs have (``input_groups``,
    the group paths of the inputs), before anything is written: a group that
    cannot be copied stops the conversion before any group is converted."""
    groups = {g.strip("/") for g in input_groups}
    return {
        path: copy_group_tables(input_paths, path, settings)
        for path, settings in copy_groups.items()
        if path.as_posix() in groups
    }


def write_copy_group(
    output_store: Any,
    group_path: str,
    group: tuple[dict[str, Any], dict[str, _Table]],
) -> None:
    """Write a group read by :py:func:`read_copy_groups` unchanged to
    ``group_path`` of the output store."""
    attrs, tables = group
    log.info(
        f"••• copying non-spatial group '{group_path}' to zarr ({len(tables)} arrays)"
    )
    retry_store_conflicts(_write)(output_store, group_path, attrs, tables)
