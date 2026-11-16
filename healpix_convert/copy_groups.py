"""Copy of the non-spatial input groups (``copy_groups`` setting)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePath
from typing import Any

import structlog
import xarray as xr

from healpix_convert.core.utils import matches_any
from healpix_convert.settings.common import CopyGroupSettings

log = structlog.get_logger()

#: encoding of an input variable kept in the copy: how its values are stored
#: (data type, packing, fill value, time units). The chunks and the codecs of
#: the input are not kept (they may be Zarr v2 codecs); a copied array is
#: written in one chunk, with the default codecs of the output.
KEPT_ENCODING = ("dtype", "scale_factor", "add_offset", "_FillValue", "units", "calendar")


def _group_dataset(datatree: xr.DataTree, path: PurePath, settings: CopyGroupSettings) -> xr.Dataset:
    ds = datatree[path.as_posix()].to_dataset(inherit=False)
    names = [str(k) for k in ds.variables if matches_any(str(k), settings.drop_variables)]
    return ds.drop_vars(names) if names else ds


def _differing_variables(ds: xr.Dataset, other: xr.Dataset) -> list[str]:
    names = set(map(str, ds.variables)) ^ set(map(str, other.variables))
    for name in set(map(str, ds.variables)) & set(map(str, other.variables)):
        if not ds[name].variable.equals(other[name].variable):
            names.add(name)
    return sorted(names)


def copy_group_dataset(
    datatrees: Sequence[xr.DataTree],
    path: PurePath,
    settings: CopyGroupSettings,
) -> xr.Dataset:
    """Return the dataset of the input group ``path`` to copy to the output
    dataset: the group of the first input, without the dropped variables, its
    values loaded and its encoding reduced to :py:data:`KEPT_ENCODING`.

    Raise a ValueError if the group is not the same in every input (variables,
    dimensions or values; attributes are taken from the first input).
    """
    ds = _group_dataset(datatrees[0], path, settings).load()

    for index, datatree in enumerate(datatrees[1:], start=1):
        other = _group_dataset(datatree, path, settings).load()
        differing = _differing_variables(ds, other)
        if differing:
            raise ValueError(
                f"copied group {path.as_posix()!r} differs between the input "
                f"datasets 0 and {index} (variables {differing}): it can only be "
                "copied if it is the same in every input (leave the differing "
                "variables out with 'drop_variables')"
            )

    for var in ds.variables.values():
        var.encoding = {k: v for k, v in var.encoding.items() if k in KEPT_ENCODING}

    return ds


def write_copy_group(
    output_store: Any,
    group_path: str,
    datatrees: Sequence[xr.DataTree],
    path: PurePath,
    settings: CopyGroupSettings,
) -> None:
    """Write the input group ``path`` unchanged to ``group_path`` of the output
    store (see :py:func:`copy_group_dataset`)."""
    ds = copy_group_dataset(datatrees, path, settings)
    log.info(
        f"••• copying non-spatial group '{group_path}' to zarr "
        f"({len(ds.variables)} variables)"
    )
    ds.to_zarr(
        output_store,
        group=group_path,
        mode="a",
        consolidated=False,
        zarr_format=3,
    )
