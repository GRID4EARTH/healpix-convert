"""
Aggregating HEALPix data onto the cells of a coarser refinement level.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Literal, TypeAlias

import numpy as np
import xarray as xr

from healpix_convert.core.healpix_conventions import Healpix

#: Supported methods for aggregating the values of the children cells. These
#: are the reductions that Xarray's coarsen applies to a block of cells and
#: that give, for a whole block, the same result as aggregating all its cells
#: at once (unlike, e.g., the median).
ReduceMethod: TypeAlias = Literal["mean", "min", "max", "sum"]

#: Encoding of an input variable reused for the aggregated ones: how the values
#: are stored (data type, fill value, CF packing), not how the chunks are
#: encoded.
VALUE_ENCODING = ("dtype", "fill_value", "_FillValue", "scale_factor", "add_offset")


def select_source_levels(
    variables: Mapping[int, Collection[str]],
    level: int,
    *,
    coarsen_from: str = "nearest",
) -> dict[str, int]:
    """Converted refinement level each variable of a derived level is
    aggregated from (variable name -> converted level).

    ``variables`` gives the variables of every level converted from the input
    data (converted level -> variable names).

    A variable is absent from the levels finer than its finest converted level
    (it is never upsampled). A variable converted at several levels, e.g., the
    copies of the finest bands that the provider of a multi-resolution product
    resampled itself, is aggregated from the nearest converted level at or
    finer than ``level`` (``"nearest"``) or from its finest converted level
    (``"finest"``). The variables of a derived level may therefore be
    aggregated from different converted levels.

    """
    finer = {lvl: set(names) for lvl, names in variables.items() if lvl >= level}
    names = {name for level_names in finer.values() for name in level_names}

    sources: dict[str, int] = {}

    for name in sorted(names):
        converted = [lvl for lvl in sorted(finer) if name in finer[lvl]]
        sources[name] = min(converted) if coarsen_from == "nearest" else max(converted)

    return sources


def coarsen_healpix_group(
    ds: xr.Dataset,
    level: int,
    *,
    healpix: Healpix,
    reduce: ReduceMethod = "mean",
) -> xr.Dataset:
    """Aggregate the data variables of a HEALPix group onto the cells of a
    coarser refinement level.

    The cells of ``ds`` must densely cover whole HEALPix cells of ``level``, in
    ascending (nested) order, as the ``healpix_cell_dense`` chunks of a
    converted group do: the cells of a coarser cell are then ``4 ** delta``
    consecutive cells, so they are aggregated by position. Nothing is grouped
    and the cell ids are not read: the aggregation is blockwise, hence lazy and
    bounded in memory.

    Each cell of the output aggregates all the cells of the input it contains
    (and no other), i.e., the mean is a block mean and not a mean of the means
    of the intermediate levels.

    Parameters
    ----------
    ds : :py:class:`xarray.Dataset`
        One (single-scale) HEALPix group of a converted dataset.
    level : int
        HEALPix refinement level of the output (coarser than the level of
        ``ds``).
    healpix : object
        HEALPix grid of ``ds`` (a ``Healpix`` object): its refinement level,
        cell dimension and cell ids coordinate.
    reduce : str, optional
        How the values of the children cells are aggregated (default: "mean").
        Missing (NaN) values are skipped.

    Returns
    -------
    :py:class:`xarray.Dataset`
        The group at the coarser refinement level, with its ``dggs`` attribute
        updated and the data type of its variables preserved.

    """
    assert healpix.indexing_scheme == "nested"
    assert healpix.refinement_level is not None
    assert isinstance(healpix.coordinate, str)

    delta = healpix.refinement_level - level
    assert delta > 0

    dim = healpix.spatial_dimension
    coordinate = healpix.coordinate
    window = 4**delta
    # the cells of the group cover whole HEALPix cells of the coarser level
    # (checked against the spatial extent of the output before any level is built)
    assert ds.sizes[dim] % window == 0

    cell_vars = [name for name, var in ds.data_vars.items() if dim in var.dims]
    # one block of cells per cell of the coarser level
    blocks = ds[cell_vars].coarsen({dim: window}, coord_func={coordinate: "min"})
    reduce_blocks = getattr(blocks, reduce)
    coarsened = reduce_blocks(keep_attrs=True)

    # the smallest cell id of a block is the first child of the coarser cell
    coarsened = coarsened.assign_coords(
        {coordinate: coarsened[coordinate] >> np.uint64(2 * delta)}
    )

    # add back the variables that are not defined on the cells (e.g., the CF
    # grid mapping) and the attributes, with the new refinement level
    coarsened = coarsened.merge(ds.drop_dims(dim), combine_attrs="override")
    coarsened.attrs = dict(ds.attrs) | {
        "dggs": healpix.model_copy(update={"refinement_level": level}).model_dump()
    }

    return encode_like(coarsened, ds.variables)


def encode_like(ds: xr.Dataset, templates: Mapping[str, xr.Variable]) -> xr.Dataset:
    """Give the variables of ``ds`` the data type and the fill value of the
    matching template (input) variables.

    Aggregating cells returns floating point values: an integer variable is
    rounded (instead of being truncated) and gets its fill value back where it
    has no value.

    The codecs of the input are not reused: the derived levels have their own
    cells and chunks, which codecs like the healpix-compress ones (dense chunks
    of one parent cell) could not encode.

    """
    ds = ds.copy()

    for name, template in templates.items():
        if name not in ds.variables:
            continue

        encoding = {k: v for k, v in template.encoding.items() if k in VALUE_ENCODING}
        # xarray would otherwise declare NaN as the missing value of the
        # floating point variables, which the converted dataset does not do
        encoding.setdefault("_FillValue", None)

        if ds[name].dtype != template.dtype:
            values = ds[name]
            if np.issubdtype(template.dtype, np.integer):
                fill_value = encoding.get("fill_value")
                values = values.fillna(0 if fill_value is None else fill_value).round()
            ds[name] = values.astype(template.dtype)

        ds[name].encoding = encoding

    return ds
