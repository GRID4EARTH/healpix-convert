"""
Aggregating HEALPix data onto the cells of a coarser refinement level.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
import xarray as xr

from healpix_convert.core.healpix_conventions import Healpix

#: Supported methods for aggregating the values of the children cells. These
#: are the reductions that Xarray's coarsen applies to a block of cells and
#: that give, for a whole block, the same result as aggregating all its cells
#: at once (unlike, e.g., the median). ``"mode"`` (the most frequent value) is
#: for categorical variables: masks, classifications.
ReduceMethod: TypeAlias = Literal["mean", "min", "max", "sum", "mode"]

#: Encoding of an input variable reused for the aggregated ones: how the values
#: are stored (data type, fill value, CF packing), not how the chunks are
#: encoded.
VALUE_ENCODING = ("dtype", "fill_value", "_FillValue", "scale_factor", "add_offset")


def nanmode(values: np.ndarray, axis: int | tuple[int, ...] = -1) -> np.ndarray:
    """Most frequent value along ``axis``, missing (NaN) values skipped.

    A tie goes to the smallest of the most frequent values, so that the result
    does not depend on the order of the cells. All values missing gives NaN.

    """
    if isinstance(axis, tuple):
        (axis,) = axis
    values = np.moveaxis(np.asarray(values), axis, -1)
    if values.shape[-1] == 0:
        return np.full(values.shape[:-1], np.nan)

    ordered = np.sort(values, axis=-1)  # NaN last
    n = ordered.shape[-1]
    position = np.broadcast_to(np.arange(n), ordered.shape)
    run_start = np.ones(ordered.shape, dtype=bool)
    run_start[..., 1:] = ordered[..., 1:] != ordered[..., :-1]
    # within a run of equal values, how many of them up to this position
    count = (
        position - np.maximum.accumulate(np.where(run_start, position, 0), axis=-1) + 1
    )
    if np.issubdtype(ordered.dtype, np.floating):
        count = np.where(np.isnan(ordered), 0, count)

    # the first maximum is the end of the earliest longest run: the smallest
    # of the most frequent values
    best = np.argmax(count, axis=-1)
    mode = np.take_along_axis(ordered, best[..., None], axis=-1)[..., 0]

    if np.issubdtype(ordered.dtype, np.floating):
        mode = np.where(count.max(axis=-1) == 0, np.nan, mode)

    return mode


def _blocks_mode(var: xr.DataArray, dim: str, window: int) -> xr.DataArray:
    """Most frequent value of each block of ``window`` consecutive cells,
    computed blockwise (lazily on Dask arrays), like Xarray's own reductions.
    """
    var = var.reset_coords(drop=True)
    windows = var.coarsen({dim: window}).construct(
        {dim: (dim, "window")}, keep_attrs=True
    )
    if windows.chunks is not None:
        windows = windows.chunk({"window": -1})

    return xr.apply_ufunc(
        nanmode,
        windows,
        input_core_dims=[["window"]],
        kwargs={"axis": -1},
        dask="parallelized",
        output_dtypes=[var.dtype],
        keep_attrs=True,
    )


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
        How the values of the children cells are aggregated (default: "mean");
        "mode" gives their most frequent value (see :py:func:`nanmode`).
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
    if reduce == "mode":
        # the structure and the coordinates of the blocks (their values are
        # replaced below, so this reduction is never computed)
        coarsened = blocks.min(keep_attrs=True)
        for name in cell_vars:
            coarsened[name] = _blocks_mode(ds[name], dim, window)
    else:
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


def codecs_encoding(codecs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Xarray's Zarr encoding of a codec pipeline given as codec settings.

    ``to_zarr`` takes a pipeline as ``filters`` (array to array codecs), one
    ``serializer`` (array to bytes) and ``compressors`` (bytes to bytes), not
    as one list: each codec is created from Zarr's codec registry and goes to
    its part.

    """
    from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec
    from zarr.registry import get_codec_class

    encoding: dict[str, Any] = {"filters": [], "compressors": []}

    for codec_settings in codecs:
        cls = get_codec_class(codec_settings["name"])
        # an empty configuration gives the codec's defaults (e.g. the byte
        # order of the "bytes" serializer)
        codec = (
            cls.from_dict(dict(codec_settings))
            if codec_settings.get("configuration")
            else cls()
        )
        if isinstance(codec, ArrayArrayCodec):
            encoding["filters"].append(codec)
        elif isinstance(codec, ArrayBytesCodec):
            encoding["serializer"] = codec
        elif isinstance(codec, BytesBytesCodec):
            encoding["compressors"].append(codec)
        else:
            raise TypeError(f"unknown kind of zarr codec: {codec!r}")

    return encoding


def encode_like(ds: xr.Dataset, templates: Mapping[str, xr.Variable]) -> xr.Dataset:
    """Give the variables of ``ds`` the data type and the fill value of the
    matching template (input) variables.

    Aggregating cells returns floating point values: an integer variable is
    rounded (instead of being truncated) and gets its fill value back where it
    has no value.

    The codecs of the input are not reused: the derived levels have their own
    cells and chunks, and the codecs of a lossy serializer would code the
    aggregated values a second time. They are set by the ``codecs`` pyramid
    setting (see :py:func:`codecs_encoding`).

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
