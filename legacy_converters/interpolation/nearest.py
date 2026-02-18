from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import affine
import numpy as np
import xarray as xr
import xdggs

import legacy_converters.accessor  # noqa: F401
from legacy_converters.crs import create_transformer

if TYPE_CHECKING:
    import pyproj
    import torch


def _nearest_pixels(
    transform: affine.Affine,
    shape: tuple[int, int],
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = target[:, 0]
    y = target[:, 1]

    nx, ny = shape
    pixel_x, pixel_y = ~transform * (x, y)

    indices_x = np.round(np.clip(pixel_x, min=0, max=nx - 1)).astype("int32")
    indices_y = np.round(np.clip(pixel_y, min=0, max=ny - 1)).astype("int32")

    return indices_x, indices_y


def _nearest_affine_torch(
    transform: affine.Affine,
    crs: pyproj.CRS,
    shape: tuple[int, int],
    cell_ids: np.ndarray,
    grid_info: xdggs.HealpixInfo,
    device: str,
) -> torch.Tensor:
    import torch

    crs_transformer = create_transformer(crs, 4326)
    target_lon, target_lat = grid_info.cell_ids2geographic(cell_ids)
    x, y = crs_transformer.transform(target_lon, target_lat, direction="INVERSE")
    target_utm = np.stack([x, y], axis=-1)
    indices_x, indices_y = _nearest_pixels(transform, shape, target_utm)

    target_size = cell_ids.size
    source_size = shape[0] * shape[1]

    columns = torch.Tensor.from_numpy(indices_x).to(device) * shape[
        0
    ] + torch.Tensor.from_numpy(indices_y).to(device)
    rows = torch.arange(target_size, dtype=columns.dtype, device=device)
    coords = torch.stack([rows, columns], axis=-1)
    values = torch.ones_like(rows, dtype="float16", device=device)

    return torch.sparse_coo_tensor(
        coords, values, size=(target_size, source_size)
    ).coalesce()


def _nearest_single_chunk(target_utm, x, y, transform):
    import sparse

    # dask.array.apply_gufunc weirdness
    if isinstance(target_utm, list):
        [target_utm] = target_utm

    source_shape = (x.size, y.size)

    new_transform = transform * affine.Affine.translation(int(x.min()), int(y.min()))

    indices_x, indices_y = _nearest_pixels(new_transform, source_shape, target_utm)

    n_cells = target_utm.shape[0]
    target_shape = (n_cells,)
    rows = np.arange(n_cells)

    weights = sparse.COO(
        coords=[rows, indices_x, indices_y],
        data=np.full_like(rows, dtype="bool", fill_value=True),
        shape=target_shape + source_shape,
        fill_value=False,
    )

    return weights


def _compute_weights_dask(
    cell_ids,
    grid_info,
    transform,
    source_shape,
    crs_transformer,
    chunks,
    source_sizes,
    source_chunks,
):
    import dask
    import dask.array
    import sparse

    def cell_ids2xy(cell_ids, grid_info, crs_transformer):
        lon, lat = grid_info.cell_ids2geographic(cell_ids)
        x, y = crs_transformer.transform(lon, lat, direction="INVERSE")

        return np.stack([x, y], axis=-1)

    cell_ids_ = dask.array.from_array(cell_ids, chunks=chunks["cells"])
    target_utm = dask.array.apply_gufunc(
        cell_ids2xy,
        "()->(j)",
        cell_ids_,
        grid_info=grid_info,
        crs_transformer=crs_transformer,
        output_sizes={"j": 2},
        meta=np.array((), dtype="float64"),
    )

    x = dask.array.arange(source_sizes["x"], chunks=source_chunks["x"])
    y = dask.array.arange(source_sizes["y"], chunks=source_chunks["y"])

    weights = dask.array.blockwise(
        partial(_nearest_single_chunk, transform=transform),
        "ikm",
        target_utm,
        "ij",
        x,
        "k",
        y,
        "m",
        dtype="bool",
        meta=sparse.COO.from_numpy(np.array((), dtype="bool"), fill_value=0),
    )

    return weights


def nearest_affine(
    source_grid: xr.Dataset,
    target_grid: xr.Dataset,
    *,
    chunks=None,
    ignored_dims=None,
    chunk_manager="dask",
) -> xr.DataArray:
    """Nearest-neighbour interpolation weights based on the affine transform

    Parameters
    ----------
    source_grid : xarray.Dataset
        The source grid. Must contain at least one variable with a ``proj:transform`` attribute.
    target_grid : xarray.Dataset
        The target grid. Must have a healpix index.

    Returns
    -------
    weights : xarray.DataArray
        The interpolation weights as a sparse matrix.
    """
    transform = source_grid.grid4earth.affine_transform(kind="center")
    nx = source_grid.sizes["x"]
    ny = source_grid.sizes["y"]
    source_shape = (nx, ny)
    source_chunks = source_grid.chunksizes
    crs = source_grid.grid4earth.crs
    cell_ids = target_grid.dggs.coord.data
    grid_info = target_grid.dggs.grid_info

    crs_transformer = create_transformer(crs, 4326)

    if ignored_dims is None:
        ignored_dims = []

    if chunks is None:
        target_lon, target_lat = grid_info.cell_ids2lonlat(cell_ids)
        x, y = crs_transformer.transform(target_lon, target_lat, direction="INVERSE")
        target_utm = np.stack([x, y], axis=-1)
        weights = _nearest_single_chunk(
            target_utm, np.arange(nx), np.arange(ny), transform
        )
    elif chunk_manager == "dask":
        weights = _compute_weights_dask(
            cell_ids,
            grid_info,
            transform,
            source_shape,
            crs_transformer,
            chunks,
            source_grid.sizes,
            source_chunks,
        )
    else:
        raise ValueError(f"unknown chunk manager: {chunk_manager}")

    source_dims = [dim for dim in source_grid.dims if dim not in ignored_dims]
    target_dims = list(target_grid.dggs.coord.dims)
    return xr.DataArray(
        weights, dims=target_dims + source_dims, coords=source_grid[source_dims].coords
    ).assign_coords(target_grid.dggs.coord.coords)
