from __future__ import annotations

from typing import TYPE_CHECKING

import affine
import numpy as np
import xarray as xr
import xdggs

from legacy_converters.crs import create_transformer
# TODO: expand_chunks not in repository (some previous merge mistake?)
# from legacy_converters.interpolation.chunks import expand_chunks

if TYPE_CHECKING:
    import pyproj
    import torch


def _bilinear_pixels(
    transform: affine.Affine,
    crs: pyproj.CRS,
    shape: tuple[int, int],
    cell_ids: np.ndarray,
    grid_info: xdggs.HealpixInfo,
) -> np.ndarray:
    crs_transformer = create_transformer(crs, 4326)

    lon, lat = grid_info.cell_ids2geographic(cell_ids)

    nx, ny = shape
    x, y = crs_transformer.transform(lon, lat, direction="INVERSE")

    pixel_x, pixel_y = ~transform * (x, y)

    valid_points = (
        (pixel_x >= 0) & (pixel_x <= nx - 1) & (pixel_y >= 0) & (pixel_y <= ny - 1)
    )

    cell_id_indices = np.arange(cell_ids.size)[valid_points]

    valid_x = pixel_x[valid_points]
    valid_y = pixel_y[valid_points]

    dx = valid_x - np.astype(valid_x, "int64")
    dy = valid_y - np.astype(valid_y, "int64")

    w11 = (1 - dx) * (1 - dy)
    w12 = (1 - dx) * dy
    w21 = dx * (1 - dy)
    w22 = dx * dy

    raw_weights = np.ravel(np.stack([w11, w12, w21, w22], axis=-1))

    minx, maxx = np.floor(valid_x), np.ceil(valid_x)
    miny, maxy = np.floor(valid_y), np.ceil(valid_y)

    neighbours = np.array(
        [
            [minx, miny],
            [minx, maxy],
            [maxx, miny],
            [maxx, maxy],
        ],
        dtype="int64",
    )
    indices = np.moveaxis(neighbours, -1, 0)

    cell_id_indices = np.ravel(
        np.broadcast_to(cell_id_indices[:, None], (valid_x.size, 4))
    )

    rows = np.ravel(indices[:, :, 0])
    columns = np.ravel(indices[:, :, 1])

    return raw_weights, cell_id_indices, rows, columns


def _bilinear_affine_torch(
    transform: affine.Affine,
    crs: pyproj.CRS,
    shape: tuple[int, int],
    cell_ids: np.ndarray,
    grid_info: xdggs.HealpixInfo,
    device: str,
) -> torch.Tensor:
    import torch

    values, cell_id_indices, rows, columns = _bilinear_pixels(
        transform, crs, shape, cell_ids, grid_info
    )
    cell_id_indices_ = torch.from_numpy(cell_id_indices)
    pixel_indices = torch.from_numpy(rows) * shape[1] + torch.from_numpy(columns)
    coords_ = torch.stack([cell_id_indices_, pixel_indices], axis=0).to(device)
    values_ = torch.from_numpy(values).double().to(device)

    target_size = cell_ids.size
    source_size = shape[0] * shape[1]

    shape = (target_size, source_size)

    return torch.sparse_coo_tensor(coords_, values_, size=(target_size, source_size))


def bilinear_affine(
    source_grid: xr.Dataset,
    target_grid: xr.Dataset,
    *,
    chunks=None,
    dtype="float64",
    ignored_dims=None,
) -> xr.DataArray:
    """Bilinear weights based on the affine transform"""
    import sparse

    def _bilinear_single_chunk(cell_ids, x, y, grid_info, transform, crs):
        source_shape = (x.size, y.size)
        new_transform = transform * affine.Affine.translation(
            int(x.min()), int(y.min())
        )
        raw_weights, cell_id_indices, rows, columns = _bilinear_pixels(
            new_transform, crs, source_shape, cell_ids, grid_info
        )

        n_cells = cell_ids.size
        target_shape = (n_cells,)

        weights = sparse.COO(
            coords=[cell_id_indices, rows, columns],
            data=raw_weights,
            shape=target_shape + source_shape,
            fill_value=0,
        )

        return weights

    transform = source_grid.grid4earth.affine_transform(kind="center")
    nx, ny = source_grid.sizes["x"], source_grid.sizes["y"]
    crs = source_grid.grid4earth.crs
    cell_ids = target_grid.dggs.coord.data
    grid_info = target_grid.dggs.grid_info

    if ignored_dims is None:
        ignored_dims = []

    if chunks is None:
        weights = _bilinear_single_chunk(
            cell_ids, np.arange(nx), np.arange(ny), grid_info, transform, crs
        )
    else:
        from functools import partial

        import dask
        import dask.array

        # TODO: expand_chunks not in repository (some previous merge mistake?)
        expanded_chunks = chunks # = expand_chunks(chunks, target_grid.sizes)
        source_chunks = source_grid.chunksizes

        cell_ids_ = dask.array.from_array(cell_ids, chunks=expanded_chunks["cells"])
        x = dask.array.arange(source_grid.sizes["x"], chunks=source_chunks["x"])
        y = dask.array.arange(source_grid.sizes["y"], chunks=source_chunks["y"])

        weights = dask.array.blockwise(
            partial(
                _bilinear_single_chunk,
                grid_info=grid_info,
                transform=transform,
                crs=crs,
            ),
            "ijk",
            cell_ids_,
            "i",
            x,
            "j",
            y,
            "k",
            dtype=dtype,
            meta=sparse.COO.from_numpy(np.array((), dtype=dtype), fill_value=0),
        )

    source_dims = [dim for dim in source_grid.dims if dim not in ignored_dims]
    target_dims = list(target_grid.dggs.coord.dims)

    return xr.DataArray(
        weights, dims=target_dims + source_dims, coords=source_grid[source_dims].coords
    ).assign_coords(target_grid.dggs.coord.coords)
