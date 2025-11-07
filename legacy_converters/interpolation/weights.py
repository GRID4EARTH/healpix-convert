import healpix_geo
import numpy as np
import sparse
import xarray as xr
import xdggs  # noqa: F401

from legacy_converters.crs import create_transformer


def nearest_affine(source_grid: xr.Dataset, target_grid: xr.Dataset) -> xr.DataArray:
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
    crs_transformer = create_transformer(source_grid.grid4earth.crs, 4326)

    transform = source_grid.grid4earth.affine_transform(kind="center")
    nx, ny = source_grid.sizes["x"], source_grid.sizes["y"]

    # TODO: use the grid_info object once that supports ellipsoids
    level = target_grid.dggs.grid_info.level
    lon, lat = healpix_geo.nested.healpix_to_lonlat(
        target_grid.dggs.coord.data, depth=level, ellipsoid="WGS84"
    )

    x, y = crs_transformer.transform(lon, lat, direction="INVERSE")

    pixel_x, pixel_y = ~transform * (x, y)
    indices_x = np.round(np.clip(pixel_x, min=0, max=nx - 1)).astype(int)
    indices_y = np.round(np.clip(pixel_y, min=0, max=ny - 1)).astype(int)

    n_cells = target_grid.dggs.coord.size
    target_shape = (n_cells,)
    source_shape = (nx, ny)
    rows = np.arange(n_cells)

    weights = sparse.COO(
        coords=[rows, indices_x, indices_y],
        data=np.ones_like(rows, dtype="float16"),
        shape=target_shape + source_shape,
        fill_value=0,
    )

    return xr.DataArray(
        weights, dims=["cells", "x", "y"], coords=source_grid[["x", "y"]].coords
    ).assign_coords(target_grid.dggs.coord.coords)


def bilinear_affine(source_grid: xr.Dataset, target_grid: xr.Dataset) -> xr.DataArray:
    """Bilinear weights based on the affine transform"""
    crs_transformer = create_transformer(source_grid.grid4earth.crs, 4326)

    transform = source_grid.grid4earth.affine_transform(kind="center")
    nx, ny = source_grid.sizes["x"], source_grid.sizes["y"]

    # TODO: use the grid_info object once that supports ellipsoids
    cell_ids = target_grid.dggs.coord.data
    level = target_grid.dggs.grid_info.level
    lon, lat = healpix_geo.nested.healpix_to_lonlat(
        cell_ids, depth=level, ellipsoid="WGS84"
    )

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
    rows = np.ravel(indices[:, :, 0])
    columns = np.ravel(indices[:, :, 1])

    cell_id_indices = np.ravel(
        np.broadcast_to(cell_id_indices[:, None], (valid_x.size, 4))
    )

    n_cells = cell_ids.size
    source_dims = ("x", "y")
    source_shape = (nx, ny)
    target_dims = target_grid.dggs.coord.dims
    target_shape = (n_cells,)

    weights = sparse.COO(
        coords=[cell_id_indices, rows, columns],
        data=raw_weights,
        shape=target_shape + source_shape,
        fill_value=0,
    )

    return xr.DataArray(
        weights, dims=target_dims + source_dims, coords=source_grid[["x", "y"]].coords
    ).assign_coords(target_grid.dggs.coord.coords)
