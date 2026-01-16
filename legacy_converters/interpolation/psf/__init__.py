import xarray as xr

from legacy_converters.interpolation.psf.kernel import (
    gaussian_filter as _gaussian_filter,
)
from legacy_converters.interpolation.psf.model import (  # noqa: F401
    HealpixToUTM,
    interpolate_to_healpix,
)


def gaussian_filter(
    source_grid: xr.Dataset,
    target_grid: xr.Dataset,
    *,
    psf_sigma: float = 5.0,
    radius_factor: float = 3.0,
    weights_threshold: float = 1e-9,
) -> xr.DataArray:
    """Construct a gaussian filter that moves from one grid to another.

    Parameters
    ----------
    source_grid : xarray.Dataset
        A dataset representing the source grid. Must have `lon` and `lat` coordinates.
    target_grid : xarray.Dataset
        A dataset representing the target grid. Must have a DGGSIndex.
    psf_sigma : float, default: 5.0
        The standard deviation used for the rotationally symmetric 2D gaussian kernel.
    radius_factor : float, default: 3.0
        A factor that together with `psf_sigma` controls the number of rings
        around a healpix cell that is used to construct the kernel.
    weights_threshold : float, default: 1e-9
        A threshold used to exclude values that are too small. Use `0.0` to
        include all values.

    Returns
    -------
    weights : xarray.DataArray
        The kernel weights for each position as a sparse matrix.
    """
    lon = source_grid["lon"]
    lat = source_grid["lat"]

    cell_ids = target_grid.dggs.coord
    grid_info = target_grid.dggs.grid_info
    level = grid_info.level
    ellipsoid = grid_info.ellipsoid

    return xr.apply_ufunc(
        _gaussian_filter,
        lon,
        lat,
        cell_ids,
        input_core_dims=[["x", "y"], ["x", "y"], ["cells"]],
        output_core_dims=[["x", "y", "cells"]],
        kwargs={
            "level": level,
            "ellipsoid": ellipsoid,
            "psf_sigma": psf_sigma,
            "radius_factor": radius_factor,
            "weights_threshold": weights_threshold,
        },
    )


def optimize_psf(gaussian_weights, ds, initial_values, *, n_iter=500, device="cpu"):
    def _interpolate(weights, arr, initial):
        print(f"PSF for variable:{arr.name}")
        return xr.apply_ufunc(
            interpolate_to_healpix,
            weights,
            arr,
            initial,
            input_core_dims=[["x", "y", "cells"], ["x", "y"], ["cells"]],
            output_core_dims=[["cells"]],
            kwargs={"n_iter": n_iter, "device": device},
        )

    interpolated_vars = {
        name: _interpolate(gaussian_weights, var, initial_values[name])
        for name, var in ds.data_vars.items()
    }

    return xr.Dataset(interpolated_vars)
