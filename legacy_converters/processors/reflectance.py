from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import healpix_geo
import numpy as np
import torch
import xarray as xr
import xbatcher  # noqa: F401
import xdggs

import legacy_converters.accessor  # noqa: F401
from legacy_converters.interpolation import bilinear_affine, matmul, nearest_affine
from legacy_converters.interpolation.psf import gaussian_filter, optimize_psf
from legacy_converters.utils import suppress_warning

if TYPE_CHECKING:
    from typing import Literal

    import affine
    import pyproj
    import zarr

initial_methods = {
    "nearest": nearest_affine,
    "bilinear": bilinear_affine,
}


@dataclass
class PSFParameters:
    # numerical model of the PSF
    sigma: float
    radius_factor: float = 3.0
    distance_metric: Literal["geodesic", "cartesian"] = "geodesic"
    initial_method: Literal["nearest", "bilinear"] = "bilinear"
    weights_threshold: float = 1e-9

    # optimization parameters
    optimizer: Literal["adam", "cg"] = "adam"
    max_iter: int = 200
    tolerance: float = 1e-7
    learning_rate: float = 1e-3
    n_iter: int = 300

    # optimizations
    device: str = "cpu"
    format: str = "coo"

    # patch merging
    patch_size: dict[str, int] = field(default_factory=lambda: {"x": 256, "y": 256})
    overlap: dict[str, int] = field(default_factory=lambda: {"x": 16, "y": 16})
    eroded_rings: int = 5


def _interpolate_torch(
    patch_bands: list[np.ndarray],
    affine: affine.Affine,
    shape: tuple[int, int],
    crs: pyproj.CRS,
    target_cell_ids: np.ndarray,
    grid_info: xdggs.HealpixInfo,
    parameters: PSFParameters,
) -> list[np.ndarray]:
    from legacy_converters.interpolation.bilinear import _bilinear_affine_torch
    from legacy_converters.interpolation.nearest import _nearest_affine_torch
    from legacy_converters.interpolation.psf.kernel import gaussian_filter
    from legacy_converters.interpolation.psf.model.conjugate_gradient import (
        least_squares_conjugate_gradient,
    )

    device = parameters.device

    initial_methods = {
        "nearest": _nearest_affine_torch,
        "bilinear": _bilinear_affine_torch,
    }

    compute_initial = initial_methods[parameters.initial_method]
    initial_value_weights: torch.Tensor = compute_initial(
        affine, crs, shape, target_cell_ids, grid_info, device=device
    )

    # torch coo tensor
    psf_model: torch.Tensor = gaussian_filter(
        affine,
        shape,
        crs,
        target_cell_ids,
        grid_info.level,
        grid_info.ellipsoid,
        psf_sigma=parameters.sigma,
        radius_factor=parameters.radius_factor,
        weights_threshold=parameters.weights_threshold,
        format=parameters.format,
        distance_metric=parameters.distance_metric,
        device=parameters.device,
    )

    interpolated_bands = []
    for band in patch_bands:
        band_ = torch.from_numpy(band).to(device)
        initial_values = torch.matmul(initial_value_weights, band_)

        interpolated = least_squares_conjugate_gradient(
            psf_model,
            band_,
            initial_values,
            max_iter=parameters.max_iter,
            tolerance=parameters.tolerance,
        )
        interpolated_bands.append(interpolated.detach().cpu().numpy().copy())

    return interpolated_bands


def _interpolate_patch_torch(
    patch: xr.Dataset,
    target_grid: xr.Dataset,
    parameters: PSFParameters,
):
    grid_info = target_grid.dggs.grid_info

    stacked = patch.stack(points=["y", "x"]).fillna(0)
    shape = (patch.sizes["y"], patch.sizes["x"])
    bands = {name: band.data for name, band in stacked.data_vars.items()}

    affine = patch.grid4earth.affine_transform(kind="center")
    crs = patch.grid4earth.crs
    if crs is None:
        raise ValueError("cannot find the crs")

    target_cell_ids = target_grid["cell_ids"]
    [dim] = target_cell_ids.dims

    result = _interpolate_torch(
        list(bands.values()),
        affine,
        shape,
        crs,
        target_cell_ids.data,
        grid_info,
        parameters,
    )

    return (
        target_grid.drop_indexes("cell_ids")
        .set_xindex("cell_ids")
        .assign(
            {
                name: xr.Variable(dim, band, old_band.attrs)
                for (name, old_band), band in zip(patch.data_vars.items(), result)
            }
        )
    )


def _interpolate_patch(patch, grid_info, *, parameters):
    # load data into memory
    patch.load()

    # target grid
    target_grid = patch.grid4earth.infer_healpix_grid(grid_info)

    # initial interpolation
    initial_weights = initial_methods[parameters.initial_method](patch, target_grid)
    # TODO: infer dims
    initial_values = matmul(patch.fillna(0), initial_weights, dims=["x", "y"])

    # construct gaussian kernel
    psf_model = gaussian_filter(
        patch,
        target_grid,
        psf_sigma=parameters.sigma,
        radius_factor=parameters.radius_factor,
        weights_threshold=parameters.weights_threshold,
        distance_metric=parameters.distance_metric,
        format={"coo": "coo", "csr": "coo"}.get(parameters.format, parameters.format),
    )

    if parameters.optimizer == "adam":
        optimizer_kwargs = {
            "n_iter": parameters.n_iter,
            "learning_rate": parameters.learning_rate,
        }
    else:
        optimizer_kwargs = {
            "max_iter": parameters.max_iter,
            "tolerance": parameters.tolerance,
        }

    # correct for PSF
    corrected = optimize_psf(
        psf_model,
        patch.drop_indexes(["x", "y"], errors="ignore"),
        initial_values,
        optimizer=parameters.optimizer,
        device=parameters.device,
        format=parameters.format,
        **optimizer_kwargs,
    )

    return corrected


def _erode(grid_info, cell_ids, rings):
    boundary = np.setdiff1d(
        np.unique(
            healpix_geo.nested.kth_neighbourhood(cell_ids, grid_info.level, ring=1),
            sorted=True,
        ),
        np.concatenate([np.array([-1], dtype="int64"), cell_ids], axis=0),
    ).astype("uint64")
    neighbourhood = healpix_geo.nested.kth_neighbourhood(
        boundary, grid_info.level, ring=rings
    )

    eroded_cells = np.setdiff1d(
        np.unique(neighbourhood, sorted=True), np.array([-1], dtype="int64")
    ).astype("uint64")

    return np.setdiff1d(cell_ids, eroded_cells)


def generate_template(ds, cell_ids, chunks):
    dims = cell_ids.dims
    shape = list(cell_ids.sizes.values())

    def _variable_template(var):
        data = np.full(shape=shape, fill_value=-1, dtype=var.data.dtype)
        encoding = {"chunks": (chunks,), "fill_value": -1}

        return xr.Variable(dims, data, var.attrs, encoding=encoding)

    variables = {
        name: _variable_template(var.variable) for name, var in ds.data_vars.items()
    }
    coords = {"cell_ids": cell_ids}

    return xr.Dataset(variables, coords=coords, attrs=ds.attrs)


def interpolate_reflectance(
    cache_store: zarr.storage.StoreLike,
    reflectance: xr.Dataset,
    grid_info: xdggs.HealpixInfo,
    *,
    psf_parameters: PSFParameters,
    limit: int = None,
    batch_size: int = None,
    cache_chunks: int = 2**20,
) -> xr.DataTree:
    """
    Interpolate the reflectance data to healpix

    Parameters
    ----------
    cache_store : zarr.storage.StoreLike
        A store that can be used to cache patches.
    reflectance : xarray.Dataset
        The reflectance dataset with a raster index.
    grid_info : xdggs.HealpixInfo
        The grid metadata.
    patch_size : mapping of str to int
        The size of the patches, by dimension.
    overlap : mapping of str to int
        The overlap between neighbouring patches, by dimension.

    Returns
    -------
    interpolated : xarray.Dataset
        The reflectance data interpolated to healpix.
    """
    bgen = reflectance.batch.generator(
        psf_parameters.patch_size,
        input_overlap=psf_parameters.overlap,
        preload_batch=False,
    )
    print(f"total number of patches: {len(bgen)}")

    target_grids = [
        patch.grid4earth.infer_healpix_grid(grid_info, index_kind="moc")
        for patch in itertools.islice(bgen, limit)
    ]

    indexes = [
        target_grid.xindexes["cell_ids"]._index._index for target_grid in target_grids
    ]

    taken_cell_ids = indexes[0]
    trimmed_cell_ids = [taken_cell_ids]
    for index in indexes[1:]:
        trimmed = index.difference(taken_cell_ids)
        trimmed_cell_ids.append(trimmed)
        taken_cell_ids = taken_cell_ids.union(trimmed)

    print("total number of cells:", taken_cell_ids.size)

    unordered_cell_ids = np.concatenate(
        [moc.cell_ids() for moc in trimmed_cell_ids], axis=0
    )
    template = generate_template(
        reflectance,
        xr.Variable("cells", unordered_cell_ids, {}, encoding={"chunks": None}),
        chunks=cache_chunks,
    )
    with suppress_warning(zarr.errors.ZarrUserWarning):
        template.to_zarr(cache_store, mode="w")

    batches = []
    start = 0
    stop = 0
    for index, (patch, target_grid, moc) in enumerate(
        zip(itertools.islice(bgen, limit), target_grids, trimmed_cell_ids)
    ):
        print(f" — processing patch #{index}")
        interpolated = _interpolate_patch_torch(
            patch,
            target_grid,
            parameters=psf_parameters,
        )
        torch.cuda.empty_cache()

        trimmed = interpolated.sel(cell_ids=moc.cell_ids())
        stop += trimmed.sizes["cells"]

        batches.append(trimmed)

        if len(batches) >= batch_size or index == len(bgen) - 1:
            full_batch = xr.concat(
                batches,
                dim="cells",
                compat="override",
                coords="minimal",
                data_vars="minimal",
            ).sortby("cell_ids")
            batches.clear()

            print(
                f"writing {full_batch.nbytes} bytes to the region between {start} and {stop}"
            )

            with suppress_warning(zarr.errors.ZarrUserWarning):
                full_batch.to_zarr(
                    cache_store, mode="a", region={"cells": slice(start, stop)}
                )
            start = stop
    restored = xr.open_dataset(cache_store, engine="zarr", chunks={}).assign_coords(
        cell_ids=lambda ds: ds["cell_ids"].compute()
    )
    return restored.sortby("cell_ids").dggs.decode(grid_info, index_kind="moc")
