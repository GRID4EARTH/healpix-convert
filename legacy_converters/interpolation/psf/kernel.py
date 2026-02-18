from __future__ import annotations

from typing import TYPE_CHECKING

import healpix_geo
import numpy as np
import sparse
import torch

from legacy_converters.crs import create_transformer
from legacy_converters.interpolation.psf.ellipsoid import (
    ellipsoid_to_geod,
    max_pixel_size,
    pointwise_distances_torch,
)

if TYPE_CHECKING:
    import affine
    import pyproj


def crop_to_domain(cell_ids, neighbours, format):
    sort_indices = np.searchsorted(cell_ids, neighbours)
    mask = (
        (neighbours >= 0)
        & (sort_indices < cell_ids.size)
        & (cell_ids[sort_indices % cell_ids.size] == neighbours)
    )

    n_values_per_row = np.sum(mask, axis=-1)
    column_indices = sort_indices[mask].ravel()
    indptr = np.cumulative_sum(n_values_per_row, include_initial=True)

    if format == "coo":
        n_rows = indptr.size - 1
        row_indices = np.repeat(np.arange(n_rows), n_values_per_row)
        return mask, row_indices, column_indices
    elif format == "csr":
        return mask, indptr, column_indices
    else:
        raise ValueError(f"unknown sparse format: {format}")


def _sparse_norm_gcxs(arr, axis):
    broadcasted_ones = sparse.GCXS(
        (np.ones_like(arr.data), arr.indices, arr.indptr),
        shape=arr.shape,
        compressed_axes=arr.compressed_axes,
        fill_value=0,
    )

    norm_arr = np.sum(arr, axis=axis)
    norm_arr.data = np.where(norm_arr.data != 0, norm_arr.data, 1)

    norm = np.reshape(norm_arr, (-1, 1)) * broadcasted_ones
    norm.fill_value = 1
    return arr / norm


def gaussian_filter(
    transform: affine.Affine,
    shape: tuple[int, int],
    crs: pyproj.CRS,
    cell_ids: np.ndarray,
    level: int,
    ellipsoid,
    psf_sigma: float,
    radius_factor: float,
    weights_threshold: float,
    format: str = "gcxs",
    distance_metric: str = "geodesic",
    device: str = "cpu",
):
    geod = ellipsoid_to_geod(ellipsoid)

    nx, ny = shape
    x, y = transform * (np.arange(nx), np.arange(ny))

    crs_transformer = create_transformer(crs, 4326)
    X, Y = np.meshgrid(x, y)
    lon, lat = crs_transformer.transform(X, Y)

    intermediate_format = {"gcxs": "csr", "csr": "coo"}.get(format, format)

    psf_radius = psf_sigma * radius_factor
    rings = np.ceil(psf_radius / max_pixel_size(level, geod)).astype(int)

    lon_utm = lon.ravel()
    lat_utm = lat.ravel()
    closest = healpix_geo.nested.lonlat_to_healpix(
        lon_utm, lat_utm, level, ellipsoid=ellipsoid
    )

    neighbours_ = healpix_geo.nested.kth_neighbourhood(closest, level, ring=rings)
    mask, *coords = crop_to_domain(cell_ids, neighbours_, format=intermediate_format)
    neighbours = neighbours_.ravel()[mask.ravel()]

    # broadcast utm_lon and utm_lat to have the same number of elements as the flattened neighbours
    broadcast_lon_utm = np.broadcast_to(
        np.expand_dims(lon_utm, axis=-1),
        neighbours_.shape,
    )
    broadcast_lat_utm = np.broadcast_to(
        np.expand_dims(lat_utm, axis=-1),
        neighbours_.shape,
    )
    lon_utm_n = broadcast_lon_utm.ravel()[mask.ravel()]
    lat_utm_n = broadcast_lat_utm.ravel()[mask.ravel()]

    # compute the distances between utm and healpix cell centers
    lon_neighbours, lat_neighbours = healpix_geo.nested.healpix_to_lonlat(
        neighbours, level, ellipsoid=ellipsoid
    )

    lon_utm_n_ = torch.from_numpy(lon_utm_n).to(device)
    lat_utm_n_ = torch.from_numpy(lat_utm_n).to(device)
    lon_neighbours_ = torch.from_numpy(lon_neighbours).to(device)
    lat_neighbours_ = torch.from_numpy(lat_neighbours).to(device)

    distances = pointwise_distances_torch(
        lon_utm_n_,
        lat_utm_n_,
        lon_neighbours_,
        lat_neighbours_,
        geod,
        metric=distance_metric,
    )

    # compute the (normalized) weights
    raw_weights = torch.exp(-0.5 * (distances / psf_sigma) ** 2)

    shape = (lon_utm.size, cell_ids.size)

    mask = raw_weights >= weights_threshold
    weights = raw_weights[mask]

    coords = [torch.from_numpy(coord).to(device)[mask] for coord in coords]

    norms = torch.bincount(coords[0], weights)
    # norms = numpy_groupies.aggregate(coords[0], weights, size=shape[1], func="sum")
    broadcasted_norms = norms[coords[0]]

    coords_ = torch.stack(coords, axis=0)
    data = weights / broadcasted_norms

    result = torch.sparse_coo_tensor(coords_, data, size=shape)
    return result
