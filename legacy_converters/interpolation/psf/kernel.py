import healpix_geo
import numpy as np
import sparse

from legacy_converters.interpolation.psf.ellipsoid import (
    ellipsoid_to_geod,
    geodesic_distances,
    max_pixel_size,
)


def crop_to_domain(cell_ids, neighbours):
    sort_indices = np.searchsorted(cell_ids, neighbours)
    mask = (
        (neighbours >= 0)
        & (sort_indices < cell_ids.size)
        & (cell_ids[sort_indices % cell_ids.size] == neighbours)
    )

    n_values_per_row = np.sum(mask, axis=-1)

    indptr = np.cumulative_sum(n_values_per_row, include_initial=True)
    column_indices = sort_indices[mask].ravel()

    return mask, indptr, column_indices


def sparse_norm(arr, axis=None):
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
    lon, lat, cell_ids, level, ellipsoid, psf_sigma, radius_factor, weights_threshold
):
    geod = ellipsoid_to_geod(ellipsoid)

    psf_radius = psf_sigma * radius_factor
    rings = np.ceil(psf_radius / max_pixel_size(level, geod)).astype(int) + 1

    lon_utm = lon.ravel()
    lat_utm = lat.ravel()
    closest = healpix_geo.nested.lonlat_to_healpix(
        lon_utm, lat_utm, level, ellipsoid=ellipsoid
    )

    neighbours_ = healpix_geo.nested.kth_neighbourhood(closest, level, ring=rings)
    mask, indptr, column_indices = crop_to_domain(cell_ids, neighbours_)
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

    distances = geodesic_distances(
        lon_utm_n, lat_utm_n, lon_neighbours, lat_neighbours, geod
    )

    # compute the (normalized) weights
    raw_weights = np.exp(-0.5 * (distances / psf_sigma) ** 2)
    weights = np.where(raw_weights >= weights_threshold, raw_weights, 0)

    shape = (lon_utm.size, cell_ids.size)

    result = sparse.GCXS(
        (weights, column_indices, indptr),
        shape=shape,
        fill_value=0,
        prune=True,
        compressed_axes=(0,),
    )

    return np.reshape(sparse_norm(result, axis=1), lon.shape + cell_ids.shape)
