import numpy as np
import pyproj
import torch


def ellipsoid_to_geod(ellipsoid):
    if isinstance(ellipsoid, str):
        return pyproj.Geod(ellps=ellipsoid)

    if isinstance(ellipsoid, dict):
        if "semimajor_axis" in ellipsoid:
            return pyproj.Geod(
                a=ellipsoid["semimajor_axis"], f=1 / ellipsoid["inverse_flattening"]
            )
        else:
            return pyproj.Geod(a=ellipsoid["radius"], sphere=True)
    else:
        if hasattr(ellipsoid, "semimajor_axis"):
            return pyproj.Geod(
                a=ellipsoid.semimajor_axis, f=1 / ellipsoid.inverse_flattening
            )
        else:
            return pyproj.Geod(a=ellipsoid.radius, sphere=True)


def geodesic_distances(from_lon, from_lat, to_lon, to_lat, geod):
    lats1, lats2 = np.broadcast_arrays(from_lat, to_lat)
    lons1, lons2 = np.broadcast_arrays(from_lon, to_lon)

    _, _, distances = geod.inv(
        lons1=lons1, lats1=lats1, lons2=lons2, lats2=lats2, return_back_azimuth=False
    )

    return distances


def cartesian_distances(from_lon, from_lat, to_lon, to_lat, geod):
    geographic_crs = pyproj.crs.CRS.from_string(f"+proj=lonlat {geod.initstring}")
    cartesian_crs = pyproj.crs.CRS.from_string(f"+proj=cart {geod.initstring}")
    transformer = pyproj.Transformer.from_crs(geographic_crs, cartesian_crs)

    from_points = np.stack(
        transformer.transform(from_lon, from_lat, np.zeros_like(from_lon)),
        axis=-1,
    )
    to_points = np.stack(
        transformer.transform(to_lon, to_lat, np.zeros_like(from_lat)), axis=-1
    )

    return np.linalg.norm(to_points - from_points, axis=-1)


def pointwise_distances(from_lon, from_lat, to_lon, to_lat, geod, *, metric):
    distance_metrics = {
        "geodesic": geodesic_distances,
        "cartesian": cartesian_distances,
    }

    func = distance_metrics.get(metric)
    if func is None:
        raise ValueError(f"unknown distance metric: {metric}")

    return func(from_lon, from_lat, to_lon, to_lat, geod)


def ellipsoidal_to_cartesian(lon, lat, semimajor_axis, eccentricity_squared):
    lon_rad = torch.deg2rad(lon)
    lat_rad = torch.deg2rad(lat)

    sine_B = torch.sin(lat_rad)
    cosine_B = torch.cos(lat_rad)

    W_squared = 1 - eccentricity_squared * sine_B**2
    N = semimajor_axis / torch.sqrt(W_squared)

    X = N * cosine_B * torch.cos(lon_rad)
    Y = N * cosine_B * torch.sin(lon_rad)
    Z = N * (1 - eccentricity_squared) * sine_B

    return torch.stack([X, Y, Z], axis=-1)


def spherical_to_cartesian(lon, lat, radius):
    lon_rad = torch.deg2rad(lon)
    lat_rad = torch.deg2rad(lat)

    cosine_lat = torch.cos(lat_rad)

    X = radius * cosine_lat * torch.cos(lon_rad)
    Y = radius * cosine_lat * torch.sin(lon_rad)
    Z = radius * torch.sin(lat_rad)

    return torch.stack([X, Y, Z], axis=-1)


def cartesian_distances_torch(from_lon, from_lat, to_lon, to_lat, geod):
    if geod.sphere:
        cartesian_from = spherical_to_cartesian(from_lon, from_lat, geod.a)
        cartesian_to = spherical_to_cartesian(to_lon, to_lat, geod.a)
    else:
        cartesian_from = ellipsoidal_to_cartesian(from_lon, from_lat, geod.a, geod.es)
        cartesian_to = ellipsoidal_to_cartesian(to_lon, to_lat, geod.a, geod.es)

    return torch.linalg.norm(cartesian_to - cartesian_from, axis=-1)


def pointwise_distances_torch(from_lon, from_lat, to_lon, to_lat, geod, *, metric):
    distance_metrics = {
        "cartesian": cartesian_distances_torch,
    }

    func = distance_metrics.get(metric)
    if func is None:
        raise ValueError(f"unknown distance metric: {metric}")

    return func(from_lon, from_lat, to_lon, to_lat, geod)


def max_pixel_size(level, geod):
    # note: this the spherical approximation using the semimajor axis to have a
    # bigger radius than necessary
    return np.sqrt(4 * np.pi * geod.a**2 / (12 * 4**level))
