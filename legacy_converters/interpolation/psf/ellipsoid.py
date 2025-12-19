import numpy as np
import pyproj


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


def max_pixel_size(level, geod):
    # note: this the spherical approximation using the semimajor axis to have a
    # bigger radius than necessary
    return np.sqrt(4 * np.pi * geod.a**2 / (12 * 4**level))
