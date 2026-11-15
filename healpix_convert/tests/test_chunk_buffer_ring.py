"""The one-cell ring that selects the input points within ``chunk_buffer_width``
of a chunk is at least as wide as the buffer."""

import healpix_geo
import numpy as np
import pyproj
import pytest

from healpix_convert.core import utils

CHUNK_LEVEL = 9

# (name, lon, lat): a chunk cell at the equator, at mid latitude, near a pole,
# and next to a corner of the base cells (lat = asin(2/3) and lon = 45 at the equator)
PLACES = [
    ("equator", 10.0, 0.0),
    ("mid-latitude", 30.0, 45.0),
    ("pole", 120.0, 89.5),
    ("south-pole", -40.0, -88.0),
    ("base-cell-corner-north", 0.0, 41.8103),
    ("base-cell-corner-equator", 45.0, 0.0),
    ("base-cell-corner-south", 90.0, -41.8103),
]
BUFFERS = [60.0, 900.0, 1500.0, 3000.0, 10_000.0]


def nominal_resolution(level):
    return utils.EARTH_RADIUS_M * np.sqrt(np.pi / 3.0) / 2.0**level


@pytest.mark.parametrize(
    "buffer, level",
    [(60.0, 16), (900.0, 12), (1500.0, 11), (3000.0, 10), (10_000.0, 8)],
)
def test_ring_level(buffer, level):
    assert utils.chunk_buffer_ring_level(buffer) == level


@pytest.mark.parametrize("buffer", np.geomspace(1.0, 2.0e6, 60))
def test_ring_level_is_the_finest_that_is_wide_enough(buffer):
    level = utils.chunk_buffer_ring_level(buffer)

    width = utils.RING_MIN_WIDTH_FACTOR * nominal_resolution(level)
    assert width >= buffer
    # a finer level would be narrower than the buffer
    assert utils.RING_MIN_WIDTH_FACTOR * nominal_resolution(level + 1) < buffer


@pytest.mark.parametrize("ellipsoid", ["sphere", "WGS84"])
@pytest.mark.parametrize("buffer", BUFFERS)
@pytest.mark.parametrize("place", PLACES, ids=[p[0] for p in PLACES])
def test_points_within_buffer_are_selected(place, buffer, ellipsoid):
    """Every point within ``buffer`` metres of the chunk cell is in the chunk
    cell or in its ring."""
    _, lon, lat = place
    if ellipsoid == "sphere":
        geod = pyproj.Geod(a=utils.EARTH_RADIUS_M, f=0.0)
    else:
        geod = pyproj.Geod(ellps="WGS84")

    (chunk_cell_id,) = healpix_geo.nested.lonlat_to_healpix(
        np.array([lon]), np.array([lat]), CHUNK_LEVEL, ellipsoid
    )
    ring_level = utils.chunk_buffer_ring_level(buffer)

    selected = np.union1d(
        healpix_geo.nested.zoom_to(
            chunk_cell_id, depth=CHUNK_LEVEL, new_depth=ring_level
        ).ravel(),
        utils.chunk_ring_cell_ids(chunk_cell_id, CHUNK_LEVEL, ring_level),
    )

    # points on the edges of the chunk cell, and points a bit less than the buffer
    # away from them, in all directions (these are within the buffer of the cell)
    vlon, vlat = healpix_geo.nested.vertices(
        np.array([chunk_cell_id], dtype=np.uint64), CHUNK_LEVEL, ellipsoid, step=16
    )
    azimuths = np.arange(0.0, 360.0, 11.25)
    plon = np.repeat(vlon.ravel(), len(azimuths))
    plat = np.repeat(vlat.ravel(), len(azimuths))
    az = np.tile(azimuths, vlon.size)
    qlon, qlat, _ = geod.fwd(plon, plat, az, np.full(plon.shape, 0.999 * buffer))

    cell_ids = healpix_geo.nested.lonlat_to_healpix(qlon, qlat, ring_level, ellipsoid)
    missing = ~np.isin(cell_ids, selected)

    assert not missing.any(), (
        f"{missing.sum()} of {missing.size} points within {buffer} m of the chunk "
        f"are outside its ring (ring level {ring_level})"
    )
