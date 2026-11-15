"""The cells that select the input points within ``chunk_buffer_width`` of a
chunk cover everything within that distance of the chunk."""

import healpix_geo
import numpy as np
import pyproj
import pytest

from healpix_convert.core import utils

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


def nominal_resolution(level):
    return utils.EARTH_RADIUS_M * np.sqrt(np.pi / 3.0) / 2.0**level


# (chunk level, buffer in metres). The ring level is finer than the chunk level
# for the first ones, coarser for the 3 km buffer on a level 12 chunk and for the
# 10 km and 20 km buffers on a level 9 chunk. The last ones are 0.72 times the
# nominal resolution of a level: just above the width of the narrowest one-cell
# rings (0.678 of it), so that a factor larger than 0.72 fails.
CASES = (
    [(9, buffer) for buffer in (60.0, 900.0, 1500.0, 3000.0)]
    + [(12, 3000.0), (9, 10_000.0), (9, 20_000.0)]
    + [(9, 0.72 * nominal_resolution(level)) for level in (10, 12, 13)]
)


@pytest.mark.parametrize(
    "buffer, level",
    [(60.0, 16), (900.0, 12), (1500.0, 11), (3000.0, 10), (10_000.0, 8)],
)
def test_ring_level(buffer, level):
    assert utils.chunk_buffer_ring_level(buffer) == level


@pytest.mark.parametrize("buffer", [5.0e6, 1.0e7, 1.0e8])
def test_ring_level_of_a_very_large_buffer(buffer):
    assert utils.chunk_buffer_ring_level(buffer) == 0


@pytest.mark.parametrize("ellipsoid", ["sphere", "WGS84"])
@pytest.mark.parametrize(
    "chunk_level, buffer", CASES, ids=[f"l{c}-{b:.0f}m" for c, b in CASES]
)
@pytest.mark.parametrize("place", PLACES, ids=[p[0] for p in PLACES])
def test_points_within_buffer_are_selected(place, chunk_level, buffer, ellipsoid):
    """Every point within ``buffer`` metres of the chunk cell is in a selected cell."""
    _, lon, lat = place
    if ellipsoid == "sphere":
        geod = pyproj.Geod(a=utils.EARTH_RADIUS_M, f=0.0)
    else:
        geod = pyproj.Geod(ellps="WGS84")

    (chunk_cell_id,) = healpix_geo.nested.lonlat_to_healpix(
        np.array([lon]), np.array([lat]), chunk_level, ellipsoid
    )
    ring_level = utils.chunk_buffer_ring_level(buffer)
    selected = utils.chunk_selection_cell_ids(chunk_cell_id, chunk_level, ring_level)

    # points on the edges of the chunk cell, and points a bit less than the buffer
    # away from them, in all directions (these are within the buffer of the cell)
    vlon, vlat = healpix_geo.nested.vertices(
        np.array([chunk_cell_id], dtype=np.uint64), chunk_level, ellipsoid, step=16
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
        f"are not selected (chunk level {chunk_level}, ring level {ring_level})"
    )


@pytest.mark.parametrize("chunk_level, ring_level", [(9, 7), (9, 9), (9, 11), (3, 0)])
def test_selection_covers_the_chunk(chunk_level, ring_level):
    """The selected cells include the chunk (its parent when the ring level is
    coarser than the chunk level, with the siblings of the chunk)."""
    (chunk_cell_id,) = healpix_geo.nested.lonlat_to_healpix(
        np.array([10.0]), np.array([20.0]), chunk_level, "sphere"
    )
    selected = utils.chunk_selection_cell_ids(chunk_cell_id, chunk_level, ring_level)
    chunk_zoomed = healpix_geo.nested.zoom_to(
        chunk_cell_id, depth=chunk_level, new_depth=ring_level
    ).ravel()

    assert np.isin(chunk_zoomed, selected).all()
    assert len(selected) == len(np.unique(selected))
