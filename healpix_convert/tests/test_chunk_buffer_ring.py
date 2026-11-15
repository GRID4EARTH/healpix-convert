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


def _curvilinear_converter(monkeypatch, ds, chunk_cell_id, chunk_level, buffer):
    """A dense chunk converter on a dataset with 2-D longitude and latitude."""
    from pathlib import PurePath

    import xarray as xr
    import zarr

    from healpix_convert.core.conversion_models import (
        ConvertStagingCache,
        InputGroupSpatialInfo,
        OutputGroupInfo,
    )
    from healpix_convert.healpix_converters import DenseChunkConverter
    from healpix_convert.settings.common import ConvertSettings

    path = PurePath("measurements/data")
    grid = {"indexing_scheme": "nested", "ellipsoid": {"name": "wgs84"}}
    settings = ConvertSettings.from_dict(
        {
            "group_settings": {
                str(path): {
                    "healpix": {**grid, "refinement_level": chunk_level + 2},
                    "chunk": {
                        "method": "healpix_cell_dense",
                        "healpix": {**grid, "refinement_level": chunk_level},
                        "chunk_buffer_width": buffer,
                    },
                    "resampler": {"name": "nearest"},
                }
            }
        }
    )
    cache = ConvertStagingCache(
        input_datatrees={"0": xr.DataTree.from_dict({str(path): ds})},
        input_spatial_groups={
            path: InputGroupSpatialInfo(**utils.extract_spatial_info_cf(ds))
        },
        output_groups=OutputGroupInfo(io_path_map={path: path}),
    )
    monkeypatch.setattr(
        "healpix_convert.core.utils.compute_output_chunk_info",
        lambda *args: (
            np.atleast_1d(np.asarray(chunk_cell_id, dtype=np.uint64)),
            np.ones(1, dtype=bool),
        ),
    )
    return DenseChunkConverter(
        path, cache=cache, settings=settings, output_store=zarr.storage.MemoryStore()
    )


# (chunk level, buffer in metres): the ring level (see ``utils.chunk_buffer_ring_level``)
# is finer than the chunk level for the first case and coarser for the second
CONVERTER_CASES = [(9, 3_000.0), (11, 20_000.0)]


@pytest.mark.parametrize("chunk_level, buffer", CONVERTER_CASES)
def test_converter_selects_points_within_buffer(monkeypatch, chunk_level, buffer):
    """The converter's own query on a curvilinear grid selects the points inside
    the chunk and all the points within the buffer of it."""
    import xarray as xr

    lon0, lat0 = 10.0, 45.0
    ring_level = utils.chunk_buffer_ring_level(buffer)
    assert (ring_level > chunk_level) == (buffer < 5_000.0)

    # a rotated, slightly sheared grid of about 6000 points, spacing about 2 km
    ii, jj = np.meshgrid(np.arange(-40, 40), np.arange(-40, 40))
    angle = np.deg2rad(23.0)
    x = 0.022 * (np.cos(angle) * ii - np.sin(angle) * jj) + 0.0003 * jj
    y = 0.016 * (np.sin(angle) * ii + np.cos(angle) * jj)
    lon, lat = lon0 + x, lat0 + y
    ds = xr.Dataset(
        {"index": (("row", "column"), np.arange(lon.size).reshape(lon.shape))},
        coords={
            "longitude": (("row", "column"), lon, {"standard_name": "longitude"}),
            "latitude": (("row", "column"), lat, {"standard_name": "latitude"}),
        },
    )

    (chunk_cell_id,) = healpix_geo.nested.lonlat_to_healpix(
        np.array([lon0]), np.array([lat0]), chunk_level, "WGS84"
    )
    converter = _curvilinear_converter(
        monkeypatch, ds, chunk_cell_id, chunk_level, buffer
    )
    selected = converter.query_input_points(chunk_cell_id)
    assert selected is not None
    selected_index = selected["index"].values

    # independent of the utils functions: distance to the densely sampled edges
    geod = pyproj.Geod(ellps="WGS84")
    vlon, vlat = healpix_geo.nested.vertices(
        np.array([chunk_cell_id], dtype=np.uint64), chunk_level, "WGS84", step=16
    )
    vlon, vlat = np.asarray(vlon).ravel(), np.asarray(vlat).ravel()
    distance = np.full(lon.size, np.inf)
    for elon, elat in zip(vlon, vlat):
        _, _, d = geod.inv(
            np.full(lon.size, elon), np.full(lon.size, elat), lon.ravel(), lat.ravel()
        )
        distance = np.minimum(distance, d)

    inside = (
        healpix_geo.nested.lonlat_to_healpix(
            lon.ravel(), lat.ravel(), chunk_level, "WGS84"
        )
        == chunk_cell_id
    )
    near = distance < 0.9 * buffer

    assert inside.sum() > 0
    assert (near & ~inside).sum() > 0
    assert np.isin(np.flatnonzero(inside), selected_index).all()
    missing = ~np.isin(np.flatnonzero(near), selected_index)
    assert not missing.any(), (
        f"{missing.sum()} of {near.sum()} points within the buffer are not selected"
    )
    assert selected_index.size < lon.size
