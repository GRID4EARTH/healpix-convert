"""Selection of the input points of a chunk from projected (x/y) input data,
for every direction of the x/y coordinates."""

from pathlib import PurePath

import affine
import healpix_geo
import numpy as np
import pyproj
import pytest
import shapely
import xarray as xr
import zarr

from healpix_convert.core import utils
from healpix_convert.core.conversion_models import (
    ConvertStagingCache,
    InputGroupSpatialInfo,
    InputSpatialInfo,
    OutputGroupInfo,
)
from healpix_convert.healpix_converters import DenseChunkConverter
from healpix_convert.settings.common import ConvertSettings

UTM31N = pyproj.CRS.from_epsg(32631)
WGS84 = pyproj.CRS.from_epsg(4326)

CHUNK_LEVEL = 11
LEVEL = 14
RESOLUTION = 60.0
BUFFER = 60.0
SIZE = 10

PATH = PurePath("data")

# one output chunk (HEALPix cell at the chunk level), in UTM zone 31N
(CELL,) = healpix_geo.nested.lonlat_to_healpix(
    3.0, 48.0, CHUNK_LEVEL, ellipsoid="WGS84"
)


def tile(descending_x: bool, ascending_y: bool) -> tuple[xr.Dataset, shapely.Polygon]:
    """A projected raster around the centre of the chunk cell, inside it.

    The x/y coordinates are the pixel centres and the spatial metadata is given
    as STAC "proj:*" attributes. By default x increases and y decreases
    (north-up raster); `descending_x` and `ascending_y` reverse them.
    """
    lon, lat = healpix_geo.nested.healpix_to_lonlat(
        CELL, CHUNK_LEVEL, ellipsoid="WGS84"
    )
    (centre,) = utils.reproject_to_multiple_crs(
        shapely.Point(lon[0], lat[0]), WGS84, [UTM31N]
    )
    half = SIZE * RESOLUTION / 2
    left, right = centre.x - half, centre.x + half
    bottom, top = centre.y - half, centre.y + half

    offsets = RESOLUTION * (np.arange(SIZE) + 0.5)
    if descending_x:
        x, a, c = right - offsets, -RESOLUTION, right
    else:
        x, a, c = left + offsets, RESOLUTION, left
    if ascending_y:
        y, e, f = bottom + offsets, RESOLUTION, bottom
    else:
        y, e, f = top - offsets, -RESOLUTION, top
    transform = affine.Affine(a, 0.0, c, 0.0, e, f)
    attrs = {"proj:code": "EPSG:32631", "proj:transform": list(transform)[:6]}

    values = np.arange(SIZE * SIZE, dtype="f4").reshape(SIZE, SIZE) + 1
    ds = xr.Dataset(
        {"var": (("y", "x"), values, attrs)},
        coords=xr.Coordinates({"x": ("x", x), "y": ("y", y)}, indexes={}),
    )
    extent = shapely.box(left, bottom, right, top)

    return ds, extent


def make_converter(monkeypatch, ds, extent):
    grid = {"indexing_scheme": "nested", "ellipsoid": {"name": "wgs84"}}
    settings = ConvertSettings.from_dict(
        {
            "group_settings": {
                str(PATH): {
                    "healpix": {**grid, "refinement_level": LEVEL},
                    "chunk": {
                        "method": "healpix_cell_dense",
                        "healpix": {**grid, "refinement_level": CHUNK_LEVEL},
                        "chunk_buffer_width": BUFFER,
                    },
                    "resampler": {"name": "nearest"},
                }
            }
        }
    )
    (extent_latlon,) = utils.reproject_to_multiple_crs(extent, UTM31N, [WGS84])
    cache = ConvertStagingCache(
        input_datatrees={"0": xr.DataTree.from_dict({str(PATH): ds})},
        input_spatial=InputSpatialInfo(
            crs=[UTM31N], geometry_latlon=[extent_latlon], geometry_xy=[extent]
        ),
        input_spatial_groups={
            PATH: InputGroupSpatialInfo(**utils.extract_spatial_info_stac(ds, UTM31N))
        },
        output_groups=OutputGroupInfo(io_path_map={PATH: PATH}),
    )
    monkeypatch.setattr(
        "healpix_convert.core.utils.compute_output_chunk_info",
        lambda *args: (np.array([CELL], dtype=np.uint64), np.array([True])),
    )
    return DenseChunkConverter(
        PATH, cache=cache, settings=settings, output_store=zarr.storage.MemoryStore()
    )


@pytest.mark.parametrize("ascending_y", [False, True], ids=["y-down", "y-up"])
@pytest.mark.parametrize("descending_x", [False, True], ids=["x-up", "x-down"])
def test_query_selects_all_points_for_any_xy_direction(
    monkeypatch, descending_x, ascending_y
) -> None:
    ds, extent = tile(descending_x, ascending_y)
    converter = make_converter(monkeypatch, ds, extent)

    # the raster lies inside the chunk cell: all its pixels are selected
    selected = converter.query_input_points(CELL)
    assert selected is not None
    assert selected.sizes["points"] == SIZE * SIZE
    np.testing.assert_array_equal(
        np.sort(selected["var"].values), np.sort(ds["var"].values.ravel())
    )

    converter.convert(0)

    var = converter.output_arrays["var"]
    assert np.any(var[:] != var.fill_value)
