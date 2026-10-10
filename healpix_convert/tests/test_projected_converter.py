"""Chunked conversion of projected (x/y) input data, e.g. EOPF Sentinel-2 groups."""

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

PATH = PurePath("measurements/reflectance")

# one output chunk (HEALPix cell at the chunk level), in UTM zone 31N
(CELL,) = healpix_geo.nested.lonlat_to_healpix(
    3.0, 48.0, CHUNK_LEVEL, ellipsoid="WGS84"
)


def chunk_polygon() -> shapely.Polygon:
    """The chunk cell polygon (x/y)."""
    lon, lat = healpix_geo.nested.vertices(CELL, CHUNK_LEVEL, ellipsoid="WGS84")
    (poly,) = utils.reproject_to_multiple_crs(
        shapely.Polygon(list(zip(lon[0], lat[0]))), WGS84, [UTM31N]
    )
    return poly


def buffered_chunk_polygon() -> shapely.Polygon:
    """The chunk cell polygon (buffered, x/y) used to select input points."""
    return shapely.buffer(chunk_polygon(), BUFFER)


def tile(
    depth: float,
    size: int = 10,
    resolution: float = RESOLUTION,
    data_shift: float = 0.0,
) -> tuple[xr.Dataset, shapely.Polygon]:
    """A projected raster ("tile") whose bottom-left corner is inside the
    buffered chunk polygon, `depth` meters below its north-east edge, so that
    the tile and the chunk overlap by a small triangle.

    Like EOPF Zarr groups, the x/y coordinates are the pixel centres and the
    spatial metadata is given as STAC "proj:*" attributes. `data_shift` moves the
    data (coordinates and transform) east of the returned extent, i.e. makes the
    extent and the transform disagree.
    """
    chunk_poly = buffered_chunk_polygon()
    xmin, ymin, xmax, ymax = chunk_poly.bounds
    left = (xmin + xmax) / 2
    # northern boundary of the buffered polygon at x = left
    vline = shapely.LineString([(left, ymin), (left, ymax)])
    bottom = shapely.intersection(vline, chunk_poly).bounds[3] - depth
    top = bottom + size * resolution

    data_left = left + data_shift
    transform = affine.Affine(resolution, 0.0, data_left, 0.0, -resolution, top)
    x = data_left + resolution * (np.arange(size) + 0.5)
    y = top - resolution * (np.arange(size) + 0.5)
    attrs = {"proj:code": "EPSG:32631", "proj:transform": list(transform)[:6]}

    ds = xr.Dataset(
        {"b01": (("y", "x"), np.ones((size, size), dtype="f4"), attrs)},
        coords=xr.Coordinates({"x": ("x", x), "y": ("y", y)}, indexes={}),
    )
    extent = shapely.box(left, bottom, left + size * resolution, top)

    return ds, extent


def make_converter(monkeypatch, ds, extent, resampler="nearest", level=LEVEL):
    grid = {"indexing_scheme": "nested", "ellipsoid": {"name": "wgs84"}}
    settings = ConvertSettings.from_dict(
        {
            "group_settings": {
                str(PATH): {
                    "healpix": {**grid, "refinement_level": level},
                    "chunk": {
                        "method": "healpix_cell_dense",
                        "healpix": {**grid, "refinement_level": CHUNK_LEVEL},
                        "chunk_buffer_width": BUFFER,
                    },
                    "resampler": {"name": resampler},
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


def assert_empty_chunk(converter, level) -> None:
    """The chunk's cell ids are written and its data values are fill values."""
    expected_cell_ids = healpix_geo.nested.zoom_to(CELL, CHUNK_LEVEL, level).ravel()
    np.testing.assert_array_equal(
        converter.output_arrays["cell_ids"][:], expected_cell_ids
    )
    b01 = converter.output_arrays["b01"]
    assert np.all(b01[:] == b01.fill_value)


def pixel_centres_in(ds: xr.Dataset, geom: shapely.Geometry) -> int:
    xx, yy = np.meshgrid(ds["x"].values, ds["y"].values)
    return int(shapely.within(shapely.points(xx.ravel(), yy.ravel()), geom).sum())


@pytest.mark.parametrize(
    ["depth", "centres_in_bbox"],
    [
        # the overlap's bounding box contains no pixel centre
        pytest.param(20.0, False, id="bbox-without-centre"),
        # the overlap's bounding box contains pixel centres, the overlap doesn't
        pytest.param(45.0, True, id="bbox-with-centres"),
    ],
)
@pytest.mark.parametrize("resampler", ["nearest", "psf"])
def test_overlap_without_pixel_centre_is_no_input(
    monkeypatch, depth, centres_in_bbox, resampler
) -> None:
    # The (buffered) chunk polygon overlaps the tile extent (pixel edges) by a
    # sliver that contains no pixel centre: there is no input point to resample.
    ds, extent = tile(depth)
    overlap = shapely.intersection(buffered_chunk_polygon(), extent)
    assert overlap.area > 0
    assert pixel_centres_in(ds, overlap) == 0
    assert (pixel_centres_in(ds, shapely.box(*overlap.bounds)) > 0) == centres_in_bbox

    converter = make_converter(monkeypatch, ds, extent, resampler=resampler)

    # the chunk is left empty (instead of failing in the resampler)
    converter.convert(0)

    assert_empty_chunk(converter, LEVEL)

    # like a chunk that does not overlap the dataset extent at all
    assert converter.query_input_points(CELL) is None


@pytest.mark.parametrize("resampler", ["nearest", "psf"])
def test_overlap_with_pixel_centres(monkeypatch, resampler) -> None:
    ds, extent = tile(500.0)
    overlap = shapely.intersection(buffered_chunk_polygon(), extent)
    n_points = pixel_centres_in(ds, overlap)
    assert n_points > 0

    converter = make_converter(monkeypatch, ds, extent, resampler=resampler)

    selected = converter.query_input_points(CELL)
    assert selected is not None
    assert selected.sizes["points"] == n_points

    converter.convert(0)

    b01 = converter.output_arrays["b01"]
    assert np.any(b01[:] != b01.fill_value)


@pytest.mark.parametrize(
    ["level", "resolution", "depth"],
    [
        # Sentinel-2 L2A: 10, 20 and 60 m groups (60 m chunk buffer)
        pytest.param(20, 10.0, 20.0, id="L20-10m"),
        pytest.param(19, 20.0, 45.0, id="L19-20m"),
        pytest.param(17, 60.0, 59.0, id="L17-60m"),
    ],
)
@pytest.mark.parametrize("resampler", ["nearest", "psf"])
def test_points_only_in_chunk_buffer(
    monkeypatch, level, resolution, depth, resampler
) -> None:
    # The tile lies outside the chunk cell but inside its buffer zone: the query
    # selects pixel centres, none of them inside the chunk cell.
    ds, extent = tile(depth, size=20, resolution=resolution)
    overlap = shapely.intersection(buffered_chunk_polygon(), extent)
    assert pixel_centres_in(ds, overlap) > 0
    assert not shapely.intersects(chunk_polygon(), extent)

    converter = make_converter(
        monkeypatch, ds, extent, resampler=resampler, level=level
    )
    selected = converter.query_input_points(CELL)
    assert selected is not None
    assert selected.sizes["points"] > 0

    # the chunk is left empty (instead of failing in the "nearest" resampler,
    # which is forced to fill the chunk's cells)
    converter.convert(0)

    assert_empty_chunk(converter, level)


def test_inconsistent_extent_and_transform_is_logged(monkeypatch, log) -> None:
    # The dataset extent overlaps the chunk widely but the data (coordinates and
    # transform) lies 20 km east of it: no pixel centre is selected, which
    # cannot come from a sliver along the extent.
    ds, extent = tile(500.0, data_shift=20_000.0)
    overlap = shapely.intersection(buffered_chunk_polygon(), extent)
    assert not shapely.buffer(overlap, -RESOLUTION).is_empty

    converter = make_converter(monkeypatch, ds, extent)
    assert converter.query_input_points(CELL) is None

    warnings = [event for event in log.events if event["level"] == "warning"]
    assert len(warnings) == 1
    assert warnings[0]["chunk_cell_id"] == CELL


def test_sliver_without_pixel_centre_is_not_a_warning(monkeypatch, log) -> None:
    ds, extent = tile(20.0)

    converter = make_converter(monkeypatch, ds, extent)
    assert converter.query_input_points(CELL) is None

    assert not [event for event in log.events if event["level"] == "warning"]
    assert [
        event
        for event in log.events
        if event["level"] == "debug" and event.get("chunk_cell_id") == CELL
    ]
