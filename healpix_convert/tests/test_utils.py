import numpy as np
import pyproj
import xarray as xr

from healpix_convert.core.utils import assign_transform_coords


def test_assign_transform_coords() -> None:
    ds = xr.Dataset(
        coords={
            "y": np.array([5399970, 5351970, 5303970]),
            "x": np.array([300030, 348030, 396030]),
        }
    )

    source_crs = pyproj.CRS.from_epsg(32630)
    target_crs = pyproj.CRS.from_epsg(4326)

    lon = np.array(
        [
            [-5.718936530344232, -5.695944792637786, -5.6734882893110905],
            [-5.066807118040187, -5.049318939665552, -5.032238145234274],
            [-4.414257908925475, -4.402285787898621, -4.390592689486813],
        ],
        dtype="float64",
    )
    lat = np.array(
        [
            [48.72065491331985, 48.28931067423311, 47.857925872687595],
            [48.73420506236285, 48.302658101913416, 47.87107394324772],
            [48.74406429191886, 48.31236976243821, 47.88064048668145],
        ],
        dtype="float64",
    )

    actual = assign_transform_coords(ds, source_crs, target_crs).drop_vars(["x", "y"])
    expected = xr.Dataset(coords={"lon": (["x", "y"], lon), "lat": (["x", "y"], lat)})

    xr.testing.assert_allclose(actual, expected)


def test_compute_output_chunk_info_world():
    import shapely

    from healpix_convert.core.utils import compute_output_chunk_info

    ids, full = compute_output_chunk_info(shapely.box(-180, -90, 180, 90), 2, "wgs84")
    np.testing.assert_array_equal(ids, np.arange(192, dtype=np.uint64))
    assert full.all()


def test_compute_output_chunk_info_regional():
    import shapely

    from healpix_convert.core.utils import compute_output_chunk_info

    ids, full = compute_output_chunk_info(shapely.box(2, 48, 2.01, 48.01), 2, "wgs84")
    assert 0 < ids.size < 192
    assert ids.size == full.size


def test_extract_spatial_info_stac_2d_xy() -> None:
    from healpix_convert.core.utils import (
        extract_spatial_info_cf,
        extract_spatial_info_stac,
    )

    # no proj:* attributes, 2-D x/y image coordinates and 2-D lat/lon coordinates
    shape = (3, 4)
    dims = ("rows", "columns")
    xx, yy = np.meshgrid(np.arange(4) * 500.0, np.arange(3) * 500.0)
    ds = xr.Dataset(
        {"var": (dims, np.zeros(shape))},
        coords={
            "x": (dims, xx),
            "y": (dims, yy),
            "longitude": (dims, xx / 1e5, {"standard_name": "longitude"}),
            "latitude": (dims, 45 + yy / 1e5, {"standard_name": "latitude"}),
        },
    )

    assert extract_spatial_info_stac(ds, pyproj.CRS.from_epsg(4326)) is None

    info = extract_spatial_info_cf(ds)
    assert info is not None
    assert info["spatial_dimensions"] == {"rows": 3, "columns": 4}
    assert not info["is_rectilinear"]
