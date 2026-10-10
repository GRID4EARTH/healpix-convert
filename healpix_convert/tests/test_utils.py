import affine
import numpy as np
import pyproj
import pytest
import xarray as xr

from healpix_convert.core.utils import (
    assign_transform_coords,
    create_raster_index,
    extract_spatial_info_stac,
)


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


# top-left corner of a small UTM grid (EPSG:32631)
_LEFT, _TOP = 300000.0, 5000040.0


def _eopf_like_dataset(resolution: float, proj_attrs: dict | None = None) -> xr.Dataset:
    """A projected x/y group like those written by EOPF: 1-dimensional x/y
    pixel-centre coordinates, opened without indexes (like `open_datatrees`)."""
    x = _LEFT + resolution * (np.arange(6) + 0.5)
    y = _TOP - resolution * (np.arange(4) + 0.5)
    return xr.Dataset(
        {"b02": (("y", "x"), np.zeros((4, 6), dtype="f4"), proj_attrs or {})},
        coords=xr.Coordinates({"x": ("x", x), "y": ("y", y)}, indexes={}),
    )


@pytest.mark.parametrize("resolution", [10.0, 20.0, 60.0])
def test_extract_spatial_info_stac_without_proj_attrs(resolution) -> None:
    # e.g. EOPF 3.0 SAFE->Zarr output: no "proj:*" attributes on the variables,
    # the transform is derived from the x/y (pixel centre) coordinates
    ds = _eopf_like_dataset(resolution)
    default_crs = pyproj.CRS.from_epsg(32631)

    info = extract_spatial_info_stac(ds, default_crs)

    assert info is not None
    assert info["crs"] == [default_crs]
    (transform,) = info["transform"]

    # the raster index rebuilt by the converters reproduces the original
    # pixel centres (no half-pixel shift)
    rebuilt = create_raster_index(ds, transform, "x", "y")
    np.testing.assert_allclose(rebuilt["x"].values, ds["x"].values, rtol=0, atol=1e-6)
    np.testing.assert_allclose(rebuilt["y"].values, ds["y"].values, rtol=0, atol=1e-6)

    # like "proj:transform": maps the top-left corner of the pixels
    expected = affine.Affine(resolution, 0.0, _LEFT, 0.0, -resolution, _TOP)
    assert transform.almost_equals(expected)


def test_extract_spatial_info_stac_transform_consistent_with_proj_attrs() -> None:
    # the same grid described by "proj:*" attributes gives the same transform
    resolution = 20.0
    proj_attrs = {
        "proj:code": "EPSG:32631",
        "proj:transform": [resolution, 0.0, _LEFT, 0.0, -resolution, _TOP],
    }
    default_crs = pyproj.CRS.from_epsg(32631)

    with_attrs = extract_spatial_info_stac(
        _eopf_like_dataset(resolution, proj_attrs), default_crs
    )
    without_attrs = extract_spatial_info_stac(
        _eopf_like_dataset(resolution), default_crs
    )

    assert with_attrs is not None and without_attrs is not None
    assert with_attrs["transform"][0].almost_equals(without_attrs["transform"][0])
