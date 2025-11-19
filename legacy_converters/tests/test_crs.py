import numpy as np
import pyproj
import pytest
import xarray as xr

from legacy_converters import crs


@pytest.mark.parametrize(
    ["crs_like", "expected"],
    (
        pytest.param(4326, pyproj.CRS.from_epsg(4326), id="crs-like"),
        pytest.param(pyproj.CRS.from_epsg(4326), pyproj.CRS.from_epsg(4326), id="crs"),
    ),
)
def test_ensure_crs(crs_like, expected):
    actual = crs.ensure_crs(crs_like)

    assert actual == expected


def test_create_transformer():
    src_crs_like = 4326
    target_crs_like = 4347

    expected = pyproj.Transformer.from_crs(
        pyproj.CRS.from_epsg(src_crs_like),
        pyproj.CRS.from_epsg(target_crs_like),
        always_xy=True,
    )

    actual = crs.create_transformer(src_crs_like, target_crs_like)

    assert actual == expected


@pytest.mark.parametrize(
    ["ds", "expected"],
    (
        pytest.param(xr.Dataset(), xr.Dataset(), id="empty"),
        pytest.param(
            xr.Dataset(coords={"time": [0, 1]}),
            xr.Dataset(coords={"time": [0, 1]}),
            id="temporal-coords",
        ),
        pytest.param(
            xr.Dataset(coords={"x": [300150, 409650], "y": [5399970, 5399850]}),
            xr.Dataset(
                coords={
                    "x": [300150, 409650],
                    "y": [5399970, 5399850],
                    "lon": (
                        ["x", "y"],
                        np.array(
                            [
                                [-5.717306839742361, -5.717248714305221],
                                [-4.229038940149932, -4.229012618606299],
                            ]
                        ),
                        {
                            "standard_name": "longitude",
                            "axis": "X",
                            "long_name": "longitude coordinate",
                            "units": "degrees_east",
                        },
                    ),
                    "lat": (
                        ["x", "y"],
                        np.array(
                            [
                                [48.72069338556936, 48.71961507426066],
                                [48.746188884221624, 48.745109611036575],
                            ]
                        ),
                        {
                            "standard_name": "latitude",
                            "axis": "Y",
                            "long_name": "latitude coordinate",
                            "units": "degrees_north",
                        },
                    ),
                }
            ),
            id="spatial-coords",
        ),
    ),
)
def test_maybe_convert(ds, expected):
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS.from_epsg(32630), pyproj.CRS.from_epsg(4326), always_xy=True
    )

    actual = crs.maybe_convert(ds, transformer)

    xr.testing.assert_allclose(actual, expected)
