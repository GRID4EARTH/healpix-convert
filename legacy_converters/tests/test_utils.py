import numpy as np
import pyproj
import xarray as xr

from legacy_converters.core.utils import assign_transform_coords


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
