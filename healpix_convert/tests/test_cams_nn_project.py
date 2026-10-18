"""Nearest-neighbour binning of the CAMS converter (``method="nn"``)."""

import healpix_geo
import numpy as np
import pytest

pytest.importorskip("requests")

from healpix_convert.converters.cams import _nn_project  # noqa: E402

LEVEL = 6


def _cams_grid():
    # the regular 0.75 degree grid of CAMS EAC4
    lat = np.arange(90.0, -90.01, -0.75)
    lon = np.arange(0.0, 360.0, 0.75)
    lon2d, lat2d = np.meshgrid(lon, lat)
    return lon2d.ravel(), lat2d.ravel()


def test_constant_field() -> None:
    lon, lat = _cams_grid()

    result = _nn_project(np.full(lon.size, 0.25), lon, lat, LEVEL)

    assert result.shape == (12 * 4**LEVEL,)
    np.testing.assert_allclose(result, 0.25)


def test_binning_on_wgs84() -> None:
    lon, lat = _cams_grid()
    values = np.cos(np.deg2rad(lat)) + np.sin(np.deg2rad(lon))

    result = _nn_project(values, lon, lat, LEVEL)

    assert not np.isnan(result).any()

    # the pixels with input points hold the mean of their points, assigned on
    # the WGS84 ellipsoid (the grid the output metadata declares)
    pix = healpix_geo.nested.lonlat_to_healpix(lon, lat, LEVEL, ellipsoid="WGS84")
    sums = np.bincount(pix.astype(np.int64), weights=values, minlength=result.size)
    counts = np.bincount(pix.astype(np.int64), minlength=result.size)
    has_points = counts > 0
    np.testing.assert_allclose(
        result[has_points], sums[has_points] / counts[has_points], rtol=1e-6
    )

    # the empty pixels are filled from their neighbours
    assert not has_points.all()
    assert result[~has_points].min() >= values.min()
    assert result[~has_points].max() <= values.max()
