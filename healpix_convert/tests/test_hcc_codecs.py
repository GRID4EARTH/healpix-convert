"""Conversion with the Sentinel-2 serializers of healpix-compress.

Skipped where healpix-compress is not installed: the package is not a
dependency of healpix-convert, which only passes codec names to zarr.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from healpix_convert import create_healpix_dataset
from healpix_convert.settings.common import ConvertSettings

pytest.importorskip("hcc.codecs.sentinel2_pyramid")

STAC_ITEM = {
    "type": "Feature",
    "stac_version": "1.1.0",
    "stac_extensions": [],
    "id": "synthetic-input",
    "bbox": [-1.0, -1.0, 1.0, 1.0],
    "geometry": {
        "type": "Polygon",
        "coordinates": [[[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]],
    },
    "properties": {"datetime": "2026-10-11T00:00:00Z"},
    "links": [],
    "assets": {},
}


@pytest.fixture
def input_path(tmp_path):
    """A tiny lat/lon reflectance dataset, with STAC discovery metadata."""
    lon = np.linspace(-1.0, 1.0, 64)
    lat = np.linspace(-1.0, 1.0, 64)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    b04 = (0.2 + 0.1 * np.sin(3 * xx) * np.cos(2 * yy)).astype("float32")

    ds = xr.Dataset(
        {"b04": (("lat", "lon"), b04)},
        coords={
            "lon": (
                "lon",
                lon,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "lat": (
                "lat",
                lat,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
        },
    )
    tree = xr.DataTree.from_dict({"/measurements": ds})
    tree.attrs["stac_discovery"] = STAC_ITEM

    path = tmp_path / "input.zarr"
    tree.to_zarr(path, mode="w", consolidated=True)

    return str(path)


def _convert(tmp_path, input_path, name, codecs):
    settings = ConvertSettings.model_validate(
        {
            "group_settings": {
                "measurements": {
                    "healpix": {
                        "refinement_level": 8,
                        "indexing_scheme": "nested",
                        "ellipsoid": {"name": "wgs84"},
                    },
                    "chunk": {
                        "method": "healpix_cell_dense",
                        "healpix": {
                            "refinement_level": 5,
                            "indexing_scheme": "nested",
                            "ellipsoid": {"name": "wgs84"},
                        },
                    },
                    "resampler": {"name": "nearest"},
                    "codecs": codecs,
                }
            }
        }
    )
    output_path = str(tmp_path / name)
    create_healpix_dataset([input_path], settings, output_path)

    return xr.open_dataset(output_path, engine="zarr", group="measurements")[
        "b04"
    ].values


@pytest.mark.parametrize(
    "codec",
    [
        pytest.param({"name": "hcc-sentinel2", "configuration": {}}, id="lossless"),
        pytest.param(
            {"name": "hcc-sentinel2-pyramid", "configuration": {"relative_step": 0.0}},
            id="pyramid-lossless",
        ),
    ],
)
def test_lossless_serializers(tmp_path, input_path, codec) -> None:
    expected = _convert(tmp_path, input_path, "reference.zarr", None)

    actual = _convert(tmp_path, input_path, "compressed.zarr", [codec])

    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    valid = ~np.isnan(expected)
    # half a digital number of the default Sentinel-2 scaling (Q = 10000)
    np.testing.assert_allclose(actual[valid], expected[valid], rtol=0, atol=0.5e-4)
