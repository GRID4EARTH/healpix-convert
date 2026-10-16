"""Cells without data of a floating output are NaN, not zeros."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
import zarr

from healpix_convert import create_healpix_dataset
from healpix_convert.settings.common import ConvertSettings

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
    "properties": {"datetime": "2026-10-16T00:00:00Z"},
    "links": [],
    "assets": {},
}


@pytest.mark.parametrize(
    ["dtype", "resampler", "expected_fill"],
    [
        pytest.param("float32", {"name": "nearest"}, np.nan, id="nearest-float"),
        pytest.param(
            "float32",
            {"name": "bilinear", "init_params": {"threshold": 0.1}},
            np.nan,
            id="bilinear",
        ),
        pytest.param("uint16", {"name": "nearest"}, 0, id="nearest-integer"),
    ],
)
def test_fill_value(tmp_path, dtype, resampler, expected_fill):
    lon = np.linspace(-1.0, 1.0, 16)
    lat = np.linspace(-1.0, 1.0, 16)
    ds = xr.Dataset(
        {"v": (("lat", "lon"), np.arange(1, 257).reshape(16, 16).astype(dtype))},
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
    input_path = str(tmp_path / "input.zarr")
    tree.to_zarr(input_path, mode="w", consolidated=True)
    settings = ConvertSettings.model_validate(
        {
            "group_settings": {
                "measurements": {
                    "healpix": {
                        "refinement_level": 7,
                        "indexing_scheme": "nested",
                        "ellipsoid": {"name": "wgs84"},
                    },
                    # dense chunks larger than the input: most cells have no data
                    "chunk": {
                        "method": "healpix_cell_dense",
                        "healpix": {
                            "refinement_level": 2,
                            "indexing_scheme": "nested",
                            "ellipsoid": {"name": "wgs84"},
                        },
                    },
                    "resampler": resampler,
                }
            }
        }
    )
    output_path = str(tmp_path / "output.zarr")
    create_healpix_dataset([input_path], settings, output_path)

    v = zarr.open_group(output_path, mode="r")["measurements/v"]
    np.testing.assert_equal(v.fill_value, expected_fill)
    if resampler["name"] == "bilinear":
        # the nearest resampler fills every cell; the bilinear one leaves the
        # cells away from the input without data
        values = v[:]
        # no zero was written for the cells without data
        assert np.isnan(values).any() and not (values == 0).any()
