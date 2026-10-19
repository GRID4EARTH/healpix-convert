"""The data type of the resampled variables is a group setting."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
import zarr
from pydantic import ValidationError

from healpix_convert import create_healpix_dataset
from healpix_convert.settings.common import ConvertSettings, HealpixGroupSettings

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
    "properties": {"datetime": "2026-10-19T00:00:00Z"},
    "links": [],
    "assets": {},
}
HEALPIX = {"indexing_scheme": "nested", "ellipsoid": {"name": "wgs84"}}
GROUP = "measurements"


def _group_settings(**kwargs) -> dict:
    return {
        "healpix": {"refinement_level": 8, **HEALPIX},
        "chunk": {
            "method": "healpix_cell_dense",
            "healpix": {"refinement_level": 5, **HEALPIX},
        },
        "resampler": {"name": "nearest"},
        **kwargs,
    }


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [(None, None), ("float16", "float16"), ("f4", "float32"), ("float64", "float64")],
)
def test_dtype_setting(dtype, expected) -> None:
    actual = HealpixGroupSettings.model_validate(_group_settings(dtype=dtype))

    assert actual.dtype == expected


@pytest.mark.parametrize(
    ("dtype", "match"),
    [
        ("int16", "must be a floating-point data type"),
        ("no-such-type", "is not a data type"),
    ],
)
def test_dtype_setting_refused(dtype, match) -> None:
    with pytest.raises(ValidationError, match=match):
        HealpixGroupSettings.model_validate(_group_settings(dtype=dtype))


def _convert(tmp_path, name: str, **group_settings) -> zarr.Array:
    lon = np.linspace(-1.0, 1.0, 32)
    lat = np.linspace(-1.0, 1.0, 32)
    xx, yy = np.meshgrid(lon, lat)
    ds = xr.Dataset(
        {"t2m": (("lat", "lon"), (280 + 10 * np.sin(xx) * np.cos(yy)).astype("f4"))},
        coords={
            "lon": ("lon", lon, {"standard_name": "longitude"}),
            "lat": ("lat", lat, {"standard_name": "latitude"}),
        },
    )
    tree = xr.DataTree.from_dict({f"/{GROUP}": ds})
    tree.attrs["stac_discovery"] = STAC_ITEM
    input_path = str(tmp_path / f"{name}-input.zarr")
    tree.to_zarr(input_path, mode="w", consolidated=True)

    settings = ConvertSettings.model_validate(
        {"group_settings": {GROUP: _group_settings(**group_settings)}}
    )
    output_path = str(tmp_path / f"{name}-output.zarr")
    create_healpix_dataset([input_path], settings, output_path)

    return zarr.open_group(output_path, mode="r")[f"{GROUP}/t2m"]


@pytest.mark.parametrize("resampler", ["nearest", "bilinear"])
def test_resampled_variables_are_written_with_the_dtype(tmp_path, resampler) -> None:
    default = _convert(tmp_path, "default", resampler={"name": resampler})
    actual = _convert(
        tmp_path, "float16", resampler={"name": resampler}, dtype="float16"
    )

    # bilinear writes float64 by default, nearest the data type of the input
    assert default.dtype == ("float64" if resampler == "bilinear" else "float32")
    assert actual.dtype == "float16"
    # the same values, cast (cells without data stay NaN)
    expected = default[:].astype("float16")
    np.testing.assert_array_equal(actual[:], expected)
    assert np.isnan(actual.fill_value)
