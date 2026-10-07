"""The input's description of its variables must survive conversion."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

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
    "properties": {"datetime": "2026-09-07T00:00:00Z"},
    "links": [],
    "assets": {},
}

# what an EOPF OLCI radiance band actually carries
BAND_ATTRS = {
    "units": "mW.m-2.sr-1.nm-1",
    "standard_name": "toa_upwelling_spectral_radiance",
    "long_name": "TOA radiance for OLCI acquisition band 01",
}


def _input(tmp_path, *, packed: bool):
    attrs = dict(BAND_ATTRS) | {"valid_min": 0, "valid_max": 1000}
    if packed:
        attrs |= {"scale_factor": 0.0136, "add_offset": 0.0}

    values = np.arange(64, dtype="uint16" if packed else "float32").reshape(8, 8)
    ds = xr.Dataset(
        {"radiance": (("lat", "lon"), values, attrs)},
        coords={
            "lon": ("lon", np.linspace(-1, 1, 8), {"standard_name": "longitude"}),
            "lat": ("lat", np.linspace(-1, 1, 8), {"standard_name": "latitude"}),
        },
    )
    tree = xr.DataTree.from_dict({"/measurements": ds})
    tree.attrs["stac_discovery"] = STAC_ITEM

    path = tmp_path / f"input-{'packed' if packed else 'plain'}.zarr"
    tree.to_zarr(path, mode="w", consolidated=True)
    return str(path)


@pytest.fixture
def settings():
    return ConvertSettings.model_validate(
        {
            "group_settings": {
                "measurements": {
                    "healpix": {"refinement_level": 4, "indexing_scheme": "nested"},
                    "chunk": {"method": "no_chunk"},
                    "resampler": {"name": "nearest"},
                }
            }
        }
    )


def _convert(tmp_path, settings, *, packed):
    out = str(tmp_path / f"out-{'packed' if packed else 'plain'}.zarr")
    create_healpix_dataset([_input(tmp_path, packed=packed)], settings, out)
    return xr.open_dataset(out, group="measurements", engine="zarr")


def test_description_survives_conversion(tmp_path, settings) -> None:
    # without this, a CF reader cannot tell what the resampled values are
    attrs = _convert(tmp_path, settings, packed=False)["radiance"].attrs

    for key, value in BAND_ATTRS.items():
        assert attrs[key] == value


def test_our_attributes_win(tmp_path, settings) -> None:
    attrs = _convert(tmp_path, settings, packed=False)["radiance"].attrs

    assert attrs["grid_mapping"] == "crs"


def test_packing_attributes_are_not_carried(tmp_path, settings) -> None:
    # the output is unpacked floats: re-applying the input's packing on read
    # would corrupt the values
    attrs = _convert(tmp_path, settings, packed=True)["radiance"].attrs

    assert "scale_factor" not in attrs
    assert "add_offset" not in attrs
    # CF defines these against the stored (packed) values
    assert "valid_min" not in attrs
    assert "valid_max" not in attrs
    # but the description is still carried
    assert attrs["standard_name"] == BAND_ATTRS["standard_name"]


def test_valid_range_kept_when_unpacked(tmp_path, settings) -> None:
    attrs = _convert(tmp_path, settings, packed=False)["radiance"].attrs

    assert attrs["valid_min"] == 0
    assert attrs["valid_max"] == 1000
