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


def _input(tmp_path, *, packed: bool, extra_attrs: dict | None = None):
    attrs = dict(BAND_ATTRS) | {"valid_min": 0, "valid_max": 1000}
    if packed:
        attrs |= {"scale_factor": 0.0136, "add_offset": 0.0}
    if extra_attrs:
        attrs |= extra_attrs

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

    suffix = "packed" if packed else "plain"
    if extra_attrs:
        suffix += "-extra"
    path = tmp_path / f"input-{suffix}.zarr"
    tree.to_zarr(path, mode="w", consolidated=True)
    return str(path)


def _settings(exclude_attrs: list[str] | None = None):
    group_settings: dict = {
        "healpix": {"refinement_level": 4, "indexing_scheme": "nested"},
        "chunk": {"method": "no_chunk"},
        "resampler": {"name": "nearest"},
    }
    if exclude_attrs is not None:
        group_settings["metadata"] = {"exclude_attrs": exclude_attrs}

    return ConvertSettings.model_validate(
        {"group_settings": {"measurements": group_settings}}
    )


@pytest.fixture
def settings():
    return _settings()


def _convert(tmp_path, settings, *, packed, extra_attrs: dict | None = None):
    out = str(tmp_path / f"out-{'packed' if packed else 'plain'}.zarr")
    input_path = _input(tmp_path, packed=packed, extra_attrs=extra_attrs)
    create_healpix_dataset([input_path], settings, out)
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


def test_excluded_attributes_are_not_propagated(tmp_path) -> None:
    # the data provider does not want the input's internal bookkeeping in the output
    extra_attrs = {"_io_config": "whatever", "history": "built by some pipeline"}
    settings = _settings(["history"])

    attrs = _convert(tmp_path, settings, packed=False, extra_attrs=extra_attrs)[
        "radiance"
    ].attrs

    assert "history" not in attrs
    # only the named attribute is dropped
    assert attrs["_io_config"] == "whatever"
    assert attrs["standard_name"] == BAND_ATTRS["standard_name"]


def test_excluded_attributes_accept_patterns(tmp_path) -> None:
    extra_attrs = {
        "_io_config": "whatever",
        "_eopf_attrs": "whatever",
        "history": "built by some pipeline",
    }
    settings = _settings(["_.*", "history"])

    attrs = _convert(tmp_path, settings, packed=False, extra_attrs=extra_attrs)[
        "radiance"
    ].attrs

    assert "_io_config" not in attrs
    assert "_eopf_attrs" not in attrs
    assert "history" not in attrs
    for key, value in BAND_ATTRS.items():
        assert attrs[key] == value


def test_excluded_attributes_do_not_drop_our_own(tmp_path) -> None:
    # exclusion applies to the input metadata, not to the attributes that
    # describe the HEALPix output
    settings = _settings(["grid_mapping", "long_name"])

    attrs = _convert(tmp_path, settings, packed=False)["radiance"].attrs

    assert attrs["grid_mapping"] == "crs"
    assert "long_name" not in attrs
