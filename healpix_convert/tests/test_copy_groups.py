"""Non-spatial groups copied unchanged to the output (``copy_groups`` setting)."""

from __future__ import annotations

import numpy as np
import pydantic
import pytest
import xarray as xr
import zarr

from healpix_convert import create_healpix_dataset
from healpix_convert.settings.common import ConvertSettings


def _stac_item(item_id: str, lat0: float) -> dict:
    return {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": [],
        "id": item_id,
        "bbox": [-1.0, lat0, 1.0, lat0 + 2.0],
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[-1, lat0], [1, lat0], [1, lat0 + 2], [-1, lat0 + 2], [-1, lat0]]
            ],
        },
        "properties": {"datetime": "2026-09-07T00:00:00Z"},
        "links": [],
        "assets": {},
    }


def _instrument(flux_offset: float = 0.0, temperature_offset: float = 0.0) -> xr.Dataset:
    """A table per band and detector, like a per-detector solar flux."""
    bands, detectors, rows = 3, 5, 4
    flux = (np.arange(bands * detectors, dtype="float32").reshape(bands, detectors) + 1000.0)
    ds = xr.Dataset(
        {
            "solar_flux": (
                ("bands", "detectors"),
                flux + np.float32(flux_offset),
                {"units": "mW.m-2.nm-1", "long_name": "in-band solar irradiance"},
            ),
            "detectors": (
                ("detectors",),
                np.arange(detectors, dtype="int64"),
                {"long_name": "instrument detectors"},
            ),
            # a packed integer table with a fill value
            "gain": (
                ("bands",),
                np.array([5.0, 10.0, np.nan], dtype="float32"),
                {"long_name": "gain"},
            ),
            # per image row: differs between the inputs
            "temperature": (
                ("rows",),
                np.arange(rows, dtype="float32") + np.float32(temperature_offset),
                {"units": "K"},
            ),
        },
        attrs={"description": "instrument tables"},
    )
    ds["gain"].encoding = {"scale_factor": 0.5, "_FillValue": np.int16(-1), "dtype": "int16"}
    return ds


def _input(tmp_path, name: str, *, lat0: float = -1.0, **instrument_kwargs) -> str:
    values = np.arange(64, dtype="float32").reshape(8, 8)
    measurements = xr.Dataset(
        {"radiance": (("lat", "lon"), values, {"units": "W"})},
        coords={
            "lon": ("lon", np.linspace(-1, 1, 8), {"standard_name": "longitude"}),
            "lat": ("lat", np.linspace(lat0, lat0 + 2, 8), {"standard_name": "latitude"}),
        },
    )
    tree = xr.DataTree.from_dict(
        {
            "/measurements": measurements,
            "/conditions/instrument": _instrument(**instrument_kwargs),
        }
    )
    tree.attrs["stac_discovery"] = _stac_item(name, lat0)
    path = tmp_path / f"{name}.zarr"
    tree.to_zarr(path, mode="w", consolidated=True)
    return str(path)


def _settings(copy_groups: dict | None = None, **extra) -> ConvertSettings:
    value: dict = {
        "group_settings": {
            "measurements": {
                "healpix": {"refinement_level": 4, "indexing_scheme": "nested"},
                "chunk": {"method": "no_chunk"},
                "resampler": {"name": "nearest"},
            }
        }
    }
    if copy_groups is not None:
        value["copy_groups"] = copy_groups
    value.update(extra)
    return ConvertSettings.model_validate(value)


def test_copy_group_unchanged(tmp_path) -> None:
    src = _input(tmp_path, "a")
    out = tmp_path / "out.zarr"

    create_healpix_dataset([src], _settings({"conditions/instrument": {}}), str(out))

    expected = xr.open_dataset(src, group="conditions/instrument", engine="zarr")
    copied = xr.open_dataset(out, group="conditions/instrument", engine="zarr")
    xr.testing.assert_identical(copied, expected)

    # stored as in the input: packed int16 with its scale and fill value
    raw = zarr.open_group(out, mode="r")["conditions/instrument/gain"]
    assert raw.dtype == np.dtype("int16")
    assert raw.attrs["scale_factor"] == 0.5
    assert raw[...].tolist() == [10, 20, -1]
    assert zarr.open_group(out, mode="r")["conditions/instrument"].attrs["description"] == (
        "instrument tables"
    )

    # in the consolidated metadata of the output
    tree = xr.open_datatree(out, engine="zarr", consolidated=True)
    assert "solar_flux" in tree["conditions/instrument"].data_vars


def test_not_listed_group_is_not_written(tmp_path) -> None:
    src = _input(tmp_path, "a")
    out = tmp_path / "out.zarr"

    create_healpix_dataset([src], _settings(), str(out))

    assert "instrument" not in zarr.open_group(out, mode="r")["conditions"]


def test_drop_variables(tmp_path) -> None:
    src = _input(tmp_path, "a")
    out = tmp_path / "out.zarr"
    settings = _settings({"conditions/instrument": {"drop_variables": ["temp.*", "gain"]}})

    create_healpix_dataset([src], settings, str(out))

    copied = xr.open_dataset(out, group="conditions/instrument", engine="zarr")
    assert sorted(copied.variables) == ["detectors", "solar_flux"]


def test_several_inputs_same_group(tmp_path) -> None:
    srcs = [_input(tmp_path, "a", lat0=-1.0), _input(tmp_path, "b", lat0=0.5)]
    out = tmp_path / "out.zarr"

    create_healpix_dataset(srcs, _settings({"conditions/instrument": {}}), str(out))

    copied = xr.open_dataset(out, group="conditions/instrument", engine="zarr")
    assert copied.sizes["detectors"] == 5  # written once, not stacked


def test_several_inputs_differing_group(tmp_path) -> None:
    srcs = [
        _input(tmp_path, "a", lat0=-1.0),
        _input(tmp_path, "b", lat0=0.5, temperature_offset=3.0),
    ]
    out = tmp_path / "out.zarr"

    with pytest.raises(ValueError, match=r"differs .*\['temperature'\]"):
        create_healpix_dataset(srcs, _settings({"conditions/instrument": {}}), str(out))

    # the differing variable left out: copied
    out2 = tmp_path / "out2.zarr"
    create_healpix_dataset(
        srcs,
        _settings({"conditions/instrument": {"drop_variables": ["temperature"]}}),
        str(out2),
    )
    copied = xr.open_dataset(out2, group="conditions/instrument", engine="zarr")
    assert "temperature" not in copied.variables


def test_spatial_group_cannot_be_copied(tmp_path) -> None:
    src = _input(tmp_path, "a")
    settings = ConvertSettings.model_validate(
        {
            "group_settings": {},
            "exclude_groups": ["measurements"],
            "copy_groups": {"conditions/instrument": {}},
        }
    )
    # a group with latitude / longitude listed in copy_groups
    tree = xr.open_datatree(src, engine="zarr")
    tree["conditions/geo"] = xr.DataTree(tree["measurements"].to_dataset())
    src2 = tmp_path / "geo.zarr"
    tree.to_zarr(src2, mode="w", consolidated=True)
    settings = ConvertSettings.model_validate(
        {
            "group_settings": settings.model_dump()["group_settings"]
            | {
                "measurements": {
                    "healpix": {"refinement_level": 4, "indexing_scheme": "nested"},
                    "chunk": {"method": "no_chunk"},
                    "resampler": {"name": "nearest"},
                }
            },
            "copy_groups": {"conditions/geo": {}},
        }
    )

    with pytest.raises(ValueError, match="converted to HEALPix, not copied"):
        create_healpix_dataset([str(src2)], settings, str(tmp_path / "out.zarr"))


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"copy_groups": {"measurements": {}}}, "both converted"),
        (
            {"copy_groups": {"conditions/instrument": {}}, "exclude_groups": ["conditions/instrument"]},
            "both excluded",
        ),
        ({"copy_groups": {"conditions/instrument": {"drop_variables": ["("]}}}, "invalid regular"),
        ({"copy_groups": {"conditions/instrument": {"unknown": 1}}}, "Extra inputs"),
    ],
)
def test_settings_validation(extra, match) -> None:
    with pytest.raises(pydantic.ValidationError, match=match):
        _settings(**extra)


def test_settings_roundtrip() -> None:
    settings = _settings({"conditions/instrument": {"drop_variables": ["x"]}})
    dumped = settings.model_dump(mode="json")
    assert dumped["copy_groups"] == {"conditions/instrument": {"drop_variables": ["x"]}}
    assert ConvertSettings.model_validate(dumped) == settings


def test_staged_conversion_copies_the_group(tmp_path) -> None:
    """The staged API (as a pipeline runs it): the group is written by the
    preparation stage and is in the consolidated metadata of the root group."""
    from healpix_convert import (
        convert_group_to_healpix,
        create_staging_cache,
        finalize_healpix_dataset,
        prepare_healpix_dataset,
    )

    src = _input(tmp_path, "a")
    out = str(tmp_path / "out.zarr")
    settings = _settings({"conditions/instrument": {"drop_variables": ["temperature"]}})
    cache = create_staging_cache([src], settings)

    prepare_healpix_dataset(cache, settings, out)
    assert "solar_flux" in zarr.open_group(out, mode="r")["conditions/instrument"]
    convert_group_to_healpix("measurements", cache=cache, settings=settings, output_path=out)
    finalize_healpix_dataset(cache, settings, out)

    root = zarr.open_group(out, mode="r", use_consolidated=True)
    consolidated = root.metadata.consolidated_metadata.flattened_metadata
    assert "conditions/instrument/solar_flux" in consolidated
    assert "conditions/instrument/temperature" not in consolidated
