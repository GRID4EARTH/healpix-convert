"""Non-spatial groups copied unchanged to the output (``copy_groups`` setting)."""

from __future__ import annotations

from pathlib import PurePath

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


def _instrument(
    flux_offset: float = 0.0, temperature_offset: float = 0.0
) -> xr.Dataset:
    """A table per band and detector, like a per-detector solar flux."""
    bands, detectors, rows = 3, 5, 4
    flux = (
        np.arange(bands * detectors, dtype="float32").reshape(bands, detectors) + 1000.0
    )
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
    ds["gain"].encoding = {
        "scale_factor": 0.5,
        "_FillValue": np.int16(-1),
        "dtype": "int16",
    }
    return ds


def _input(
    tmp_path,
    name: str,
    *,
    lat0: float = -1.0,
    zarr_format: int = 3,
    **instrument_kwargs,
) -> str:
    values = np.arange(64, dtype="float32").reshape(8, 8)
    measurements = xr.Dataset(
        {"radiance": (("lat", "lon"), values, {"units": "W"})},
        coords={
            "lon": ("lon", np.linspace(-1, 1, 8), {"standard_name": "longitude"}),
            "lat": (
                "lat",
                np.linspace(lat0, lat0 + 2, 8),
                {"standard_name": "latitude"},
            ),
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
    tree.to_zarr(path, mode="w", consolidated=True, zarr_format=zarr_format)
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
    assert zarr.open_group(out, mode="r")["conditions/instrument"].attrs[
        "description"
    ] == ("instrument tables")

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
    settings = _settings(
        {"conditions/instrument": {"drop_variables": ["temp.*", "gain"]}}
    )

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
    # a group with latitude / longitude listed in copy_groups
    tree = xr.open_datatree(src, engine="zarr")
    tree["conditions/geo"] = xr.DataTree(tree["measurements"].to_dataset())
    src2 = tmp_path / "geo.zarr"
    tree.to_zarr(src2, mode="w", consolidated=True)
    settings = _settings({"conditions/geo": {}})

    with pytest.raises(ValueError, match="converted to HEALPix, not copied"):
        create_healpix_dataset([str(src2)], settings, str(tmp_path / "out.zarr"))


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"copy_groups": {"measurements": {}}}, "both converted"),
        (
            {
                "copy_groups": {"conditions/instrument": {}},
                "exclude_groups": ["conditions/instrument"],
            },
            "both excluded",
        ),
        (
            {"copy_groups": {"conditions/instrument": {"drop_variables": ["("]}}},
            "invalid regular",
        ),
        ({"copy_groups": {"conditions/instrument": {"unknown": 1}}}, "Extra inputs"),
        (
            {
                "copy_groups": {"conditions/instrument": {}},
                "exclude_groups": ["conditions"],
            },
            "parent 'conditions' is excluded",
        ),
        (
            {
                "copy_groups": {"measurements/extra": {}},
                "multiscale_settings": {"groups": ["measurements"]},
            },
            "is a multiscale group",
        ),
        ({"copy_groups": {"/": {}}}, "root group cannot be copied"),
        (
            {
                "copy_groups": {"conditions/instrument": {}},
                "exclude_groups": ["/conditions"],
            },
            "parent 'conditions' is excluded",
        ),
    ],
)
def test_settings_validation(extra, match) -> None:
    with pytest.raises(pydantic.ValidationError, match=match):
        _settings(**extra)


def test_settings_roundtrip() -> None:
    settings = _settings({"conditions/instrument": {"drop_variables": ["x"]}})
    dumped = settings.model_dump(mode="json")
    assert dumped["copy_groups"] == {
        "conditions/instrument": {"drop_variables": ["x"], "per_input_variables": []}
    }
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
    convert_group_to_healpix(
        "measurements", cache=cache, settings=settings, output_path=out
    )
    finalize_healpix_dataset(cache, settings, out)

    root = zarr.open_group(out, mode="r", use_consolidated=True)
    consolidated = root.metadata.consolidated_metadata.flattened_metadata
    assert "conditions/instrument/solar_flux" in consolidated
    assert "conditions/instrument/temperature" not in consolidated


def test_leading_slash_is_the_same_path(tmp_path) -> None:
    settings = _settings({"/conditions/instrument/": {}})
    assert list(settings.copy_groups) == [PurePath("conditions/instrument")]


def _tables_input(tmp_path, name: str, variables: dict, *, zarr_format: int = 3) -> str:
    """An input whose copied group holds the given raw variables (no CF decoding
    when written)."""
    values = np.arange(64, dtype="float32").reshape(8, 8)
    measurements = xr.Dataset(
        {"radiance": (("lat", "lon"), values)},
        coords={
            "lon": ("lon", np.linspace(-1, 1, 8), {"standard_name": "longitude"}),
            "lat": ("lat", np.linspace(-1, 1, 8), {"standard_name": "latitude"}),
        },
    )
    tables = xr.Dataset()
    for vname, (dims, data, attrs, encoding) in variables.items():
        tables[vname] = xr.Variable(dims, data, attrs)
        tables[vname].encoding = encoding
    tree = xr.DataTree.from_dict(
        {"/measurements": measurements, "/conditions/tables": tables}
    )
    tree.attrs["stac_discovery"] = _stac_item(name, -1.0)
    path = tmp_path / f"{name}.zarr"
    tree.to_zarr(path, mode="w", consolidated=True, zarr_format=zarr_format)
    return str(path)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_stored_values_are_copied_as_they_are(tmp_path, zarr_format) -> None:
    """missing_value, 64-bit integers with a fill value, packed integers: the
    stored values and the reading attributes are those of the input."""
    big = np.array([2**60 + 1, 5, -1], dtype="int64")
    src = _tables_input(
        tmp_path,
        f"t{zarr_format}",
        {
            "with_missing": (
                ("n",),
                np.array([1, 2, -999], "int16"),
                {"missing_value": -999},
                {},
            ),
            "big": (("n",), big, {}, {"_FillValue": np.int64(-1)}),
            "packed": (
                ("n",),
                np.array([0.5, np.nan, 2.0]),
                {"units": "W"},
                {"dtype": "int16", "scale_factor": 0.5, "_FillValue": np.int16(-1)},
            ),
            "scalar": ((), np.float32(3.5), {}, {}),
        },
        zarr_format=zarr_format,
    )
    out = tmp_path / "out.zarr"
    create_healpix_dataset([src], _settings({"conditions/tables": {}}), str(out))

    raw_in = xr.open_dataset(
        src, group="conditions/tables", engine="zarr", decode_cf=False
    )
    raw_out = xr.open_dataset(
        out, group="conditions/tables", engine="zarr", decode_cf=False
    )
    for name in ("with_missing", "big", "packed", "scalar"):
        assert raw_out[name].dtype == raw_in[name].dtype, name
        np.testing.assert_array_equal(raw_out[name].values, raw_in[name].values)
        for key in ("missing_value", "scale_factor", "_FillValue", "units"):
            assert (key in raw_out[name].attrs) == (key in raw_in[name].attrs), (
                name,
                key,
            )
            if key in raw_in[name].attrs:
                np.testing.assert_array_equal(
                    raw_out[name].attrs[key], raw_in[name].attrs[key]
                )
    assert raw_out["big"].values[0] == 2**60 + 1
    # decoded as the input
    xr.testing.assert_identical(
        xr.open_dataset(out, group="conditions/tables", engine="zarr"),
        xr.open_dataset(src, group="conditions/tables", engine="zarr"),
    )


def test_string_variable_is_refused_early(tmp_path) -> None:
    """Refused before anything is converted or written: no partial output."""
    src = _tables_input(
        tmp_path,
        "s",
        {"names": (("n",), np.array(["a", "bb", "ccc"], dtype=object), {}, {})},
    )
    out = tmp_path / "o.zarr"
    with pytest.raises(ValueError, match="'names' has the data type .*not supported"):
        create_healpix_dataset([src], _settings({"conditions/tables": {}}), str(out))
    assert not out.exists()
    # left out: copied
    create_healpix_dataset(
        [src],
        _settings({"conditions/tables": {"drop_variables": ["names"]}}),
        str(tmp_path / "o2.zarr"),
    )


def test_differing_scale_is_a_difference(tmp_path) -> None:
    def tables(scale):
        return {
            "g": (
                ("n",),
                np.array([1.0, 2.0]),
                {},
                {"dtype": "int16", "scale_factor": scale},
            )
        }

    a = _tables_input(tmp_path, "a", tables(0.5))
    b = _tables_input(tmp_path, "b", tables(0.25))
    with pytest.raises(ValueError, match=r"differs .*\['g'\]"):
        create_healpix_dataset(
            [a, b], _settings({"conditions/tables": {}}), str(tmp_path / "o.zarr")
        )


def _raw_zarr_input(
    tmp_path, name: str, zarr_format: int, arrays: dict, group_attrs=None
) -> str:
    """An input written with Zarr itself (as a product is), so that the stored
    values, fill values and attributes are exactly those given."""
    path = _input(tmp_path, f"{name}-v{zarr_format}", zarr_format=zarr_format)
    root = zarr.open_group(path, mode="a", zarr_format=zarr_format)
    group = root.require_group("conditions/tables")
    group.attrs.update(group_attrs or {})
    for vname, (values, fill, attrs, dims) in arrays.items():
        values = np.asarray(values)
        kwargs = {"dimension_names": dims} if zarr_format == 3 else {}
        attributes = dict(attrs)
        if zarr_format == 2:
            attributes["_ARRAY_DIMENSIONS"] = list(dims)
        a = group.create_array(
            vname,
            shape=values.shape,
            dtype=values.dtype,
            chunks=tuple(max(1, n) for n in values.shape),
            fill_value=fill,
            attributes=attributes,
            **kwargs,
        )
        a[...] = values
    zarr.consolidate_metadata(path)
    return path


ARRAYS = {
    "nan_and_fill": (np.array([1.0, np.nan, 3.0]), -999.0, {"units": "K"}, ("n",)),
    "flag": (np.array([True, False, True]), False, {}, ("n",)),
    "when": (
        np.array(["2020-01-01", "NaT", "2020-01-03"], dtype="datetime64[ns]"),
        None,
        {},
        ("n",),
    ),
    "u64": (
        np.array([2**64 - 2, 7, 0], dtype="uint64"),
        0,
        {"flag_values": [1, 2]},
        ("n",),
    ),
    "f16": (np.array([0.5, 1.5, 2.5], dtype="float16"), None, {}, ("n",)),
    "empty": (np.zeros((0,), dtype="int32"), -1, {}, ("m",)),
}


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_arrays_copied_as_stored_by_zarr(tmp_path, zarr_format) -> None:
    """The arrays as Zarr stores them: values (NaN beside a fill value, booleans,
    datetimes with NaT, 64-bit unsigned, float16, empty), data type, fill value,
    attributes, dimension names; and read by xarray as the input."""
    src = _raw_zarr_input(
        tmp_path, "z", zarr_format, ARRAYS, group_attrs={"title": "tables"}
    )
    out = tmp_path / "out.zarr"
    create_healpix_dataset([src], _settings({"conditions/tables": {}}), str(out))

    gin = zarr.open_group(src, mode="r")["conditions/tables"]
    gout = zarr.open_group(out, mode="r")["conditions/tables"]
    assert dict(gout.attrs) == {"title": "tables"}
    for name in ARRAYS:
        a, b = gin[name], gout[name]
        assert b.dtype == a.dtype, name
        np.testing.assert_array_equal(b[...], a[...])
        if zarr_format == 3:
            assert dict(b.attrs) == dict(a.attrs), name
            np.testing.assert_array_equal(
                np.asarray(b.fill_value), np.asarray(a.fill_value)
            )
            assert b.metadata.dimension_names == a.metadata.dimension_names, name
        else:
            assert b.metadata.dimension_names == tuple(
                a.attrs["_ARRAY_DIMENSIONS"]
            ), name
    assert np.isnan(
        gout["nan_and_fill"][1]
    )  # the NaN is not replaced by the fill value
    # a boolean is read as booleans (its fill value False is not a missing value)
    assert (
        xr.open_dataset(out, group="conditions/tables", engine="zarr")["flag"].dtype
        == bool
    )
    for name in ("nan_and_fill", "u64", "f16"):
        xr.testing.assert_identical(
            xr.open_dataset(out, group="conditions/tables", engine="zarr")[name],
            xr.open_dataset(src, group="conditions/tables", engine="zarr")[name],
        )


def test_group_of_attributes_only_keeps_them(tmp_path) -> None:
    src = _raw_zarr_input(tmp_path, "a", 3, {}, group_attrs={"keep": "me"})
    out = tmp_path / "out.zarr"
    create_healpix_dataset([src], _settings({"conditions/tables": {}}), str(out))
    assert dict(zarr.open_group(out, mode="r")["conditions/tables"].attrs) == {
        "keep": "me"
    }


def test_identical_inputs_with_nat_and_differing_attributes(tmp_path) -> None:
    arrays = {
        "when": ARRAYS["when"],
        "v": (np.array([1, 2]), -1, {"valid_range": [0, 10]}, ("k",)),
    }
    a = _raw_zarr_input(tmp_path, "a", 3, arrays)
    b = _raw_zarr_input(tmp_path, "b", 3, arrays)
    out = tmp_path / "out.zarr"
    create_healpix_dataset(
        [a, b], _settings({"conditions/tables": {}}), str(out)
    )  # NaT == NaT

    arrays2 = dict(arrays, v=(np.array([1, 2]), -1, {"valid_range": [0, 5]}, ("k",)))
    c = _raw_zarr_input(tmp_path, "c", 3, arrays2)
    with pytest.raises(ValueError, match=r"differs .*\['v'\]"):
        create_healpix_dataset(
            [a, c], _settings({"conditions/tables": {}}), str(tmp_path / "o2.zarr")
        )


def test_dry_run_copies_into_memory(tmp_path) -> None:
    src = _input(tmp_path, "a")
    tree = create_healpix_dataset(
        [src],
        _settings({"conditions/instrument": {}}),
        str(tmp_path / "out.zarr"),
        dry_run=True,
    )
    assert "solar_flux" in tree["conditions/instrument"].data_vars


@pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[ns]"])
def test_v2_datetime_with_fill_value_is_refused_by_name(tmp_path, dtype) -> None:
    values = np.array([1, 2], dtype="i8").astype(dtype)
    src = _raw_zarr_input(tmp_path, "d", 2, {"t": (values, values[0], {}, ("n",))})
    out = tmp_path / "o.zarr"
    with pytest.raises(ValueError, match="variable 't': a Zarr v2 .* fill value"):
        create_healpix_dataset([src], _settings({"conditions/tables": {}}), str(out))
    assert not out.exists()


@pytest.mark.parametrize(
    "other",
    [
        {"v": (np.array([1, 2], "int32"), -1, {}, ("n",))},  # fill value differs
        {"v": (np.array([1, 2], "int64"), -2, {}, ("n",))},  # data type differs
        {
            "v": (np.array([1, 2], "int32"), -2, {"a": [1, [2, 3]]}, ("n",))
        },  # ragged attribute
    ],
)
def test_inputs_differing_in_fill_dtype_or_attributes(tmp_path, other) -> None:
    a = _raw_zarr_input(
        tmp_path,
        "a",
        3,
        {"v": (np.array([1, 2], "int32"), -2, {"a": [1, [2, 4]]}, ("n",))},
    )
    b = _raw_zarr_input(tmp_path, "b", 3, other)
    with pytest.raises(ValueError, match=r"differs .*\['v'\]"):
        create_healpix_dataset(
            [a, b], _settings({"conditions/tables": {}}), str(tmp_path / "o.zarr")
        )


def test_nested_attributes_with_nan_compare_equal(tmp_path) -> None:
    arrays = {
        "v": (
            np.array([1.0]),
            None,
            {"meta": {"v": float("nan"), "l": [1, [2, 3]]}},
            ("n",),
        )
    }
    a = _raw_zarr_input(tmp_path, "a", 3, arrays)
    b = _raw_zarr_input(tmp_path, "b", 3, arrays)
    create_healpix_dataset(
        [a, b], _settings({"conditions/tables": {}}), str(tmp_path / "o.zarr")
    )


def test_all_variables_dropped_keeps_the_group_attributes(tmp_path) -> None:
    src = _raw_zarr_input(
        tmp_path, "a", 3, {"v": (np.array([1]), 0, {}, ("n",))}, group_attrs={"k": 1}
    )
    out = tmp_path / "o.zarr"
    create_healpix_dataset(
        [src], _settings({"conditions/tables": {"drop_variables": [".*"]}}), str(out)
    )
    g = zarr.open_group(out, mode="r")["conditions/tables"]
    assert dict(g.attrs) == {"k": 1} and not list(g.arrays())


def test_inputs_differing_in_group_attributes(tmp_path) -> None:
    arrays = {"v": (np.array([1, 2], "int32"), -1, {}, ("n",))}
    a = _raw_zarr_input(tmp_path, "a", 3, arrays, group_attrs={"made": "x"})
    b = _raw_zarr_input(tmp_path, "b", 3, arrays, group_attrs={"made": "y"})
    out = tmp_path / "o.zarr"
    with pytest.raises(ValueError, match="attributes of the group"):
        create_healpix_dataset([a, b], _settings({"conditions/tables": {}}), str(out))
    assert not out.exists()  # checked before anything is written


def test_differing_inputs_leave_no_output(tmp_path) -> None:
    srcs = [
        _input(tmp_path, "a", lat0=-1.0),
        _input(tmp_path, "b", lat0=0.5, temperature_offset=3.0),
    ]
    out = tmp_path / "out.zarr"
    with pytest.raises(ValueError, match="differs"):
        create_healpix_dataset(srcs, _settings({"conditions/instrument": {}}), str(out))
    assert not out.exists()


def test_copied_group_has_its_own_consolidated_metadata(tmp_path) -> None:
    src = _input(tmp_path, "a")
    out = tmp_path / "out.zarr"
    create_healpix_dataset([src], _settings({"conditions/instrument": {}}), str(out))
    group = zarr.open_group(
        out, path="conditions/instrument", mode="r", use_consolidated=True
    )
    assert "solar_flux" in group.metadata.consolidated_metadata.metadata


def test_per_input_variables_one_sub_group_per_input(tmp_path) -> None:
    # the variable that differs between the inputs is copied once per input, into
    # inputs/<input name>; the others once, and must still be the same
    srcs = [
        _input(tmp_path, "frame_a", lat0=-1.0),
        _input(tmp_path, "frame_b", lat0=0.5, temperature_offset=3.0),
    ]
    out = tmp_path / "out.zarr"
    settings = _settings(
        {"conditions/instrument": {"per_input_variables": ["temp.*"]}}
    )

    create_healpix_dataset(srcs, settings, str(out))

    root = zarr.open_group(out, mode="r")
    group = root["conditions/instrument"]
    assert "temperature" not in group  # not at the group level
    assert sorted(a for a, _ in group.arrays()) == ["detectors", "gain", "solar_flux"]
    for name, offset in (("frame_a", 0.0), ("frame_b", 3.0)):
        array = group[f"inputs/{name}/temperature"]
        assert array[...].tolist() == [offset + i for i in range(4)]
        assert array.attrs["units"] == "K"
        assert tuple(array.metadata.dimension_names) == ("rows",)
    # in the consolidated metadata of the output
    tree = xr.open_datatree(out, engine="zarr", consolidated=True)
    assert "temperature" in tree["conditions/instrument/inputs/frame_b"].data_vars

    # a shared variable that differs is still refused
    srcs2 = [
        _input(tmp_path, "frame_c", lat0=-1.0),
        _input(tmp_path, "frame_d", lat0=0.5, flux_offset=1.0),
    ]
    with pytest.raises(ValueError, match=r"differs .*\['solar_flux'\]"):
        create_healpix_dataset(srcs2, settings, str(tmp_path / "out2.zarr"))


def test_per_input_variables_single_input_same_layout(tmp_path) -> None:
    # one input gives one sub-group: the layout does not depend on the number of inputs
    src = _input(tmp_path, "only_frame")
    out = tmp_path / "out.zarr"

    create_healpix_dataset(
        [src],
        _settings({"conditions/instrument": {"per_input_variables": ["temperature"]}}),
        str(out),
    )

    group = zarr.open_group(out, mode="r")["conditions/instrument"]
    assert group["inputs/only_frame/temperature"].shape == (4,)
    assert "temperature" not in group


def test_per_input_variables_input_name_is_the_last_component(tmp_path) -> None:
    from healpix_convert.copy_groups import input_name

    assert input_name("https://example.org/a/S3A_SL_1_RBT_x.zarr") == "S3A_SL_1_RBT_x"
    assert input_name("/data/S3A_SL_1_RBT_x.zarr/") == "S3A_SL_1_RBT_x"
    assert input_name("plain") == "plain"


def test_per_input_variables_same_input_name_is_refused(tmp_path) -> None:
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    srcs = [_input(tmp_path / "x", "same"), _input(tmp_path / "y", "same", lat0=0.5)]

    with pytest.raises(ValueError, match="same name"):
        create_healpix_dataset(
            srcs,
            _settings({"conditions/instrument": {"per_input_variables": ["temperature"]}}),
            str(tmp_path / "out.zarr"),
        )


def test_per_input_variables_invalid_pattern() -> None:
    with pytest.raises(pydantic.ValidationError, match="invalid regular expression"):
        _settings({"conditions/instrument": {"per_input_variables": ["("]}})


def test_per_input_variables_all_variables_of_a_group(tmp_path) -> None:
    # a group that holds only per-row tables (e.g. times per row): every variable per input
    srcs = [
        _input(tmp_path, "frame_a", lat0=-1.0),
        _input(tmp_path, "frame_b", lat0=0.5, temperature_offset=3.0, flux_offset=2.0),
    ]
    out = tmp_path / "out.zarr"

    create_healpix_dataset(
        srcs,
        _settings({"conditions/instrument": {"per_input_variables": [".*"]}}),
        str(out),
    )

    group = zarr.open_group(out, mode="r")["conditions/instrument"]
    assert not list(group.arrays())
    assert sorted(a for a, _ in group["inputs/frame_b"].arrays()) == [
        "detectors",
        "gain",
        "solar_flux",
        "temperature",
    ]
    assert group["inputs/frame_b/solar_flux"][0, 0] == 1002.0
