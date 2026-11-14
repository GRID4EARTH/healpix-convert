"""Variables left out (``drop_variables``) and written as the value of the nearest
input point (``nearest_variables``)."""

from __future__ import annotations

from pathlib import PurePath

import numpy as np
import pytest
import shapely
import xarray as xr
import zarr

from healpix_convert import (
    ConvertSettings,
    ConvertStagingCache,
    convert_group_to_healpix,
    create_healpix_dataset,
    create_staging_cache,
    finalize_healpix_dataset,
    generate_pyramids,
    prepare_healpix_dataset,
)
from healpix_convert.core.utils import drop_matching_variables, drop_variables_from_groups
from healpix_convert.tests.test_batch_resample import STAC_ITEM

HEALPIX = {"indexing_scheme": "nested", "ellipsoid": {"name": "wgs84"}}
FULL_EXTENT = shapely.box(-180.0, -90.0, 180.0, 90.0)
CHUNK = {
    "method": "healpix_cell_dense",
    "healpix": {"refinement_level": 1, **HEALPIX},
    "chunk_buffer_width": 0.0,
}
T0 = 1_786_528_000_000_000_000  # a time in nanoseconds: beyond float32 (and float64 integers)
ROWS, COLUMNS = 24, 48


def _write_input(path):
    """A swath-like image grid: 2-D lat/lon, the row / column indices as dimension
    coordinates, projected distances x / y, and the time of each row (int64 ns)."""
    lon = np.linspace(-60.0, 60.0, COLUMNS)
    lat = np.linspace(-40.0, 40.0, ROWS)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    rows = np.arange(ROWS, dtype="int64") + 1000
    columns = np.arange(COLUMNS, dtype="int64") + 10
    ds = xr.Dataset(
        {
            "a": (("rows", "columns"), np.cos(np.deg2rad(xx)) + yy / 100.0),
            "time_stamp": ("rows", T0 + 300_000_123 * np.arange(ROWS, dtype="int64")),
        },
        coords={
            "rows": ("rows", rows),
            "columns": ("columns", columns),
            "latitude": (("rows", "columns"), yy, {"standard_name": "latitude"}),
            "longitude": (("rows", "columns"), xx, {"standard_name": "longitude"}),
            "x": (("rows", "columns"), xx * 1e5),
            "y": (("rows", "columns"), yy * 1e5),
        },
    )
    tree = xr.DataTree.from_dict({"/measurements": ds})
    tree.attrs["stac_discovery"] = STAC_ITEM | {
        "geometry": shapely.geometry.mapping(FULL_EXTENT),
        "bbox": [-180.0, -90.0, 180.0, 90.0],
    }
    tree.to_zarr(path, mode="w", consolidated=True)
    return str(path)


def _settings(own_levels=True, derived=(2,), **group) -> dict:
    gs = {
        "healpix": {"refinement_level": 4, **HEALPIX},
        "chunk": CHUNK,
        "resampler": {"name": "bilinear"},
        "output_dtype": "float32",
        "drop_variables": ["x", "y"],
        "nearest_variables": ["rows", "columns", "time_stamp"],
        **group,
    }
    settings: dict = {"group_settings": {"measurements": gs}}
    if own_levels:
        settings["multiscale_settings"] = {
            "groups": ["measurements"],
            "pyramid_groups": {"measurements": {"derived_levels": list(derived)}},
        }
    return settings


def test_settings():
    settings = ConvertSettings.model_validate(_settings())
    gs = settings.group_settings[PurePath("measurements/4")]
    assert gs.drop_variables == ["x", "y"]
    assert gs.nearest_variables == ["rows", "columns", "time_stamp"]
    assert settings.drop_variables_groups() == {PurePath("measurements"): ["x", "y"]}
    dumped = settings.model_dump(mode="json")
    assert ConvertSettings.model_validate(dumped) == settings
    with pytest.raises(ValueError, match="invalid regular expression"):
        ConvertSettings.model_validate(_settings(drop_variables=["("]))


def test_nearest_variables_need_dense_chunks():
    with pytest.raises(ValueError, match="requires the 'healpix_cell_dense'"):
        ConvertSettings.model_validate(
            _settings(own_levels=False, chunk={"method": "no_chunk"})
        )


def test_drop_helpers():
    ds = xr.Dataset({"a": ("n", [1.0]), "x2": ("n", [2.0])}, coords={"x": ("n", [0.0])})
    assert set(drop_matching_variables(ds, ["x.*"]).variables) == {"a"}
    tree = xr.DataTree.from_dict({"/g": ds})
    assert set(drop_variables_from_groups(tree, {"g": ["x"]})["g"].variables) == {"a", "x2"}
    assert drop_variables_from_groups(tree, {"other": ["x"]}) is tree


def _check_store(output):
    root = zarr.open_group(output, mode="r")
    level = root["measurements/4"]
    names = set(level.array_keys())
    assert "x" not in names and "y" not in names
    assert {"a", "rows", "columns", "time_stamp"} <= names

    a = level["a"][:]
    rows, columns, time = level["rows"][:], level["columns"][:], level["time_stamp"][:]
    assert rows.dtype == np.int64 and columns.dtype == np.int64 and time.dtype == np.int64
    observed = np.isfinite(a)
    assert observed.any() and (~observed).any()
    # cells off the input: fill value -1 (0 is an index), not the edge pixel
    assert (rows[~observed] == -1).all() and (columns[~observed] == -1).all()
    assert (time[~observed] == -1).all()
    # on the input: indices of the grid, and the exact time of the row (no float rounding)
    assert ((rows[observed] >= 1000) & (rows[observed] < 1000 + ROWS)).all()
    assert ((columns[observed] >= 10) & (columns[observed] < 10 + COLUMNS)).all()
    np.testing.assert_array_equal(
        time[observed], T0 + 300_000_123 * (rows[observed] - 1000)
    )

    # the nearest pixel: within about a pixel plus a level-4 cell (~3.7 degrees,
    # wider in longitude off the equator) of the cell centre
    import healpix_geo

    lon_c, lat_c = healpix_geo.nested.healpix_to_lonlat(
        level["cell_ids"][:][observed], 4, ellipsoid="WGS84"
    )
    lat_p = np.linspace(-40.0, 40.0, ROWS)[rows[observed] - 1000]
    lon_p = np.linspace(-60.0, 60.0, COLUMNS)[columns[observed] - 10]
    assert np.abs(lat_p - lat_c).max() <= 80.0 / (ROWS - 1) + 3.7
    dlon = (lon_p - np.asarray(lon_c) + 180.0) % 360.0 - 180.0  # healpix-geo: 0..360
    assert np.abs(dlon).max() <= 120.0 / (COLUMNS - 1) + 3.7 / np.cos(np.deg2rad(45.0))
    return root


def test_one_call_conversion_and_pyramid(tmp_path):
    input_path = _write_input(tmp_path / "input.zarr")
    output = str(tmp_path / "out.zarr")
    create_healpix_dataset([input_path], _settings(), output)

    root = _check_store(output)
    derived = set(root["measurements/2"].array_keys())
    assert "a" in derived
    assert not derived & {"rows", "columns", "time_stamp", "x", "y"}
    # the multiscales metadata: the derived level is aggregated from "a" only
    layout = root["measurements"].attrs["multiscales"]["layout"]
    names = {item["asset"] for item in layout}
    assert not any(n.endswith(("/rows", "/columns", "/time_stamp")) and n.startswith("2/") for n in names)


def test_multi_stage_from_json(tmp_path):
    """Stages on other machines: the cache read back from JSON drops x / y again."""
    input_path = _write_input(tmp_path / "input.zarr")
    output = str(tmp_path / "out.zarr")
    raw = _settings()

    cache = create_staging_cache([input_path], raw)
    prepare_healpix_dataset(cache, raw, output)
    reloaded = ConvertStagingCache.from_json(cache.to_json())
    assert reloaded.input_group_drop_variables == {PurePath("measurements"): ["x", "y"]}
    dt = next(iter(reloaded.input_datatrees.values()))
    assert "x" not in dt["measurements/4"].variables
    convert_group_to_healpix("measurements/4", cache=reloaded, settings=raw, output_path=output)
    generate_pyramids(reloaded, raw, output)
    finalize_healpix_dataset(reloaded, raw, output)
    _check_store(output)


def test_without_the_settings_as_before(tmp_path):
    """Without the settings, every variable is resampled with the group's resampler
    and cast to ``output_dtype`` (unchanged behaviour)."""
    input_path = _write_input(tmp_path / "input.zarr")
    output = str(tmp_path / "out.zarr")
    create_healpix_dataset(
        [input_path], _settings(drop_variables=[], nearest_variables=[]), output
    )
    level = zarr.open_group(output, mode="r")["measurements/4"]
    assert {"x", "y", "rows", "columns", "time_stamp"} <= set(level.array_keys())
    assert level["rows"].dtype == np.float32


def _write_indices(path):
    """A group of per-pixel integer indices: a sensor index (int16, stored with a
    fill value: decoded as floats with NaN) and a segment index (int8), converted
    with the nearest resampler."""
    lon = np.linspace(-60.0, 60.0, COLUMNS)
    lat = np.linspace(-40.0, 40.0, ROWS)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    sensor = (np.arange(COLUMNS, dtype="int16")[None, :] * np.ones((ROWS, 1), "int16"))
    sensor[0, 0] = -1  # one pixel without a sensor (the stored fill value)
    segment = (np.arange(ROWS, dtype="int8")[:, None] % 5) * np.ones((1, COLUMNS), "int8")
    ds = xr.Dataset(
        {
            "sensor_index": (("rows", "columns"), sensor),
            "segment_index": (("rows", "columns"), segment),
        },
        coords={
            "latitude": (("rows", "columns"), yy, {"standard_name": "latitude"}),
            "longitude": (("rows", "columns"), xx, {"standard_name": "longitude"}),
        },
    )
    ds["sensor_index"].encoding["_FillValue"] = np.int16(-1)
    tree = xr.DataTree.from_dict({"/indices": ds})
    tree.attrs["stac_discovery"] = STAC_ITEM | {
        "geometry": shapely.geometry.mapping(FULL_EXTENT),
        "bbox": [-180.0, -90.0, 180.0, 90.0],
    }
    tree.to_zarr(path, mode="w", consolidated=True)
    return str(path)


def test_integer_indices(tmp_path):
    """The nearest resampler, every variable nearest: the stored integer types and
    fill values, no value off the input (``max_distance_m``)."""
    input_path = _write_indices(tmp_path / "input.zarr")
    output = str(tmp_path / "out.zarr")
    settings = {
        "group_settings": {
            "indices": {
                "healpix": {"refinement_level": 4, **HEALPIX},
                "chunk": CHUNK,
                "resampler": {"name": "nearest", "init_params": {"max_distance_m": 300_000.0}},
                "nearest_variables": [".*"],
            }
        }
    }
    create_healpix_dataset([input_path], settings, output)
    group = zarr.open_group(output, mode="r")["indices"]
    sensor, segment = group["sensor_index"], group["segment_index"]
    assert sensor.dtype == np.int16 and sensor.fill_value == -1
    assert segment.dtype == np.int8 and segment.fill_value == -1
    d, f = sensor[:], segment[:]
    seen = f != -1
    assert seen.any() and (~seen).any()  # cells off the grid stay empty
    assert ((d[seen] >= -1) & (d[seen] < COLUMNS)).all()
    assert ((f[seen] >= 0) & (f[seen] < 5)).all()
