"""A group that takes its latitude / longitude from another group (``coordinates_group``)."""

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
from healpix_convert.core.utils import add_coordinates_from, add_coordinates_from_groups
from healpix_convert.tests.test_batch_resample import STAC_ITEM

HEALPIX = {"indexing_scheme": "nested", "ellipsoid": {"name": "wgs84"}}
FULL_EXTENT = shapely.box(-180.0, -90.0, 180.0, 90.0)
CHUNK = {
    "method": "healpix_cell_dense",
    "healpix": {"refinement_level": 1, **HEALPIX},
    "chunk_buffer_width": 0.0,
}


def _grid():
    """A curvilinear (2-D lat/lon) image grid, as a satellite swath."""
    rows, columns = 24, 48
    lon = np.linspace(-170.0, 170.0, columns)
    lat = np.linspace(-70.0, 70.0, rows)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    return xx + 0.5 * yy / 70.0, yy


def _write_input(path):
    """``measurements``: values and the lat/lon of the image grid; ``quality``:
    flags on the same grid without coordinates; ``tiepoints``: values on a
    coarser grid with lat/lon; ``tiepoints_other``: values on that grid without."""
    lon2, lat2 = _grid()
    # the pixel index (exact in float32): tells which pixel a cell took
    value = np.arange(lat2.size, dtype="f8").reshape(lat2.shape)
    latlon = {
        "latitude": (("rows", "columns"), lat2, {"standard_name": "latitude"}),
        "longitude": (("rows", "columns"), lon2, {"standard_name": "longitude"}),
        # an image coordinate of the group: not taken by the other group
        "x": (("rows", "columns"), np.arange(lat2.size, dtype="f8").reshape(lat2.shape)),
    }
    measurements = xr.Dataset({"a": (("rows", "columns"), value)}, coords=latlon)
    flags = (np.arange(lat2.size) % 7).reshape(lat2.shape).astype("uint16")
    quality = xr.Dataset({"flag": (("rows", "columns"), flags)})
    tlon, tlat = lon2[::4, ::4], lat2[::4, ::4]
    tie = {
        "latitude": (("t_rows", "t_columns"), tlat, {"standard_name": "latitude"}),
        "longitude": (("t_rows", "t_columns"), tlon, {"standard_name": "longitude"}),
    }
    tiepoints = xr.Dataset({"angle": (("t_rows", "t_columns"), tlat + 90.0)}, coords=tie)
    tiepoints_other = xr.Dataset({"angle_o": (("t_rows", "t_columns"), tlon + 180.0)})
    tree = xr.DataTree.from_dict(
        {
            "/measurements": measurements,
            "/quality": quality,
            "/conditions/tiepoints": tiepoints,
            "/conditions/tiepoints_other": tiepoints_other,
        }
    )
    tree.attrs["stac_discovery"] = STAC_ITEM | {
        "geometry": shapely.geometry.mapping(FULL_EXTENT),
        "bbox": [-180.0, -90.0, 180.0, 90.0],
    }
    tree.to_zarr(path, mode="w", consolidated=True)
    return str(path)


def _group(level=3, resampler="nearest", **extra):
    return {
        "healpix": {"refinement_level": level, **HEALPIX},
        "chunk": CHUNK,
        "resampler": {"name": resampler},
        **extra,
    }


def _settings(own_levels=False) -> dict:
    """Without ``own_levels``: plain groups. With it: ``measurements`` and
    ``quality`` are multiscale groups converted from their own variables (their
    levels are ``measurements/3`` and ``quality/3``)."""
    settings = {
        "group_settings": {
            "measurements": _group(),
            "quality": _group(coordinates_group="measurements"),
            "conditions/tiepoints": _group(level=2),
            "conditions/tiepoints_other": _group(
                level=2, coordinates_group="conditions/tiepoints"
            ),
        },
    }
    if own_levels:
        settings["multiscale_settings"] = {"groups": ["measurements", "quality"]}
    return settings


def test_setting_round_trip():
    settings = ConvertSettings.model_validate(_settings())
    gs = settings.group_settings[PurePath("quality")]
    assert gs.coordinates_group == PurePath("measurements")
    dumped = settings.model_dump(mode="json")
    assert dumped["group_settings"]["quality"]["coordinates_group"] == "measurements"
    assert dumped["group_settings"]["measurements"]["coordinates_group"] is None
    assert ConvertSettings.model_validate(dumped) == settings
    assert settings.coordinates_groups() == {
        PurePath("quality"): PurePath("measurements"),
        PurePath("conditions/tiepoints_other"): PurePath("conditions/tiepoints"),
    }


def test_coordinates_groups_of_own_level_groups():
    """The settings of an own-variables multiscale group move to its level
    group; the coordinates go to the input group (the multiscale group)."""
    settings = ConvertSettings.model_validate(_settings(own_levels=True))
    assert PurePath("quality/3") in settings.group_settings
    assert settings.coordinates_groups()[PurePath("quality")] == PurePath("measurements")


def test_staging_cache_finds_the_group(tmp_path):
    input_path = _write_input(tmp_path / "input.zarr")
    cache = create_staging_cache([input_path], _settings())

    assert PurePath("quality") in cache.input_spatial_groups
    assert PurePath("conditions/tiepoints_other") in cache.input_spatial_groups
    info = cache.input_spatial_groups[PurePath("quality")]
    assert info.spatial_arrays == ["flag"]  # only the lat/lon were taken
    dt = next(iter(cache.input_datatrees.values()))
    assert {"latitude", "longitude"} <= set(dt["quality"].coords)
    assert "x" not in dt["quality"].variables


def test_converted_like_the_coordinates_group(tmp_path):
    """Same grid, same resampler: the flags land on the cells of the values of
    the coordinates group, with their integer type."""
    input_path = _write_input(tmp_path / "input.zarr")
    output = str(tmp_path / "out.zarr")
    create_healpix_dataset([input_path], _settings(), output)

    root = zarr.open_group(output, mode="r")
    a = root["measurements/a"][:]
    flag = root["quality/flag"][:]
    assert root["quality/flag"].dtype == np.uint16
    np.testing.assert_array_equal(root["quality/cell_ids"][:], root["measurements/cell_ids"][:])
    observed = np.isfinite(a)
    assert observed.any()
    # nearest: each cell takes the flag of the pixel whose value (index) it takes
    np.testing.assert_array_equal(flag[observed], a[observed].astype("int64") % 7)
    assert np.isfinite(root["conditions/tiepoints_other/angle_o"][:]).any()


def test_own_level_groups_multi_stage_and_json(tmp_path):
    """Multiscale groups converted from their own variables, the
    flags taking the coordinates of the measurements, a cache read back from
    JSON (the stages run on other machines)."""
    input_path = _write_input(tmp_path / "input.zarr")
    output = str(tmp_path / "out.zarr")
    raw = _settings(own_levels=True)

    cache = create_staging_cache([input_path], raw)
    prepare_healpix_dataset(cache, raw, output)
    reloaded = ConvertStagingCache.from_json(cache.to_json())
    assert reloaded.input_group_coordinates == cache.input_group_coordinates
    dt = next(iter(reloaded.input_datatrees.values()))
    assert {"latitude", "longitude"} <= set(dt["quality/3"].coords)
    for group in raw["group_settings"]:
        path = group if group.startswith("conditions") else f"{group}/3"
        convert_group_to_healpix(path, cache=reloaded, settings=raw, output_path=output)
    generate_pyramids(reloaded, raw, output)
    finalize_healpix_dataset(reloaded, raw, output)

    root = zarr.open_group(output, mode="r")
    assert sorted(root["quality"].group_keys()) == ["3"]
    a = root["measurements/3/a"][:]
    flag = root["quality/3/flag"][:]
    assert np.isfinite(a).any() and flag.dtype == np.uint16
    # validating the cache again (in-memory trees, moved already) changes nothing
    ConvertStagingCache.model_validate(reloaded)


def test_distributed_conversion_adds_the_coordinates(tmp_path):
    """With a dask client and ``load_input_data``, the converter reopens its input
    group from the stores: it takes the coordinates again."""
    distributed = pytest.importorskip("distributed")
    input_path = _write_input(tmp_path / "input.zarr")
    output = str(tmp_path / "out.zarr")
    raw = _settings()

    cache = create_staging_cache([input_path], raw)
    prepare_healpix_dataset(cache, raw, output)
    with distributed.Client(
        processes=False, n_workers=1, threads_per_worker=2, dashboard_address=None
    ):
        convert_group_to_healpix(
            "quality", cache=cache, settings=raw, output_path=output, load_input_data=True
        )
    serial = str(tmp_path / "serial.zarr")
    create_healpix_dataset([input_path], raw, serial)
    np.testing.assert_array_equal(
        zarr.open_group(output, mode="r")["quality/flag"][:],
        zarr.open_group(serial, mode="r")["quality/flag"][:],
    )


def test_mismatched_dimensions():
    ds = xr.Dataset({"flag": (("rows", "columns"), np.zeros((2, 3)))})
    source = xr.Dataset(
        coords={"latitude": (("rows", "columns"), np.zeros((2, 4)), {"standard_name": "latitude"})}
    )
    with pytest.raises(ValueError, match="dimension 'columns' of size 4"):
        add_coordinates_from(ds, source, "quality", "measurements")


def test_source_without_latlon():
    ds = xr.Dataset({"flag": ("n", np.zeros(3))})
    with pytest.raises(ValueError, match="no latitude / longitude"):
        add_coordinates_from(ds, xr.Dataset({"b": ("n", np.zeros(3))}), "q", "m")


def test_missing_coordinates_group_and_skipped_groups():
    tree = xr.DataTree.from_dict({"/q": xr.Dataset({"flag": ("n", np.zeros(3))})})
    with pytest.raises(ValueError, match="not in the input dataset"):
        add_coordinates_from_groups(tree, {"q": "m"})
    # a group missing from the tree (filtered out) is skipped
    assert add_coordinates_from_groups(tree, {"other": "m"}) is tree
