"""A multiscale group converted from its own variables (no child groups in the input)."""

from __future__ import annotations

from pathlib import PurePath

import numpy as np
import pytest
import shapely
import xarray as xr
import zarr

from healpix_convert import (
    ConvertSettings,
    convert_group_to_healpix,
    create_healpix_dataset,
    create_staging_cache,
    finalize_healpix_dataset,
    generate_pyramids,
    prepare_healpix_dataset,
)
from healpix_convert.tests.test_batch_resample import STAC_ITEM

GROUP = "measurements"
HEALPIX = {"indexing_scheme": "nested", "ellipsoid": {"name": "wgs84"}}
FULL_EXTENT = shapely.box(-180.0, -90.0, 180.0, 90.0)
CHUNK = {
    "method": "healpix_cell_dense",
    "healpix": {"refinement_level": 1, **HEALPIX},
    "chunk_buffer_width": 0.0,
}


def _write_input(path, with_child=True):
    """``measurements`` holds the variables itself (no child group to convert),
    plus a child group left out of the conversion."""
    lon = np.linspace(-180.0, 180.0, 64)
    lat = np.linspace(-80.0, 80.0, 32)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    coords = {
        "lon": ("lon", lon, {"standard_name": "longitude", "units": "degrees_east"}),
        "lat": ("lat", lat, {"standard_name": "latitude", "units": "degrees_north"}),
    }
    ds = xr.Dataset(
        {
            "a": (("lat", "lon"), np.sin(np.deg2rad(xx)) * np.cos(np.deg2rad(yy))),
            "b": (("lat", "lon"), np.cos(np.deg2rad(xx)) + 0.5),
        },
        coords=coords,
        attrs={"title": "own variables"},
    )
    nodes = {f"/{GROUP}": ds}
    if with_child:
        nodes[f"/{GROUP}/extra"] = xr.Dataset({"c": ("n", np.arange(3.0))})
    tree = xr.DataTree.from_dict(nodes)
    tree.attrs["stac_discovery"] = STAC_ITEM | {
        "geometry": shapely.geometry.mapping(FULL_EXTENT),
        "bbox": [-180.0, -90.0, 180.0, 90.0],
    }
    tree.to_zarr(path, mode="w", consolidated=True)
    return str(path)


def _settings(derived_levels=(2, 1)) -> dict:
    return {
        "exclude_groups": [f"{GROUP}/extra"],
        "multiscale_settings": {
            "groups": [GROUP],
            "pyramid_groups": {GROUP: {"derived_levels": list(derived_levels)}},
        },
        "group_settings": {
            GROUP: {
                "healpix": {"refinement_level": 3, **HEALPIX},
                "chunk": CHUNK,
                "resampler": {"name": "nearest"},
            }
        },
    }


def test_settings_move_to_the_level_group():
    settings = ConvertSettings.model_validate(_settings())

    assert list(settings.group_settings) == [PurePath(f"{GROUP}/3")]
    assert settings.own_level_groups() == {PurePath(f"{GROUP}/3"): PurePath(GROUP)}
    assert settings.native_refinement_levels(PurePath(GROUP)) == {
        3: PurePath(f"{GROUP}/3")
    }
    # validating the dumped settings again changes nothing
    again = ConvertSettings.model_validate(settings.model_dump())
    assert again.group_settings == settings.group_settings


def test_settings_level_group_given_twice():
    raw = _settings()
    raw["group_settings"][f"{GROUP}/3"] = raw["group_settings"][GROUP]
    with pytest.raises(ValueError, match="has settings of its own already"):
        ConvertSettings.model_validate(raw)


def test_group_not_multiscale_unchanged():
    raw = _settings()
    raw["multiscale_settings"] = {}
    settings = ConvertSettings.model_validate(raw)
    assert list(settings.group_settings) == [PurePath(GROUP)]
    assert settings.own_level_groups() == {}


def test_staging_cache_moves_the_variables(tmp_path):
    input_path = _write_input(tmp_path / "input.zarr")
    settings = ConvertSettings.model_validate(_settings())

    cache = create_staging_cache([input_path], settings)

    dt = next(iter(cache.input_datatrees.values()))
    assert not dt[GROUP].to_dataset(inherit=False).data_vars
    assert dt[GROUP].attrs["title"] == "own variables"
    assert set(dt[f"{GROUP}/3"].data_vars) == {"a", "b"}
    assert "extra" in dt[GROUP].children
    assert PurePath(f"{GROUP}/3") in cache.input_spatial_groups
    assert cache.output_groups.io_path_map[PurePath(f"{GROUP}/3")] == PurePath(
        f"{GROUP}/3"
    )


def test_one_call_conversion(tmp_path):
    input_path = _write_input(tmp_path / "input.zarr")
    output_path = str(tmp_path / "converted.zarr")

    dt = create_healpix_dataset([input_path], _settings(), output_path)

    assert set(dt[GROUP].children) == {"3", "2", "1"}
    assert set(dt[GROUP]["3"].data_vars) >= {"a", "b"}
    assets = [item["asset"] for item in dt[GROUP].attrs["multiscales"]["layout"]]
    assert assets == ["3", "2", "1"]
    converted = dt[GROUP]["3"].to_dataset()
    parents = converted["cell_ids"].values >> np.uint64(4)
    expected = [
        np.nanmean(converted["a"].values[parents == p]) for p in np.unique(parents)
    ]
    np.testing.assert_allclose(dt[GROUP]["1"]["a"].values, expected, atol=1e-12)


def test_multi_stage_conversion(tmp_path):
    """The stages of a workflow: each one validates the same settings again."""
    input_path = _write_input(tmp_path / "input.zarr", with_child=False)
    output_path = str(tmp_path / "converted.zarr")
    raw = _settings(derived_levels=(1,))
    raw["exclude_groups"] = []

    cache = create_staging_cache([input_path], raw)
    prepare_healpix_dataset(cache, raw, output_path)
    convert_group_to_healpix(
        f"{GROUP}/3", cache=cache, settings=raw, output_path=output_path
    )
    generate_pyramids(cache, raw, output_path)
    finalize_healpix_dataset(cache, raw, output_path)

    root = zarr.open_group(output_path, mode="r")
    assert sorted(root[GROUP].group_keys()) == ["1", "3"]
    a = root[f"{GROUP}/3/a"][:]
    assert np.isfinite(a).all()
    assert "multiscales" in root[GROUP].attrs


def test_staging_cache_from_json(tmp_path):
    """A cache read back from JSON (a multi-stage workflow on several machines)
    reopens the inputs from their paths: the level group is made again."""
    from healpix_convert import ConvertStagingCache

    input_path = _write_input(tmp_path / "input.zarr", with_child=False)
    output_path = str(tmp_path / "converted.zarr")
    raw = _settings(derived_levels=(1,))
    raw["exclude_groups"] = []

    cache = create_staging_cache([input_path], raw)
    prepare_healpix_dataset(cache, raw, output_path)

    reloaded = ConvertStagingCache.from_json(cache.to_json())
    dt = next(iter(reloaded.input_datatrees.values()))
    assert set(dt[f"{GROUP}/3"].data_vars) == {"a", "b"}
    assert not dt[GROUP].to_dataset(inherit=False).data_vars

    convert_group_to_healpix(
        f"{GROUP}/3", cache=reloaded, settings=raw, output_path=output_path
    )
    generate_pyramids(reloaded, raw, output_path)
    finalize_healpix_dataset(reloaded, raw, output_path)

    root = zarr.open_group(output_path, mode="r")
    assert np.isfinite(root[f"{GROUP}/3/a"][:]).all()
    assert np.isfinite(root[f"{GROUP}/1/a"][:]).all()


def test_move_group_variables_to_child_leaves_other_trees_alone():
    from healpix_convert.core.utils import move_group_variables_to_child

    tree = xr.DataTree.from_dict(
        {"/g": xr.Dataset(), "/g/child": xr.Dataset({"x": ("n", [1.0, 2.0])})}
    )
    # the group has no variables, the child exists: nothing to move, same object
    assert move_group_variables_to_child(tree, [("g", "g/child")]) is tree
    assert move_group_variables_to_child(tree, [("g", "g/3")]) is tree


def test_cache_records_the_source_group(tmp_path):
    from healpix_convert import ConvertStagingCache

    input_path = _write_input(tmp_path / "input.zarr")
    cache = create_staging_cache([input_path], _settings())
    assert cache.input_group_sources == {PurePath(f"{GROUP}/3"): PurePath(GROUP)}
    assert cache.input_group(f"{GROUP}/3") == GROUP
    assert cache.input_group("other/group") == "other/group"
    reloaded = ConvertStagingCache.from_json(cache.to_json())
    assert reloaded.input_group_sources == cache.input_group_sources


def test_distributed_conversion_reopens_the_source_group(tmp_path):
    """With a dask client and ``load_input_data``, the converter reopens its input
    group from the input stores: the source group of the level group."""
    distributed = pytest.importorskip("distributed")

    input_path = _write_input(tmp_path / "input.zarr", with_child=False)
    output_path = str(tmp_path / "converted.zarr")
    raw = _settings(derived_levels=(1,))
    raw["exclude_groups"] = []

    cache = create_staging_cache([input_path], raw)
    prepare_healpix_dataset(cache, raw, output_path)
    with distributed.Client(
        processes=False, n_workers=1, threads_per_worker=2, dashboard_address=None
    ):
        convert_group_to_healpix(
            f"{GROUP}/3",
            cache=cache,
            settings=raw,
            output_path=output_path,
            load_input_data=True,
        )
    generate_pyramids(cache, raw, output_path)
    finalize_healpix_dataset(cache, raw, output_path)

    serial = str(tmp_path / "serial.zarr")
    create_healpix_dataset([input_path], raw, serial)
    a = zarr.open_group(output_path, mode="r")[f"{GROUP}/3/a"][:]
    b = zarr.open_group(serial, mode="r")[f"{GROUP}/3/a"][:]
    np.testing.assert_array_equal(a, b)
