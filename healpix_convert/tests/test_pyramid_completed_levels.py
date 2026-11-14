"""Tests for the derived levels of a pyramid that are native levels too: the
variables that the converted group of such a level does not hold are
aggregated from the finer levels and added to it.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import zarr

from healpix_convert.tests.test_pyramid import (
    GROUP,
    _cache_and_settings,
    _dense_cells,
    _generate_pyramids,
    _layout,
    _layout_assets,
    _level_dataset,
    _settings,
    _variable_names,
    _write_converted,
    converted_multi_level,  # noqa: F401  (fixture)
)


@pytest.fixture
def converted_resolutions(tmp_path):
    """A full-earth dataset converted at three levels, each variable at one
    level only (like the bands of a product with several resolutions): "a" and
    "b" at the level 3, "c" at the level 2, "d" at the level 1.
    """
    cell_ids = {level: _dense_cells(level) for level in (3, 2, 1)}
    rng = np.random.default_rng(0)
    data = {
        3: {
            "a": np.arange(cell_ids[3].size, dtype="float64"),
            "b": rng.normal(size=cell_ids[3].size),
        },
        2: {"c": rng.normal(size=cell_ids[2].size)},
        1: {"d": rng.normal(size=cell_ids[1].size)},
    }
    datasets = {
        level: _level_dataset(level, cell_ids[level], **data[level]) for level in data
    }
    path = _write_converted(tmp_path / "converted.zarr", datasets)

    return path, cell_ids, data


def _objects(path: str, prefix: str) -> dict[str, bytes]:
    """The stored objects of an array or a group (relative path -> bytes)."""
    root = pathlib.Path(path) / prefix

    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _block_mean(values: np.ndarray, depth: int) -> np.ndarray:
    return values.reshape(-1, 4**depth).mean(axis=1)


def test_missing_variables_are_added_to_the_native_levels(converted_resolutions):
    path, _, data = converted_resolutions

    dt = _generate_pyramids(
        *_cache_and_settings(
            [3, 2, 1], {"derived_levels": [2, 1, 0]}, variables=_variable_names(data)
        ),
        path,
    )

    assert set(dt[GROUP].children) == {"3", "2", "1", "0"}

    level2 = dt[GROUP]["2"].to_dataset()
    assert {"a", "b", "c"} <= set(level2.data_vars)
    assert "d" not in level2
    np.testing.assert_allclose(level2["a"].values, _block_mean(data[3]["a"], 1))
    np.testing.assert_allclose(level2["b"].values, _block_mean(data[3]["b"], 1))
    np.testing.assert_allclose(level2["c"].values, data[2]["c"])

    level1 = dt[GROUP]["1"].to_dataset()
    assert {"a", "b", "c", "d"} <= set(level1.data_vars)
    np.testing.assert_allclose(level1["a"].values, _block_mean(data[3]["a"], 2))
    np.testing.assert_allclose(level1["b"].values, _block_mean(data[3]["b"], 2))
    np.testing.assert_allclose(level1["c"].values, _block_mean(data[2]["c"], 1))
    np.testing.assert_allclose(level1["d"].values, data[1]["d"])

    # a derived level that is not a native one holds every variable, as before
    level0 = dt[GROUP]["0"].to_dataset()
    np.testing.assert_allclose(level0["a"].values, _block_mean(data[3]["a"], 3))
    np.testing.assert_allclose(level0["c"].values, _block_mean(data[2]["c"], 2))
    np.testing.assert_allclose(level0["d"].values, _block_mean(data[1]["d"], 1))

    # the cells of the added variables are the ones of the converted group
    assert level2["a"].dims == level2["c"].dims
    assert level2["a"].attrs.get("grid_mapping") == level2["c"].attrs.get(
        "grid_mapping"
    )


def test_converted_arrays_and_attributes_are_left_as_written(converted_resolutions):
    path, _, data = converted_resolutions
    before = {
        name: _objects(path, f"{GROUP}/2/{name}") for name in ("c", "cell_ids")
    }
    attrs = zarr.open_group(path, path=f"{GROUP}/2", mode="r").attrs.asdict()

    _generate_pyramids(
        *_cache_and_settings(
            [3, 2, 1], {"derived_levels": [2, 1]}, variables=_variable_names(data)
        ),
        path,
    )

    for name, objects in before.items():
        assert _objects(path, f"{GROUP}/2/{name}") == objects, name
    group = zarr.open_group(path, path=f"{GROUP}/2", mode="r")
    assert group.attrs.asdict() == attrs
    assert {"a", "b", "c"} <= set(group.array_keys())


def test_consolidated_metadata_lists_the_added_arrays(converted_resolutions):
    path, _, data = converted_resolutions

    _generate_pyramids(
        *_cache_and_settings(
            [3, 2, 1], {"derived_levels": [2, 1]}, variables=_variable_names(data)
        ),
        path,
    )

    root = zarr.open_consolidated(path, mode="r")
    assert {"a", "b", "c"} <= set(root[f"{GROUP}/2"].array_keys())
    assert {"a", "b", "c", "d"} <= set(root[f"{GROUP}/1"].array_keys())


def test_nothing_is_added_to_a_level_that_holds_every_variable(
    converted_multi_level,  # noqa: F811
):
    """The provider's copy of "a" at the level 2 is kept: the level 2 holds
    every variable of the finer level, so it is left as it is."""
    path, cell_ids, data = converted_multi_level
    before = _objects(path, f"{GROUP}/2")

    dt = _generate_pyramids(
        *_cache_and_settings(
            [3, 2], {"derived_levels": [2, 1]}, variables=_variable_names(data)
        ),
        path,
    )

    after = _objects(path, f"{GROUP}/2")
    # the group metadata may be rewritten by the consolidation: the arrays are not
    assert {k: v for k, v in after.items() if "/" in k} == {
        k: v for k, v in before.items() if "/" in k
    }
    np.testing.assert_allclose(dt[GROUP]["2"]["a"].values, data[2]["a"])
    np.testing.assert_allclose(
        dt[GROUP]["1"]["a"].values, np.full(cell_ids[2].size // 4, 100.0)
    )


def test_coarsen_from_finest_keeps_the_converted_copy(
    converted_multi_level,  # noqa: F811
):
    """``coarsen_from`` tells where an added variable comes from, not whether a
    converted one is replaced: the copy of "a" at the level 2 stays."""
    path, _, data = converted_multi_level

    dt = _generate_pyramids(
        *_cache_and_settings(
            [3, 2],
            {"derived_levels": [2, 1], "coarsen_from": "finest"},
            variables=_variable_names(data),
        ),
        path,
    )

    np.testing.assert_allclose(dt[GROUP]["2"]["a"].values, data[2]["a"])
    np.testing.assert_allclose(dt[GROUP]["1"]["a"].values, _block_mean(data[3]["a"], 2))


def test_multiscales_metadata_describes_the_added_arrays(converted_resolutions):
    _, _, data = converted_resolutions
    cache, _ = _cache_and_settings(
        [3, 2, 1], {"derived_levels": [2, 1, 0]}, variables=_variable_names(data)
    )
    multiscales = cache.output_groups.multiscale_groups[pathlib.PurePath(GROUP)]

    assets = _layout_assets(multiscales)
    assert {"3/a", "3/b", "2/a", "2/b", "2/c", "1/a", "1/b", "1/c", "1/d"} <= set(
        assets
    )
    assert "2/d" not in assets

    # converted arrays have no source, added ones name the array they come from
    assert "derived_from" not in _layout(multiscales, "2/c")
    assert _layout(multiscales, "2/a")["derived_from"] == "3/a"
    assert _layout(multiscales, "2/a")["transform"] == {"scale": [2.0]}
    assert _layout(multiscales, "1/a")["derived_from"] == "3/a"
    assert _layout(multiscales, "1/c")["derived_from"] == "2/c"
    assert "derived_from" not in _layout(multiscales, "1/d")
    assert _layout(multiscales, "0/d")["derived_from"] == "1/d"


def test_asset_type_group_cannot_describe_a_completed_level(converted_resolutions):
    _, _, data = converted_resolutions

    with pytest.raises(ValueError, match="aggregated from a finer"):
        _cache_and_settings(
            [3, 2, 1],
            {"derived_levels": [2]},
            variables=_variable_names(data),
            asset_type="group",
        )


def test_the_finest_native_level_cannot_be_derived():
    with pytest.raises(ValueError, match="must be coarser than the finest"):
        _settings([3, 2], {"derived_levels": [3, 1]})


def test_a_completed_level_cannot_go_in_a_subgroup():
    with pytest.raises(ValueError, match="subgroup_name"):
        _settings([3, 2], {"derived_levels": [2, 1], "subgroup_name": "pyramids"})

    # without a native level among the derived ones, the subgroup is fine
    _settings([3, 2], {"derived_levels": [1], "subgroup_name": "pyramids"})


def test_in_memory_same_as_on_the_store(converted_resolutions, tmp_path, monkeypatch):
    import shutil

    monkeypatch.setenv("HEALPIX_CONVERT_PYRAMID_PROCESSES", "2")
    path, _, data = converted_resolutions
    pyramid = {"derived_levels": [2, 1, 0]}
    variables = _variable_names(data)
    copy = str(tmp_path / "in_memory.zarr")
    shutil.copytree(path, copy)

    native_attrs = {
        level: zarr.open_group(path, path=f"{GROUP}/{level}", mode="r").attrs.asdict()
        for level in (2, 1)
    }

    _generate_pyramids(
        *_cache_and_settings([3, 2, 1], pyramid, variables=variables), path
    )
    _generate_pyramids(
        *_cache_and_settings([3, 2, 1], pyramid, variables=variables),
        copy,
        in_memory=True,
    )

    for level in (2, 1, 0):
        a = zarr.open_group(path, path=f"{GROUP}/{level}", mode="r")
        b = zarr.open_group(copy, path=f"{GROUP}/{level}", mode="r")
        assert sorted(a.array_keys()) == sorted(b.array_keys())
        if level in native_attrs:
            # the attributes of a completed native level are left as written
            assert a.attrs.asdict() == native_attrs[level]
            assert b.attrs.asdict() == native_attrs[level]
        for name in a.array_keys():
            assert a[name].metadata.to_dict() == b[name].metadata.to_dict(), name
            np.testing.assert_allclose(
                b[name][...], a[name][...], rtol=1e-12, err_msg=name
            )


def test_generating_the_pyramid_again_gives_the_same_levels(tmp_path):
    """The variables added to a completed level are not taken for converted
    ones by a second run: the coarser levels are still aggregated from the
    converted level (a block mean differs from a mean of means where cells are
    missing), and the added arrays are written again with the same values."""
    cell_ids = {level: _dense_cells(level) for level in (3, 2)}
    a = np.arange(cell_ids[3].size, dtype="float64")
    a[::5] = np.nan
    data = {3: {"a": a}, 2: {"c": np.ones(cell_ids[2].size)}}
    path = _write_converted(
        tmp_path / "converted.zarr",
        {level: _level_dataset(level, cell_ids[level], **data[level]) for level in data},
    )
    args = ([3, 2], {"derived_levels": [2, 1]})
    variables = _variable_names(data)

    first = _generate_pyramids(*_cache_and_settings(*args, variables=variables), path)
    expected = {level: first[GROUP][str(level)]["a"].values.copy() for level in (2, 1)}
    np.testing.assert_allclose(
        expected[1], np.nanmean(a.reshape(-1, 16), axis=1), equal_nan=True
    )

    second = _generate_pyramids(*_cache_and_settings(*args, variables=variables), path)

    for level in (2, 1):
        np.testing.assert_array_equal(
            second[GROUP][str(level)]["a"].values, expected[level]
        )
    np.testing.assert_allclose(second[GROUP]["2"]["c"].values, data[2]["c"])


def test_only_native_levels_as_derived_levels(converted_resolutions):
    path, _, data = converted_resolutions

    dt = _generate_pyramids(
        *_cache_and_settings(
            [3, 2, 1], {"derived_levels": [2, 1]}, variables=_variable_names(data)
        ),
        path,
    )

    assert set(dt[GROUP].children) == {"3", "2", "1"}
    np.testing.assert_allclose(dt[GROUP]["1"]["a"].values, _block_mean(data[3]["a"], 2))
