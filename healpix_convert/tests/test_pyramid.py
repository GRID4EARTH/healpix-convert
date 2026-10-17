"""Generating pyramids of coarser HEALPix refinement levels."""

from __future__ import annotations

from pathlib import PurePath

import healpix_geo.nested as hpn
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
    generate_pyramids,
    prepare_healpix_dataset,
)
from healpix_convert.cache import (
    MIN_CHUNK_DEPTH,
    _chunk_cells,
    _coarsest_whole_level,
    _pyramid_chunk_level,
    _set_pyramid_groups,
)
from healpix_convert.core import utils
from healpix_convert.core.conversion_models import OutputGroupInfo, OutputSpatialInfo
from healpix_convert.core.healpix_conventions import Healpix
from healpix_convert.core.metadata import (
    CF_GRID_MAPPING_VARIABLE,
    get_cf_grid_mapping_attrs,
    get_healpix_group_attrs,
    get_multiscale_group_attrs,
)
from healpix_convert.core.multiscales_conventions import HealpixMultiscales
from healpix_convert.pyramid import create_pyramid_groups
from healpix_convert.settings.conventions import MetadataSettings
from healpix_convert.tests.test_batch_resample import STAC_ITEM

GROUP = "measurements"
METADATA = MetadataSettings()
HEALPIX = {"indexing_scheme": "nested", "ellipsoid": {"name": "wgs84"}}
FULL_EXTENT = shapely.box(-180.0, -90.0, 180.0, 90.0)
CHUNK_LEVEL = 1


# --- a converted dataset, its staging cache and its conversion settings,
# --- written and built by hand (see the end of this module for the conversion
# --- of an actual input dataset)


def _healpix(level: int) -> Healpix:
    return Healpix.model_validate({"refinement_level": level, **HEALPIX})


def _dense_cells(level: int, extent: shapely.Polygon = FULL_EXTENT) -> np.ndarray:
    """The cells a dense conversion of ``extent`` writes at the given level:
    all the children of the chunk cells covering it.
    """
    chunk_cells, _ = utils.compute_output_chunk_info(extent, CHUNK_LEVEL, "wgs84")
    cell_ids = hpn.zoom_to(np.sort(chunk_cells), CHUNK_LEVEL, level)

    return np.sort(cell_ids.ravel())


def _level_dataset(level: int, cell_ids, **variables) -> xr.Dataset:
    """One (single-scale) HEALPix group of a converted dataset."""
    healpix = _healpix(level)
    chunk_size = 4 ** (level - CHUNK_LEVEL)

    ds = xr.Dataset(
        {name: ("cells", values) for name, values in variables.items()}
        # the CF grid mapping variable, like the one written by the conversion
        | {
            CF_GRID_MAPPING_VARIABLE: (
                (),
                np.int8(0),
                get_cf_grid_mapping_attrs(healpix, METADATA),
            )
        },
        coords={"cell_ids": ("cells", np.asarray(cell_ids, dtype="uint64"))},
        attrs=get_healpix_group_attrs(healpix, METADATA),
    )

    # one chunk per chunk cell, densely filled with its children
    return ds.chunk({"cells": chunk_size})


def _layout_assets(multiscales) -> list[str]:
    """Assets of a multiscales layout, given as metadata or as a model."""
    if isinstance(multiscales, HealpixMultiscales):
        multiscales = multiscales.model_dump()

    return [item["asset"] for item in multiscales["layout"]]


def _block_means(values, parents) -> list[float]:
    """Mean of the values of the cells of each parent cell, coarser cells in
    ascending (nested) order.
    """
    return [np.nanmean(values[parents == cell_id]) for cell_id in np.unique(parents)]


def _multiscales(levels) -> HealpixMultiscales:
    """Multiscales metadata of the levels converted from the input data."""
    layout = [
        {"asset": f"{GROUP}/{level}", "dggs": _healpix(level)}
        for level in sorted(levels, reverse=True)
    ]

    return HealpixMultiscales.model_validate({"layout": layout})


def _write_converted(path, levels: dict[int, xr.Dataset], **to_zarr_kwargs) -> str:
    """Write a converted dataset: one multiscale group with one child group per
    refinement level converted from the input data.
    """
    group_attrs = {"description": "converted data"} | get_multiscale_group_attrs(
        _multiscales(levels), METADATA
    )
    nodes = {f"/{GROUP}": xr.Dataset(attrs=group_attrs)}

    for level, ds in levels.items():
        nodes[f"/{GROUP}/{level}"] = ds

    tree = xr.DataTree.from_dict(nodes)
    tree.attrs["stac_discovery"] = STAC_ITEM
    tree.to_zarr(path, mode="w", zarr_format=3, consolidated=True, **to_zarr_kwargs)

    return str(path)


def _cache_and_settings(
    levels,
    pyramid: dict | None = None,
    extent: shapely.Polygon = FULL_EXTENT,
    chunk_method: str = "dense",
) -> tuple[ConvertStagingCache, ConvertSettings]:
    """The staging cache and the settings of that conversion: where the
    children of the multiscale group have been written, the spatial extent of
    the output and the pyramid settings of the group.

    The structure of the pyramids is computed by the staging cache, as
    :py:func:`~healpix_convert.create_staging_cache` does.

    """
    settings = _settings(levels, pyramid, chunk_method)
    output_spatial = OutputSpatialInfo(geometry_latlon=extent)
    io_path_map = {PurePath(GROUP): PurePath(GROUP)}

    for level in levels:
        io_path_map[PurePath(f"{GROUP}/l{level}")] = PurePath(f"{GROUP}/{level}")

    pyramid_multiscale_groups = _set_pyramid_groups(
        settings, io_path_map, output_spatial
    )
    multiscale_groups = {PurePath(GROUP): _multiscales(levels)}

    for group_path, info in pyramid_multiscale_groups.items():
        if group_path == io_path_map[info.input_multiscale_path]:
            # the group holding the pyramid is described by its whole layout
            multiscale_groups[info.input_multiscale_path] = info.multiscales_metadata

    cache = ConvertStagingCache(
        output_spatial=output_spatial,
        output_groups=OutputGroupInfo(
            multiscale_groups=multiscale_groups,
            pyramid_multiscale_groups=pyramid_multiscale_groups,
            io_path_map=io_path_map,
        ),
    )

    return cache, settings


def _settings(levels, pyramid: dict | None = None, chunk_method: str = "dense"):
    """The settings of that conversion, with the pyramid settings of the
    multiscale group.
    """
    if chunk_method == "dense":
        chunk: dict = {
            "method": "healpix_cell_dense",
            "healpix": {"refinement_level": CHUNK_LEVEL, **HEALPIX},
        }
    else:
        chunk = {
            "method": "healpix_cell_sparse",
            "healpix": {"refinement_level": CHUNK_LEVEL, **HEALPIX},
            "chunk_size": 16,
        }

    group_settings = {}

    for level in levels:
        group_settings[f"{GROUP}/l{level}"] = {
            "healpix": {"refinement_level": level, **HEALPIX},
            "chunk": chunk,
            "resampler": {"name": "nearest"},
        }

    return ConvertSettings.model_validate(
        {
            "multiscale_settings": {
                "groups": [GROUP],
                "pyramid_groups": {} if pyramid is None else {GROUP: pyramid},
            },
            "group_settings": group_settings,
        }
    )


def _generate_pyramids(cache, settings, path):
    """The two stages that write a pyramid of a hand-written dataset: creating
    its groups and metadata (as preparing the output dataset does), then
    aggregating the converted data onto them.
    """
    store = zarr.open_group(path, mode="a").store

    for input_path, multiscales in cache.output_groups.multiscale_groups.items():
        group = zarr.open_group(
            store,
            path=cache.output_groups.io_path_map[input_path].as_posix(),
            mode="a",
        )
        group.attrs.update(get_multiscale_group_attrs(multiscales, settings.metadata))

    create_pyramid_groups(store, cache, settings)

    return generate_pyramids(cache, settings, path)


@pytest.fixture
def converted(tmp_path):
    """A full-earth dataset converted at the level 3, with dense chunks of one
    level-1 cell each (16 cells per chunk), holding a float variable (some
    cells have no data) and an integer one.
    """
    cell_ids = _dense_cells(3)
    rng = np.random.default_rng(42)
    temperature = rng.random(cell_ids.size)
    temperature[rng.random(cell_ids.size) < 0.1] = np.nan
    counts = rng.integers(0, 100, cell_ids.size).astype("int16")

    path = _write_converted(
        tmp_path / "converted.zarr",
        {3: _level_dataset(3, cell_ids, temperature=temperature, counts=counts)},
    )

    return path, cell_ids, temperature, counts


@pytest.fixture
def converted_multi_level(tmp_path):
    """A full-earth dataset converted at the levels 3 and 2, where "a" is also
    stored at the level 2 (a copy resampled by the data provider).
    """
    cell_ids = {level: _dense_cells(level) for level in (3, 2)}
    data = {
        3: {"a": np.arange(cell_ids[3].size, dtype="float64")},
        # the provider's copy of "a" is not the mean of the level 3 cells, and
        # "b" is only stored at the level 2
        2: {
            "a": np.full(cell_ids[2].size, 100.0),
            "b": np.arange(cell_ids[2].size, dtype="float64"),
        },
    }
    datasets = {}

    for level in data:
        datasets[level] = _level_dataset(level, cell_ids[level], **data[level])

    path = _write_converted(tmp_path / "converted.zarr", datasets)

    return path, cell_ids, data


def test_pyramid_levels_and_metadata(converted):
    path, cell_ids, _, _ = converted
    cache, settings = _cache_and_settings([3], {"min_refinement_level": 1})

    dt = _generate_pyramids(cache, settings, path)

    # the levels generated next to the level converted from the input data
    assert set(dt[GROUP].children) == {"3", "2", "1"}
    assert dt[GROUP]["3"].sizes["cells"] == cell_ids.size
    assert dt[GROUP]["2"].sizes["cells"] == cell_ids.size // 4
    assert dt[GROUP]["1"].sizes["cells"] == cell_ids.size // 16

    # the multiscale group declares the whole pyramid, and keeps its attributes
    multiscales = dt[GROUP].attrs["multiscales"]
    assert _layout_assets(multiscales) == [f"{GROUP}/3", f"{GROUP}/2", f"{GROUP}/1"]
    assert [item["dggs"]["refinement_level"] for item in multiscales["layout"]] == [
        3,
        2,
        1,
    ]
    assert dt[GROUP].attrs["description"] == "converted data"
    assert {c["name"] for c in dt[GROUP].attrs["zarr_conventions"]} == {
        "multiscales",
        "dggs",
    }

    # each generated group describes its own HEALPix grid
    for level in (2, 1):
        group = dt[GROUP][str(level)]
        assert group.attrs["dggs"] == _healpix(level).model_dump()
        assert group[CF_GRID_MAPPING_VARIABLE].attrs == get_cf_grid_mapping_attrs(
            _healpix(level), METADATA
        )

    # the dataset (STAC item included) is otherwise left as it is
    assert dt.attrs["stac_discovery"] == STAC_ITEM


def test_consolidated_metadata_is_up_to_date(converted):
    path, *_ = converted

    _generate_pyramids(*_cache_and_settings([3], {"min_refinement_level": 1}), path)

    # the stage consolidates what it wrote: a reader that uses the consolidated
    # metadata of the dataset sees the generated levels (the stage itself never
    # reads it, since it is not up to date while the dataset is written)
    dt = xr.open_datatree(path, engine="zarr", consolidated=True)
    assert set(dt[GROUP].children) == {"3", "2", "1"}


def test_native_levels_are_not_rewritten(tmp_path):
    """The converted levels are not touched, so the codecs they are written
    with, lossy ones included, are kept as they are.
    """
    cell_ids = _dense_cells(3)
    values = np.arange(cell_ids.size, dtype="float64")
    path = _write_converted(
        tmp_path / "converted.zarr",
        {3: _level_dataset(3, cell_ids, a=values)},
        encoding={
            f"/{GROUP}/3": {
                "a": {
                    # a lossy filter: the values are stored as hundredths
                    "filters": [
                        {
                            "name": "numcodecs.fixedscaleoffset",
                            "configuration": {
                                "offset": 0,
                                "scale": 100,
                                "dtype": "<f8",
                                "astype": "<i4",
                            },
                        }
                    ],
                    "compressors": [{"name": "zstd", "configuration": {"level": 9}}],
                }
            }
        },
    )
    native = zarr.open_group(path, mode="r")[f"{GROUP}/3"]["a"]
    metadata = native.metadata.to_dict()
    chunk = native.store.get_sync(f"{GROUP}/3/a/c/0")

    _generate_pyramids(*_cache_and_settings([3], {"min_refinement_level": 2}), path)

    rewritten = zarr.open_group(path, mode="r")[f"{GROUP}/3"]["a"]
    assert rewritten.metadata.to_dict() == metadata
    assert rewritten.store.get_sync(f"{GROUP}/3/a/c/0") == chunk

    # the generated level aggregates the values as they are stored
    level2 = xr.open_zarr(path, group=f"{GROUP}/2")
    np.testing.assert_allclose(
        level2["a"].values, values.reshape(-1, 4).mean(axis=1), atol=0.01
    )


def test_generated_levels_are_block_means(converted):
    path, cell_ids, temperature, counts = converted

    _generate_pyramids(*_cache_and_settings([3], {"min_refinement_level": 0}), path)

    for level in (2, 1, 0):
        actual = xr.open_zarr(path, group=f"{GROUP}/{level}")
        parents = cell_ids >> np.uint64(2 * (3 - level))
        expected_cell_ids = np.unique(parents)
        np.testing.assert_array_equal(actual["cell_ids"].values, expected_cell_ids)

        for cell_id, mean, count in zip(
            expected_cell_ids, actual["temperature"].values, actual["counts"].values
        ):
            # the mean of all the cells of the native level, not a mean of means
            children = parents == cell_id
            np.testing.assert_allclose(mean, np.nanmean(temperature[children]))
            # an integer variable keeps its data type, rounded (not truncated)
            assert count == np.rint(np.mean(counts[children])).astype("int16")


def test_block_mean_differs_from_mean_of_means(tmp_path):
    # one level-1 cell whose first level-2 cell has a single valid level-3 cell:
    # the mean of means would weigh that cell as much as the 4 valid cells of
    # the other level-2 cells
    cell_ids = _dense_cells(3)
    values = np.ones(cell_ids.size)
    values[1:4] = np.nan
    values[0] = 9.0
    path = _write_converted(
        tmp_path / "converted.zarr", {3: _level_dataset(3, cell_ids, a=values)}
    )

    _generate_pyramids(*_cache_and_settings([3], {"min_refinement_level": 1}), path)

    level1 = xr.open_zarr(path, group=f"{GROUP}/1")
    np.testing.assert_allclose(level1["a"].values[0], np.nanmean(values[:16]))
    assert level1["a"].values[0] != np.nanmean(
        [np.nanmean(values[i : i + 4]) for i in range(0, 16, 4)]
    )


def test_reduce_method(converted):
    path, cell_ids, temperature, _ = converted

    _generate_pyramids(
        *_cache_and_settings([3], {"min_refinement_level": 2, "reduce": "max"}), path
    )

    level2 = xr.open_zarr(path, group=f"{GROUP}/2")
    parents = cell_ids >> np.uint64(2)
    expected = [np.nanmax(temperature[parents == cell]) for cell in np.unique(parents)]
    np.testing.assert_allclose(level2["temperature"].values, expected)


def test_generated_chunks_are_healpix_cells(converted):
    path, *_ = converted
    cache, settings = _cache_and_settings([3], {"min_refinement_level": 1})

    dt = _generate_pyramids(cache, settings, path)

    # the chunks hold the cells of one HEALPix cell, as the staging cache says
    info = cache.output_groups.pyramid_multiscale_groups[PurePath(GROUP)]
    for level_path, (healpix, chunk_level) in info.pyramid_levels.items():
        written = dt[level_path.as_posix()]["temperature"].encoding["chunks"]
        assert written == (4 ** (healpix.refinement_level - chunk_level),)


def test_pyramid_chunk_level():
    # a level-20 conversion with dense chunks of one level-12 cell each: the
    # chunks of a generated level hold as many cells as the converted ones
    # (4 ** 8 cells, the children of a cell 8 levels above)
    assert _pyramid_chunk_level(19, 20, 12) == 11
    assert _pyramid_chunk_level(13, 20, 12) == 5
    # ... until the chunk cells are the base cells
    assert _pyramid_chunk_level(8, 20, 12) == 0
    assert _pyramid_chunk_level(4, 20, 12) == 0

    # a conversion with small chunks still gets reasonably sized ones
    assert _pyramid_chunk_level(18, 20, 19) == 18 - MIN_CHUNK_DEPTH


def test_pyramid_in_a_sibling_group(converted):
    path, cell_ids, temperature, _ = converted

    dt = _generate_pyramids(
        *_cache_and_settings([3], {"min_refinement_level": 2, "name": "pyramid"}), path
    )

    # the generated levels are written next to the multiscale group, which is
    # left as it is
    assert set(dt[GROUP].children) == {"3"}
    assert _layout_assets(dt[GROUP].attrs["multiscales"]) == [f"{GROUP}/3"]
    assert set(dt["pyramid"].children) == {"2"}

    # the layout of the new group describes the whole pyramid: the native level
    # (in the original group) followed by the generated ones
    assert _layout_assets(dt["pyramid"].attrs["multiscales"]) == [
        f"{GROUP}/3",
        "pyramid/2",
    ]
    parents = cell_ids >> np.uint64(2)
    expected = [np.nanmean(temperature[parents == cell]) for cell in np.unique(parents)]
    np.testing.assert_allclose(dt["pyramid/2"]["temperature"].values, expected)


def test_multiple_native_levels(converted_multi_level):
    path, cell_ids, data = converted_multi_level

    dt = _generate_pyramids(
        *_cache_and_settings([3, 2], {"min_refinement_level": 1}), path
    )

    assert set(dt[GROUP].children) == {"3", "2", "1"}

    # the levels converted from the input data are left as they are
    level2 = dt[GROUP]["2"].to_dataset()
    np.testing.assert_allclose(level2["a"].values, data[2]["a"])
    np.testing.assert_allclose(level2["b"].values, data[2]["b"])

    # the level 1 is aggregated from the level 2 (the copy of "a" included,
    # "nearest" being the default)
    level1 = dt[GROUP]["1"].to_dataset()
    np.testing.assert_allclose(
        level1["a"].values, np.full(cell_ids[2].size // 4, 100.0)
    )
    np.testing.assert_allclose(
        level1["b"].values, data[2]["b"].reshape(-1, 4).mean(axis=1)
    )


def test_coarsen_from_finest(converted_multi_level):
    path, _, data = converted_multi_level

    dt = _generate_pyramids(
        *_cache_and_settings(
            [3, 2], {"min_refinement_level": 1, "coarsen_from": "finest"}
        ),
        path,
    )

    # the copy of "a" resampled by the provider is ignored: the level 1 is
    # aggregated from the level 3
    np.testing.assert_allclose(
        dt[GROUP]["1"]["a"].values, data[3]["a"].reshape(-1, 16).mean(axis=1)
    )


def test_sparse_chunks_are_not_supported():
    with pytest.raises(ValueError, match="requires the 'healpix_cell_dense' method"):
        _settings([3], {"min_refinement_level": 1}, chunk_method="sparse")


def test_partial_extent_coarsest_level(tmp_path):
    # an extent covering 2 of the 4 level-1 chunk cells of a level-0 cell: the
    # cells of the level 0 are not covered entirely
    extent = shapely.box(0.0, 0.0, 20.0, 20.0)
    cell_ids = _dense_cells(3, extent)
    path = _write_converted(
        tmp_path / "converted.zarr",
        {3: _level_dataset(3, cell_ids, a=np.ones(cell_ids.size))},
    )

    # down to the chunk level: fine
    dt = _generate_pyramids(
        *_cache_and_settings([3], {"min_refinement_level": CHUNK_LEVEL}, extent), path
    )
    assert set(dt[GROUP].children) == {"3", "2", "1"}

    # the staging cache is where that is checked, before anything is written
    with pytest.raises(ValueError, match="do not cover whole HEALPix cells"):
        _cache_and_settings([3], {"min_refinement_level": 0}, extent)


def test_no_pyramid_settings(converted):
    path, *_ = converted

    # nothing to generate: the stage does nothing
    dt = _generate_pyramids(*_cache_and_settings([3]), path)

    assert set(dt[GROUP].children) == {"3"}


# --- the whole conversion workflow, from an actual input dataset


def _write_input(path):
    """An input dataset with two spatial groups, converted at two different
    HEALPix refinement levels.
    """
    lon = np.linspace(-180.0, 180.0, 64)
    lat = np.linspace(-80.0, 80.0, 32)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    coords = {
        "lon": ("lon", lon, {"standard_name": "longitude", "units": "degrees_east"}),
        "lat": ("lat", lat, {"standard_name": "latitude", "units": "degrees_north"}),
    }
    fine = xr.Dataset(
        {"a": (("lat", "lon"), np.sin(np.deg2rad(xx)) * np.cos(np.deg2rad(yy)))},
        coords=coords,
    )
    coarse = xr.Dataset(
        {"b": (("lat", "lon"), np.cos(np.deg2rad(xx)) + 0.5)}, coords=coords
    )

    tree = xr.DataTree.from_dict({f"/{GROUP}/fine": fine, f"/{GROUP}/coarse": coarse})
    tree.attrs["stac_discovery"] = STAC_ITEM | {
        "geometry": shapely.geometry.mapping(FULL_EXTENT),
        "bbox": [-180.0, -90.0, 180.0, 90.0],
    }
    tree.to_zarr(path, mode="w", consolidated=True)

    return str(path)


def _workflow_settings(min_refinement_level: int = 0) -> ConvertSettings:
    """Convert two input groups at the levels 3 and 2, with dense chunks of one
    level-1 cell each, and a pyramid down to the given level.
    """
    chunk = {
        "method": "healpix_cell_dense",
        "healpix": {"refinement_level": CHUNK_LEVEL, **HEALPIX},
        "chunk_buffer_width": 0.0,
    }
    group_settings = {}

    for name, level in (("fine", 3), ("coarse", 2)):
        group_settings[f"{GROUP}/{name}"] = {
            "healpix": {"refinement_level": level, **HEALPIX},
            "chunk": chunk,
            "resampler": {"name": "nearest"},
        }

    return ConvertSettings.model_validate(
        {
            "multiscale_settings": {
                "groups": [GROUP],
                "pyramid_groups": {
                    GROUP: {"min_refinement_level": min_refinement_level}
                },
            },
            "group_settings": group_settings,
        }
    )


@pytest.fixture
def prepared(tmp_path):
    """An output dataset prepared from that input: all its groups, arrays and
    metadata, with no data yet.
    """
    input_path = _write_input(tmp_path / "input.zarr")
    output_path = str(tmp_path / "converted.zarr")
    settings = _workflow_settings()

    cache = create_staging_cache([input_path], settings)
    prepare_healpix_dataset(cache, settings, output_path)

    return cache, settings, output_path


@pytest.fixture
def converted_workflow(prepared):
    """... and its data converted (but no pyramid level aggregated yet)."""
    cache, settings, output_path = prepared

    for group in (f"{GROUP}/fine", f"{GROUP}/coarse"):
        convert_group_to_healpix(
            group, cache=cache, settings=settings, output_path=output_path
        )

    return cache, settings, output_path


def test_prepare_creates_the_pyramid_groups(prepared):
    cache, settings, output_path = prepared
    root = zarr.open_group(output_path, mode="r")

    # the generated levels are groups of the output dataset, like the converted
    # ones, and the multiscale group describes the whole pyramid
    assert sorted(root[GROUP].group_keys()) == ["0", "1", "2", "3"]
    assert _layout_assets(root[GROUP].attrs["multiscales"]) == [
        f"{GROUP}/3",
        f"{GROUP}/2",
        f"{GROUP}/1",
        f"{GROUP}/0",
    ]

    for level in (1, 0):
        group = root[f"{GROUP}/{level}"]
        assert group.attrs["dggs"]["refinement_level"] == level
        # ... with no array yet: those are written with the aggregated data
        assert list(group.array_keys()) == []


def test_prepare_dry_run_creates_the_pyramid_groups(tmp_path):
    input_path = _write_input(tmp_path / "input.zarr")
    settings = _workflow_settings()
    cache = create_staging_cache([input_path], settings)

    dt = prepare_healpix_dataset(
        cache, settings, str(tmp_path / "converted.zarr"), dry_run=True
    )

    # the structure of the output dataset, pyramid groups included, without
    # writing anything
    assert set(dt[GROUP].children) == {"3", "2", "1", "0"}
    assert _layout_assets(dt[GROUP].attrs["multiscales"]) == [
        f"{GROUP}/3",
        f"{GROUP}/2",
        f"{GROUP}/1",
        f"{GROUP}/0",
    ]
    assert not dt[f"{GROUP}/0"].to_dataset().variables
    assert not (tmp_path / "converted.zarr").exists()


def test_create_healpix_dataset_fills_the_pyramids(tmp_path):
    input_path = _write_input(tmp_path / "input.zarr")
    output_path = str(tmp_path / "converted.zarr")

    dt = create_healpix_dataset(
        [input_path], _workflow_settings(min_refinement_level=1), output_path
    )

    # a conversion in one call aggregates the pyramid levels as well
    assert set(dt[GROUP].children) == {"3", "2", "1"}
    converted = dt[GROUP]["3"].to_dataset()
    parents = converted["cell_ids"].values >> np.uint64(4)
    expected = _block_means(converted["a"].values, parents)
    np.testing.assert_allclose(dt[GROUP]["1"]["a"].values, expected, atol=1e-12)


def test_conversion_workflow(converted_workflow):
    cache, settings, output_path = converted_workflow

    dt = generate_pyramids(cache, settings, output_path)

    # the converted levels (3 and 2) and the generated ones (1 and 0)
    assert set(dt[GROUP].children) == {"3", "2", "1", "0"}
    assert _layout_assets(dt[GROUP].attrs["multiscales"]) == [
        f"{GROUP}/3",
        f"{GROUP}/2",
        f"{GROUP}/1",
        f"{GROUP}/0",
    ]

    converted = dt[GROUP]["3"].to_dataset()
    for level in (1, 0):
        generated = dt[GROUP][str(level)].to_dataset()
        parents = converted["cell_ids"].values >> np.uint64(2 * (3 - level))
        np.testing.assert_array_equal(generated["cell_ids"].values, np.unique(parents))
        # "a" is aggregated from the level 3 and "b" from the level 2
        assert set(generated.data_vars) == {"a", "b", CF_GRID_MAPPING_VARIABLE}
        expected = _block_means(converted["a"].values, parents)
        np.testing.assert_allclose(generated["a"].values, expected, atol=1e-12)


def test_pyramid_structure_comes_from_the_staging_cache(converted_workflow):
    """The staging cache holds the structure of the pyramids: the levels, their
    groups, their HEALPix grid and their chunks.
    """
    cache, settings, output_path = converted_workflow
    info = cache.output_groups.pyramid_multiscale_groups[PurePath(GROUP)]
    dt = utils.open_datatrees([output_path], chunks={})[0]

    assert info.input_multiscale_path == PurePath(GROUP)
    assert _layout_assets(info.multiscales_metadata) == [
        f"{GROUP}/3",
        f"{GROUP}/2",
        f"{GROUP}/1",
        f"{GROUP}/0",
    ]

    # the converted levels, with the HEALPix grid and the chunks they were
    # converted with
    assert {path.as_posix() for path in info.native_levels} == {
        f"{GROUP}/3",
        f"{GROUP}/2",
    }
    for level_path, (healpix, chunk_level) in info.native_levels.items():
        written = dt[level_path.as_posix()]["cell_ids"]
        assert written.size == 12 * 4**healpix.refinement_level
        assert chunk_level == CHUNK_LEVEL
        # the cells of the level are the children of its chunk cells
        chunk_cells = _chunk_cells(
            cache.output_spatial.geometry_latlon,
            settings.group_settings[PurePath(f"{GROUP}/fine")].chunk,
        )
        np.testing.assert_array_equal(
            hpn.zoom_to(chunk_cells, CHUNK_LEVEL, healpix.refinement_level).ravel(),
            written.values,
        )
        # a full-earth extent covers whole HEALPix cells at every level
        assert _coarsest_whole_level(chunk_cells, CHUNK_LEVEL) == 0

    # the generated levels, with the grid and the chunks they are written with
    assert {path.as_posix() for path in info.pyramid_levels} == {
        f"{GROUP}/1",
        f"{GROUP}/0",
    }
    for level_path, (healpix, chunk_level) in info.pyramid_levels.items():
        assert level_path == PurePath(GROUP) / str(healpix.refinement_level)
        assert chunk_level == 0

    # the cache, which the conversion stages are given, round-trips
    assert ConvertStagingCache.from_json(cache.to_json()).output_groups == (
        cache.output_groups
    )
