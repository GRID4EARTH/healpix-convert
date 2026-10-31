"""Finalizing the output converted dataset (the last stage of the conversion)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import shapely
import xarray as xr

from healpix_convert import (
    ConvertSettings,
    convert_group_to_healpix,
    create_healpix_dataset,
    create_staging_cache,
    finalize_healpix_dataset,
    generate_pyramids,
    prepare_healpix_dataset,
)
from healpix_convert.finalize import consolidation_paths
from healpix_convert.tests.test_batch_resample import STAC_ITEM

#: The group converted on its own, outside of any multiscale group.
GROUP = "measurements"

#: The multiscale group, with one child group per converted level and a pyramid
#: of one derived level.
MULTISCALE_GROUP = "levels"

#: A group of the input dataset holding no data, carried to the output as it is.
EMPTY_GROUP = "provenance"

HEALPIX = {"indexing_scheme": "nested", "ellipsoid": {"name": "wgs84"}}
FULL_EXTENT = shapely.box(-180.0, -90.0, 180.0, 90.0)

#: The groups of the output dataset, by the setting that consolidates them.
OUTPUT_GROUPS = {
    "root_group": {""},
    "healpix_groups": {GROUP, f"{MULTISCALE_GROUP}/3", f"{MULTISCALE_GROUP}/2"},
    "multiscale_groups": {MULTISCALE_GROUP},
    "derived_level_groups": {f"{MULTISCALE_GROUP}/1"},
}

ALL_OUTPUT_GROUPS = set().union(*OUTPUT_GROUPS.values())


def _write_input(path) -> str:
    """An input dataset with a spatial group converted on its own, a multiscale
    group whose two children are converted at different refinement levels, and
    a group holding no data.
    """
    lon = np.linspace(-180.0, 180.0, 64)
    lat = np.linspace(-80.0, 80.0, 32)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    coords = {
        "lon": ("lon", lon, {"standard_name": "longitude", "units": "degrees_east"}),
        "lat": ("lat", lat, {"standard_name": "latitude", "units": "degrees_north"}),
    }

    def dataset(name):
        values = np.sin(np.deg2rad(xx)) * np.cos(np.deg2rad(yy))
        return xr.Dataset({name: (("lat", "lon"), values)}, coords=coords)

    tree = xr.DataTree.from_dict(
        {
            f"/{GROUP}": dataset("a"),
            f"/{MULTISCALE_GROUP}/fine": dataset("b"),
            f"/{MULTISCALE_GROUP}/coarse": dataset("c"),
            f"/{EMPTY_GROUP}": xr.Dataset(attrs={"source": "a test"}),
        }
    )
    tree.attrs["stac_discovery"] = STAC_ITEM | {
        "geometry": shapely.geometry.mapping(FULL_EXTENT),
        "bbox": [-180.0, -90.0, 180.0, 90.0],
    }
    tree.to_zarr(path, mode="w", consolidated=True)

    return str(path)


def _settings(consolidate: dict | None = None) -> ConvertSettings:
    """Convert that input, with the given ``consolidate`` metadata settings
    (their defaults if none are given).
    """
    chunk = {
        "method": "healpix_cell_dense",
        "healpix": {"refinement_level": 1, **HEALPIX},
        "chunk_buffer_width": 0.0,
    }
    group_settings = {
        GROUP: {
            "healpix": {"refinement_level": 4, **HEALPIX},
            "chunk": {"method": "no_chunk"},
            "resampler": {"name": "nearest"},
        }
    }

    for name, level in (("fine", 3), ("coarse", 2)):
        group_settings[f"{MULTISCALE_GROUP}/{name}"] = {
            "healpix": {"refinement_level": level, **HEALPIX},
            "chunk": chunk,
            "resampler": {"name": "nearest"},
        }

    metadata = {} if consolidate is None else {"consolidate": consolidate}

    return ConvertSettings.model_validate(
        {
            "metadata": metadata,
            "multiscale_settings": {
                "groups": [MULTISCALE_GROUP],
                "pyramid_groups": {MULTISCALE_GROUP: {"derived_levels": [1]}},
            },
            "group_settings": group_settings,
        }
    )


def _consolidated_groups(output_path: str) -> set[str]:
    """Paths of the output groups whose own Zarr metadata document holds
    consolidated metadata (the root group being the empty path).

    The metadata of a group consolidated by one of its parents is in the
    document of that parent, not in its own.

    """
    paths = set()

    for document in Path(output_path).rglob("zarr.json"):
        metadata = json.loads(document.read_text())
        if metadata["node_type"] != "group":
            continue
        if "consolidated_metadata" not in metadata:
            continue
        path = document.parent.relative_to(output_path).as_posix()
        paths.add("" if path == "." else path)

    return paths


@pytest.fixture
def input_path(tmp_path) -> str:
    return _write_input(tmp_path / "input.zarr")


@pytest.fixture
def converted(tmp_path, input_path):
    """That input dataset converted by the staged API, up to (but without) the
    last stage: no metadata is consolidated yet.
    """

    def convert(settings: ConvertSettings) -> tuple:
        output_path = str(tmp_path / "output.zarr")
        cache = create_staging_cache([input_path], settings)

        prepare_healpix_dataset(cache, settings, output_path)
        for group in (GROUP, f"{MULTISCALE_GROUP}/fine", f"{MULTISCALE_GROUP}/coarse"):
            convert_group_to_healpix(
                group, cache=cache, settings=settings, output_path=output_path
            )
        generate_pyramids(cache, settings, output_path)

        return cache, settings, output_path

    return convert


def test_consolidation_paths_are_the_output_group_paths(input_path) -> None:
    settings = _settings()
    cache = create_staging_cache([input_path], settings)

    paths = consolidation_paths(cache, settings)

    # the root group first, then the paths of the groups in the output dataset
    # (not the paths of the converted levels in the input dataset)
    assert paths[0] == ""
    assert set(paths[1:]) == ALL_OUTPUT_GROUPS - {""}


def test_consolidation_paths_are_unique(input_path) -> None:
    settings = _settings({"groups": ["/", f"/{MULTISCALE_GROUP}", EMPTY_GROUP]})
    cache = create_staging_cache([input_path], settings)

    paths = consolidation_paths(cache, settings)

    # a group named by several settings is consolidated once, and a path is
    # normalized the way Zarr addresses a group
    assert sorted(paths) == sorted(ALL_OUTPUT_GROUPS | {EMPTY_GROUP})


def test_nothing_is_consolidated_before_the_last_stage(converted) -> None:
    *_, output_path = converted(_settings())

    assert _consolidated_groups(output_path) == set()


def test_every_output_group_is_consolidated_by_default(converted) -> None:
    cache, settings, output_path = converted(_settings())

    dt = finalize_healpix_dataset(cache, settings, output_path)

    assert _consolidated_groups(output_path) == ALL_OUTPUT_GROUPS
    # the consolidated metadata of the root group describes the whole dataset,
    # the derived level of the pyramid included
    assert set(dt[MULTISCALE_GROUP].children) == {"3", "2", "1"}
    assert dt[f"{MULTISCALE_GROUP}/1"]["b"].sizes["cells"] == 48


@pytest.mark.parametrize("setting", list(OUTPUT_GROUPS))
def test_a_setting_consolidates_its_own_groups(converted, setting: str) -> None:
    cache, settings, output_path = converted(
        _settings({name: name == setting for name in OUTPUT_GROUPS})
    )

    finalize_healpix_dataset(cache, settings, output_path)

    assert _consolidated_groups(output_path) == OUTPUT_GROUPS[setting]


def test_no_group_is_consolidated(converted) -> None:
    cache, settings, output_path = converted(
        _settings({name: False for name in OUTPUT_GROUPS})
    )

    dt = finalize_healpix_dataset(cache, settings, output_path)

    assert _consolidated_groups(output_path) == set()
    # the output dataset is still readable, without consolidated metadata
    assert set(dt[MULTISCALE_GROUP].children) == {"3", "2", "1"}


def test_user_defined_groups_are_consolidated(converted) -> None:
    consolidate = {name: False for name in OUTPUT_GROUPS}
    cache, settings, output_path = converted(
        _settings(consolidate | {"groups": [f"/{EMPTY_GROUP}", GROUP]})
    )

    finalize_healpix_dataset(cache, settings, output_path)

    # a group of the output dataset that no other setting names, given with or
    # without a leading separator
    assert _consolidated_groups(output_path) == {EMPTY_GROUP, GROUP}


def test_an_unknown_user_defined_group_is_reported_by_the_staging_cache(
    input_path, log
) -> None:
    settings = _settings({"groups": [EMPTY_GROUP, "missing"]})

    create_staging_cache([input_path], settings)

    # the structure of the output dataset is known there, before anything is
    # converted
    warnings = [event for event in log.events if event["level"] == "warning"]
    assert len(warnings) == 1
    assert "does not hold" in warnings[0]["event"]
    assert "missing" in warnings[0]["event"]
    assert EMPTY_GROUP not in warnings[0]["event"]


def test_an_unknown_user_defined_group_is_skipped(converted) -> None:
    consolidate = {name: False for name in OUTPUT_GROUPS}
    cache, settings, output_path = converted(
        _settings(consolidate | {"groups": [EMPTY_GROUP, "missing"]})
    )

    finalize_healpix_dataset(cache, settings, output_path)

    # the group the output dataset does not hold is skipped, the others are
    # consolidated
    assert _consolidated_groups(output_path) == {EMPTY_GROUP}


def test_the_one_call_conversion_finalizes_the_output(tmp_path, input_path) -> None:
    output_path = str(tmp_path / "output.zarr")
    settings = _settings({"healpix_groups": False, "derived_level_groups": False})

    dt = create_healpix_dataset([input_path], settings, output_path)

    # a conversion in one call runs the last stage as well, with the same
    # settings
    assert _consolidated_groups(output_path) == {"", MULTISCALE_GROUP}
    assert set(dt[MULTISCALE_GROUP].children) == {"3", "2", "1"}
