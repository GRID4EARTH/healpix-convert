from __future__ import annotations

from typing import Any

import pytest
import shapely
import xarray as xr
from pydantic import ValidationError
from pytest_structlog import StructuredLogCapture

from healpix_convert.core import stac
from healpix_convert.core.conversion_models import OutputSpatialInfo
from healpix_convert.core.stac import StacItem
from healpix_convert.settings.stac import STACSettings


def generate_items(n_items, *, equal):
    stac_extensions = []
    possible_item_ids = [f"item-{n}" for n in range(n_items)]
    bbox = []
    geometry = {}

    properties = {}
    links = []
    assets = {}

    if equal:
        item = {
            "stac_extensions": stac_extensions,
            "id": possible_item_ids[0],
            "bbox": bbox,
            "geometry": geometry,
            "properties": properties,
            "links": links,
            "assets": assets,
        }

        return [item for _ in range(n_items)]
    else:
        raise NotImplementedError


@pytest.mark.parametrize(
    "input_attrs",
    (
        pytest.param(generate_items(n_items=3, equal=True), id="3"),
        pytest.param(generate_items(n_items=1, equal=True), id="1"),
    ),
)
def test_extract_stac_metadata(input_attrs: list[dict[str, Any]]) -> None:
    input_datatrees = {
        f"group-{index}": xr.DataTree(xr.Dataset(attrs={"stac_discovery": attrs}))
        for index, attrs in enumerate(input_attrs)
    }

    actual = stac._extract_stac_metadata(input_datatrees)
    expected = {
        n: StacItem(**attrs) for n, attrs in zip(input_datatrees.keys(), input_attrs)
    }
    assert actual == expected


def test_extract_stac_metadata_empty(log: StructuredLogCapture) -> None:
    input_attrs = generate_items(n_items=3, equal=True)
    input_attrs[1] = {}

    input_datatrees = {
        f"group-{index}": xr.DataTree(
            xr.Dataset(attrs={"stac_discovery": attrs} if attrs else {})
        )
        for index, attrs in enumerate(input_attrs)
    }
    group_name = list(input_datatrees)[1]

    actual = stac._extract_stac_metadata(input_datatrees)
    expected = {
        n: StacItem(**attrs)
        for n, attrs in zip(input_datatrees.keys(), input_attrs)
        if attrs
    }

    assert actual == expected
    assert log.events == [
        log.warning("failed to extract STAC metadata from dataset", dataset=group_name)
    ]


@pytest.mark.parametrize(
    "values",
    (
        [3, 3, 3],
        ["a", "a", "a"],
        range(6),
    ),
)
def test_merge_no_conflicts(values):
    if len(set(values)) != 1:
        message = (
            "unexpectedly mismatching attributes"
            if values
            else "empty set of attributes"
        )
        with pytest.raises(ValueError, match=message):
            stac._merge_no_conflicts(values)
        return

    actual = stac._merge_no_conflicts(values)
    assert actual == values[0]


@pytest.fixture
def spatial_info():
    return OutputSpatialInfo(
        geometry_latlon=shapely.box(0, 0, 1, 1), bbox=[0.0, 0.0, 1.0, 1.0]
    )


def _items(*overrides: dict[str, Any]) -> dict[str, StacItem]:
    """Input items differing only by the fields each override names."""
    base: dict[str, Any] = {
        "stac_extensions": [],
        "id": "in",
        "bbox": [0.0, 0.0, 1.0, 1.0],
        "geometry": {"type": "Polygon", "coordinates": []},
        "properties": {},
        "links": [],
        "assets": {},
    }

    return {
        f"in-{index}": StacItem(**(base | override))
        for index, override in enumerate(overrides)
    }


def _merged(items, spatial_info, **strategies) -> StacItem:
    # exclude_assets drops every input asset by default: the merge is what is
    # under test here, so let the assets through
    settings = STACSettings(exclude_assets=[], **strategies)

    return stac.merge_stac_items(items, spatial_info, "out.zarr", settings)


SHARED_LINK = {"rel": "license", "href": "https://example.org/license"}


SHARED_EXTENSION = "https://stac-extensions.github.io/projection/v1.1.0/schema.json"
EO_EXTENSION = "https://stac-extensions.github.io/eo/v1.1.0/schema.json"
SAR_EXTENSION = "https://stac-extensions.github.io/sar/v1.0.0/schema.json"


@pytest.fixture
def differing_items():
    return _items(
        {
            "stac_extensions": [SHARED_EXTENSION, EO_EXTENSION],
            "properties": {"platform": "s2a", "datetime": "A", "only_a": 1},
            "links": [SHARED_LINK, {"rel": "self", "href": "a.json"}],
            "assets": {"data": {"href": "a.zarr"}, "only_a": {"href": "a.png"}},
        },
        {
            "stac_extensions": [SHARED_EXTENSION, SAR_EXTENSION],
            "properties": {"platform": "s2a", "datetime": "B", "only_b": 2},
            "links": [SHARED_LINK, {"rel": "self", "href": "b.json"}],
            "assets": {"data": {"href": "b.zarr"}, "only_b": {"href": "b.png"}},
        },
    )


def test_merge_pick_first_is_the_default(differing_items, spatial_info) -> None:
    merged = _merged(differing_items, spatial_info)
    first = next(iter(differing_items.values()))

    assert merged.properties == first.properties
    assert merged.stac_extensions == first.stac_extensions
    assert merged.links == first.links
    assert merged.assets == first.assets


def test_merge_join_inner_keeps_what_the_inputs_agree_on(
    differing_items, spatial_info
) -> None:
    merged = _merged(
        differing_items,
        spatial_info,
        merge_properties="join_inner",
        merge_extensions="join_inner",
        merge_links="join_inner",
        merge_assets="join_inner",
    )

    # `datetime` is in both but differs, so it is not something they share
    assert merged.properties == {"platform": "s2a"}
    assert merged.stac_extensions == [SHARED_EXTENSION]
    assert merged.links == [SHARED_LINK]
    assert merged.assets == {}


def test_merge_join_outer_keeps_everything(differing_items, spatial_info) -> None:
    merged = _merged(
        differing_items,
        spatial_info,
        merge_extensions="join_outer",
        merge_links="join_outer",
    )

    assert merged.stac_extensions == [SHARED_EXTENSION, EO_EXTENSION, SAR_EXTENSION]
    assert merged.links == [
        SHARED_LINK,
        {"rel": "self", "href": "a.json"},
        {"rel": "self", "href": "b.json"},
    ]


def test_join_outer_is_not_offered_for_mappings() -> None:
    # a union of mappings has to invent a rule for the keys whose value differs
    for field in ("merge_properties", "merge_assets"):
        with pytest.raises(ValidationError, match=".*'pick_first' or 'join_inner'.*"):
            STACSettings(**{field: "join_outer"})


@pytest.mark.parametrize("strategy", ["pick_first", "join_inner", "join_outer"])
def test_merge_strategies_agree_on_a_single_input(strategy, spatial_info) -> None:
    items = _items(
        {
            "stac_extensions": [SHARED_EXTENSION],
            "properties": {"datetime": "A"},
            "links": [SHARED_LINK],
            "assets": {"data": {"href": "a.zarr"}},
        }
    )
    only = next(iter(items.values()))

    strategies = {"merge_extensions": strategy, "merge_links": strategy}
    if strategy != "join_outer":
        strategies |= {"merge_properties": strategy, "merge_assets": strategy}

    merged = _merged(items, spatial_info, **strategies)

    assert merged.properties == only.properties
    assert merged.stac_extensions == only.stac_extensions
    assert merged.links == only.links
    assert merged.assets == only.assets


def test_merge_strategies_are_independent(differing_items, spatial_info) -> None:
    merged = _merged(differing_items, spatial_info, merge_links="join_outer")

    first = next(iter(differing_items.values()))
    assert merged.properties == first.properties
    assert len(merged.links) == 3
