from __future__ import annotations

from typing import Any

import pytest
import xarray as xr
from pytest_structlog import StructuredLogCapture

from legacy_converters import stac_operations
from legacy_converters.core.stac import StacItem


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

    actual = stac_operations._extract_stac_metadata(input_datatrees)
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

    actual = stac_operations._extract_stac_metadata(input_datatrees)
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
            stac_operations._merge_no_conflicts(values)
        return

    actual = stac_operations._merge_no_conflicts(values)
    assert actual == values[0]
