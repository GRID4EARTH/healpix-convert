from __future__ import annotations

import itertools
import json
from typing import Any

import shapely
import structlog
import xarray as xr

from legacy_converters.core.conversion_models import OutputSpatialInfo
from legacy_converters.core.stac import StacItem

log = structlog.get_logger()


def _extract_stac_metadata(
    input_datatrees: dict[str, xr.DataTree],
) -> dict[str, StacItem]:
    collected_metadata = {}
    for name, datatree in input_datatrees.items():
        metadata = datatree.attrs.get("stac_discovery")
        if metadata is None:
            # FIXME: maybe raise an error instead?
            log.warning("failed to extract STAC metadata from dataset", dataset=name)
            continue
        collected_metadata[name] = StacItem(**metadata)

    return collected_metadata


def _merge_no_conflicts(values: list[Any]) -> Any:
    if len(values) == 0:
        raise ValueError("empty set of attributes")

    initial = values[0]
    if any(initial != value for value in values[1:]):
        raise ValueError("unexpectedly mismatching attributes")

    return values[0]


def _merge_properties(values: list[dict[str, Any]]) -> dict[str, Any]:
    return values[0]


def _merge_links(values: list[dict[str, Any]]) -> dict[str, Any]:
    return values[0]


def _merge_assets(values: list[dict[str, Any]]) -> dict[str, Any]:
    return values[0]


def _collect_keys(mappings: list[StacItem]):
    return list(
        dict.fromkeys(
            itertools.chain.from_iterable(
                mapping.model_dump().keys() for mapping in mappings
            )
        )
    )


def _merge_stac_items(
    items: dict[str, StacItem], spatial_info: OutputSpatialInfo
) -> StacItem:
    if len(items) == 0:
        raise ValueError("no STAC metadata found")

    # special-case properties, links, assets
    merge_strategies = {
        "properties": _merge_properties,
        "links": _merge_links,
        "assets": _merge_assets,
        "geometry": lambda _: json.loads(
            shapely.to_geojson(spatial_info.geometry_latlon)
        ),
        "bbox": lambda _: spatial_info.bbox,
    }

    merged_item = {}
    for name in _collect_keys(items.values()):
        merge_strategy = merge_strategies.get(name, _merge_no_conflicts)
        try:
            merged_item[name] = merge_strategy(
                [getattr(item, name) for item in items.values()]
            )
        except ValueError as e:
            log.error("attribute merging failed", name=name, error=str(e))
            raise

    return StacItem(**merged_item)
