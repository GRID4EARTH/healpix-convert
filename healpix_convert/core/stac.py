"""
The STAC discovery metadata of the input and output datasets: the item model,
the merging of the input items into the output one, and the provenance of the
conversion.
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import PurePath
from typing import TYPE_CHECKING, Any, Literal

import shapely
import structlog
import xarray as xr
from pydantic import BaseModel, ConfigDict

from healpix_convert.core.conversion_models import OutputSpatialInfo
from healpix_convert.core.json_schema import convert_settings_schema_url

if TYPE_CHECKING:
    # only for the annotations: `core` must not depend on `settings` at runtime
    from healpix_convert.settings.common import ConvertSettings
    from healpix_convert.settings.stac import STACSettings

log = structlog.get_logger()


class StacItem(BaseModel):
    """Common structure of STAC items."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["Feature"] = "Feature"
    """geojson type."""

    stac_version: Literal["1.0.0", "1.1.0"] = "1.0.0"
    """Used STAC version."""

    stac_extensions: list[str]
    """Used STAC extensions."""

    id: str
    """ID of the item."""

    bbox: list[float]
    """Spatial bounding box (lat-lon coordinates)"""

    # TODO: validate further
    geometry: dict[str, object]
    """Spatial footprint (lat-lon geometry)"""

    # TODO: validate further
    properties: dict[str, object]
    """Additional properties"""

    # TODO: validate further
    links: list[object]
    """Links to additional resources"""

    # TODO: validate further
    assets: dict[str, object]
    """Links to data files"""


STAC_PROCESSING_SCHEMA_URL = (
    "https://stac-extensions.github.io/processing/v1.2.0/schema.json"
)

#: Libraries whose versions are worth recording: the converter itself and the
#: resampling engine that decides the actual data values.
PROVENANCE_PACKAGES = ("healpix-convert", "healpix-resample")


def _package_versions() -> dict[str, str]:
    import importlib.metadata

    versions = {}
    for package in PROVENANCE_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            # not installed (e.g. an optional resampler backend): recording a
            # wrong version would be worse than recording none
            continue

    return versions


def format_stac_derived_links(input_paths: Iterable[str]) -> list[dict[str, Any]]:
    """STAC links pointing back at the datasets the output was derived from.

    `derived_from` is how STAC records that one item was produced from others,
    so the inputs stay findable — the paths are the object-store locations the
    conversion actually read.

    """
    return [
        {"rel": "derived_from", "href": str(path), "type": "application/vnd+zarr"}
        for path in input_paths
    ]


def format_stac_processing(
    settings: ConvertSettings,
    *,
    input_ids: Iterable[str] = (),
    input_paths: Iterable[str] = (),
    processing_datetime: datetime | None = None,
    include_conversion_settings: bool = True,
) -> dict[str, Any]:
    """Provenance properties for the converted dataset, following the STAC
    processing extension.

    Records what is needed to tell two outputs of the same input apart: the
    conversion settings (including the resampler and its parameters), the
    versions that produced them, the inputs, and when.

    Parameters
    ----------
    settings : :py:class:`~healpix_convert.settings.common.ConvertSettings`
        Conversion settings, recorded verbatim.
    input_ids : iterable of str, optional
        STAC ids of the input items the output was derived from.
    processing_datetime : :py:class:`datetime.datetime`, optional
        Defaults to the current UTC time.
    include_conversion_settings : bool, optional
        If True (the default), record the settings as ``processing:expression``.

    """
    if processing_datetime is None:
        processing_datetime = datetime.now(UTC)

    versions = _package_versions()

    properties: dict[str, Any] = {
        "processing:datetime": processing_datetime.isoformat(),
        "processing:software": versions,
    }

    if include_conversion_settings:
        # the settings are what make two outputs of one input differ, so they
        # are recorded rather than summarised
        properties["processing:expression"] = {
            # the schema of the recorded version, not the latest one: the
            # settings may not validate against the schema of a later version
            "format": convert_settings_schema_url(versions.get("healpix-convert")),
            "expression": settings.to_dict(),
        }

    # name both what the inputs *are* (STAC ids) and where they were read
    # from, so the output can be traced back to the actual data
    described = [str(p) for p in input_paths] or [str(i) for i in input_ids]
    if described:
        properties["processing:lineage"] = (
            "Resampled onto HEALPix by healpix-convert from " + ", ".join(described)
        )

    return properties


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


def _merge_mapping(values: list[Mapping[str, Any]], strategy: str) -> dict[str, Any]:
    """Merge the keyed STAC metadata of the input items (properties, assets).

    ``join_inner`` keeps what every input agrees on, key and value: an entry
    that one input spells differently is not something they have in common.
    """
    if strategy == "pick_first" or len(values) == 1:
        return dict(values[0])

    merged: dict[str, Any] = {}

    others = values[1:]
    for key, value in values[0].items():
        if all(key in other and other[key] == value for other in others):
            merged[key] = value

    return merged


def _merge_sequence(values: list[Sequence[Any]], strategy: str) -> list[Any]:
    """Merge the listed STAC metadata of the input items (extensions, links).

    Items are merged whole: two of them are the same one when they are equal,
    which is all a link has to go on. ``join_inner`` keeps what every input
    has, ``join_outer`` what any input has, in the order they first appear.
    """
    if strategy == "pick_first" or len(values) == 1:
        return list(values[0])

    merged: list[Any] = []

    if strategy == "join_outer":
        for items in values:
            for item in items:
                if item not in merged:
                    merged.append(item)

        return merged

    others = values[1:]
    for item in values[0]:
        if item not in merged and all(item in other for other in others):
            merged.append(item)

    return merged


def _collect_keys(mappings: Iterable[StacItem]):
    return list(
        dict.fromkeys(
            itertools.chain.from_iterable(
                mapping.model_dump().keys() for mapping in mappings
            )
        )
    )


def merge_stac_items(
    items: dict[str, StacItem],
    spatial_info: OutputSpatialInfo,
    output_path: str,
    stac_settings: STACSettings | None = None,
) -> StacItem:
    """Merge the STAC items of the input datasets into the output's one.

    The geometry, the bounding box and the id describe the output. The
    properties, the extensions, the links and the assets are merged with the
    strategy the settings select for each of them (all ``pick_first`` without
    settings). Everything else must match across the inputs.

    """
    if len(items) == 0:
        raise ValueError("no STAC metadata found")

    item_id = re.sub("[^a-zA-Z0-9-]", "_", PurePath(output_path).stem)

    if stac_settings is None:
        merge_properties = merge_extensions = "pick_first"
        merge_links = merge_assets = "pick_first"
    else:
        merge_properties = stac_settings.merge_properties
        merge_extensions = stac_settings.merge_extensions
        merge_links = stac_settings.merge_links
        merge_assets = stac_settings.merge_assets

    # special-case properties, extensions, links, assets
    merge_strategies = {
        "properties": lambda v: _merge_mapping(v, merge_properties),
        "stac_extensions": lambda v: _merge_sequence(v, merge_extensions),
        "links": lambda v: _merge_sequence(v, merge_links),
        "assets": lambda v: _merge_mapping(v, merge_assets),
        "geometry": lambda _: json.loads(
            shapely.to_geojson(spatial_info.geometry_latlon)
        ),
        "bbox": lambda _: spatial_info.bbox,
        "id": lambda _: item_id,
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
