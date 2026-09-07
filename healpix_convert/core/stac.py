"""
Pydantic model classes for stac discovery metadata
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    # only for the annotation: `core` must not depend on `settings` at runtime
    from healpix_convert.settings.common import ConvertSettings


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


PROCESSING_EXTENSION = (
    "https://stac-extensions.github.io/processing/v1.2.0/schema.json"
)

#: Libraries whose versions are worth recording: the converter itself and the
#: resampling engine that decides the actual data values.
PROVENANCE_PACKAGES = ("healpix-convert", "healpix-resample")

#: `format` of the processing:expression object holding the conversion settings.
SETTINGS_EXPRESSION_FORMAT = "healpix-convert/convert-settings"


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


def derived_from_links(input_paths: Iterable[str]) -> list[dict[str, Any]]:
    """STAC links pointing back at the datasets the output was derived from.

    `derived_from` is how STAC records that one item was produced from others,
    so the inputs stay findable — the paths are the object-store locations the
    conversion actually read.

    """
    return [
        {"rel": "derived_from", "href": str(path), "type": "application/vnd+zarr"}
        for path in input_paths
    ]


def processing_properties(
    settings: ConvertSettings,
    *,
    input_ids: Iterable[str] = (),
    input_paths: Iterable[str] = (),
    processing_datetime: datetime | None = None,
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

    """
    if processing_datetime is None:
        processing_datetime = datetime.now(UTC)

    properties: dict[str, Any] = {
        "processing:datetime": processing_datetime.isoformat(),
        "processing:software": _package_versions(),
        # the settings are what make two outputs of one input differ, so they
        # are recorded rather than summarised
        "processing:expression": {
            "format": SETTINGS_EXPRESSION_FORMAT,
            "expression": settings.to_dict(),
        },
    }

    # name both what the inputs *are* (STAC ids) and where they were read
    # from, so the output can be traced back to the actual data
    described = [str(p) for p in input_paths] or [str(i) for i in input_ids]
    if described:
        properties["processing:lineage"] = (
            "Resampled onto HEALPix by healpix-convert from " + ", ".join(described)
        )

    return properties
