"""
Pydantic model classes for stac discovery metadata
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


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
