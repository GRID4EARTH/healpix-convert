"""
Pydantic model classes for Zarr Multiscales conventions
"""

from __future__ import annotations

from pathlib import PurePath
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_serializer
from pydantic.experimental.missing_sentinel import MISSING

from healpix_convert.core.healpix_conventions import Healpix

_UUID: str = "d35379db-88df-4056-af3a-620245f8e347"
_SCHEMA_URL = "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v1/schema.json"
_SPEC_URL = "https://github.com/zarr-conventions/multiscales/blob/v1/README.md"
_DESCR = "Multiscale layout of zarr datasets"


class MultiscalesZarrConvention(BaseModel):
    """Multiscales Zarr convention metadata.

    See https://github.com/zarr-conventions/multiscales

    """

    # TODO: use computed_field instead?
    # (https://github.com/pydantic/pydantic/issues/1927#issuecomment-2888510681)
    # this looks pretty ugly
    uuid: Literal["d35379db-88df-4056-af3a-620245f8e347"] = _UUID
    name: Literal["multiscales"] = "multiscales"
    schema_url: Literal[
        "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v1/schema.json"
    ] = _SCHEMA_URL
    spec_url: Literal[
        "https://github.com/zarr-conventions/multiscales/blob/v1/README.md"
    ] = _SPEC_URL
    description: Literal["Multiscale layout of zarr datasets"] = _DESCR


class HealpixMultiscalesLayoutItem(BaseModel):
    """Metadata for a single Healpix multiscales layout item."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    asset: PurePath
    """Name of the single-scale group (healpix level)."""

    dggs: Healpix | MISSING = MISSING
    """HEALPix grid settings (absolute positioning)."""

    @field_serializer("asset")
    def serialize_group_settings(self, value):
        # always convert paths to strings (unix-style)
        return value.as_posix()


class HealpixMultiscales(BaseModel):
    """Metadata for a Healpix multiscales group."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    layout: list[HealpixMultiscalesLayoutItem]
    """Multiscale description."""
