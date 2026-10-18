"""
Pydantic model classes for Zarr Multiscales conventions
"""

from __future__ import annotations

from pathlib import PurePath
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_serializer, model_validator

from healpix_convert.core.healpix_conventions import Healpix
from healpix_convert.core.pydantic_compat import MISSING

_UUID: str = "d35379db-88df-4056-af3a-620245f8e347"
_SCHEMA_URL = "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v0.1/schema.json"
_SPEC_URL = "https://github.com/zarr-conventions/multiscales/blob/v0.1/README.md"
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
        "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v0.1/schema.json"
    ] = _SCHEMA_URL
    spec_url: Literal[
        "https://github.com/zarr-conventions/multiscales/blob/v0.1/README.md"
    ] = _SPEC_URL
    description: Literal["Multiscale layout of zarr datasets"] = _DESCR


class MultiscalesTransform(BaseModel):
    """Coordinate transformation of a multiscales layout item, from the level
    it is derived from.
    """

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    scale: list[float]
    """Scale factor per axis, from the source level to this one.

    A HEALPix group has a single axis, the cell dimension, so there is a
    single factor: the length of a cell relative to a cell of the source
    level, i.e. ``2 ** (source refinement level - this refinement level)``.
    HEALPix tessellates a 2D surface, so one cell of this level covers the
    square of that, the number of cells of the source level it aggregates.
    """


class HealpixMultiscalesLayoutItem(BaseModel):
    """Metadata for a single Healpix multiscales layout item."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    asset: PurePath
    """Path of the single-scale group (healpix level), relative to the group
    holding the multiscales metadata.
    """

    dggs: Healpix | MISSING = MISSING
    """HEALPix grid settings (absolute positioning)."""

    derived_from: PurePath | MISSING = MISSING
    """Path of the single-scale group this level is aggregated from, relative
    to the group holding the multiscales metadata.

    Only given for a derived level, i.e., a level of a pyramid that is
    aggregated from a level converted from the input data.
    """

    transform: MultiscalesTransform | MISSING = MISSING
    """Transformation from the level given by :py:attr:`derived_from` to this
    one. The convention requires it whenever ``derived_from`` is given.
    """

    resampling_method: str | MISSING = MISSING
    """How the values of the source cells are aggregated into one cell of this
    level (derived levels only).
    """

    @field_serializer("asset", "derived_from")
    def serialize_group_settings(self, value):
        # always convert paths to strings (unix-style)
        return value.as_posix() if isinstance(value, PurePath) else value

    @model_validator(mode="after")
    def ensure_transform_of_derived_level(self) -> HealpixMultiscalesLayoutItem:
        if self.derived_from is not MISSING and self.transform is MISSING:
            raise ValueError("field 'transform' is required with 'derived_from'")

        return self


class HealpixMultiscales(BaseModel):
    """Metadata for a Healpix multiscales group."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    layout: list[HealpixMultiscalesLayoutItem]
    """Multiscale description."""

    resampling_method: str | MISSING = MISSING
    """Default method the derived levels are aggregated with.

    Given for a group holding a pyramid, whose derived levels are all
    aggregated with the same method.
    """
