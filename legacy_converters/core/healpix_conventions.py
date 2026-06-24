"""
Pydantic model classes for DGGS / HEALPix conventions
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator
from pydantic.experimental.missing_sentinel import MISSING

_UUID: str = "7b255807-140c-42ca-97f6-7a1cfecdbc38"
_SCHEMA_URL = (
    "https://raw.githubusercontent.com/zarr-conventions/dggs/refs/tags/v1/schema.json"
)
_SPEC_URL = "https://github.com/zarr-conventions/dggs/blob/v1/README.md"
_DESCR = "Discrete Global Grid Systems convention for zarr"


class DGGSZarrConvention(BaseModel):
    """DGGS Zarr convention metadata.

    See https://github.com/zarr-conventions/dggs

    """

    # TODO: use computed_field instead?
    # (https://github.com/pydantic/pydantic/issues/1927#issuecomment-2888510681)
    # this looks pretty ugly
    uuid: Literal["7b255807-140c-42ca-97f6-7a1cfecdbc38"] = _UUID
    name: Literal["dggs"] = "dggs"
    schema_url: Literal[
        "https://raw.githubusercontent.com/zarr-conventions/dggs/refs/tags/v1/schema.json"
    ] = _SCHEMA_URL
    spec_url: Literal["https://github.com/zarr-conventions/dggs/blob/v1/README.md"] = (
        _SPEC_URL
    )
    description: Literal["Discrete Global Grid Systems convention for zarr"] = _DESCR


class WGS84Ellipsoid(BaseModel):
    """Singleton model defining the WGS84 ellipsoid.

    Compliant with the DGGS Zarr convention.

    """

    name: Literal["wgs84"] = "wgs84"

    @computed_field
    @property
    def semimajor_axis(self) -> float:
        # workaround as Python's Literal doesn't support float
        return 6378137.0

    @computed_field
    @property
    def inverse_flattening(self) -> float:
        # workaround as Python's Literal doesn't support float
        return 298.257223563


class Healpix(BaseModel):
    """HEALPix grid.

    Compliant with the DGGS Zarr convention.

    """

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    name: Literal["healpix"] = "healpix"

    refinement_level: Annotated[int | None, Field(ge=0, le=29)]
    """HEALPix refinement level (must be defined for "nested" and "ring" indexing schemes)."""

    indexing_scheme: Literal["nested", "ring", "zuniq"] = "nested"
    """HEALPix indexing scheme."""

    ellipsoid: Annotated[WGS84Ellipsoid, Field(discriminator="name")] | MISSING = (
        MISSING
    )
    """Reference system of the DGGS (if not given, a sphere is assumed)."""

    spatial_dimension: str = "cells"
    """Name of the spatial dimension"""

    coordinate: str | MISSING = "cell_ids"
    """Points to the array containing the cell ids (optional for homogeneous full-earth coverage)."""

    compression: Literal["none", "compacted", "ranges"] | MISSING = "none"
    """Cell id compression method."""

    @model_validator(mode="after")
    def ensure_defined_fields(self) -> Healpix:
        if self.refinement_level is None:
            if self.indexing_scheme in ("nested", "ring"):
                raise ValueError(
                    f"refinement level must be defined for {self.indexing_scheme!r} indexing scheme"
                )
            if self.compression != "none":
                raise ValueError(
                    "refinement level is undefined but compression is not 'none'"
                )
            if self.coordinate is MISSING:
                raise ValueError(
                    "field 'coordinate' is omitted and refinement level is undefined"
                )

        return self


class CFHealpixGridMapping(BaseModel):
    """CF-conventions HEALPix grid mapping (CF 1.13+)."""

    model_config = ConfigDict(frozen=True)

    grid_mapping_name: Literal["healpix"] = "healpix"

    refinement_level: Annotated[int | MISSING, Field(ge=0, le=29)] = MISSING
    """HEALPix refinement level (must be given for "nested" and "ring" indexing schemes)."""

    indexing_scheme: Literal["nested", "ring", "zuniq"] = "nested"
    """HEALPix indexing scheme."""

    reference_ellipsoid_name: str | MISSING = MISSING
    semi_major_axis: float | MISSING = MISSING
    inverse_flattening: float | MISSING = MISSING

    @model_validator(mode="after")
    def validate_indexing_scheme_vs_refinement_level(self) -> CFHealpixGridMapping:
        if (
            self.indexing_scheme in ("nested", "ring")
            and self.refinement_level is MISSING
        ):
            raise ValueError(
                f"refinement level must be defined for {self.indexing_scheme!r} indexing scheme"
            )
        if self.indexing_scheme == "zuniq" and isinstance(self.refinement_level, int):
            raise ValueError(
                f"refinement level must be omitted for {self.indexing_scheme!r} indexing scheme"
            )

        return self

    @classmethod
    def from_healpix(cls, hp: Healpix) -> CFHealpixGridMapping:
        """Build a CF grid mapping from a :class:`Healpix` grid description."""
        kwargs: dict = {
            "refinement_level": hp.refinement_level,
            "indexing_scheme": hp.indexing_scheme,
        }
        ellipsoid = hp.ellipsoid
        if ellipsoid is not MISSING:
            kwargs["reference_ellipsoid_name"] = ellipsoid.name.upper()
            kwargs["semi_major_axis"] = ellipsoid.semimajor_axis
            kwargs["inverse_flattening"] = ellipsoid.inverse_flattening

        return cls(**kwargs)
