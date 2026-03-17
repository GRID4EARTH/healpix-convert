"""
Pydantic model classes for conversion settings.
"""

from __future__ import annotations

from pathlib import PurePath
from typing import Annotated, Any, Literal, TypeAlias

import structlog
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    field_serializer,
    model_validator,
)

from legacy_converters.core.healpix_conventions import Healpix
from legacy_converters.settings. import _CONVERT_SETTINGS

log = structlog.get_logger()


class BaseResamplerSettings(BaseModel):

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True, extra="allow")

    @model_validator(mode="before")
    @classmethod
    def raise_warning_extra_fields(cls, values: dict[str, Any]) -> dict[str, Any]:
        extra_fields = set(values) - set(cls.model_fields)
        if extra_fields:
            log.warning(
                f"The following user settings {extra_fields!r} are given to {cls.__name__}. "
                "this is temporarily allowed (early development stage) but will be ignored "
                "or forbidden in the future as a list of valid setting fields are added here "
                "for each resampler."
            )

        return values


class KNearestNeighborsResamplerSettings(BaseResamplerSettings):
    """Settings for resampling data onto HEALPix using k-nearest neighbor interpolation."""

    name: Literal["k-nearest"] = "k-nearest"


class NearestResamplerSettings(BaseResamplerSettings):
    """Settings for resampling data onto HEALPix using nearest neighbor (k=1) interpolation."""

    name: Literal["nearest"] = "nearest"


class BilinearResamplerSettings(BaseResamplerSettings):
    """Settings for resampling data onto HEALPix using bilinear (k=4) interpolation."""

    name: Literal["bilinear"] = "bilinear"


class PSFResamplerSettings(BaseResamplerSettings):
    """Settings for resampling data onto HEALPix using PSF interpolation."""

    name: Literal["psf"] = "psf"


class CellPointResamplerSettings(BaseResamplerSettings):
    """Settings for resampling data as HEALPix "cell-points" (i.e., cells with maximum
    refinement level 29).
    """

    name: Literal["cell-point"] = "cell-point"


ResamplerSettings: TypeAlias = (
    KNearestNeighborsResamplerSettings
    | NearestResamplerSettings
    | BilinearResamplerSettings
    | PSFResamplerSettings
    | CellPointResamplerSettings
)


class HealpixGroupSettings(BaseModel):
    """Settings for converting a single Zarr group onto HEALPix."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    healpix: Healpix
    """Output HEALPix grid settings."""

    chunk: bool = True
    """Whether or not data resampled on HEALPix is chunked."""

    resampler: Annotated[ResamplerSettings, Field(discriminator="name")]
    """Resampling method name and settings."""

    @model_validator(mode="after")
    def validate_cell_point_resampler_level(self) -> HealpixGroupSettings:
        if self.resampler.name == "cell-point" and self.healpix.refinement_level != 29:
            raise ValueError(
                "HEALPix refinement level must be equal to 29 when the 'cell-point' resampler "
                "is used."
            )

        return self


class MultiscalesGroupSettings(BaseModel):
    """Settings for a multiscale Zarr group."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    multiscales: bool = True
    """If True, add multiscales metadata to this group.

    The Zarr group must contain children groups with defined
    HEALPix conversion settings.

    The metadata is compliant with the "multiscales" Zarr convention.
    """

    add_healpix_positioning: bool = True
    """If True, add HEALPix grid settings as absolute positioning
    information in each multiscale layout entry.
    """


def get_group_settings_tag(v: Any) -> str | None:
    if isinstance(v, dict):
        if "multiscales" in v:
            return "multiscales_group"
        elif "healpix" in v and "resampler" in v:
            return "healpix_group"
        else:
            return None
    if isinstance(v, BaseModel):
        if hasattr(v, "multiscales"):
            return "multiscales_group"
        elif hasattr(v, "healpix") and hasattr(v, "resampler"):
            return "healpix_group"
        else:
            return None
    else:
        return None


class ConvertSettings(BaseModel):
    """General settings for converting input Zarr dataset(s) onto HEALPix."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    healpix_chunks: Healpix
    """Common grid settings for output HEALPix data chunks."""

    exclude_groups: list[PurePath] = Field(default_factory=list)
    """Groups to exclude from the conversion (won't be included in output)."""

    group_settings: dict[
        PurePath,
        Annotated[
            (
                Annotated[MultiscalesGroupSettings, Tag("multiscales_group")]
                | Annotated[HealpixGroupSettings, Tag("healpix_group")]
            ),
            Discriminator(get_group_settings_tag),
        ],
    ]
    """Conversion settings per group in the input Zarr dataset(s)."""

    @field_serializer("exclude_groups")
    def serialize_exclude_groups(self, value):
        # always convert paths to strings (unix-style)
        return [v.as_posix() for v in value]

    @field_serializer("group_settings")
    def serialize_group_settings(self, value):
        # always convert paths to strings (unix-style)
        return {k.as_posix(): v for k, v in value.items()}

    @model_validator(mode="after")
    def validate_healpix_chunks(self) -> ConvertSettings:
        if self.healpix_chunks.refinement_level is None:
            raise ValueError("refinement level must be defined for HEALPix chunks.")

        if self.healpix_chunks.indexing_scheme != "nested":
            raise ValueError(
                "only the 'nested' indexing scheme is currently supported for HEALPix chunks"
            )

        # Ensure that refinement level for one Zarr group is not lower (coarser)
        # than the refinement level set for Zarr chunks.
        chunk_level = self.healpix_chunks.refinement_level

        for name, settings in self.group_settings.items():
            if not isinstance(settings, HealpixGroupSettings):
                continue

            level = settings.healpix.refinement_level

            if settings.chunk and level is not None and level < chunk_level:
                raise ValueError(
                    f"found output HEALPix refinement level {level} set in group {name}, "
                    f"which is lower than the chunk refinement level {chunk_level}."
                )

        return self

    @model_validator(mode="after")
    def validate_multiscales(self) -> ConvertSettings:
        # TODO: check multiscale settings children groups
        # - there exists >1 (or >=1?) direct child paths with HealpixGroupSetting instance
        # - all those instances have the same Healpix settings except refinement level
        # - all those instances have the same resampler settings (at least same method)

        return self


def get_settings(name: str) -> ConvertSettings:
    """Get default conversion settings.

    Parameters
    ----------
    name : str
        Name of the product.

    Returns
    -------
    :py:class:`ConvertSettings`

    """
    if name == "-":
        return ConvertSettings.model_validate(_CONVERT_SETTINGS)
    else:
        raise ValueError(f"settings not found for {name!r}")
