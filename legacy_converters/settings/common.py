"""
Pydantic model classes for conversion settings.
"""

from __future__ import annotations

from pathlib import PurePath
from typing import Annotated, Any, Literal, TypeAlias

import structlog
from pydantic import BaseModel, ConfigDict, Field, model_validator

from legacy_converters.core.healpix_conventions import Healpix

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

    min_cell_total_weight: Annotated[float, Field(gt=0.0)] = 0.1
    """Minimum total weight (sum) required to fill a HEALPix cell with a value."""


class NearestResamplerSettings(BaseResamplerSettings):
    """Settings for resampling data onto HEALPix using nearest neighbor (k=1) interpolation."""

    name: Literal["nearest"] = "nearest"


class BilinearResamplerSettings(BaseResamplerSettings):
    """Settings for resampling data onto HEALPix using bilinear (k=4) interpolation."""

    name: Literal["bilinear"] = "bilinear"

    min_cell_total_weight: Annotated[float, Field(gt=0.0)] = 0.1
    """Minimum total weight (sum) required to fill a HEALPix cell with a value."""


class PSFResamplerSettings(BaseResamplerSettings):
    """Settings for resampling data onto HEALPix using PSF interpolation."""

    name: Literal["psf"] = "psf"

    min_cell_total_weight: Annotated[float, Field(gt=0.0)] = 0.1
    """Minimum total weight (sum) required to fill a HEALPix cell with a value."""


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


class ConvertGroupSettings(BaseModel):
    """Settings for converting a single Zarr group onto HEALPix."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    healpix: Healpix
    """Output HEALPix grid settings."""

    resampler: Annotated[ResamplerSettings, Field(discriminator="name")]
    """Resampling method name and settings."""

    @model_validator(mode="after")
    def validate_cell_point_resampler_level(self) -> ConvertGroupSettings:
        if self.resampler.name == "cell-point" and self.healpix.refinement_level != 29:
            raise ValueError(
                "HEALPix refinement level must be equal to 29 when the 'cell-point' resampler "
                "is used."
            )

        return self


class ConvertSettings(BaseModel):
    """General settings for converting input Zarr dataset(s) onto HEALPix."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    chunk_refinement_level: Annotated[int, Field(gt=0)]
    """HEALPix refinement level used to delineate chunks in the output Zarr dataset."""

    group_settings: dict[PurePath, ConvertGroupSettings]
    """Conversion settings per group in the input Zarr dataset(s)."""

    @model_validator(mode="after")
    def validate_chunk_refinement_level(self) -> ConvertSettings:
        """Ensure that the given chunk refinement level is greater or equal to
        the output HEALPix refinement levels given in each group settings."""
        for name, settings in self.group_settings.items():
            level = settings.healpix.refinement_level
            if level is not None and level < self.chunk_refinement_level:
                raise ValueError(
                    f"found output HEALPix refinement level {level} set in group {name}, "
                    f"which is lower than the chunk refinement level {self.chunk_refinement_level}."
                )

        return self
