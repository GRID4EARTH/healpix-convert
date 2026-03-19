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
    computed_field,
    field_serializer,
    model_validator,
)

from legacy_converters.core.healpix_conventions import Healpix
from legacy_converters.settings. import (
    __CONVERT_SETTINGS,
    __CONVERT_SETTINGS,
)
from legacy_converters.settings. import (
    _CONVERT_SETTINGS,
    __ERR_CONVERT_SETTINGS,
    __RBT_CONVERT_SETTINGS,
)

log = structlog.get_logger()


class BaseResamplerSettings(BaseModel):
    """Base class settings for healpix-resample."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    init_params: dict[str, Any] = Field(default_factory=dict)
    """Parameters passed to the resampler constructor."""

    resample_params: dict[str, dict[str, Any] | Any] = Field(default_factory=dict)
    """Parameters passed to the resampler's `resample()` method.

    Parameter values may be given either like:
    - a common value used for all arrays (data variables) in the group
    - a value for each array in the group (dictionary where keys are
      variable names)

    """


def broadcast_params(
    params: dict[str, dict[str, Any] | Any],
    var_names: list[str],
) -> dict[str, dict[str, Any]]:
    """Broadcast `params` values for all variables given by `var_names`.

    Returns a dictionary where keys are variable names and values are
    dictionaries of parameter values.
    """

    broadcasted: dict[str, dict[str, Any]] = {vname: {} for vname in var_names}

    for pname, pval in params.items():
        if isinstance(pval, dict):
            missing_vars = set(var_names) - set(pval)
            if missing_vars:
                raise ValueError(
                    f"missing parameter values for variables {missing_vars}"
                )
            for vname, vval in pval.items():
                broadcasted[vname][pname] = vval
        else:
            for vname in var_names:
                broadcasted[vname][pname] = pval

    return dict(broadcasted)


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


class HealpixChunkSettings(BaseModel):
    """Settings for chunking output data along the HEALPix cell dimension.

    These settings are defined for each input Zarr group to convert to HEALPix.

    """

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    method: Literal["healpix_cell", "user_defined", "no_chunk"] = "healpix_cell"
    """Chunking method.

    - "healpix_cell" (default): chunks are defined as HEALPix cells on a low
      (coarse) refinement level. The healpix grid settings for chunks are
      defined in the global conversion settings under the "healpix_chunks" field
      (see :py:class:`ConvertSettings`). The same chunk cells are thus used for
      all Zarr groups to convert to HEALPix.

    - "user_defined": chunks have a fixed size defined by the "chunk_size"
      field in these (group) settings.

    - "no_chunk": data on HEALPix doesn't need to be chunked (e.g., groups
      with small data size).

    """

    # TODO: MISSING may be better than None as default value
    # but dask (tokenize) doesn't like it
    chunk_size: int | None = None
    """User-defined chunk (fixed) size for 'user_defined' method."""

    @computed_field
    @property
    def is_fixed_size(self) -> bool:
        """Returns True if chunks have a uniform, fixed size."""
        # TODO: eventually use regular field instead
        # (when we support a method with variable chunk sizes)
        return True

    @model_validator(mode="after")
    def validate(self) -> HealpixChunkSettings:
        # TODO: add test
        if self.method == "user_defined" and self.chunk_size is None:
            raise ValueError(f"'chunk_size' must be defined for method {self.method!r}")

        return self


class HealpixGroupSettings(BaseModel):
    """Settings for converting a single Zarr group onto HEALPix."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    healpix: Healpix
    """Output HEALPix grid settings."""

    chunk: Annotated[
        HealpixChunkSettings, Field(default_factory=lambda: HealpixChunkSettings())
    ]
    """Settings for chunking data along the HEALPix cell dimension."""

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

    @model_validator(mode="after")
    def validate_chunk_vs_indexing_scheme(self) -> HealpixGroupSettings:
        # TODO: add test
        if (
            self.chunk.method == "healpix_cell"
            and self.healpix.indexing_scheme != "nested"
        ):
            raise ValueError(
                "chunk method 'healpix_cell' only supports the 'nested' indexing scheme."
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

    # TODO: may be optional
    # not needed if no group in `group_settings` uses the "healpix_cell" method
    # (note: if using MISSING, beware dask/distributed doesn't like it for tokenize)
    healpix_chunks: Healpix
    """Common grid settings for output HEALPix data chunked using the
    "healpix_cell" method (see :py:class:`HealpixChunkSettings`).
    """

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
            has_chunks = settings.chunk.method != "no_chunk"

            if has_chunks and level is not None and level < chunk_level:
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
        return ConvertSettings.model_validate(__CONVERT_SETTINGS)

    elif name == "-":
        return ConvertSettings.model_validate(__CONVERT_SETTINGS)
    elif name == "--l1-efr":
        return ConvertSettings.model_validate(_CONVERT_SETTINGS)
    elif name == "--l1-err":
        return ConvertSettings.model_validate(__ERR_CONVERT_SETTINGS)
    elif name == "--l1-rbt":
        return ConvertSettings.model_validate(__RBT_CONVERT_SETTINGS)

    else:
        raise ValueError(f"settings not found for {name!r}")
