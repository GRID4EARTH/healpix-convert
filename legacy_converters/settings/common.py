"""
Pydantic model classes for conversion settings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import PurePath
from typing import Annotated, Any, Literal, TypeAlias

import structlog
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_serializer,
    model_validator,
)

# pydantic requires typing_extensions.TypedDict (not typing.TypedDict) on Python < 3.12
from typing_extensions import TypedDict

from legacy_converters.core.healpix_conventions import Healpix
from legacy_converters.settings.cams import CAMS_CONVERT_SETTINGS
from legacy_converters.settings.climatedt import CDT_SFC_CONVERT_SETTINGS
from legacy_converters.settings.era5 import ERA5_CONVERT_SETTINGS
from legacy_converters.settings. import (
    __CONVERT_SETTINGS,
    __CONVERT_SETTINGS,
)
from legacy_converters.settings. import (
    _CONVERT_SETTINGS,
    __ERR_CONVERT_SETTINGS,
    __LFR_CONVERT_SETTINGS,
    __LRR_CONVERT_SETTINGS,
    __FRP_CONVERT_SETTINGS,
    __LST_CONVERT_SETTINGS,
    __RBT_CONVERT_SETTINGS,
)
from legacy_converters.settings._psf import (
    _PSF_CONVERT_SETTINGS,
    __ERR_PSF_CONVERT_SETTINGS,
    __LFR_PSF_CONVERT_SETTINGS,
    __LRR_PSF_CONVERT_SETTINGS,
    __FRP_PSF_CONVERT_SETTINGS,
    __LST_PSF_CONVERT_SETTINGS,
    __RBT_PSF_CONVERT_SETTINGS,
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


class BaseChunkSettings(ABC, BaseModel):
    """Base abstract class for chunking strategy and settings."""

    method: str

    @property
    @abstractmethod
    def is_fixed_size(self) -> bool:
        """Returns True if chunks have a uniform, fixed size."""
        ...

    @property
    @abstractmethod
    def is_healpix_aligned(self) -> bool:
        """Returns True if chunk bounds and/or content are aligned with HEALPix cells."""
        ...


class NoChunkSettings(BaseChunkSettings):
    """Do not chunk output data along the HEALPix cell dimension."""

    method: Literal["no_chunk"] = "no_chunk"

    @property
    def is_fixed_size(self) -> bool:
        return True

    @property
    def is_healpix_aligned(self) -> bool:
        return False


class HealpixUniformChunkSettings(BaseChunkSettings):
    """Common class for fixed-size chunks aligned with HEALPix at a given refinement level."""

    healpix: Healpix
    """HEALPix grid settings used to define "chunk" cells."""

    @property
    def is_fixed_size(self) -> bool:
        return True

    @property
    def is_healpix_aligned(self) -> bool:
        return True

    @model_validator(mode="after")
    def validate_healpix_settings(self) -> HealpixUniformChunkSettings:
        # TODO: add tests
        if self.healpix.refinement_level is None:
            raise ValueError("refinement level must be defined for HEALPix chunks.")

        if self.healpix.indexing_scheme != "nested":
            raise ValueError(
                f"chunk method {self.method!r} only supports the 'nested' indexing scheme."
            )

        return self


class HealpixDenseChunkSettings(HealpixUniformChunkSettings):
    r"""Chunk output data along the HEALPix cell dimension according to
    HEALPix cells given at a parent, "coarse" refinement level (dense version).

    Each of those "chunk" cells are densely populated with adjacent children
    HEALPix cells all at the same "fine" refinement level.

    Chunk size is thus fixed and is given by the following formula:

    .. math:: 4^{(\Delta - \delta)}

    where :math:`\Delta` is the parent level (output chunks) and :math:`\delta` is
    the child level (output data values).

    This chunking strategy works only with the "nested" HEALPIX cell indexing scheme.

    """

    method: Literal["healpix_cell_dense"] = "healpix_cell_dense"

    chunk_buffer_width: float = 60.0
    """Fixed, uniform width to apply around the edges of a HEALPix coarse
    "chunk" cell.

    This parameter is used for spatial filtering of input data given in a
    projected coordinate reference system (CRS), prior to resampling it onto the
    HEALPix cells within the chunk. The buffer width value should thus be given
    in the same CRS units (usually in meters) and greater or equal to zero.

    Set the value to zero if no buffer is needed (e.g., some resampling methods
    do not require it).

    """


class HealpixSparseChunkSettings(HealpixUniformChunkSettings):
    r"""Chunk output data along the HEALPix cell dimension according to
    HEALPix cells given at a parent, "coarse" refinement level (sparse version).

    Each of those "chunk" cells are populated with sparsely distributed children
    HEALPix cells given at a "fine" refinement level.

    Chunk size is user defined. There's no strict constraint on the minimum
    size, which should be large enough such that a parent "chunk" cell can
    contain all output data values (children cells). If all children cells are
    at the same refinement level, the maximum size is given by the following
    formula:

    .. math:: 4^{(\Delta - \delta)}

    where :math:`\Delta` is the parent level (output chunks) and :math:`\delta` is
    the child level (output data values).

    This chunking strategy works only with the "nested" HEALPIX cell indexing scheme.

    """

    method: Literal["healpix_cell_sparse"] = "healpix_cell_sparse"

    chunk_size: int
    """User-defined, uniform chunk size."""


ChunkSettings: TypeAlias = (
    NoChunkSettings | HealpixDenseChunkSettings | HealpixSparseChunkSettings
)


class CodecSettings(TypedDict):
    """Settings for a single Zarr codec.

    This class follows Zarr's codec (extension) metadata model.

    """

    name: str
    configuration: dict[str, Any]


class HealpixGroupSettings(BaseModel):
    """Settings for converting a single Zarr group onto HEALPix."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    healpix: Healpix
    """Output HEALPix grid settings."""

    chunk: Annotated[ChunkSettings, Field(discriminator="method")]
    """Settings for chunking data along the HEALPix cell dimension."""

    resampler: Annotated[ResamplerSettings, Field(discriminator="name")]
    """Resampling method name and settings."""

    codecs: list[CodecSettings] | None = Field(default=None)

    @model_validator(mode="after")
    def validate_cell_point_resampler_level(self) -> HealpixGroupSettings:
        if self.resampler.name == "cell-point" and self.healpix.refinement_level != 29:
            raise ValueError(
                "HEALPix refinement level must be equal to 29 when the 'cell-point' resampler "
                "is used."
            )

        return self

    @model_validator(mode="after")
    def validate_healpix_chunk_vs_data(self) -> HealpixGroupSettings:
        # validate healpix settings given for output chunks vs. output resampled data
        # TODO: add test
        chunk_healpix: Healpix | None = getattr(self.chunk, "healpix", None)
        if chunk_healpix is None:
            return self

        if self.healpix.indexing_scheme != "nested":
            raise ValueError(
                f"chunk method {self.chunk.method!r} only supports the 'nested' indexing scheme."
            )

        chunk_level = chunk_healpix.refinement_level
        level = self.healpix.refinement_level

        if level is not None and chunk_level is not None and level < chunk_level:
            raise ValueError(
                f"found output HEALPix refinement level {level} "
                f"that is lower than the chunk refinement level {chunk_level}."
            )

        return self


class MultiscaleSettings(BaseModel):
    """Settings for a multiscale Zarr groups."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    groups: list[PurePath] = Field(default_factory=list)
    """List of all multiscale groups in input dataset(s).

    Each Zarr group must contain children groups with defined
    HEALPix conversion settings.

    The metadata that will be added to the corresponding ouput group
    is compliant with the "multiscales" Zarr convention.
    """

    add_healpix_positioning: bool = True
    """If True, add HEALPix grid settings as absolute positioning
    information in each multiscale layout entry.
    """

    @field_serializer("groups")
    def serialize_groups(self, value):
        # always convert paths to strings (unix-style)
        return [v.as_posix() for v in value]


class ConvertSettings(BaseModel):
    """General settings for converting input Zarr dataset(s) onto HEALPix."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    exclude_groups: list[PurePath] = Field(default_factory=list)
    """Groups to exclude from the conversion (won't be included in output)."""

    multiscale_settings: MultiscaleSettings = Field(default_factory=MultiscaleSettings)
    """Settings for multiscale Zarr groups."""

    group_settings: dict[PurePath, HealpixGroupSettings]
    """Conversion settings per group in the input Zarr dataset(s)."""

    def filter_and_replace(self, func: Callable[[str, dict], dict]) -> ConvertSettings:
        """Filter and replace conversion settings by applying a specified
        function over the settings of each group.

        Parameters
        ----------
        func: callable
            A function which accepts a string (group path) and a dictionary
            (group settings) as arguments and that returns a dictionary (new
            group settings).

        Returns
        -------
        :py:class:`ConvertSettings`
            New conversion settings (validated).

        """
        settings = self.model_dump()

        new_group_settings = {}

        for group_name, group_settings in settings["group_settings"].items():
            new_group_settings[group_name] = func(group_name, group_settings)

        settings["group_settings"] = new_group_settings

        return ConvertSettings.model_validate(settings)

    @field_serializer("exclude_groups")
    def serialize_exclude_groups(self, value):
        # always convert paths to strings (unix-style)
        return [v.as_posix() for v in value]

    @field_serializer("group_settings")
    def serialize_group_settings(self, value):
        # always convert paths to strings (unix-style)
        return {k.as_posix(): v for k, v in value.items()}

    @model_validator(mode="after")
    def validate_multiscales(self) -> ConvertSettings:
        # TODO: check multiscale settings children groups
        # - there exists >1 (or >=1?) direct child paths with HealpixGroupSetting instance
        # - all those instances have the same Healpix settings except refinement level
        # - all those instances have the same resampler settings (at least same method)

        return self

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ConvertSettings:
        """Create a new settings object from a dictionary."""
        return cls.model_validate(value)

    def to_dict(self) -> dict[str, Any]:
        """Return settings as a Python dictionary."""
        return self.model_dump()

    @classmethod
    def from_json(cls, value: str) -> ConvertSettings:
        """Create a new settings object from a JSON-formatted string."""
        return cls.model_validate_json(value)

    def to_json(self) -> str:
        """Return settings as a JSON-formatted string."""
        return self.model_dump_json()


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
    elif name == "--l2-lfr":
        return ConvertSettings.model_validate(__LFR_CONVERT_SETTINGS)
    elif name == "--l2-lrr":
        return ConvertSettings.model_validate(__LRR_CONVERT_SETTINGS)
    elif name == "--l1-rbt":
        return ConvertSettings.model_validate(__RBT_CONVERT_SETTINGS)
    elif name == "--l2-lst":
        return ConvertSettings.model_validate(__LST_CONVERT_SETTINGS)
    elif name == "--l2-frp":
        return ConvertSettings.model_validate(__FRP_CONVERT_SETTINGS)
    elif name == "--l1-efr-psf":
        return ConvertSettings.model_validate(_PSF_CONVERT_SETTINGS)
    elif name == "--l1-err-psf":
        return ConvertSettings.model_validate(__ERR_PSF_CONVERT_SETTINGS)
    elif name == "--l2-lfr-psf":
        return ConvertSettings.model_validate(__LFR_PSF_CONVERT_SETTINGS)
    elif name == "--l2-lrr-psf":
        return ConvertSettings.model_validate(__LRR_PSF_CONVERT_SETTINGS)
    elif name == "--l1-rbt-psf":
        return ConvertSettings.model_validate(__RBT_PSF_CONVERT_SETTINGS)
    elif name == "--l2-lst-psf":
        return ConvertSettings.model_validate(__LST_PSF_CONVERT_SETTINGS)
    elif name == "--l2-frp-psf":
        return ConvertSettings.model_validate(__FRP_PSF_CONVERT_SETTINGS)
    elif name == "era5":
        return ConvertSettings.model_validate(ERA5_CONVERT_SETTINGS)
    elif name == "cams-eac4":
        return ConvertSettings.model_validate(CAMS_CONVERT_SETTINGS)
    elif name == "climatedt-sfc":
        return ConvertSettings.model_validate(CDT_SFC_CONVERT_SETTINGS)

    else:
        raise ValueError(f"settings not found for {name!r}")


def validate_convert_settings(settings: dict | ConvertSettings) -> ConvertSettings:
    if not isinstance(settings, ConvertSettings):
        try:
            settings = ConvertSettings.model_validate(settings)
        except ValidationError as err:
            log.error(
                "error while validating conversion settings:\n"
                + "\n".join(e["msg"] for e in err.errors())
            )
            raise

    return settings
