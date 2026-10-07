"""
Pydantic model classes for selecting the conventions used in the output dataset.

The conversion is not opinionated about which conventions the output Zarr dataset
follows: the data provider selects them via
:py:attr:`~healpix_convert.settings.common.ConvertSettings.metadata`, e.g.,

.. code-block:: python

    {
        "zarr_conventions": [
            "dggs",
            "multiscales",
            {"name": "cf", "version": "1.13", "configuration": {"healpix_grid_mapping": True}},
        ]
    }

Only established, versioned conventions are supported (see
:py:obj:`ConventionSelector`).

"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic.experimental.missing_sentinel import MISSING

#: Oldest version of the CF conventions this package knows how to write, and the
#: one selected by default.
#:
#: CF 1.13 is the first version that defines the HEALPix grid mapping (see
#: `CF issue #433 <https://github.com/cf-convention/cf-conventions/pull/605>`_),
#: on which the output datasets rely to describe their grid.
CF_MIN_VERSION: str = "1.13"


def _parse_version(value: str) -> tuple[int, ...]:
    """Parse a dotted version string into a comparable tuple of integers."""
    try:
        return tuple(int(part) for part in value.split("."))
    except ValueError:
        raise ValueError(f"invalid version number {value!r}") from None


class ConventionConfiguration(BaseModel):
    """Base class for the configuration (options) of a convention.

    Conventions that define no option are configured by this class, which
    accepts none.

    """

    model_config = ConfigDict(
        frozen=True, use_attribute_docstrings=True, extra="forbid"
    )


class BaseConventionSelector(BaseModel):
    """Base class for selecting one convention followed by the output dataset.

    Follows Zarr's extension metadata model: a ``name`` and, optionally, a
    ``version`` and a ``configuration``. The latter two are left unset for the
    conventions that define neither (and are then absent from the serialized
    settings).

    """

    model_config = ConfigDict(
        frozen=True, use_attribute_docstrings=True, extra="forbid"
    )

    name: str
    """Name of the convention."""

    version: str | MISSING = MISSING
    """Version of the convention followed by the output dataset."""

    configuration: ConventionConfiguration | MISSING = MISSING
    """Options of the convention."""


class DGGSConventionSelector(BaseConventionSelector):
    """Select the DGGS Zarr convention.

    See https://github.com/zarr-conventions/dggs

    """

    name: Literal["dggs"] = "dggs"


class MultiscalesConventionSelector(BaseConventionSelector):
    """Select the Multiscales Zarr convention.

    See https://github.com/zarr-conventions/multiscales

    """

    name: Literal["multiscales"] = "multiscales"


class CFConfiguration(ConventionConfiguration):
    """Configuration of the CF conventions."""

    healpix_grid_mapping: bool = True
    """If True, emit the CF HEALPix grid mapping (CF 1.13+) in each HEALPix group."""


class CFConventionSelector(BaseConventionSelector):
    """Select the CF conventions.

    See https://cfconventions.org

    """

    name: Literal["cf"] = "cf"

    version: str = CF_MIN_VERSION
    """Version of the CF conventions followed by the output dataset.

    Must be :py:obj:`CF_MIN_VERSION` or newer (the default).
    """

    configuration: CFConfiguration = Field(default_factory=CFConfiguration)

    @field_validator("version")
    @classmethod
    def validate_supported_version(cls, value: str) -> str:
        if _parse_version(value) < _parse_version(CF_MIN_VERSION):
            raise ValueError(
                f"unsupported CF version {value!r} "
                f"(must be {CF_MIN_VERSION} or newer: the HEALPix grid mapping "
                f"is not defined in older versions)"
            )

        return value


def _expand_shorthand(value: Any) -> Any:
    """Accept a convention given as a plain name (e.g., ``"dggs"``) and
    normalize the spelling of convention names (lookup is case-insensitive).
    """
    if isinstance(value, str):
        value = {"name": value}

    if isinstance(value, dict) and isinstance(value.get("name"), str):
        value = value | {"name": value["name"].lower()}

    return value


ConventionSelector: TypeAlias = Annotated[
    DGGSConventionSelector | MultiscalesConventionSelector | CFConventionSelector,
    Field(discriminator="name"),
    BeforeValidator(_expand_shorthand),
]


def _default_zarr_conventions() -> list[Any]:
    return [
        DGGSConventionSelector(),
        MultiscalesConventionSelector(),
        CFConventionSelector(),
    ]


class MetadataSettings(BaseModel):
    """Settings for the metadata written in the output Zarr dataset."""

    model_config = ConfigDict(
        frozen=True, use_attribute_docstrings=True, extra="forbid"
    )

    zarr_conventions: list[ConventionSelector] = Field(
        default_factory=_default_zarr_conventions
    )
    """Conventions followed by the output Zarr dataset.

    Each item is either a convention name (e.g., ``"dggs"``) or a mapping with
    a ``name`` and, optionally, a ``version`` and a ``configuration``
    (see :py:obj:`ConventionSelector`).
    """

    @model_validator(mode="after")
    def validate_unique_conventions(self) -> MetadataSettings:
        names = [selector.name for selector in self.zarr_conventions]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"duplicate convention(s) {sorted(duplicates)}")

        return self

    def get(self, name: str) -> ConventionSelector | None:
        """Return the selected convention with the given name, or None if that
        convention is not selected.
        """
        for selector in self.zarr_conventions:
            if selector.name == name.lower():
                return selector

        return None

    def has(self, name: str) -> bool:
        """Returns True if the convention with the given name is selected."""
        return self.get(name) is not None
