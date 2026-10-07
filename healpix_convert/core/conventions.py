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
            {"name": "CF", "version": "1.13", "configuration": {"healpix_grid_mapping": True}},
        ]
    }

Only established, versioned conventions are supported (see :py:obj:`ConventionSpec`).

Migrating to a newer version of the CF conventions
--------------------------------------------------

Writers never test CF version numbers: they ask whether a *feature* is available
(:py:meth:`CFConventionSpec.supports`). Supporting a new CF version therefore means:

1. add it to :py:obj:`CF_SUPPORTED_VERSIONS` (ordered, oldest first: the last one
   is the default);
2. declare in :py:obj:`CF_VERSION_FEATURES` which of the features used here that
   version provides;
3. only if the *content* of a feature changed in that version, update the model
   that emits it — e.g.
   :py:class:`~healpix_convert.core.healpix_conventions.CFHealpixGridMapping`
   for ``"healpix_grid_mapping"``.

Nothing else has to be touched, and a version that does not provide a requested
feature is rejected at validation time instead of silently writing metadata that
the declared version does not define.

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

#: Canonical spelling of the supported convention names (lookup is case-insensitive).
_CANONICAL_NAMES = {"dggs": "dggs", "multiscales": "multiscales", "cf": "CF"}

#: Versions of the CF conventions this package knows how to write, oldest first.
#: The last one is used by default.
CF_SUPPORTED_VERSIONS: tuple[str, ...] = ("1.13",)

#: CF features used by this package, per CF version.
#:
#: - ``"healpix_grid_mapping"``: the HEALPix grid mapping introduced by
#:   `CF issue #433 <https://github.com/cf-convention/cf-conventions/pull/605>`_.
CF_VERSION_FEATURES: dict[str, frozenset[str]] = {
    "1.13": frozenset({"healpix_grid_mapping"}),
}

#: All CF features known to this package.
CF_FEATURES: frozenset[str] = frozenset().union(*CF_VERSION_FEATURES.values())


class DGGSConventionSpec(BaseModel):
    """Select the DGGS Zarr convention.

    See https://github.com/zarr-conventions/dggs

    """

    model_config = ConfigDict(
        frozen=True, use_attribute_docstrings=True, extra="forbid"
    )

    name: Literal["dggs"] = "dggs"


class MultiscalesConventionSpec(BaseModel):
    """Select the Multiscales Zarr convention.

    See https://github.com/zarr-conventions/multiscales

    """

    model_config = ConfigDict(
        frozen=True, use_attribute_docstrings=True, extra="forbid"
    )

    name: Literal["multiscales"] = "multiscales"


class CFConfiguration(BaseModel):
    """Configuration of the CF conventions."""

    model_config = ConfigDict(
        frozen=True, use_attribute_docstrings=True, extra="forbid"
    )

    healpix_grid_mapping: bool = True
    """If True, emit the CF HEALPix grid mapping (CF 1.13+) in each HEALPix group."""


class CFConventionSpec(BaseModel):
    """Select the CF conventions.

    See https://cfconventions.org

    """

    model_config = ConfigDict(
        frozen=True, use_attribute_docstrings=True, extra="forbid"
    )

    name: Literal["CF"] = "CF"

    version: str = CF_SUPPORTED_VERSIONS[-1]
    """Version of the CF conventions followed by the output dataset.

    Must be one of :py:obj:`CF_SUPPORTED_VERSIONS` (defaults to the most recent one).
    """

    configuration: CFConfiguration = Field(default_factory=CFConfiguration)

    @field_validator("version")
    @classmethod
    def validate_supported_version(cls, value: str) -> str:
        if value not in CF_SUPPORTED_VERSIONS:
            raise ValueError(
                f"unsupported CF version {value!r} "
                f"(supported: {', '.join(CF_SUPPORTED_VERSIONS)})"
            )

        return value

    @model_validator(mode="after")
    def validate_configuration_vs_version(self) -> CFConventionSpec:
        if self.configuration.healpix_grid_mapping and not self.supports(
            "healpix_grid_mapping"
        ):
            versions = [
                version
                for version, features in CF_VERSION_FEATURES.items()
                if "healpix_grid_mapping" in features
            ]
            raise ValueError(
                f"the HEALPix grid mapping is not defined in CF {self.version} "
                f"(introduced in CF {min(versions)})"
            )

        return self

    def supports(self, feature: str) -> bool:
        """Returns True if the selected CF version provides the given feature.

        See :py:obj:`CF_VERSION_FEATURES`.

        """
        if feature not in CF_FEATURES:
            raise ValueError(
                f"unknown CF feature {feature!r} (known: {', '.join(sorted(CF_FEATURES))})"
            )

        return feature in CF_VERSION_FEATURES.get(self.version, frozenset())


def _expand_shorthand(value: Any) -> Any:
    """Accept a convention given as a plain name (e.g., ``"dggs"``) and
    normalize the spelling of known convention names.
    """
    if isinstance(value, str):
        value = {"name": value}

    if isinstance(value, dict) and isinstance(value.get("name"), str):
        name = value["name"]
        canonical = _CANONICAL_NAMES.get(name.lower())
        if canonical is not None and canonical != name:
            value = value | {"name": canonical}

    return value


ConventionSpec: TypeAlias = Annotated[
    DGGSConventionSpec | MultiscalesConventionSpec | CFConventionSpec,
    Field(discriminator="name"),
    BeforeValidator(_expand_shorthand),
]


def _default_zarr_conventions() -> list[Any]:
    return [
        DGGSConventionSpec(),
        MultiscalesConventionSpec(),
        CFConventionSpec(),
    ]


class MetadataSettings(BaseModel):
    """Settings for the metadata written in the output Zarr dataset."""

    model_config = ConfigDict(
        frozen=True, use_attribute_docstrings=True, extra="forbid"
    )

    zarr_conventions: list[ConventionSpec] = Field(
        default_factory=_default_zarr_conventions
    )
    """Conventions followed by the output Zarr dataset.

    Each item is either a convention name or a mapping with a ``name`` and,
    optionally, a ``version`` and a ``configuration`` (following Zarr's
    extension metadata model).
    """

    @model_validator(mode="after")
    def validate_unique_conventions(self) -> MetadataSettings:
        names = [spec.name for spec in self.zarr_conventions]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"duplicate convention(s) {sorted(duplicates)}")

        return self

    def get(self, name: str) -> ConventionSpec | None:
        """Return the selected convention with the given name, or None if that
        convention is not selected.
        """
        canonical = _CANONICAL_NAMES.get(name.lower(), name)
        for spec in self.zarr_conventions:
            if spec.name == canonical:
                return spec

        return None

    def has(self, name: str) -> bool:
        """Returns True if the convention with the given name is selected."""
        return self.get(name) is not None

    @property
    def cf(self) -> CFConventionSpec | None:
        """The selected CF conventions, or None if CF is not selected."""
        spec = self.get("CF")
        assert spec is None or isinstance(spec, CFConventionSpec)
        return spec

    @property
    def conventions_attr(self) -> str | None:
        """Value of the CF ``Conventions`` attribute (e.g., ``"CF-1.13"``),
        or None if the CF conventions are not selected.
        """
        cf = self.cf
        return None if cf is None else f"CF-{cf.version}"

    @property
    def write_cf_grid_mapping(self) -> bool:
        """Returns True if the CF HEALPix grid mapping must be emitted."""
        cf = self.cf
        if cf is None:
            return False

        return cf.configuration.healpix_grid_mapping and cf.supports(
            "healpix_grid_mapping"
        )
