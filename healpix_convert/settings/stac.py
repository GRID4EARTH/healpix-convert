"""
Pydantic model classes for the STAC metadata written in the output dataset.

The STAC item of the output is merged from the items of the input datasets and
completed with the provenance of the conversion. The settings here are a basic
control mechanism over the propagated part of that item: they drop and set
properties and extensions by name, and nothing else. In particular the item is
not validated against the extensions it lists, so keeping the properties and
the extensions consistent with each other is left to the data provider.

The STAC metadata written by healpix-convert itself (the properties and the
extension of the STAC processing extension, describing the conversion) keeps
precedence: it is written after these settings are applied. The one exception
is ``include_conversion_settings``, which decides whether the settings
themselves are recorded.

"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from healpix_convert.core.utils import check_patterns, matches_any


class STACSettings(BaseModel):
    """Settings for the STAC metadata (``stac_discovery``) written in the root
    group of the output dataset.
    """

    model_config = ConfigDict(
        frozen=True, use_attribute_docstrings=True, extra="forbid"
    )

    include_conversion_settings: bool = True
    """If True (the default), the conversion settings are recorded in the output
    as the ``processing:expression`` STAC property.

    Set it to False to leave them out, e.g., because they are large or because
    they name internal paths that should not be published. The rest of the
    provenance (``processing:datetime``, ``processing:software``,
    ``processing:lineage``) is written either way.
    """

    exclude_props: list[str] = Field(default_factory=list)
    """Names of the input STAC properties that must not be propagated to the output.

    Each item is either the exact name of a property or a regular expression
    matched against the whole property name (e.g., ``"internal:.*"`` excludes
    every property of an ``internal`` namespace).

    The properties written by healpix-convert itself (``processing:*``) keep
    precedence.
    """

    override_props: dict[str, Any] = Field(default_factory=dict)
    """STAC properties set in the output, overriding the propagated ones.

    An override wins over ``exclude_props``. The properties written by
    healpix-convert itself (``processing:*``) keep precedence.
    """

    exclude_extensions: list[str] = Field(default_factory=list)
    """Input STAC extensions that must not be listed in the output.

    Each item is either the exact URL of an extension schema or a regular
    expression matched against the whole URL (e.g.,
    ``"https://stac-extensions.github.io/projection/.*"`` excludes any version
    of the projection extension).

    The STAC processing extension, listed by healpix-convert itself, keeps
    precedence.
    """

    override_extensions: list[str] = Field(default_factory=list)
    """URLs of the STAC extension schemas listed in the output, in addition to
    the propagated ones.

    An override wins over ``exclude_extensions``.
    """

    @field_validator("exclude_props", "exclude_extensions")
    @classmethod
    def validate_patterns(cls, value: list[str]) -> list[str]:
        check_patterns(value)

        return value

    def is_excluded_prop(self, name: str) -> bool:
        """Returns True if the input STAC property with the given name must not
        be propagated to the output.
        """
        return matches_any(name, self.exclude_props)

    def is_excluded_extension(self, url: str) -> bool:
        """Returns True if the input STAC extension with the given schema URL
        must not be propagated to the output.
        """
        return matches_any(url, self.exclude_extensions)

    def filter_props(self, props: Mapping[str, Any]) -> dict[str, Any]:
        """Return the propagated STAC properties: the given ones without the
        excluded ones, updated with ``override_props``.
        """
        filtered = {k: v for k, v in props.items() if not self.is_excluded_prop(k)}
        filtered.update(self.override_props)

        return filtered

    def filter_extensions(self, extensions: Iterable[str]) -> list[str]:
        """Return the propagated STAC extensions: the given ones without the
        excluded ones, followed by the ``override_extensions`` that are not
        listed yet.
        """
        filtered = [url for url in extensions if not self.is_excluded_extension(url)]
        filtered += [url for url in self.override_extensions if url not in filtered]

        return filtered
