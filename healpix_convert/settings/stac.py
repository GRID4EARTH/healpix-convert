"""
Pydantic model classes for the STAC metadata written in the output dataset.

The STAC item of the output is merged from the items of the input datasets and
completed with the provenance of the conversion. The settings here are a basic
control mechanism over the propagated part of that item: they drop and set
properties, extensions, links and assets by name, and nothing else. In
particular the item is not validated against the extensions it lists, so
keeping the properties, the extensions and the assets consistent with each
other is left to the data provider.

The input's assets are the one thing not propagated by default: they point at
the input files, which the output does not hold.

The STAC metadata written by healpix-convert itself (the properties, the links
and the extension of the STAC processing extension, describing the conversion)
keeps precedence: it is written after these settings are applied. The one
exception is ``include_conversion_settings``, which decides whether the
settings themselves are recorded.

"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator

from healpix_convert.core.utils import check_patterns, matches_any

#: How the listed STAC metadata of several input datasets (extensions, links)
#: is merged into the output's: ``"pick_first"`` keeps what the first input
#: dataset has, ignoring the others; ``"join_inner"`` what every input dataset
#: has (their intersection); ``"join_outer"`` what any input dataset has
#: (their union).
MergeStrategy: TypeAlias = Literal["pick_first", "join_inner", "join_outer"]

#: How the keyed STAC metadata of several input datasets (properties, assets)
#: is merged into the output's. There is no ``"join_outer"``: a union of
#: mappings has to invent a rule for the keys whose value differs across the
#: inputs, and any such rule describes the output less well than leaving the
#: entry out.
MappingMergeStrategy: TypeAlias = Literal["pick_first", "join_inner"]


def _link_rel(link: Any) -> str:
    """Return the ``rel`` of a STAC link, the name it is excluded by.

    STAC links have no identifier: ``rel`` is the one field that says what a
    link is for, so it is what the patterns are matched against. A link
    without one matches the empty string.
    """
    if isinstance(link, Mapping):
        return str(link.get("rel", ""))

    return ""


def _filter_mapping(
    values: Mapping[str, Any],
    exclude: Iterable[str],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """Drop the entries whose key matches ``exclude``, then apply ``override``."""
    filtered = {k: v for k, v in values.items() if not matches_any(k, exclude)}
    filtered.update(override)

    return filtered


def _filter_sequence(
    values: Iterable[Any],
    exclude: Iterable[str],
    override: Iterable[Any],
    key=str,
) -> list[Any]:
    """Drop the items whose ``key`` matches ``exclude``, then append the items
    of ``override`` that are not in the result yet.
    """
    filtered = [v for v in values if not matches_any(key(v), exclude)]
    filtered += [v for v in override if v not in filtered]

    return filtered


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

    merge_properties: MappingMergeStrategy = "pick_first"
    """How the STAC properties of several input datasets are merged.

    ``"join_inner"`` keeps the properties every input has with the same value,
    i.e., what they agree on: a ``datetime`` that differs across the inputs
    does not describe the output either.

    With a single input dataset both strategies are equivalent.
    """

    merge_extensions: MergeStrategy = "pick_first"
    """How the STAC extensions of several input datasets are merged.

    ``"join_outer"`` lists every extension any input declares, which is what
    keeps the output item from dropping a namespace its properties still use.
    """

    merge_links: MergeStrategy = "pick_first"
    """How the STAC links of several input datasets are merged.

    Links have no identifier, so they are merged whole: two links are the same
    one when they are equal. ``"join_inner"`` keeps the links every input has
    and ``"join_outer"`` every link any input has, in the order they first
    appear.
    """

    merge_assets: MappingMergeStrategy = "pick_first"
    """How the STAC assets of several input datasets are merged.

    Merged like the properties, by asset key. The result is then filtered by
    ``exclude_assets``, which drops every input asset by default.
    """

    exclude_properties: list[str] = Field(default_factory=list)
    """Names of the input STAC properties that must not be propagated to the output.

    Each item is either the exact name of a property or a regular expression
    matched against the whole property name (e.g., ``"internal:.*"`` excludes
    every property of an ``internal`` namespace).

    The properties written by healpix-convert itself (``processing:*``) keep
    precedence.
    """

    override_properties: dict[str, Any] = Field(default_factory=dict)
    """STAC properties set in the output, overriding the propagated ones.

    An override wins over ``exclude_properties``. The properties written by
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

    exclude_links: list[str] = Field(default_factory=list)
    """Input STAC links that must not be propagated to the output.

    Links have no identifier, so each item is matched against the whole ``rel``
    of a link: either an exact relation type (e.g., ``"via"``) or a regular
    expression (e.g., ``"self|parent|root"``).

    The ``derived_from`` links written by healpix-convert itself keep
    precedence.
    """

    override_links: list[dict[str, Any]] = Field(default_factory=list)
    """STAC links written in the output, in addition to the propagated ones.

    Each item is a whole link object (``rel``, ``href``, and whatever else the
    link carries). An override wins over ``exclude_links``.
    """

    exclude_assets: list[str] = Field(default=[".*"])
    """Keys of the input STAC assets that must not be propagated to the output.

    Each item is either the exact key of an asset or a regular expression
    matched against the whole key (e.g., ``"thumbnail.*"``).

    Every input asset is excluded by default: the assets of the input point at
    the input files, which the output does not hold, so propagating them would
    have the output item advertise files that are not its own. Set this to an
    empty list to keep them, or describe the output's own files with
    ``override_assets``.
    """

    override_assets: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """STAC assets set in the output, overriding the propagated ones.

    Each key is the key of an asset and each value the whole asset object
    (``href``, ``type``, ``roles``, etc.). An override wins over
    ``exclude_assets``.
    """

    @field_validator(
        "exclude_properties", "exclude_extensions", "exclude_links", "exclude_assets"
    )
    @classmethod
    def validate_patterns(cls, value: list[str]) -> list[str]:
        check_patterns(value)

        return value

    def filter_properties(self, properties: Mapping[str, Any]) -> dict[str, Any]:
        """Return the propagated STAC properties: the given ones without the
        excluded ones, updated with ``override_properties``.
        """
        return _filter_mapping(
            properties, self.exclude_properties, self.override_properties
        )

    def filter_assets(self, assets: Mapping[str, Any]) -> dict[str, Any]:
        """Return the propagated STAC assets: the given ones without the
        excluded ones, updated with ``override_assets``.
        """
        return _filter_mapping(assets, self.exclude_assets, self.override_assets)

    def filter_extensions(self, extensions: Iterable[str]) -> list[str]:
        """Return the propagated STAC extensions: the given ones without the
        excluded ones, followed by the ``override_extensions`` that are not
        listed yet.
        """
        return _filter_sequence(
            extensions, self.exclude_extensions, self.override_extensions
        )

    def filter_links(self, links: Iterable[Any]) -> list[Any]:
        """Return the propagated STAC links: the given ones whose ``rel`` is not
        excluded, followed by the ``override_links`` that are not listed yet.
        """
        return _filter_sequence(
            links, self.exclude_links, self.override_links, key=_link_rel
        )
