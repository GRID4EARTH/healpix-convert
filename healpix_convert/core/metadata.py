"""
Writing output (meta)data according to the selected conventions.

This is the single place where :py:class:`MetadataSettings` is turned into actual
Zarr attributes, so that all conversion paths (the shared HEALPix converters as
well as the ERA5 / CAMS / ClimateDT converters, which build their Zarr groups
themselves) emit consistent metadata.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from healpix_convert.core.healpix_conventions import (
    CFHealpixGridMapping,
    DGGSZarrConvention,
    Healpix,
)
from healpix_convert.core.multiscales_conventions import (
    HealpixMultiscales,
    MultiscalesZarrConvention,
)
from healpix_convert.settings.conventions import (
    CFConventionSelector,
    MetadataSettings,
)

if TYPE_CHECKING:
    import zarr

#: Name of the CF grid mapping variable written in each HEALPix group.
CF_GRID_MAPPING_VARIABLE = "crs"


def _get_cf_selector(settings: MetadataSettings) -> CFConventionSelector | None:
    """The selected CF conventions, or None if CF is not selected."""
    selector = settings.get("cf")
    assert selector is None or isinstance(selector, CFConventionSelector)
    return selector


def _get_cf_conventions_attr(settings: MetadataSettings) -> str | None:
    """Value of the CF ``Conventions`` attribute (e.g., ``"CF-1.13"``), or None
    if the CF conventions are not selected.
    """
    cf = _get_cf_selector(settings)
    return None if cf is None else f"CF-{cf.version}"


def _cf_grid_mapping_selected(settings: MetadataSettings) -> bool:
    """Returns True if the CF HEALPix grid mapping must be emitted."""
    cf = _get_cf_selector(settings)
    return cf is not None and cf.configuration.healpix_grid_mapping


def get_multiscale_group_attrs(
    multiscales: HealpixMultiscales, settings: MetadataSettings
) -> dict[str, Any]:
    """Build the convention attributes of an output multiscale Zarr group.

    The DGGS convention is declared here as well: it applies to the single-scale
    (HEALPix) children groups of a multiscale group.

    Conventions that are not selected in ``settings`` are skipped, so the
    returned attributes may be empty.

    NOTE: the CF conventions are not (yet) listed in the ``zarr_conventions``
    attribute: unlike the conventions above, https://github.com/zarr-conventions/CF
    has no tagged release from which a uuid and schema url can be referenced.
    CF compliance is declared via the CF ``Conventions`` attribute instead
    (see :py:func:`write_root_conventions`).

    """
    attrs: dict[str, Any] = {}

    conventions = []
    if settings.has("multiscales"):
        conventions.append(MultiscalesZarrConvention().model_dump())
    if settings.has("dggs"):
        conventions.append(DGGSZarrConvention().model_dump())

    if conventions:
        attrs["zarr_conventions"] = conventions

    if settings.has("multiscales"):
        attrs["multiscales"] = multiscales.model_dump()

    return attrs


def get_healpix_group_attrs(
    healpix: Healpix, settings: MetadataSettings
) -> dict[str, Any]:
    """Build the convention attributes of an output (single-scale) HEALPix Zarr group.

    Returns empty attributes if the DGGS convention is not selected in ``settings``.

    """
    if not settings.has("dggs"):
        return {}

    return {
        "zarr_conventions": [DGGSZarrConvention().model_dump()],
        "dggs": healpix.model_dump(),
    }


def write_root_conventions(group: zarr.Group, settings: MetadataSettings) -> None:
    """Write the CF ``Conventions`` attribute in the root group of the output dataset.

    Does nothing if the CF conventions are not selected. The attribute is written
    once, in the root group only: the conventions apply to the whole dataset and
    duplicating them per group would let them get stale.

    """
    conventions_attr = _get_cf_conventions_attr(settings)
    if conventions_attr is not None:
        group.attrs["Conventions"] = conventions_attr


def get_cf_cell_id_attrs(settings: MetadataSettings) -> dict[str, Any]:
    """CF attributes of the HEALPix cell ids coordinate (empty if CF is not selected)."""
    if not _cf_grid_mapping_selected(settings):
        return {}

    return {"standard_name": "healpix_index", "units": "1"}


def get_cf_grid_mapping_attrs(
    healpix: Healpix, settings: MetadataSettings
) -> dict[str, Any] | None:
    """CF HEALPix grid mapping attributes, or None if no grid mapping variable
    must be written.
    """
    if not _cf_grid_mapping_selected(settings):
        return None

    return CFHealpixGridMapping.from_healpix(healpix).model_dump()


def get_cf_data_variable_attrs(
    healpix: Healpix, settings: MetadataSettings
) -> dict[str, Any]:
    """CF attributes of a resampled data variable (empty if CF is not selected).

    If ``healpix`` describes a cell ids coordinate, it is named in the CF
    ``coordinates`` attribute: without it, readers have no way to tell that the
    cell ids array is an auxiliary coordinate rather than data.

    """
    if not _cf_grid_mapping_selected(settings):
        return {}

    attrs = {"grid_mapping": CF_GRID_MAPPING_VARIABLE}

    if isinstance(healpix.coordinate, str):
        attrs["coordinates"] = healpix.coordinate

    return attrs
