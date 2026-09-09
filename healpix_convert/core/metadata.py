"""
Writing output (meta)data according to the selected conventions.

This is the single place where :py:class:`MetadataSettings` is turned into actual
Zarr attributes, so that all conversion paths (the shared HEALPix converters as
well as the ERA5 / CAMS / ClimateDT converters, which build their Zarr groups
themselves) emit consistent metadata.

"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from healpix_convert.core.conventions import MetadataSettings
from healpix_convert.core.healpix_conventions import (
    CFHealpixGridMapping,
    DGGSZarrConvention,
    Healpix,
    write_cf_grid_mapping,
)
from healpix_convert.core.multiscales_conventions import (
    HealpixMultiscales,
    MultiscalesZarrConvention,
)

if TYPE_CHECKING:
    import zarr

#: Name of the CF grid mapping variable written in each HEALPix group.
CF_GRID_MAPPING_VARIABLE = "crs"

_ZARR_CONVENTION_MODELS = {
    "dggs": DGGSZarrConvention,
    "multiscales": MultiscalesZarrConvention,
    # NOTE: the CF conventions are not (yet) listed in the "zarr_conventions"
    # attribute: unlike the conventions above, https://github.com/zarr-conventions/CF
    # has no tagged release from which a uuid and schema url can be referenced.
    # CF compliance is declared via the CF "Conventions" attribute instead
    # (see `write_root_conventions`).
}


def group_conventions_attrs(
    metadata: MetadataSettings,
    *,
    declare: Iterable[str] = ("dggs", "multiscales"),
    dggs: Healpix | None = None,
    multiscales: HealpixMultiscales | None = None,
) -> dict[str, Any]:
    """Build the convention attributes of a single output Zarr group.

    Parameters
    ----------
    metadata : :py:class:`MetadataSettings`
        Metadata settings, i.e., the conventions selected for the output dataset.
    declare : iterable of str, optional
        Names of the conventions that apply to this group (or its subtree) and
        that must be listed in its ``zarr_conventions`` attribute, in that order.
        Conventions that are not selected in ``metadata`` are skipped.
    dggs : :py:class:`Healpix`, optional
        If given (and if the DGGS convention is selected), the HEALPix grid
        description written as the group's ``dggs`` attribute.
    multiscales : :py:class:`HealpixMultiscales`, optional
        If given (and if the Multiscales convention is selected), the multiscale
        description written as the group's ``multiscales`` attribute.

    Returns
    -------
    dict
        Group attributes (may be empty if no convention is selected).

    """
    attrs: dict[str, Any] = {}

    conventions = [
        _ZARR_CONVENTION_MODELS[name]().model_dump()
        for name in declare
        if name in _ZARR_CONVENTION_MODELS and metadata.has(name)
    ]
    if conventions:
        attrs["zarr_conventions"] = conventions

    if dggs is not None and metadata.has("dggs"):
        attrs["dggs"] = dggs.model_dump()

    if multiscales is not None and metadata.has("multiscales"):
        attrs["multiscales"] = multiscales.model_dump()

    return attrs


def write_group_conventions(
    group: zarr.Group,
    metadata: MetadataSettings,
    *,
    declare: Iterable[str] = ("dggs", "multiscales"),
    dggs: Healpix | None = None,
    multiscales: HealpixMultiscales | None = None,
) -> None:
    """Write the convention attributes of an existing output Zarr group.

    See :py:func:`group_conventions_attrs`.

    """
    group.attrs.update(
        group_conventions_attrs(
            metadata, declare=declare, dggs=dggs, multiscales=multiscales
        )
    )


def write_root_conventions(group: zarr.Group, metadata: MetadataSettings) -> None:
    """Write the CF ``Conventions`` attribute in the root group of the output dataset.

    Does nothing if the CF conventions are not selected. The attribute is written
    once, in the root group only: the conventions apply to the whole dataset and
    duplicating them per group would let them get stale.

    """
    conventions_attr = metadata.conventions_attr
    if conventions_attr is not None:
        group.attrs["Conventions"] = conventions_attr


def cf_cell_id_attrs(metadata: MetadataSettings) -> dict[str, Any]:
    """CF attributes of the HEALPix cell ids coordinate (empty if CF is not selected)."""
    if not metadata.write_cf_grid_mapping:
        return {}

    return {"standard_name": "healpix_index", "units": "1"}


def cf_grid_mapping_attrs(
    metadata: MetadataSettings, healpix: Healpix
) -> dict[str, Any] | None:
    """CF HEALPix grid mapping attributes, or None if no grid mapping variable
    must be written.
    """
    if not metadata.write_cf_grid_mapping:
        return None

    return CFHealpixGridMapping.from_healpix(healpix).model_dump()


def cf_data_variable_attrs(
    metadata: MetadataSettings, healpix: Healpix | None = None
) -> dict[str, Any]:
    """CF attributes of a resampled data variable (empty if CF is not selected).

    If ``healpix`` is given and describes a cell ids coordinate, it is named in
    the CF ``coordinates`` attribute: without it, readers have no way to tell
    that the cell ids array is an auxiliary coordinate rather than data.

    """
    if not metadata.write_cf_grid_mapping:
        return {}

    attrs = {"grid_mapping": CF_GRID_MAPPING_VARIABLE}

    if healpix is not None and isinstance(healpix.coordinate, str):
        attrs["coordinates"] = healpix.coordinate

    return attrs


def maybe_write_cf_grid_mapping(
    group: zarr.Group,
    metadata: MetadataSettings,
    healpix: Healpix,
    data_variables: Iterable[str],
    *,
    cell_id_coord: str = "cell_ids",
) -> None:
    """Emit the CF HEALPix grid mapping in an existing HEALPix group, if the CF
    conventions are selected.

    Thin wrapper around
    :py:func:`healpix_convert.core.healpix_conventions.write_cf_grid_mapping`
    for the converters that build their Zarr groups themselves.

    """
    if not metadata.write_cf_grid_mapping:
        return

    write_cf_grid_mapping(group, healpix, data_variables, cell_id_coord=cell_id_coord)
