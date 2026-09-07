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


#: CF ``cell_methods`` for each resampling method, describing what a cell value
#: represents.
#:
#: Interpolating resamplers estimate the value at the cell centre, which CF
#: describes as ``point``: "if the data value pertains to the gridpoint alone,
#: rather than to an interval, area or n-dimensional volume of non-zero size".
#:
#: PSF resampling instead weights the input by the instrument response over an
#: area, so it is a mean rather than a point — but a *weighted* one, which CF
#: has no standard method for. The parenthesis qualifies it without overclaiming:
#: tools parsing only the method still read an areal quantity, while a reader is
#: not told it is a plain average. CF allows this form, and drops the ``comment:``
#: keyword when there is no standardized information to precede it (CF 7.3.2,
#: whose own example is ``lat: mean (area-weighted)``).
#:
#: See CF conventions section 7.3 ("Cell Methods") and Appendix E.
CELL_METHODS = {
    "nearest": "area: point",
    "k-nearest": "area: point",
    "bilinear": "area: point",
    "cell-point": "area: point",
    "psf": "area: mean (point spread function weighted)",
}


def cf_cell_methods(metadata: MetadataSettings, resampler_name: str) -> str | None:
    """CF ``cell_methods`` for a resampled variable, or None if CF is not selected.

    Raises for an unknown resampling method rather than silently omitting the
    attribute: a new resampler must state what its values represent.

    """
    if metadata.cf is None:
        return None

    if resampler_name not in CELL_METHODS:
        raise ValueError(
            f"no CF cell_methods defined for resampling method {resampler_name!r} "
            f"(known: {', '.join(sorted(CELL_METHODS))})"
        )

    return CELL_METHODS[resampler_name]


def cf_data_variable_attrs(
    metadata: MetadataSettings,
    healpix: Healpix | None = None,
    resampler_name: str | None = None,
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

    if resampler_name is not None:
        cell_methods = cf_cell_methods(metadata, resampler_name)
        if cell_methods is not None:
            attrs["cell_methods"] = cell_methods

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
