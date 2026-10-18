"""Writing the pyramids of coarser HEALPix refinement levels.

The last stage of the conversion aggregates the levels of the multiscale groups
of the output dataset onto coarser HEALPix refinement levels, each written in a
group of the multiscale group (the converted levels are never rewritten).
Viewers like Gridlook then pick the level from the zoom.

The structure of the pyramids — which levels are generated, in which groups,
with which HEALPix grid and chunks, and the multiscales metadata describing
them — is computed once when the conversion staging cache is created (see
:py:class:`~healpix_convert.core.conversion_models.PyramidMultiscaleGroupInfo`).
What is left here is writing it: :py:func:`create_pyramid_groups` writes the
(empty) groups and their metadata when the output dataset is prepared, like the
rest of its structure, and :py:func:`write_pyramid_levels` aggregates the
converted data onto them, which creates and fills their arrays.

Only groups converted with the ``healpix_cell_dense`` chunk method are
supported (the conversion settings enforce it): the cells of such a chunk are
all the children of one HEALPix cell at a coarser ("chunk") level, in ascending
(nested) order, so the cells of a coarser cell are consecutive cells of the
converted arrays. Aggregating them is then a matter of position (see
:py:func:`~healpix_convert.core.coarsen.coarsen_healpix_group`): nothing has to
be grouped and no cell id is ever read.

"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePath
from typing import Any

import dask
import structlog
import xarray as xr
import zarr.api.synchronous as zarr

import healpix_convert.core.utils as utils
from healpix_convert.core.coarsen import coarsen_healpix_group, encode_like
from healpix_convert.core.conversion_models import (
    ConvertStagingCache,
    HealpixLevelInfo,
    PyramidMultiscaleGroupInfo,
)
from healpix_convert.core.healpix_conventions import CFHealpixGridMapping, Healpix
from healpix_convert.core.metadata import (
    CF_GRID_MAPPING_VARIABLE,
    get_healpix_group_attrs,
)
from healpix_convert.settings.common import ConvertSettings, PyramidSettings

log = structlog.get_logger()


def create_pyramid_groups(
    output_store: zarr.StoreLike,
    cache: ConvertStagingCache,
    settings: ConvertSettings,
) -> None:
    """Create the (empty) Zarr groups of the generated pyramid levels.

    The arrays of those groups are created when the converted data is
    aggregated onto them (see :py:func:`write_pyramid_levels`). The multiscales
    metadata describing a pyramid is written with the multiscale group that
    holds it, like the metadata of any other output group.

    Parameters
    ----------
    output_store : Zarr store
        Store of the output Zarr dataset.
    cache : :py:class:`ConvertStagingCache`
        The conversion staging cache, which holds the structure of the
        pyramids.
    settings : :py:class:`ConvertSettings`
        The conversion settings (the conventions written here).

    """
    for path, info in cache.output_groups.pyramid_multiscale_groups.items():
        for level_path, (healpix, _) in info.pyramid_levels.items():
            # the group named by the pyramid settings, if any, is created along
            # the way
            group = zarr.open_group(output_store, path=level_path.as_posix(), mode="a")
            group.attrs.update(get_healpix_group_attrs(healpix, settings.metadata))

        log.info(
            f"••• created the pyramid groups of '{path.as_posix()}': "
            f"levels {_levels(info.pyramid_levels)}"
        )


def write_pyramid_levels(
    output_store: zarr.StoreLike,
    cache: ConvertStagingCache,
    settings: ConvertSettings,
    *,
    output_storage_options: dict[str, Any] | None = None,
) -> None:
    """Aggregate the converted data onto the generated levels of the pyramids
    and write it in their groups, creating their arrays.

    The groups themselves and the multiscales metadata of the groups holding
    them are written by :py:func:`create_pyramid_groups`. The converted levels
    are not rewritten.

    Parameters
    ----------
    output_store : Zarr store
        Store of the output Zarr dataset, with the converted groups written.
    cache : :py:class:`ConvertStagingCache`
        The conversion staging cache, which holds the structure of the
        pyramids.
    settings : :py:class:`ConvertSettings`
        The conversion settings.
    output_storage_options : dict, optional
        Storage options passed to Zarr.

    """
    # the groups of the output dataset are read from the store itself: its
    # consolidated metadata, if any, is not up to date while it is written
    dt = utils.open_datatrees(
        [output_store],
        chunks={},
        storage_options=output_storage_options,
        consolidated=False,
    )[0]

    writes = []

    for path, info in cache.output_groups.pyramid_multiscale_groups.items():
        pyramid = settings.multiscale_settings.pyramid_groups[path]
        log.info(
            f"••• aggregating the pyramid levels of '{path.as_posix()}' "
            f"({pyramid.reduce} of the converted levels "
            f"{_levels(info.native_levels)}): levels {_levels(info.pyramid_levels)}"
        )

        for level_path, ds in _build_pyramid(dt, info, pyramid, settings).items():
            writes.append(
                ds.to_zarr(
                    output_store,
                    group=level_path.as_posix(),
                    mode="a",
                    zarr_format=3,
                    # the Zarr metadata of the output dataset is consolidated
                    # once, by the conversion stage that calls this
                    consolidated=False,
                    compute=False,
                )
            )

    # all the levels are written in one Dask computation: a converted level is
    # read once and feeds every level aggregated from it
    dask.compute(*writes)


def _levels(levels: Mapping[PurePath, HealpixLevelInfo]) -> list[int]:
    """Refinement levels of the given level groups, finest first."""
    return sorted(
        (healpix.refinement_level for healpix, _ in levels.values()),  # type: ignore[misc]
        reverse=True,
    )


def _native_datasets(
    dt: xr.DataTree, info: PyramidMultiscaleGroupInfo
) -> dict[int, xr.Dataset]:
    """The converted groups of a pyramid, with lazy (Dask) arrays
    (refinement level -> dataset).
    """
    datasets = {}

    for level_path, (healpix, _) in info.native_levels.items():
        assert healpix.refinement_level is not None
        datasets[healpix.refinement_level] = dt[level_path.as_posix()].to_dataset(
            inherit=False
        )

    return datasets


def _variable_sources(
    datasets: Mapping[int, xr.Dataset], level: int, *, coarsen_from: str
) -> dict[int, list[str]]:
    """Converted level each data variable of a generated level is aggregated
    from (converted level -> variable names).

    A variable is absent from the levels finer than its finest converted level
    (it is never upsampled). A variable converted at several levels, e.g., the
    copies of the finest bands that the provider of a multi-resolution product
    resampled itself, is aggregated from the nearest converted level at or
    finer than ``level`` (``"nearest"``), or always from its finest converted
    level (``"finest"``).

    """
    finer_levels = sorted(lvl for lvl in datasets if lvl >= level)
    data_vars = {lvl: set(datasets[lvl].data_vars) for lvl in finer_levels}
    names = {name for lvl in finer_levels for name in data_vars[lvl]}

    sources: dict[int, list[str]] = {}

    for name in sorted(names):
        converted = [lvl for lvl in finer_levels if name in data_vars[lvl]]
        source = min(converted) if coarsen_from == "nearest" else max(converted)
        sources.setdefault(source, []).append(name)

    return sources


def _build_pyramid(
    dt: xr.DataTree,
    info: PyramidMultiscaleGroupInfo,
    pyramid: PyramidSettings,
    settings: ConvertSettings,
) -> dict[PurePath, xr.Dataset]:
    """Build the datasets of the generated levels of one pyramid
    (group path -> dataset).
    """
    datasets = _native_datasets(dt, info)
    native_healpix = {
        healpix.refinement_level: healpix for healpix, _ in info.native_levels.values()
    }

    # the data type and the fill value of a variable are the ones it has at its
    # finest converted level
    templates: dict[str, xr.Variable] = {}
    for level in sorted(datasets, reverse=True):
        for name, var in datasets[level].variables.items():
            templates.setdefault(name, var)

    generated = {}

    for level_path, (healpix, chunk_level) in info.pyramid_levels.items():
        assert healpix.refinement_level is not None

        ds = _build_level(
            datasets,
            healpix,
            native_healpix=native_healpix,
            pyramid=pyramid,
            templates=templates,
            # a chunk holds the children of one HEALPix cell of the chunk level
            chunk_size=4 ** (healpix.refinement_level - chunk_level),
        )
        ds = _set_cf_grid_mapping(ds, healpix)
        ds.attrs = get_healpix_group_attrs(healpix, settings.metadata)
        generated[level_path] = ds

    return generated


def _build_level(
    datasets: Mapping[int, xr.Dataset],
    healpix: Healpix,
    *,
    native_healpix: Mapping[int, Healpix],
    pyramid: PyramidSettings,
    templates: Mapping[str, xr.Variable],
    chunk_size: int,
) -> xr.Dataset:
    """Build the dataset of one generated level of a pyramid, from the
    converted levels of its multiscale group.
    """
    level = healpix.refinement_level
    assert level is not None

    sources = _variable_sources(datasets, level, coarsen_from=pyramid.coarsen_from)
    parts = []

    for source, names in sources.items():
        parts.append(
            coarsen_healpix_group(
                datasets[source][names],
                level,
                healpix=native_healpix[source],
                reduce=pyramid.reduce,
            )
        )

    if len(parts) == 1:
        ds = parts[0]
    else:
        # the converted levels of a group share their chunk cells (same spatial
        # extent, same chunk level), so the cells of the level are the same
        # whichever level its variables are aggregated from: the parts need no
        # alignment, and the cell ids are not read to merge them
        ds = xr.merge(
            parts, join="override", compat="override", combine_attrs="override"
        )

    ds = encode_like(ds, templates)

    return ds.chunk({healpix.spatial_dimension: chunk_size})


def _set_cf_grid_mapping(ds: xr.Dataset, healpix: Healpix) -> xr.Dataset:
    """Describe the HEALPix grid of a generated level in the CF grid mapping
    variable of the converted group, if it has one.

    The grid mapping is propagated rather than selected in the metadata
    settings: the data variables of the converted levels point at it, and so do
    the aggregated ones.

    """
    if CF_GRID_MAPPING_VARIABLE not in ds:
        return ds

    ds[CF_GRID_MAPPING_VARIABLE].attrs = CFHealpixGridMapping.from_healpix(
        healpix
    ).model_dump()

    return ds
