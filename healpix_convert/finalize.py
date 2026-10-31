"""Finalizing the output converted dataset.

The last stage of the conversion (see
:py:func:`~healpix_convert.finalize_healpix_dataset`) does what is left once
every group of the output dataset has been written: it consolidates the Zarr
metadata of the output dataset, as the ``consolidate`` metadata settings ask
(see
:py:class:`~healpix_convert.settings.common.ConsolidateMetadataSettings`).

Consolidating the metadata of a group copies the metadata of all its
descendants into its own metadata document: such a copy is stale as soon as a
node is written next to it, which is why it is written here and not by the
stages that write the groups and the arrays of the output dataset.

"""

from __future__ import annotations

from pathlib import PurePath

import structlog
import zarr.api.synchronous as zarr

from healpix_convert.core.conversion_models import (
    ROOT_GROUP_PATH,
    ConvertStagingCache,
    zarr_group_path,
)
from healpix_convert.settings.common import ConvertSettings

log = structlog.get_logger()


def consolidation_paths(
    cache: ConvertStagingCache, settings: ConvertSettings
) -> list[str]:
    """Return the paths of the output groups whose metadata is consolidated,
    the root group first and without duplicates.

    Parameters
    ----------
    cache : :py:class:`ConvertStagingCache`
        The conversion staging cache, which holds the structure of the output
        dataset.
    settings : :py:class:`ConvertSettings`
        The conversion settings (the ``consolidate`` metadata settings here).

    """
    consolidate = settings.metadata.consolidate
    output_groups = cache.output_groups
    paths: list[PurePath | str] = []

    if consolidate.root_group:
        paths.append(ROOT_GROUP_PATH)

    if consolidate.healpix_groups:
        for input_path in cache.input_spatial_groups:
            paths.append(output_groups.io_path_map[input_path])

    if consolidate.multiscale_groups:
        for input_path in output_groups.multiscale_groups:
            paths.append(output_groups.io_path_map[input_path])

    if consolidate.derived_level_groups:
        for info in output_groups.pyramid_multiscale_groups.values():
            paths.extend(info.derived_levels)

    paths.extend(consolidate.groups)

    # the same group may be named by several settings, e.g. a user-defined path
    return list(dict.fromkeys(zarr_group_path(path) for path in paths))


def consolidate_output_metadata(
    output_store: zarr.StoreLike,
    cache: ConvertStagingCache,
    settings: ConvertSettings,
) -> None:
    """Consolidate the Zarr metadata of the output groups named by the
    ``consolidate`` metadata settings.

    Parameters
    ----------
    output_store : Zarr store
        Store of the output Zarr dataset, fully written.
    cache : :py:class:`ConvertStagingCache`
        The conversion staging cache, which holds the structure of the output
        dataset.
    settings : :py:class:`ConvertSettings`
        The conversion settings.

    Notes
    -----
    A named group that the output dataset does not hold is skipped with a
    warning (creating the staging cache has warned about it already).

    """
    paths = consolidation_paths(cache, settings)

    if not paths:
        log.info("••• no output group to consolidate the Zarr metadata of.")
        return

    log.info(f"••• consolidating the Zarr metadata of {len(paths)} output group(s)...")

    for path in paths:
        name = path or "/"

        try:
            zarr.consolidate_metadata(output_store, path=path)
        except FileNotFoundError:
            log.warning(
                f"••• skip group '{name}', which does not exist in the output dataset"
            )
        else:
            log.info(f"••• consolidated the metadata of group '{name}'")
