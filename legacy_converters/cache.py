"""Conversion staging cache functions."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePath

import shapely
import structlog
import xarray as xr

import legacy_converters.core.utils as utils
from legacy_converters.core.conversion_models import (
    ConvertStagingCache,
    InputGroupSpatialInfo,
    InputSpatialInfo,
    OutputGroupInfo,
    OutputSpatialInfo,
)
from legacy_converters.core.multiscales_conventions import HealpixMultiscales
from legacy_converters.settings.common import (
    ConvertSettings,
    HealpixGroupSettings,
    validate_convert_settings,
)

log = structlog.get_logger()


def _check_and_filter_input_groups(
    datatrees: list[xr.DataTree],
    groups: str | Sequence[str] | None,
) -> list[xr.DataTree]:
    """Return xr.DataTree objects with user group selection.

    - check that all input datasets have the same hierarchical structure
    - filter input datasets with user-defined groups

    """
    dt0 = datatrees[0]

    if any(not dt0.isomorphic(dt) for dt in datatrees[1:]):
        raise ValueError(
            "input Zarr datasets must all have the same hierarchical structure (groups)"
        )

    if groups is not None:
        if isinstance(groups, str):
            dt0 = dt0.match(groups)
        else:
            lgroups = list(groups)
            dt0 = dt0.filter(lambda dt: dt.path in lgroups)

    return [dt.filter_like(dt0) for dt in datatrees]


def _extract_root_spatial_info(datatrees: list[xr.DataTree]) -> InputSpatialInfo:
    """Extract spatial information from input Zarr datasets (root group)."""

    spatial_info = InputSpatialInfo()

    for dt in datatrees:
        spatial_info.crs.append(utils.get_crs_from_stac(dt))
        spatial_info.geometry_latlon.append(utils.get_wgs84_polygon_from_stac(dt))
        spatial_info.geometry_xy.append(utils.get_proj_polygon_from_stac(dt))

    return spatial_info


def _extract_spatial_info(ds: xr.Dataset) -> InputGroupSpatialInfo | None:
    """Extract spatial information in a group of a Zarr input dataset.

    Return None if no spatial information is found.
    """

    info = utils.extract_spatial_info_stac(ds)

    if info is None:
        info = utils.extract_spatial_info_cf(ds)

    if info is not None:
        info = InputGroupSpatialInfo(**info)

    return info


def _find_input_spatial_groups(
    datatrees: list[xr.DataTree],
    settings: ConvertSettings,
) -> dict[PurePath, InputGroupSpatialInfo]:
    """Find spatial groups in input datasets, i.e., groups to convert to HEALPix.

    - extract and return spatial information for each group found
    - check that spatial information is consistent across all input datasets
    - check that HEALPix conversion settings are given for each group found
    """
    spatial_groups = {}

    for path, dts in xr.group_subtrees(*datatrees):
        no_data = [not dt.has_data for dt in dts]
        if PurePath(path) in settings.exclude_groups:
            continue
        if all(no_data):
            # a spatial group must have data
            continue
        elif any(no_data):
            raise ValueError(
                f"inconsistent group {path!r} across input datasets "
                "(non-compatible data vs. no data)"
            )

        spatial_info = _extract_spatial_info(dts[0].dataset)

        for dt in dts[1:]:
            info = _extract_spatial_info(dt.dataset)
            if spatial_info is None and info is None:
                continue
            elif spatial_info is not None and info is not None:
                try:
                    spatial_info += info
                except ValueError:
                    raise ValueError(
                        f"inconsistent group {path!r} across input datasets "
                        "(non-compatible spatial properties)"
                    )
            else:
                raise ValueError(
                    f"inconsistent group {path!r} across input datasets "
                    "(non-compatible spatial vs. non-spatial)"
                )

        if spatial_info is not None:
            path = PurePath(path)
            if not isinstance(settings.group_settings.get(path), HealpixGroupSettings):
                raise KeyError(
                    f"HEALPix conversion settings are missing for group {path!r}"
                )
            spatial_groups[PurePath(path)] = spatial_info

    return spatial_groups


def _set_output_groups(
    datatrees: list[xr.DataTree],
    input_spatial_groups: dict[PurePath, InputGroupSpatialInfo],
    settings: ConvertSettings,
) -> OutputGroupInfo:
    multiscale_groups: dict[PurePath, HealpixMultiscales] = {}
    excluded_groups: list[PurePath] = []
    io_path_map: dict[PurePath, PurePath] = {}

    multiscales_settings = settings.multiscale_settings

    for path, dts in xr.group_subtrees(*datatrees):
        path = PurePath(path)

        if path in settings.exclude_groups:
            excluded_groups.append(path)
            continue

        if path not in io_path_map:
            io_path_map[path] = path

        # datatrees isomorphism has been checked before
        dt0 = dts[0]

        # configure multiscales
        if path in multiscales_settings.groups:
            layouts: list[dict] = []

            for child_dt in dt0.children.values():
                child_path = PurePath(child_dt.path.lstrip("/"))
                child_settings = settings.group_settings.get(child_path)

                if isinstance(child_settings, HealpixGroupSettings):
                    group_output_path = child_path.parent / str(
                        child_settings.healpix.refinement_level
                    )
                    io_path_map[child_path] = group_output_path
                    layout = {"asset": group_output_path}
                    if multiscales_settings.add_healpix_positioning:
                        layout["dggs"] = child_settings.healpix
                    layouts.append(layout)

            if layouts:
                multiscale_groups[path] = HealpixMultiscales.model_validate(
                    {"layout": layouts}
                )
            else:
                raise ValueError(
                    f"multiscale group {path} has no child group to convert to HEALPix"
                )

    return OutputGroupInfo(
        multiscale_groups=multiscale_groups,
        excluded_groups=excluded_groups,
        io_path_map=io_path_map,
    )


def _set_output_spatial_info(
    input_spatial: InputSpatialInfo,
    output_extent: dict | shapely.Polygon | None,
) -> OutputSpatialInfo:

    if output_extent is None:
        geometry = utils.compute_output_extent_latlon(*input_spatial.get_extents())
    elif isinstance(output_extent, shapely.Polygon):
        geometry = output_extent
    else:
        geometry = shapely.geometry.shape(output_extent)
        if not isinstance(output_extent, shapely.Polygon):
            raise ValueError("output extent must represent a single polygon.")

    return OutputSpatialInfo(geometry_latlon=geometry)


def create_staging_cache(
    input_paths: Sequence[str],
    settings: dict | ConvertSettings,
    *,
    groups: str | Sequence[str] | None = None,
    output_extent: dict | shapely.Polygon | None = None,
) -> ConvertStagingCache:
    """Create a temporary (staging) conversion data cache by parsing input dataset(s)
    and conversion settings.

    Useful for testing and debugging or for multi-stage conversion workflows.

    Parameters
    ----------
    input_paths : list of paths
        Path(s) to the input Zarr dataset(s) to convert to HEALPix.
    settings : dict or object
        Healpix conversion settings.
    groups : str or list, optional
        Groups (paths) in the input datasets to process. It can be
        either a string (single group or unix-like pattern) or a
        list of groups. By default, all groups found in input datasets
        are processed.
    output_extent : dict or :py:class:`shapely.Polygon`, optional
        Spatial extent of the output HEALPix dataset, given as a
        (GEOJSON-like or object) polygon with lat-lon coordinates.
        If not given (default), the spatial extent will be the union
        of the spatial extents of the input datasets (single polygon).

    Returns
    -------
    :py:class:`ConvertStagingCache`
        A reusable data cache that contains information extracted from
        the input dataset(s) and/or pre-computed for the output dataset.

    """
    cache = ConvertStagingCache()
    settings = validate_convert_settings(settings)

    # --- open and filter input Zarr datasets
    log.info("••• reading Zarr datasets...")

    input_datatrees = _check_and_filter_input_groups(
        utils.open_datatrees(list(input_paths)), groups
    )
    cache.input_datatrees = {path: dt for path, dt in zip(input_paths, input_datatrees)}

    # --- extract (root) spatial info in input Zarr datasets
    log.info("••• extracting spatial info from input datasets (root group)...")

    cache.input_spatial = _extract_root_spatial_info(input_datatrees)

    unique_crss = set(cache.input_spatial.crs)
    log.info(
        f"found {len(unique_crss)} CRS(s) in input datasets:\n"
        + "\n".join(f"  - {crs.name}" for crs in unique_crss)
    )

    # --- compute or set output spatial extent
    cache.output_spatial = _set_output_spatial_info(cache.input_spatial, output_extent)

    log.info(
        "spatial extent of the output HEALPix dataset (WGS84):\n"
        f"  - geometry: {cache.output_spatial.geometry_latlon}\n"
        f"  - bbox: {cache.output_spatial.bbox}"
    )

    # --- detect input spatial groups (to convert to HEALPix)
    log.info(
        "••• detecting spatial groups in input datasets (to convert to HEALPix)..."
    )

    cache.input_spatial_groups = _find_input_spatial_groups(input_datatrees, settings)

    log.info(
        f"found {len(cache.input_spatial_groups)} spatial groups to convert to HEALPix:\n"
        + "\n".join(f"  - {path}" for path in cache.input_spatial_groups)
    )

    # --- configure HEALPix output groups
    log.info("••• configuring groups in the output HEALPix dataset...")

    cache.output_groups = _set_output_groups(
        input_datatrees,
        cache.input_spatial_groups,
        settings,
    )

    log.info(
        f"configured {len(cache.output_groups.multiscale_groups)} multiscale groups:\n"
        + "\n".join(f"  - {path}" for path in cache.output_groups.multiscale_groups)
    )
    log.info(
        f"skip {len(cache.output_groups.excluded_groups)} group(s) in output dataset:\n"
        + "\n".join(f"  - {path}" for path in cache.output_groups.excluded_groups)
    )
    log.info(
        "output HEALPix dataset groups:\n"
        + "\n".join(f"  - {path}" for path in cache.output_groups.io_path_map.values())
    )

    log.info("••• finished creating conversion staging cache.")

    return cache
