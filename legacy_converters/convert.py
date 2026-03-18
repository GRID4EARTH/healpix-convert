from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import cast

import affine
import distributed
import healpix_geo
import healpix_resample
import numpy as np
import pydantic
import pyproj
import shapely
import structlog
import xarray as xr
import zarr.api.synchronous as zarr

import legacy_converters.core.utils as utils
from legacy_converters.core.healpix_conventions import DGGSZarrConvention, Healpix
from legacy_converters.core.multiscales_conventions import (
    HealpixMultiscales,
    MultiscalesZarrConvention,
)
from legacy_converters.settings.common import (
    ConvertSettings,
    HealpixGroupSettings,
    MultiscalesGroupSettings,
)

log = structlog.get_logger()


@dataclass
class InputSpatialInfo:
    """Spatial information extracted from input Zarr datasets (root group),"""

    crs: list[pyproj.CRS] = field(default_factory=list)
    """CRS, one per input dataset."""

    geometry_latlon: list[shapely.Polygon] = field(default_factory=list)
    """Spatial extent (lat-lon coordinates), one per input dataset."""

    geometry_xy: list[shapely.Polygon | None] = field(default_factory=list)
    """Spatial extent (x-y projected), one per input dataset."""

    def get_extents(self) -> tuple[list[shapely.Polygon], list[pyproj.CRS]]:
        """Return spatial extents (geometry + crs)."""
        geoms = []
        for crs, geom_latlon, geom_xy in zip(
            self.crs, self.geometry_latlon, self.geometry_xy
        ):
            if crs.is_projected:
                if geom_xy is not None:
                    geoms.append(geom_xy)
                else:
                    raise ValueError("x-y geometry is missing for projected input CRS")
            else:
                geoms.append(geom_latlon)

        return geoms, self.crs


@dataclass
class InputGroupSpatialInfo:
    """Spatial information extracted from an input Zarr group.

    All spatial arrays and/or metadata will be removed in the
    output group converted to HEALPix.
    """

    crs: list[pyproj.CRS]
    """CRS of the input group, one per input Zarr dataset."""

    transform: list[affine.Affine] | None
    """Affine transforms (if found, if projected CRS), one per input Zarr dataset"""

    spatial_dimensions: dict[str, int]
    """Spatial dimension names and their size."""

    spatial_coordinates: list[str]
    """Spatial coordinate names."""

    spatial_attrs: list[str]
    """Spatial (group) attribute names."""

    spatial_arrays: list[str]
    """Spatial arrays (data variables) in group."""

    spatial_var_attrs: dict[str, list[str]]
    """Spatial (data-)variable attribute names."""

    def __add__(self, other: InputGroupSpatialInfo) -> InputGroupSpatialInfo:
        # validate and add another input dataset to the group spatial info
        invalid = (
            self.spatial_dimensions != other.spatial_dimensions
            or self.spatial_coordinates != other.spatial_coordinates
            or self.spatial_arrays != other.spatial_arrays
            or self.is_projected is not other.is_projected
        )

        if self.transform is None and other.transform is None:
            transform = None
        elif self.transform is not None and other.transform is not None:
            transform = self.transform + other.transform
        else:
            invalid = True

        if invalid:
            raise ValueError("incompatible spatial info")

        return InputGroupSpatialInfo(
            crs=self.crs + other.crs,
            transform=transform,
            spatial_dimensions=self.spatial_dimensions,
            spatial_coordinates=self.spatial_coordinates,
            spatial_attrs=self.spatial_attrs,
            spatial_arrays=self.spatial_arrays,
            spatial_var_attrs=self.spatial_var_attrs,
        )

    @property
    def is_projected(self) -> bool:
        return all(crs.is_projected for crs in self.crs)


@dataclass
class OutputSpatialInfo:
    """Spatial information of the output HEALPix Zarr dataset."""

    geometry_latlon: shapely.Polygon = field(default_factory=shapely.Polygon)
    """Spatial extent (lat-lon coordinates)."""

    @property
    def bbox(self) -> list:
        return shapely.bounds(self.geometry_latlon)


@dataclass
class OutputChunkInfo:
    """Chunk (meta)data of the output HEALPix Zarr dataset."""

    healpix: Healpix = field(default_factory=lambda: Healpix(refinement_level=10))

    cell_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    """HEALPix cell ids (nested) of chunks in the output Zarr dataset."""

    is_full: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.bool))
    """Boolean mask for output chunks.

    (if True, a chunk has all its elements filled by data values).
    """


@dataclass
class OutputGroupInfo:
    """Information about groups in the output Zarr dataset."""

    root_path: PurePath = field(default_factory=PurePath)
    """Path to the output Zarr dataset (root group)."""

    multiscale_groups: dict[PurePath, HealpixMultiscales] = field(default_factory=dict)
    """Multiscales groups (including metadata)."""

    excluded_groups: list[PurePath] = field(default_factory=list)
    """Input dataset groups excluded from the output dataset."""

    io_path_map: dict[PurePath, PurePath] = field(default_factory=dict)
    """Mapping input->output group paths."""


@dataclass
class ConvertTempData:
    """Stores temporary data and objects reused during the conversion."""

    input_datatrees: list[xr.DataTree] = field(default_factory=list)
    """Input Zarr datasets opened as raw xarray.DataTree objects."""

    input_spatial: InputSpatialInfo = field(default_factory=InputSpatialInfo)
    """Spatial information of input Zarr datasets (root group)."""

    input_spatial_groups: dict[PurePath, InputGroupSpatialInfo] = field(
        default_factory=dict
    )
    """Spatial groups (to convert to HEALPix) in input datasets."""

    output_spatial: OutputSpatialInfo = field(default_factory=OutputSpatialInfo)
    """Spatial information of the output HEALPix Zarr datasets."""

    output_chunks: OutputChunkInfo = field(default_factory=OutputChunkInfo)
    """Chunk (meta)data of the output HEALPix Zarr dataset."""

    output_groups: OutputGroupInfo = field(default_factory=OutputGroupInfo)
    """Information about groups in the output Zarr dataset."""


def _validate_convert_settings(settings: dict | ConvertSettings) -> ConvertSettings:
    if not isinstance(settings, ConvertSettings):
        try:
            settings = ConvertSettings.model_validate(settings)
        except pydantic.ValidationError as err:
            log.error(
                "error while validating conversion settings:\n"
                + "\n".join(e["msg"] for e in err.errors())
            )
            raise

    return settings


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
    output_path: str,
) -> OutputGroupInfo:
    multiscale_groups: dict[PurePath, HealpixMultiscales] = {}
    excluded_groups: list[PurePath] = []
    io_path_map: dict[PurePath, PurePath] = {}

    for path, dts in xr.group_subtrees(*datatrees):
        path = PurePath(path)

        if path in settings.exclude_groups:
            excluded_groups.append(path)
            continue

        if path not in io_path_map:
            io_path_map[path] = path

        multiscales_settings = settings.group_settings.get(path)

        if not isinstance(multiscales_settings, MultiscalesGroupSettings):
            continue

        # datatrees isomorphism has been checked before
        dt0 = dts[0]
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
        root_path=PurePath(output_path),
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


def _set_output_chunks(
    output_spatial: OutputSpatialInfo,
    healpix_chunks: Healpix,
) -> OutputChunkInfo:
    chunk_cell_ids, chunk_is_full = utils.compute_output_chunk_info(
        output_spatial.geometry_latlon,
        cast(int, healpix_chunks.refinement_level),
        healpix_chunks.ellipsoid.name,
    )

    return OutputChunkInfo(
        healpix=healpix_chunks, cell_ids=chunk_cell_ids, is_full=chunk_is_full
    )


def _convert_group_to_healpix(
    path: PurePath,
    data: ConvertTempData,
    settings: ConvertSettings,
):
    log.info(f"••• converting group {path} to HEALPix...")

    spatial_info = data.input_spatial
    group_datasets = [dt[str(path)].dataset for dt in data.input_datatrees]
    group_spatial_info = data.input_spatial_groups[path]
    group_settings = settings.group_settings[path]

    assert isinstance(group_settings, HealpixGroupSettings)
    healpix = group_settings.healpix
    chunk_info = data.output_chunks

    if not group_spatial_info.is_projected:
        log.warning("conversion of lat-lon group not yet implemented (almost there!)")
        return
    if group_settings.chunk is False:
        log.warning("non-chunked group conversion not yet implemented (almost there!)")
        return

    # chunk (fixed) size
    # only "nested" is currently supported for fixed-size chunk processing
    assert healpix.indexing_scheme == "nested"
    assert chunk_info.healpix.indexing_scheme == "nested"

    level = healpix.refinement_level
    chunk_level = chunk_info.healpix.refinement_level
    assert level is not None and chunk_level is not None
    chunk_size: int = 4 ** (level - chunk_level)

    log.info(f"using chunked conversion with fixed chunk size of {chunk_size}")
    log.info(f"resampling data on HEALPix using {group_settings.resampler.name} method")

    # maybe create dataset raster indexes (input datasets with projected CRS)
    if group_spatial_info.is_projected:
        transforms = group_spatial_info.transform
        assert transforms is not None
        spatial_dims = group_spatial_info.spatial_dimensions

        group_datasets = [
            utils.create_raster_index(ds, tr, *spatial_dims)
            for ds, tr in zip(group_datasets, transforms)
        ]

    # create Zarr group with metadata
    group_path = data.output_groups.io_path_map[path]
    zgroup = zarr.create_group(
        str(data.output_groups.root_path),
        path=str(group_path),
        attributes={
            "zarr_conventions": [DGGSZarrConvention().model_dump()],
            "dggs": healpix.model_dump(),
        },
        overwrite=True,
    )

    # create Zarr arrays in the group (values filled chunk by chunk right after)
    cell_dim = healpix.spatial_dimension
    cell_dim_size = chunk_info.cell_ids.size * chunk_size
    zarrays: dict[str, zarr.Array] = {}

    zarrays[healpix.coordinate] = zgroup.create_array(
        name=healpix.coordinate,
        shape=(cell_dim_size,),
        dtype=chunk_info.cell_ids.dtype,
        chunks=(chunk_size,),
        dimension_names=(cell_dim,),
    )

    for name, var in group_datasets[0].coords.items():
        if name not in group_spatial_info.spatial_coordinates:
            # TODO: use dask.array.to_zarr() if var is chunked?
            zarrays[name] = zgroup.create_array(
                name=name,
                data=var.values(),
                chunks=var.chunks,
                dimension_names=var.dims,
            )
    for name, var in group_datasets[0].data_vars.items():
        if name in group_spatial_info.spatial_arrays:
            ...
            # TODO: how to handle non-spatial dimensions?
            if group_settings.resampler.name in ["bilinear", "psf"]:
                dtype = np.float64
            else:
                dtype = var.dtype
            zarrays[name] = zgroup.create_array(
                name=name,
                shape=(cell_dim_size,),
                dtype=dtype,
                chunks=(chunk_size,),
                dimension_names=(cell_dim,),
            )
        else:
            # write array unchanged in output group
            # TODO: use dask.array.to_zarr() if var is chunked?
            zarrays[name] = zgroup.create_array(
                name=name,
                data=var.values(),
                chunks=var.chunks,
                dimension_names=var.dims,
            )
    try:
        client = distributed.Client.current()
    except ValueError:
        client = None

    chunk_index_range = range(data.output_chunks.cell_ids.size)

    if client is not None:
        log.info("converting chunks in parallel using Dask/Distributed")
        # use a wrapper func as passing _convert_one_chunk kwargs to client.map doesn't work
        # (dask/distributed does not like pickling `group_datasets`)
        func = lambda idx: _convert_one_chunk(
            idx,
            chunk_info=chunk_info,
            spatial_info=spatial_info,
            group_spatial_info=group_spatial_info,
            group_settings=group_settings,
            datasets=group_datasets,
            zarrays=zarrays,
        )
        futures = client.map(func, list(chunk_index_range))
        client.gather(futures)
    else:
        log.warning("converting chunks serially (it may take a while)")
        for chunk_index in chunk_index_range:
            _convert_one_chunk(
                chunk_index,
                chunk_info=chunk_info,
                spatial_info=spatial_info,
                group_spatial_info=group_spatial_info,
                group_settings=group_settings,
                datasets=group_datasets,
                zarrays=zarrays,
            )


_RESAMPLER_NAME_CLS: dict[str, type] = {
    "k-nearest": healpix_resample.KNeighborsResampler,
    "nearest": healpix_resample.NearestResampler,
    "bilinear": healpix_resample.BilinearResampler,
    "psf": healpix_resample.PSFResampler,
    "cell-point": healpix_resample.CellPointResampler,
}


def _query_chunk_input_points_projected(
    chunk_cell_id: int,
    datasets: list[xr.Dataset],
    chunk_info: OutputChunkInfo,
    spatial_info: InputSpatialInfo,
    group_spatial_info: InputGroupSpatialInfo,
) -> xr.Dataset | None:
    """Query input `datasets`, keep points used to resample data on `cell_ids`.

    Returns a single dataset with 1-dimensional point data which has:
    - a "points" dimension
    - lat(points) and lon(points) coordinates

    Returns None if query result has no input point.

    This function works with projected points (e.g.,  UTM) and
    implements the following procedure:
    - extract output HEALPix cell ids enveloppe as a polygon
    - prepare each input dataset:
      - re-project the enveloppe polygon in the dataset's projected CRS
      - create a fixed-width (meters) buffer around the polygon
      - get overlapping zone between the buffer polygon and the input Zarr dataset coverage
        in projected CRS coordinates
      - maybe exclude input dataset with overlapping zone is empty
      - slice input dataset (in projected coordinates) using the bounding box of the
        overlapping zone.
    - flatten (stack) input datasets and convert x/y coordinates to lat-lon.

    """
    # fixed (re-projected) chunk polygon buffer width, in meter
    # TODO: make it resolution dependent and/or configurable?
    buffer_width_meters = 60

    # compute chunk cell polygon
    lon, lat = healpix_geo.nested.vertices(
        chunk_cell_id,
        chunk_info.healpix.refinement_level,
        ellipsoid=chunk_info.healpix.ellipsoid.name.upper(),
    )
    chunk_poly_latlon = shapely.Polygon(list(zip(lon[0], lat[0])))

    chunk_polys = utils.reproject_to_multiple_crs(
        chunk_poly_latlon, pyproj.CRS.from_epsg(4326), group_spatial_info.crs
    )

    # Add buffer around polygon
    # TODO: Buffer may not be needed depending on resampler?
    #  - for nearest or groupby, healpix-resample looks within cell only.
    chunk_polys = [shapely.buffer(poly, buffer_width_meters) for poly in chunk_polys]

    # compute intersections with input dataset spatial extents
    input_extents = spatial_info.get_extents()[0]
    overlap_polys = [
        shapely.intersection(chunk_poly, input_extent)
        for chunk_poly, input_extent in zip(chunk_polys, input_extents)
    ]

    # prepare input datasets:
    # - filter and crop input datasets based on overlapping coverage
    #   between (buffered) chunk cell and dataset extents
    # - convert to lat/lon (wgs84) if needed
    # - flatten spatial dimensions
    prepared_datasets = []
    for ds, poly in zip(datasets, overlap_polys):
        if poly.is_empty:
            continue

        xmin, ymin, xmax, ymax = shapely.bounds(poly)

        # TODO: slice bounds vs. north/south hemisphere?
        ds_cropped = ds.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
        ds_with_latlon = ds_cropped.grid4earth.convert_to(4326).stack(
            points=list(group_spatial_info.spatial_dimensions), create_index=False
        )

        prepared_datasets.append(ds_with_latlon)

    if not len(prepared_datasets):
        # No input data found for the current chunk
        return None

    ds_input_points = xr.concat(prepared_datasets, dim="points")

    # TODO: assert (optional) that all points are within the cell-buffered polygon
    # FIXME: results are bad here
    lon = ds_input_points.lon.values
    lat = ds_input_points.lat.values
    spoints = shapely.points(lat, lon)
    points_in_chunk_cell = np.count_nonzero(shapely.within(spoints, chunk_poly_latlon))
    log.info(f"{spoints.size} input points / {points_in_chunk_cell.size} in chunk cell")

    return ds_input_points


def _convert_one_chunk(
    chunk_index: int,
    *,
    chunk_info: OutputChunkInfo,
    spatial_info: InputSpatialInfo,
    group_spatial_info: InputGroupSpatialInfo,
    group_settings: HealpixGroupSettings,
    datasets: list[xr.Dataset],
    zarrays: dict[str, zarr.Array],
):
    chunk_cell_id = chunk_info.cell_ids[chunk_index]

    cell_ids = healpix_geo.nested.zoom_to(
        chunk_cell_id,
        chunk_info.healpix.refinement_level,
        group_settings.healpix.refinement_level,
    )[0]

    chunk_slice = slice(chunk_index, chunk_index + cell_ids.size - 1)
    zarrays[group_settings.healpix.coordinate][chunk_slice] = cell_ids

    if group_spatial_info.is_projected:
        ds_input_points = _query_chunk_input_points_projected(
            chunk_cell_id,
            datasets,
            chunk_info,
            spatial_info,
            group_spatial_info,
        )
    else:
        # TODO: implement alternative input point query strategy(ies) working
        # with other kind of data (e.g., lat/lon)
        ds_input_points = None

    # no input data point for current chunk
    if ds_input_points is None:
        return

    resampler_params = group_settings.resampler.model_dump()
    resampler_name = resampler_params.pop("name")
    resampler_cls = _RESAMPLER_NAME_CLS[resampler_name]
    # TODO: pass array dtype to resampler
    # in case all arrays to resample have the same dtype
    resampler = resampler_cls(
        lon_deg=ds_input_points.lon.values,
        lat_deg=ds_input_points.lat.values,
        level=group_settings.healpix.refinement_level,
        # out_cell_ids=cell_ids,
        nest=group_settings.healpix.indexing_scheme == "nested",
        ellipsoid=group_settings.healpix.ellipsoid.name.upper(),
        **resampler_params,
    )

    # resample arrays (data variables)
    for name, var in ds_input_points.data_vars.items():
        if name not in group_spatial_info.spatial_arrays:
            continue

        # initialize cell data to nodata (TODO: get the right fill_value and dtype)
        # index/fill the array below with data values returned by the resampler
        cell_data = np.full(cell_ids.size, np.nan)

        data = var.values.astype("f")

        scale_factor = var.attrs.get("scale_factor")
        add_offset = var.attrs.get("add_offset")
        if scale_factor is not None:
            data *= scale_factor
        if add_offset is not None:
            data += add_offset

        res = resampler.resample(data)

        zarray = zarrays[str(name)]

        # assume nested indexing scheme -> chunk cell ids is a range
        in_chunk = (res.cell_ids >= cell_ids[0]) & (res.cell_ids <= cell_ids[-1])
        res_indices = (res.cell_ids - cell_ids[0]).astype("int")

        cell_data[res_indices[in_chunk]] = res.cell_data[in_chunk]
        cell_data[np.isnan(cell_data)] = zarray.fill_value
        zarrays[str(name)][chunk_slice] = cell_data.astype(cell_data.dtype)


def create_healpix_dataset(
    input_paths: Sequence[str],
    settings: dict | ConvertSettings,
    output_path: str,
    *,
    groups: str | Sequence[str] | None = None,
    output_extent: dict | shapely.Polygon | None = None,
) -> xr.DataTree:
    """Create a new Zarr dataset with data resampled on the HEALPix grid.

    Parameters
    ----------
    input_paths : list of paths
        Path(s) to the input Zarr dataset(s) to convert to HEALPix.
    output_path : str
        Path to the HEALPix output Zarr dataset.
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
    :py:class:`xarray.DataTree`
        The output Zarr dataset with HEALPix data, returned as an Xarray
        DataTree object.

    """
    data = ConvertTempData()

    log.info(f"••• start converting {len(input_paths)} input datasets into HEALPix...")

    log.info("••• reading conversion settings...")
    settings = _validate_convert_settings(settings)

    # --- open and filter input Zarr datasets
    log.info("••• reading Zarr datasets...")

    data.input_datatrees = _check_and_filter_input_groups(
        utils.open_datatrees(list(input_paths)), groups
    )

    # --- extract (root) spatial info in input Zarr datasets
    log.info("••• extracting spatial info from input datasets (root group)...")

    data.input_spatial = _extract_root_spatial_info(data.input_datatrees)

    unique_crss = set(data.input_spatial.crs)
    log.info(
        f"found {len(unique_crss)} CRS(s) in input datasets:\n"
        + "\n".join(f"  - {crs.name}" for crs in unique_crss)
    )

    # --- compute or set output spatial extent
    data.output_spatial = _set_output_spatial_info(data.input_spatial, output_extent)

    log.info(
        "spatial extent of the output HEALPix dataset (WGS84):\n"
        f"  - geometry: {data.output_spatial.geometry_latlon}\n"
        f"  - bbox: {data.output_spatial.bbox}"
    )

    # --- compute output chunk (coarse) HEALPix cell ids
    data.output_chunks = _set_output_chunks(
        data.output_spatial, settings.healpix_chunks
    )

    log.info(
        f"using fixed-size chunks along the HEALPix cell dimension in the output Zarr dataset\n"
        f"  - one chunk = one level-{settings.healpix_chunks.refinement_level} HEALPix cell\n"
        f"  - total: {data.output_chunks.cell_ids.size} chunk/cells"
    )

    # --- detect input spatial groups (to convert to HEALPix)
    log.info(
        "••• detecting spatial groups in input datasets (to convert to HEALPix)..."
    )

    data.input_spatial_groups = _find_input_spatial_groups(
        data.input_datatrees, settings
    )

    log.info(
        f"found {len(data.input_spatial_groups)} spatial groups to convert to HEALPix:\n"
        + "\n".join(f"  - {path}" for path in data.input_spatial_groups)
    )

    # --- configure HEALPix output groups
    log.info("••• configuring groups in the output HEALPix dataset...")

    data.output_groups = _set_output_groups(
        data.input_datatrees,
        data.input_spatial_groups,
        settings,
        output_path,
    )

    log.info(
        f"configured {len(data.output_groups.multiscale_groups)} multiscale groups:\n"
        + "\n".join(f"  - {path}" for path in data.output_groups.multiscale_groups)
    )
    log.info(
        f"skip {len(data.output_groups.excluded_groups)} group(s) in output dataset:\n"
        + "\n".join(f"  - {path}" for path in data.output_groups.excluded_groups)
    )
    log.info(
        "output HEALPix dataset groups:\n"
        + "\n".join(f"  - {path}" for path in data.output_groups.io_path_map.values())
    )

    # --- create output Zarr dataset
    # TODO: check for any existing output Zarr dataset
    # TODO: write root metadata (STAC / STAC discovery attributes, etc.)
    root_path = PurePath(output_path)
    log.info(f"••• creating {root_path} Zarr dataset...")
    zarr.open_group(str(root_path), mode="w")

    # --- process groups
    log.info("••• processing groups...")

    for path in data.input_datatrees[0].groups:
        path = PurePath(path.lstrip("/"))

        if path in data.output_groups.excluded_groups:
            continue
        if str(path) == ".":
            # root path (skip it here if attributes are written above)
            continue

        group_path_rel = str(data.output_groups.io_path_map[path])

        if path in data.output_groups.multiscale_groups:
            log.info(f"writing group '{group_path_rel}' to Zarr")
            multiscales_obj = data.output_groups.multiscale_groups[path]
            zarr.create_group(
                str(root_path),
                path=group_path_rel,
                attributes={
                    "zarr_conventions": [
                        MultiscalesZarrConvention().model_dump(),
                        DGGSZarrConvention().model_dump(),
                    ],
                    "multiscales": multiscales_obj.model_dump(),
                },
            )
        elif path in data.input_spatial_groups:
            log.info(f"writing group '{group_path_rel}' to Zarr")
            _convert_group_to_healpix(path, data, settings)
        elif not data.input_datatrees[0][str(path)].has_data:
            zarr.create_group(str(root_path), path=group_path_rel)
        else:
            # non-spatial group (skip for now)
            log.warning(
                f"skip writing non-spatial group '{group_path_rel}' to Zarr "
                "(not yet supported)"
            )

    # --- consolidate Zarr metadata
    log.info("••• consolidating output Zarr metadata...")
    zarr.consolidate_metadata(str(root_path))
    # TODO: consolidate metadata for each group/node

    log.info("••• done!")

    return xr.open_datatree(str(root_path))
