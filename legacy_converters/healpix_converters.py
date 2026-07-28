from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import PurePath
from typing import Any, NotRequired, TypedDict, Unpack, cast

import dask.distributed
import healpix_geo
import healpix_resample
import numpy as np
import numpy.typing as npt
import pyproj
import shapely
import structlog
import xarray as xr
import zarr.api.synchronous as zarr
from zarr.api.asynchronous import JSON

import legacy_converters.core.utils as utils
from legacy_converters.core.conversion_models import (
    ConvertStagingCache,
    InputGroupSpatialInfo,
    InputSpatialInfo,
    OutputSpatialInfo,
)
from legacy_converters.core.healpix_conventions import (
    CFHealpixGridMapping,
    DGGSZarrConvention,
    Healpix,
    WGS84Ellipsoid,
)
from legacy_converters.settings.common import (
    ConvertSettings,
    HealpixDenseChunkSettings,
    HealpixGroupSettings,
    HealpixSparseChunkSettings,
    HealpixUniformChunkSettings,
    NoChunkSettings,
    broadcast_params,
)

log = structlog.get_logger()


class ZarrCreateArrayKwargs(TypedDict):
    shape: tuple[int, ...]
    dtype: npt.DTypeLike | str
    data: NotRequired[np.ndarray]
    chunks: NotRequired[tuple[int, ...] | None]
    dimension_names: NotRequired[Iterable[str]]
    codecs: NotRequired[Iterable[dict[str, JSON]] | None]
    attributes: NotRequired[dict[str, JSON] | None]


class HealpixGroupConverter(ABC):
    """Base (abstract) class for resampling data in a Zarr group to HEALPix."""

    root_spatial_info: InputSpatialInfo
    spatial_info: InputGroupSpatialInfo
    settings: HealpixGroupSettings
    healpix: Healpix
    group_path: str
    input_paths: list[str]
    datasets: list[xr.Dataset]
    output_store: zarr.StoreLike
    output_path: PurePath
    output_spatial: OutputSpatialInfo
    output_group: zarr.Group
    output_arrays: dict[str, zarr.Array]

    def __init__(
        self,
        path: PurePath,
        *,
        cache: ConvertStagingCache,
        settings: ConvertSettings,
        output_store: str | zarr.StoreLike,
        output_storage_options: dict[str, Any] | None = None,
    ):
        self.root_spatial_info = cache.input_spatial
        self.spatial_info = cache.input_spatial_groups[path]

        group_settings = settings.group_settings[path]
        assert isinstance(group_settings, HealpixGroupSettings)
        self.settings = group_settings
        self.healpix = self.settings.healpix

        self.group_path = str(path)
        self.input_paths = [str(p) for p in cache.input_datatrees]
        self.datasets = [
            dt[self.group_path].dataset for dt in cache.input_datatrees.values()
        ]

        # maybe create dataset raster indexes (input datasets with projected CRS)
        if self.spatial_info.is_projected:
            transforms = self.spatial_info.transform
            assert transforms is not None
            spatial_dims = self.spatial_info.spatial_dimensions

            self.datasets = [
                utils.create_raster_index(ds, tr, *spatial_dims)
                for ds, tr in zip(self.datasets, transforms)
            ]

        if not isinstance(output_store, str):
            root_group = zarr.open_group(
                output_store, mode="a", storage_options=output_storage_options
            )
            output_store = root_group.store

        self.output_store = output_store
        self.output_path = cache.output_groups.io_path_map[path]
        self.output_spatial = cache.output_spatial
        self.output_group = zarr.open_group(
            self.output_store,
            path=str(self.output_path),
            mode="a",
            attributes={
                "zarr_conventions": [DGGSZarrConvention().model_dump()],
                "dggs": self.healpix.model_dump(),
            },
        )

        self.__post_init__()

        self.output_arrays = self._get_arrays()

    def __post_init__(self):
        pass

    @property
    def cell_dim(self) -> str:
        """Returns the name of the HEALPix cell dimension."""
        return self.healpix.spatial_dimension

    @property
    @abstractmethod
    def cell_dim_size(self) -> int:
        """Returns the number of elements along the HEALPix cell dimension.

        Must be implemented in subclasses.

        """
        ...

    @property
    @abstractmethod
    def cell_dim_chunk_size(self) -> int:
        """Returns the fixed chunk size along the HEALPix cell dimension.

        Must be implemented in subclasses.

        """
        ...

    def _get_arrays(self) -> dict[str, zarr.Array]:
        """Get the output zarr arrays, create them if they don't exist yet."""

        zarrays: dict[str, zarr.Array] = {}

        def _get_maybe_create_array(name: str, **kwargs: Unpack[ZarrCreateArrayKwargs]):
            arr = self.output_group.get(name)
            if arr is None:
                path = (
                    f"{self.output_group.path}/{name}"
                    if self.output_group.path
                    else name
                )
                arr = zarr.create(path=path, store=self.output_group.store, **kwargs)

            assert isinstance(arr, zarr.Array)
            zarrays[name] = arr

        if isinstance(self.healpix.coordinate, str):
            _get_maybe_create_array(
                self.healpix.coordinate,
                shape=(self.cell_dim_size,),
                dtype=np.uint64,
                chunks=(self.cell_dim_chunk_size,),
                dimension_names=(self.cell_dim,),
                fill_value=np.iinfo(np.uint64).max,
                codecs=None,
                # CF 1.13 conventions
                # TODO: more flexible handling of attributes
                attributes={"standard_name": "healpix_index", "units": "1"},
            )

        # CF HEALPix grid mapping variable
        # TODO: more flexible handling of metadata
        cf_hp = CFHealpixGridMapping.from_healpix(self.healpix)
        _get_maybe_create_array(
            "crs",
            shape=(),
            data=np.array(0, dtype=np.int8),
            dtype=np.int8,
            attributes=cf_hp.model_dump(),
        )

        ds0 = self.datasets[0]
        spatial_dims = set(self.spatial_info.spatial_dimensions)

        for name, var in ds0.variables.items():
            if name in self.spatial_info.spatial_coordinates:
                # legacy spatial coordinates are replaced by one HEALPix cell ids coordinate
                continue
            elif name in self.spatial_info.spatial_arrays:
                extra_dims = [d for d in var.dims if d not in spatial_dims]

                if extra_dims:
                    extra_shape = [ds0.sizes[d] for d in extra_dims]
                    var_chunk_sizes = var.encoding.get(
                        "preferred_chunks", {d: ds0.sizes[d] for d in var.dims}
                    )
                    extra_chunk_sizes = [var_chunk_sizes[d] for d in extra_dims]
                    # similarly to how "core" dimensions are handled in numpy
                    # generalized universal functions, the HEALPix cell
                    # dimension is moved to the last axis in the output array.
                    dims = tuple(extra_dims + [self.cell_dim])
                    shape = tuple(extra_shape + [self.cell_dim_size])
                    chunks = tuple(extra_chunk_sizes + [self.cell_dim_chunk_size])
                else:
                    dims = (self.cell_dim,)
                    shape = (self.cell_dim_size,)
                    chunks = (self.cell_dim_chunk_size,)

                if self.settings.resampler.name in ["bilinear", "psf"]:
                    dtype = np.float64
                else:
                    dtype = var.dtype

                _get_maybe_create_array(
                    str(name),
                    shape=shape,
                    dtype=dtype,
                    chunks=chunks,
                    dimension_names=[str(d) for d in dims],
                    codecs=cast(Iterable[dict[str, JSON]], self.settings.codecs),
                    # CF 1.13 conventions
                    # TODO: more flexible handling of attributes
                    attributes={"grid_mapping": "crs"},
                )
            else:
                # write array unchanged in output group
                # TODO: use dask.array.to_zarr() if var is chunked? (avoid memory blow-up)
                _get_maybe_create_array(
                    str(name),
                    shape=var.shape,
                    dtype=var.dtype,
                    data=var.values,
                    chunks=var.encoding.get("chunks"),
                    dimension_names=[str(d) for d in var.dims],
                    codecs=None,
                )

        return zarrays

    def load_datasets(self, client: dask.distributed.Client | None = None) -> None:
        """Preload all input datasets or persist them in distributed memory when using dask."""
        if client is not None:
            # re-open input datasets with chunks={} -> set chunks aligned with zarr
            reopened_datasets = utils.open_datasets(
                self.input_paths, self.group_path, chunks={}
            )
            self.datasets = [ds.persist(scheduler=client) for ds in reopened_datasets]
            # block until all data is loaded.
            for ds in self.datasets:
                dask.distributed.wait(ds)
        else:
            self.datasets = [ds.load() for ds in self.datasets]

    def query_input_points(self, chunk_cell_id: int | None) -> xr.Dataset | None:
        """Query input data points to resample as HEALPix cell data.

        Implementation is optional (implement it only if this is relevant).

        Parameters
        ----------
        chunk_cell_id : int, optional
           If provided, this method should return data points for one output chunk
           only (coarse HEALPix cell). If `None`, this method should return all input
           data points.

        Returns
        -------
        xarray.Dataset or None
           An Xarray Dataset that must have `lat` and `lon` 1-dimensional coordinates.
           Or None if no input point is found (for the given chunk).

        """
        raise NotImplementedError

    @abstractmethod
    def convert(self, chunk_index: int | None = None):
        """Convert data to HEALPix and write it in the output zarr arrays.

        Must be implemented in subclasses.

        Parameters
        ----------
        chunk_index : int, optional
            Convert and write only the output chunk given by the index.

        """
        ...


_RESAMPLER_NAME_CLS: dict[str, type] = {
    "k-nearest": healpix_resample.KNeighborsResampler,
    "nearest": healpix_resample.NearestResampler,
    "bilinear": healpix_resample.BilinearResampler,
    "psf": healpix_resample.PSFResampler,
    "cell-point": healpix_resample.CellPointResampler,
}


class UniformChunkConverter(HealpixGroupConverter, ABC):
    """Common class for chunked conversion where chunks have a fixed-size and are aligned with
    HEALPix cells at a given refinement level.

    This class works only with the "nested" HEALPIX cell indexing scheme.

    """

    chunk_healpix: Healpix
    chunk_cell_ids: np.ndarray
    _input_chunk_cell_ids: list[np.ndarray]

    def __post_init__(self):
        assert isinstance(self.settings.chunk, HealpixUniformChunkSettings)
        self.chunk_healpix = self.settings.chunk.healpix

        assert self.healpix.indexing_scheme == "nested"
        assert self.chunk_healpix.indexing_scheme == "nested"

        self._compute_chunk_cell_ids()

        cell_dim = self.healpix.spatial_dimension
        log.info(
            f"fixed chunk size along the {cell_dim!r} dimension: {self.cell_dim_chunk_size}:\n"
            f"  - each chunk represents a HEALPix cell at level {self.chunk_healpix.refinement_level}\n"
            f"  - total: {self.chunk_cell_ids.size} chunk/cells"
        )

        # cache for querying input lat-lon points inside an healpix chunk cell
        self._input_chunk_cell_ids = []

    def _compute_chunk_cell_ids(self):
        assert self.chunk_healpix.refinement_level is not None

        self.chunk_cell_ids, _ = utils.compute_output_chunk_info(
            self.output_spatial.geometry_latlon,
            self.chunk_healpix.refinement_level,
            self.chunk_healpix.ellipsoid.name,
        )

    def _compute_input_chunk_cell_ids(self, level=None):
        if level is None:
            level = self.chunk_healpix.refinement_level

        lon_name, lat_name = self.spatial_info.spatial_coordinates
        cell_ids = []

        for ds in self.datasets:
            cell_ids.append(
                healpix_geo.nested.lonlat_to_healpix(
                    ds[lon_name].values,
                    ds[lat_name].values,
                    level,
                    self.chunk_healpix.ellipsoid.name.upper(),
                )
            )

        return cell_ids

    @property
    def input_chunk_cell_ids(self):
        if not len(self._input_chunk_cell_ids):
            self._input_chunk_cell_ids = self._compute_input_chunk_cell_ids()

        return self._input_chunk_cell_ids

    @property
    @abstractmethod
    def chunk_buffer(self) -> int | float: ...

    def _query_chunk_input_points_latlon_curvilinear(
        self, chunk_cell_id: int
    ) -> xr.Dataset | None:
        """Query input lat-lon points on a curvilinear grid (e.g., ).

        Assumes 2-dimensional longitude and latitude coordinates present in the datasets.

        Only input points located within the HEALPix chunk cell are selected. If
        the chunk buffer is non zero, the selection of input points also include
        an outer ring of HEALPix cells around the the chunk at a level determined
        from the buffer width.

        """
        lon_name, lat_name = self.spatial_info.spatial_coordinates
        ds0 = self.datasets[0]
        rdim, cdim = ds0[lon_name].dims

        if self.chunk_buffer > 0:
            ring_level = utils.resolution_to_level(self.chunk_buffer, round_type="ceil")
            input_ring_cell_ids = self._compute_input_chunk_cell_ids(level=ring_level)
        else:
            ring_level = 0
            input_ring_cell_ids = [np.array([], dtype=np.uint64)] * len(self.datasets)

        prepared_datasets = []
        for ds, cids, rids in zip(
            self.datasets, self.input_chunk_cell_ids, input_ring_cell_ids
        ):
            if self.chunk_buffer > 0:
                # TODO: far from optimal (a function in healpix_geo for getting the ring would be better)
                chunk_zoomed = healpix_geo.nested.zoom_to(
                    chunk_cell_id,
                    depth=self.chunk_healpix.refinement_level,
                    new_depth=ring_level,
                ).ravel()

                chunk_with_ring = healpix_geo.nested.kth_neighbourhood(
                    chunk_zoomed, ring_level, ring=1
                )
                chunk_with_ring = np.unique(chunk_with_ring)

                start_id = np.searchsorted(
                    chunk_with_ring, chunk_zoomed[0], side="left"
                )
                stop_id = np.searchsorted(
                    chunk_with_ring, chunk_zoomed[-1], side="right"
                )
                ring_cell_ids = np.delete(chunk_with_ring, slice(start_id, stop_id))

                is_selected = np.logical_or(
                    cids == chunk_cell_id, np.isin(rids, ring_cell_ids)
                )
            else:
                # fastpath
                is_selected = cids == chunk_cell_id

            ridx, cidx = np.nonzero(is_selected)

            if len(ridx) and len(cidx):
                # note: the Xarray advanced indexing operation below already broadcasts
                # the Dataset variables that only have a subset of the spatial dimensions
                # (i.e., either rows or columns)
                ds_clipped = ds.isel(
                    {
                        rdim: xr.Variable("points", ridx),
                        cdim: xr.Variable("points", cidx),
                    }
                )
                prepared_datasets.append(ds_clipped)

        if not len(prepared_datasets):
            # No input data found for the current chunk
            return None

        ds_input_points = xr.concat(prepared_datasets, dim="points")
        ds_input_points["lon"] = ds_input_points[lon_name]
        ds_input_points["lat"] = ds_input_points[lat_name]

        return ds_input_points

    def _query_chunk_input_points_projected(
        self, chunk_cell_id: int
    ) -> xr.Dataset | None:
        """
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
        crs_wgs84 = pyproj.CRS.from_epsg(4326)

        # compute chunk cell polygon
        lon, lat = healpix_geo.nested.vertices(
            chunk_cell_id,
            self.chunk_healpix.refinement_level,
            ellipsoid=self.chunk_healpix.ellipsoid.name.upper(),
        )
        chunk_poly_latlon = shapely.Polygon(list(zip(lon[0], lat[0])))

        chunk_polys = utils.reproject_to_multiple_crs(
            chunk_poly_latlon, crs_wgs84, self.spatial_info.crs
        )

        # Maybe add buffer around polygon
        if self.chunk_buffer > 0.0:
            chunk_polys = [
                shapely.buffer(poly, self.chunk_buffer) for poly in chunk_polys
            ]

        # compute intersections with input dataset spatial extents
        input_extents = self.root_spatial_info.get_extents()[0]
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
        for ds, crs, poly in zip(self.datasets, self.spatial_info.crs, overlap_polys):
            if poly.is_empty:
                continue

            xmin, ymin, xmax, ymax = shapely.bounds(poly)

            ds_clipped = (
                ds
                # TODO: check slice bounds for north vs. south hemisphere?
                .sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
                .compute()
                .pipe(lambda ds: utils.assign_transform_coords(ds, crs, crs_wgs84))
                # note: the Xarray stack operation below already broadcasts
                # the Dataset variables that only have a subset of the spatial dimensions
                # (i.e., either x or y)
                .stack(
                    points=list(self.spatial_info.spatial_dimensions),
                    create_index=False,
                )
            )

            x = ds_clipped.x.values
            y = ds_clipped.y.values
            points = shapely.points(x, y)
            in_poly = shapely.within(points, poly)

            ds_clipped = ds_clipped.isel(points=in_poly)

            prepared_datasets.append(ds_clipped)

        if not len(prepared_datasets):
            # No input data found for the current chunk
            return None

        ds_input_points = xr.concat(prepared_datasets, dim="points")
        return ds_input_points

    def query_input_points(self, chunk_cell_id: int | None) -> xr.Dataset | None:
        # this class supports chunked conversion only
        assert chunk_cell_id is not None

        if self.spatial_info.is_projected:
            return self._query_chunk_input_points_projected(chunk_cell_id)
        elif not self.spatial_info.is_rectilinear:
            return self._query_chunk_input_points_latlon_curvilinear(chunk_cell_id)
        else:
            log.warning(
                "selection of chunk input data not supported or not yet implemented"
            )
            return


class DenseChunkConverter(UniformChunkConverter):
    """Chunked conversion of input data to HEALPix, where output chunks correspond
    to "coarse" HEALPix cells densely populated by finer HEALPIX cells at a same level
    (continuous cell id range).

    This class works only with the "nested" HEALPIX cell indexing scheme.

    """

    @property
    def cell_dim_size(self) -> int:
        return self.chunk_cell_ids.size * self.cell_dim_chunk_size

    @property
    def cell_dim_chunk_size(self) -> int:
        level = self.healpix.refinement_level
        chunk_level = self.chunk_healpix.refinement_level
        assert level is not None and chunk_level is not None
        return 4 ** (level - chunk_level)

    @property
    def chunk_buffer(self):
        assert isinstance(self.settings.chunk, HealpixDenseChunkSettings)
        return self.settings.chunk.chunk_buffer_width

    def convert(self, chunk_index: int | None = None):
        # this class supports chunked conversion only
        assert chunk_index is not None

        n_chunks = self.chunk_cell_ids.size

        chunk_cell_id = self.chunk_cell_ids[chunk_index]

        log.debug(
            f"processing chunk {chunk_index + 1}/{n_chunks} (cell id {chunk_cell_id})"
        )

        cell_ids = healpix_geo.nested.zoom_to(
            chunk_cell_id,
            self.chunk_healpix.refinement_level,
            self.healpix.refinement_level,
        )[0]

        start = chunk_index * cell_ids.size
        stop = start + cell_ids.size
        chunk_slice = slice(start, stop)
        self.output_arrays[self.healpix.coordinate][chunk_slice] = cell_ids

        log.debug("query input points")
        ds_input_points = self.query_input_points(chunk_cell_id)

        # no input data point for current chunk
        if ds_input_points is None:
            return

        resampler_settings = self.settings.resampler

        # For the "nearest" resample method: give output cell ids the resampler (forcing)
        # For the other methods: best to let the resampler algorithm do the lookup
        if resampler_settings.name == "nearest":
            out_cell_ids = cell_ids
        else:
            out_cell_ids = None

        resampler_cls = _RESAMPLER_NAME_CLS[resampler_settings.name]
        # TODO: pass array dtype to resampler
        # in case all arrays to resample have the same dtype

        log.debug("initialize resampler")
        resampler = resampler_cls(
            lon_deg=ds_input_points.lon.values,
            lat_deg=ds_input_points.lat.values,
            level=self.healpix.refinement_level,
            verbose=False,
            out_cell_ids=out_cell_ids,
            nest=self.healpix.indexing_scheme == "nested",
            ellipsoid=self.healpix.ellipsoid.name.upper(),
            **dict(resampler_settings.init_params),
        )
        # resample arrays (data variables)
        var_names = [
            str(name)
            for name in ds_input_points.variables
            if name in self.spatial_info.spatial_arrays
        ]
        resample_params = broadcast_params(
            resampler_settings.resample_params, var_names
        )

        log.debug(f"resample {len(var_names)} arrays")
        for name in var_names:
            var = ds_input_points.variables[name]

            assert "points" in var.dims
            extra_dims = [d for d in var.dims if d != "points"]
            extra_dim_sizes = [var.sizes[d] for d in extra_dims]

            # make sure the "points" core dimension is moved to the last axis
            # maybe stack non-spatial dimensions, to pass to `.resample()` as a single "batch" dimension
            if extra_dims:
                if len(extra_dims) == 1:
                    var = var.transpose(extra_dims[0], "points")
                else:
                    var = var.stack(batch=extra_dims).transpose("batch", "points")

                cell_data_shape = (var.shape[0], cell_ids.size)
            else:
                cell_data_shape = (cell_ids.size,)

            # initialize cell data to nodata (TODO: get the right fill_value and dtype)
            # index/fill the array below with data values returned by the resampler
            cell_data = np.full(cell_data_shape, np.nan)

            data = var.values.astype("f")

            res = resampler.resample(data, **resample_params[name])

            zarray = self.output_arrays[str(name)]

            # assume nested indexing scheme -> chunk cell ids is a range
            in_chunk = (res.cell_ids >= cell_ids[0]) & (res.cell_ids <= cell_ids[-1])
            res_indices = (res.cell_ids - cell_ids[0]).astype("int")

            cell_data[..., res_indices[in_chunk]] = res.cell_data[..., in_chunk]
            cell_data[..., np.isnan(cell_data)] = zarray.fill_value

            # maybe unstack non-spatial dimensions from the resampled batch dimension
            if len(extra_dims) > 1:
                cell_data = cell_data.reshape(tuple(extra_dim_sizes + [cell_ids.size]))

            zarray[..., chunk_slice] = cell_data.astype(cell_data.dtype)


class SparseChunkConverter(UniformChunkConverter):
    """Chunked conversion of input data to HEALPix, where output chunks correspond
    to "coarse" HEALPix cells sparsely populated by finer HEALPIX cells at a same level.

    This class works only with the "nested" HEALPIX cell indexing scheme.

    """

    @property
    def cell_dim_size(self) -> int:
        return self.chunk_cell_ids.size * self.cell_dim_chunk_size

    @property
    def cell_dim_chunk_size(self) -> int:
        assert isinstance(self.settings.chunk, HealpixSparseChunkSettings)
        return self.settings.chunk.chunk_size

    @property
    def chunk_buffer(self):
        return 0

    def convert(self, chunk_index: int | None = None):
        # this class supports chunked conversion only
        assert chunk_index is not None

        n_chunks = self.chunk_cell_ids.size

        chunk_cell_id = self.chunk_cell_ids[chunk_index]

        log.debug(
            f"processing chunk {chunk_index + 1}/{n_chunks} (cell id {chunk_cell_id})"
        )

        log.debug("query input points")
        ds_input_points = self.query_input_points(chunk_cell_id)

        # no input data point for current chunk
        if ds_input_points is None:
            return

        log.debug("initialize resampler")
        resampler_settings = self.settings.resampler

        resampler_cls = _RESAMPLER_NAME_CLS[resampler_settings.name]
        # TODO: pass array dtype to resampler
        # in case all arrays to resample have the same dtype

        resampler = resampler_cls(
            lon_deg=ds_input_points.lon.values,
            lat_deg=ds_input_points.lat.values,
            # level=self.healpix.refinement_level,  # assumes cell-point (level=29)
            verbose=False,
            nest=self.healpix.indexing_scheme == "nested",
            ellipsoid=self.healpix.ellipsoid.name.upper(),
            **dict(resampler_settings.init_params),
        )

        cell_ids = resampler.get_cell_ids()
        start = chunk_index * self.cell_dim_chunk_size
        stop = start + cell_ids.size
        chunk_slice = slice(start, stop)

        self.output_arrays[self.healpix.coordinate][chunk_slice] = cell_ids

        # resample arrays (data variables)
        var_names = [
            str(name)
            for name in ds_input_points.variables
            if name in self.spatial_info.spatial_arrays
        ]
        resample_params = broadcast_params(
            resampler_settings.resample_params, var_names
        )

        log.debug(f"resample {len(var_names)} arrays")
        for name in var_names:
            var = ds_input_points.variables[name]

            assert "points" in var.dims
            extra_dims = [d for d in var.dims if d != "points"]
            extra_dim_sizes = [var.sizes[d] for d in extra_dims]

            # make sure the "points" core dimension is moved to the last axis
            # maybe stack non-spatial dimensions, to pass to `.resample()` as a single "batch" dimension
            if extra_dims:
                if len(extra_dims) == 1:
                    var = var.transpose(extra_dims[0], "points")
                else:
                    var = var.stack(batch=extra_dims).transpose("batch", "points")

            data = var.values.astype("f")

            res = resampler.resample(data, **resample_params[name])

            # maybe unstack non-spatial dimensions from the resampled batch dimension
            if len(extra_dims) > 1:
                cell_data = res.cell_data.reshape(
                    tuple(extra_dim_sizes + [cell_ids.size])
                )
            else:
                cell_data = res.cell_data

            zarray = self.output_arrays[str(name)]
            zarray[..., chunk_slice] = cell_data.astype(var.dtype)

        del resampler


class NoChunkConverter(HealpixGroupConverter):
    """Conversion of input data to HEALPix, where the output consists in
    one single chunk of HEALPix cells that are  fully covering the extent
    (polygon) of the output Zarr dataset at a fixed refinement level given
    by the group conversion settings.

    This class works only with the "nested" HEALPIX cell indexing scheme.

    """

    cell_ids: np.ndarray
    ellipsoid: str

    def __post_init__(self):
        assert isinstance(self.settings.chunk, NoChunkSettings)
        assert self.healpix.indexing_scheme == "nested"

        if isinstance(self.healpix.ellipsoid, WGS84Ellipsoid):
            self.ellipsoid = self.healpix.ellipsoid.name.upper()
        else:
            # TODO: is sphere really used here? If yes, need a fix below
            # in `query_input_points` where input x/y coordinates are converted
            # to lat/lon on the WGS84 datum.
            self.ellipsoid = "sphere"

        self.cell_ids, _, _ = healpix_geo.nested.polygon_coverage(
            shapely.coordinates.get_coordinates(self.output_spatial.geometry_latlon),
            self.healpix.refinement_level,
            ellipsoid=self.ellipsoid,
            flat=True,
        )

        cell_dim = self.healpix.spatial_dimension
        log.info(f"one single chunk along the {cell_dim!r} dimension.")

    @property
    def cell_dim_size(self) -> int:
        return self.cell_ids.size

    @property
    def cell_dim_chunk_size(self) -> int:
        return self.cell_dim_size

    def query_input_points(self, chunk_cell_id: int | None) -> xr.Dataset:
        # this class does not support conversion chunk by chunk
        assert chunk_cell_id is None

        crs_wgs84 = pyproj.CRS.from_epsg(4326)

        def maybe_convert_to_latlon(ds: xr.Dataset, crs: pyproj.CRS):
            if self.spatial_info.is_projected:
                return utils.assign_transform_coords(ds, crs, crs_wgs84)
            else:
                lon_name, lat_name = self.spatial_info.spatial_coordinates
                return ds.assign_coords(lon=ds[lon_name], lat=ds[lat_name])

        prepared_datasets = []
        for ds, crs in zip(self.datasets, self.spatial_info.crs):
            ds_latlon_flat = (
                ds.compute().pipe(lambda ds: maybe_convert_to_latlon(ds, crs))
                # note: the Xarray stack operation below already broadcasts
                # the Dataset variables that only have a subset of the spatial dimensions
                .stack(
                    points=list(self.spatial_info.spatial_dimensions),
                    create_index=False,
                )
            )

            prepared_datasets.append(ds_latlon_flat)

        ds_input_points = xr.concat(prepared_datasets, dim="points")
        return ds_input_points

    def convert(self, chunk_index: int | None = None):
        # this class does not support conversion chunk by chunk
        assert chunk_index is None

        cell_ids = self.cell_ids

        self.output_arrays[self.healpix.coordinate][:] = cell_ids

        log.debug("query input points")
        ds_input_points = self.query_input_points(None)

        resampler_settings = self.settings.resampler

        # For the "nearest" resample method: give output cell ids the resampler (forcing)
        # For the other methods: best to let the resampler algorithm do the lookup
        if resampler_settings.name == "nearest":
            out_cell_ids = self.cell_ids
        else:
            out_cell_ids = None

        resampler_cls = _RESAMPLER_NAME_CLS[resampler_settings.name]
        # TODO: pass array dtype to resampler
        # in case all arrays to resample have the same dtype

        log.debug("initialize resampler")
        resampler = resampler_cls(
            lon_deg=ds_input_points.lon.values,
            lat_deg=ds_input_points.lat.values,
            level=self.healpix.refinement_level,
            verbose=False,
            out_cell_ids=out_cell_ids,
            nest=self.healpix.indexing_scheme == "nested",
            ellipsoid=self.ellipsoid,
            **dict(resampler_settings.init_params),
        )
        # resample arrays (data variables)
        var_names = [
            str(name)
            for name in ds_input_points.variables
            if name in self.spatial_info.spatial_arrays
        ]
        resample_params = broadcast_params(
            resampler_settings.resample_params, var_names
        )

        log.debug(f"resample {len(var_names)} arrays")
        for name in var_names:
            var = ds_input_points.variables[name]

            assert "points" in var.dims
            extra_dims = [d for d in var.dims if d != "points"]
            extra_dim_sizes = [var.sizes[d] for d in extra_dims]

            # make sure the "points" core dimension is moved to the last axis
            # maybe stack non-spatial dimensions, to pass to `.resample()` as a single "batch" dimension
            if extra_dims:
                if len(extra_dims) == 1:
                    var = var.transpose(extra_dims[0], "points")
                else:
                    var = var.stack(batch=extra_dims).transpose("batch", "points")

                cell_data_shape = (var.shape[0], cell_ids.size)
            else:
                cell_data_shape = (cell_ids.size,)

            # initialize cell data to nodata (TODO: get the right fill_value and dtype)
            # index/fill the array below with data values returned by the resampler
            cell_data = np.full(cell_data_shape, np.nan)

            data = var.values.astype("f")

            res = resampler.resample(data, **resample_params[name])

            zarray = self.output_arrays[str(name)]

            _, res_cell_data_idx, cell_data_idx = np.intersect1d(
                res.cell_ids, cell_ids, assume_unique=True, return_indices=True
            )

            cell_data[..., cell_data_idx] = res.cell_data[..., res_cell_data_idx]
            cell_data[..., np.isnan(cell_data)] = zarray.fill_value

            # maybe unstack non-spatial dimensions from the resampled batch dimension
            if len(extra_dims) > 1:
                cell_data = cell_data.reshape(tuple(extra_dim_sizes + [cell_ids.size]))

            zarray[..., :] = cell_data.astype(cell_data.dtype)


_CONVERTER_NAME_CLS: dict[str, type[HealpixGroupConverter]] = {
    "no_chunk": NoChunkConverter,
    "healpix_cell_dense": DenseChunkConverter,
    "healpix_cell_sparse": SparseChunkConverter,
}


def init_converter(
    path: PurePath,
    *,
    cache: ConvertStagingCache,
    settings: ConvertSettings,
    output_store: str | zarr.StoreLike,
    output_storage_options: dict[str, Any] | None = None,
) -> HealpixGroupConverter:
    group_settings = settings.group_settings[path]
    cls = _CONVERTER_NAME_CLS[group_settings.chunk.method]

    return cls(
        path,
        cache=cache,
        settings=settings,
        output_store=output_store,
        output_storage_options=output_storage_options,
    )
