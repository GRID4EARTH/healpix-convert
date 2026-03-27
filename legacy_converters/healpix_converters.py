from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import PurePath
from typing import Any, TypedDict, Unpack

import healpix_geo
import healpix_resample
import numpy as np
import pyproj
import shapely
import structlog
import xarray as xr
import zarr.api.synchronous as zarr

import legacy_converters.core.utils as utils
from legacy_converters.core.conversion_models import (
    ConvertStagingCache,
    InputGroupSpatialInfo,
    InputSpatialInfo,
    OutputChunkInfo,
)
from legacy_converters.core.healpix_conventions import DGGSZarrConvention, Healpix
from legacy_converters.settings.common import (
    ConvertSettings,
    HealpixGroupSettings,
    broadcast_params,
)

log = structlog.get_logger()


class ZarrCreateArrayKwargs(TypedDict):
    shape: tuple[int, ...]
    dtype: np.dtype
    data: np.ndarray
    chunks: tuple[int, ...]
    dimension_names: Iterable[str]


class HealpixGroupConverter(ABC):
    """Base (abstract) class for resampling data in a Zarr group to HEALPix."""

    root_spatial_info: InputSpatialInfo
    spatial_info: InputGroupSpatialInfo
    chunk_info: OutputChunkInfo
    settings: HealpixGroupSettings
    healpix: Healpix
    datasets: list[xr.Dataset]
    output_store: zarr.StoreLike
    output_path: PurePath
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
        self.chunk_info = cache.output_chunks

        group_settings = settings.group_settings[path]
        assert isinstance(group_settings, HealpixGroupSettings)
        self.settings = group_settings
        self.healpix = self.settings.healpix

        self.datasets = [dt[str(path)].dataset for dt in cache.input_datatrees.values()]

        # TODO: remove
        if not self.spatial_info.is_projected:
            log.warning(
                "conversion of lat-lon group not yet implemented (almost there!)"
            )
            return

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
                arr = self.output_group.create_array(name=name, **kwargs)
            assert isinstance(arr, zarr.Array)
            zarrays[name] = arr

        _get_maybe_create_array(
            self.healpix.coordinate,
            shape=(self.cell_dim_size,),
            dtype=np.int64,
            chunks=(self.cell_dim_chunk_size,),
            dimension_names=(self.cell_dim,),
        )

        ds0 = self.datasets[0]

        for name, var in ds0.coords.items():
            if name not in self.spatial_info.spatial_coordinates:
                # TODO: use dask.array.to_zarr() if var is chunked?
                _get_maybe_create_array(
                    str(name),
                    data=var.values,
                    chunks=var.chunks,
                    dimension_names=var.dims,
                )
        for name, var in ds0.data_vars.items():
            if name in self.spatial_info.spatial_arrays:
                # TODO: how to handle non-spatial dimensions?
                if self.settings.resampler.name in ["bilinear", "psf"]:
                    dtype = np.float64
                else:
                    dtype = var.dtype
                _get_maybe_create_array(
                    str(name),
                    shape=(self.cell_dim_size,),
                    dtype=dtype,
                    chunks=(self.cell_dim_chunk_size,),
                    dimension_names=(self.cell_dim,),
                )
            else:
                # write array unchanged in output group
                # TODO: use dask.array.to_zarr() if var is chunked?
                _get_maybe_create_array(
                    str(name),
                    data=var.values,
                    chunks=var.chunks,
                    dimension_names=var.dims,
                )

        return zarrays

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


class DenseChunkConverter(HealpixGroupConverter):
    """Chunked conversion of input data to HEALPix, where output chunks correspond
    to "coarse" HEALPix cells densely populated by finer HEALPIX cells at a same level
    (continuous cell id range).

    This class works only with the "nested" HEALPIX cell indexing scheme.

    """

    def __post_init__(self):
        assert self.settings.chunk.method == "healpix_cell"
        assert self.healpix.indexing_scheme == "nested"
        assert self.chunk_info.healpix.indexing_scheme == "nested"

        level = self.healpix.refinement_level
        chunk_level = self.chunk_info.healpix.refinement_level
        assert level is not None and chunk_level is not None
        self._chunk_size: int = 4 ** (level - chunk_level)

        # TODO: make it configurable
        self.poly_buffer: float = 60

    @property
    def cell_dim_size(self) -> int:
        return self.chunk_info.cell_ids.size * self._chunk_size

    @property
    def cell_dim_chunk_size(self) -> int:
        return self._chunk_size

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
        # compute chunk cell polygon
        lon, lat = healpix_geo.nested.vertices(
            chunk_cell_id,
            self.chunk_info.healpix.refinement_level,
            ellipsoid=self.chunk_info.healpix.ellipsoid.name.upper(),
        )
        chunk_poly_latlon = shapely.Polygon(list(zip(lon[0], lat[0])))

        chunk_polys = utils.reproject_to_multiple_crs(
            chunk_poly_latlon, pyproj.CRS.from_epsg(4326), self.spatial_info.crs
        )

        # Add buffer around polygon
        # TODO: Buffer may not be needed depending on resampler?
        #  - for nearest or groupby, healpix-resample looks within cell only.
        chunk_polys = [shapely.buffer(poly, self.poly_buffer) for poly in chunk_polys]

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
        for ds, poly in zip(self.datasets, overlap_polys):
            if poly.is_empty:
                continue

            xmin, ymin, xmax, ymax = shapely.bounds(poly)

            ds_clipped = (
                ds
                # TODO: check slice bounds for north vs. south hemisphere?
                .sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
                .grid4earth.convert_to(4326)
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
        else:
            log.warning(
                "conversion of lat-lon group not yet implemented (almost there!)"
            )
            return

    def convert(self, chunk_index: int | None = None):
        # this class supports chunked conversion only
        assert chunk_index is not None

        n_chunks = self.chunk_info.cell_ids.size

        chunk_cell_id = self.chunk_info.cell_ids[chunk_index]

        log.debug(
            f"processing chunk {chunk_index + 1}/{n_chunks} (cell id {chunk_cell_id})"
        )

        cell_ids = healpix_geo.nested.zoom_to(
            chunk_cell_id,
            self.chunk_info.healpix.refinement_level,
            self.healpix.refinement_level,
        )[0]

        start = chunk_index * cell_ids.size
        stop = start + cell_ids.size
        chunk_slice = slice(start, stop)
        self.output_arrays[self.healpix.coordinate][chunk_slice] = cell_ids

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
            for name in ds_input_points.data_vars
            if name in self.spatial_info.spatial_arrays
        ]
        resample_params = broadcast_params(
            resampler_settings.resample_params, var_names
        )

        for name in var_names:
            var = ds_input_points[name]

            # initialize cell data to nodata (TODO: get the right fill_value and dtype)
            # index/fill the array below with data values returned by the resampler
            cell_data = np.full(cell_ids.size, np.nan)

            data = var.values.astype("f")

            res = resampler.resample(data, **resample_params[name])

            zarray = self.output_arrays[str(name)]

            # assume nested indexing scheme -> chunk cell ids is a range
            in_chunk = (res.cell_ids >= cell_ids[0]) & (res.cell_ids <= cell_ids[-1])
            res_indices = (res.cell_ids - cell_ids[0]).astype("int")

            cell_data[res_indices[in_chunk]] = res.cell_data[in_chunk]
            cell_data[np.isnan(cell_data)] = zarray.fill_value
            self.output_arrays[str(name)][chunk_slice] = cell_data.astype(
                cell_data.dtype
            )
