from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import pyproj
import xarray as xr
from affine import Affine

from legacy_converters.crs import CRSLike, create_transformer, maybe_convert

if TYPE_CHECKING:
    from typing import Literal

    import xdggs


def _maybe_create_raster_index(ds):
    import rasterix

    transform = ds.grid4earth.affine_transform(kind="corner")
    if transform is None:
        return ds

    x_dim = "x"
    y_dim = "y"
    width = ds.sizes[x_dim]
    height = ds.sizes[y_dim]

    raster_index = rasterix.RasterIndex.from_transform(
        transform,
        width=width,
        height=height,
        x_dim=x_dim,
        y_dim=y_dim,
    )

    coords = xr.Coordinates.from_xindex(raster_index)

    return ds.assign_coords(coords)


@xr.register_datatree_accessor("grid4earth")
class DataTreeConverterAccessor:
    def __init__(self, dt: xr.DataTree):
        self._dt = dt

    def _infer_crs_code(self) -> str:
        return self._dt.attrs["other_metadata"]["horizontal_CRS_code"]

    @cached_property
    def crs(self) -> pyproj.CRS:
        """The EOPF tree's spatial CRS"""
        return pyproj.CRS.from_user_input(self._infer_crs_code())

    def create_raster_indexes(self):
        return self._dt.map_over_datasets(_maybe_create_raster_index)

    def convert_to(self, target_crs: CRSLike) -> xr.DataTree:
        """Attach spatial coordinates in the target CRS

        Parameters
        ----------
        target_crs : int or str or pyproj.CRS
            The target CRS, as interpreted by :py:func:`pyproj.CRS.from_user_input`.

        Returns
        -------
        xarray.DataTree
            The current object with new coordinates.
        """
        transformer = create_transformer(self.crs, target_crs)

        return xr.DataTree.from_dict(
            {
                path: maybe_convert(node.ds, transformer)
                for path, node in self._dt.subtree_with_keys
            }
        )


def _search_attribute(ds: xr.Dataset, name: str) -> int | str | list:
    values = {
        var_name: var.attrs[name]
        for var_name, var in ds.data_vars.items()
        if name in var.attrs
    }
    unique_values = {tuple(v) if isinstance(v, list) else v for v in values.values()}
    if len(unique_values) > 1:
        raise ValueError(f"disagreement in {name}")

    return next(iter(values.values()), None)


@xr.register_dataset_accessor("grid4earth")
class DatasetConverterAccessor:
    def __init__(self, ds: xr.Dataset):
        self._ds = ds

    def _infer_crs_code(self) -> str | None:
        return _search_attribute(self._ds, "proj:epsg")

    def _infer_affine_transform(self) -> Affine | None:
        return _search_attribute(self._ds, "proj:transform")

    def _infer_bounding_box(self) -> str | None:
        return _search_attribute(self._ds, "proj:bbox")

    @cached_property
    def crs(self) -> pyproj.CRS | None:
        crs_code = self._infer_crs_code()
        if crs_code is None:
            return crs_code

        return pyproj.CRS.from_user_input(crs_code)

    @property
    def bbox(self) -> tuple[float, ...] | None:
        index = self._ds.xindexes.get("x")
        if index is not None and hasattr(index, "xy_dims"):
            return index.bbox

        return self._infer_bounding_box()

    def affine_transform(
        self, kind: Literal["corner", "center"] | None = None
    ) -> Affine | None:
        if kind not in {"corner", "center", None}:
            raise ValueError(f"Unknown transform kind requested: {kind!r}")

        index = self._ds.xindexes.get("x")
        if index is not None and hasattr(index, "transform"):
            # raster index
            if kind in ["corner", None]:
                return index.transform()
            else:
                return index.center_transform()

        values = self._infer_affine_transform()
        if values is None:
            return None

        affine = Affine(*values)
        if kind in {"corner", None}:
            return affine * Affine.translation(-0.5, -0.5)

        return affine

    def minimum_bounding_rectangle(self) -> np.ndarray:
        transform = self.affine_transform(kind="corner")
        if transform is None:
            # TODO: what do we do here? Infer from the coordinates?
            raise ValueError(
                "no affine transform found, but this is required to compute the MBR."
            )

        nx = self._ds.sizes["x"]
        ny = self._ds.sizes["y"]

        x = np.array([0, nx, nx, 0], dtype="uint64")
        y = np.array([0, 0, ny, ny], dtype="uint64")

        coords_x, coords_y = transform * (x, y)

        return np.stack([coords_x, coords_y], axis=-1)

    def infer_healpix_grid(self, grid_info: xdggs.HealpixInfo) -> xr.Dataset:
        import healpix_geo
        import xdggs  # noqa: F401

        if grid_info.indexing_scheme != "nested":
            raise ValueError(
                "Rasterizing is only supported for the `nested` indexing scheme."
            )

        try:
            mbr = self.minimum_bounding_rectangle()
        except ValueError as e:
            raise ValueError(
                "Can't determine the minimum bounding rectangle"
                " needed to define the corresponding healpix grid."
            ) from e
        transformer = create_transformer(self.crs, 4326)
        vertices = np.stack(transformer.transform(mbr[:, 0], mbr[:, 1]), axis=-1)

        cell_ids, _, _ = healpix_geo.nested.polygon_coverage(
            vertices, grid_info.level, ellipsoid="WGS84", flat=True
        )

        return xr.Dataset(coords={"cell_ids": ("cells", cell_ids)}).dggs.decode(
            grid_info
        )
