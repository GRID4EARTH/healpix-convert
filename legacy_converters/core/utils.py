"""Utility methods used internally for converting datasets onto HEALPix."""

from collections import Counter
from typing import Any

import affine
import healpix_geo
import numpy as np
import pyproj
import rasterix
import shapely
import xarray as xr


def open_datatrees(paths: list[str], chunks: dict | None = None) -> list[xr.DataTree]:
    """Open input Zarr datasets as "raw" xarray.DataTree objects.

    Fast-version (disable most decoding and index creation).
    """
    return [
        xr.open_datatree(
            path,
            engine="zarr",
            chunks=chunks,
            decode_cf=True,
            decode_times=False,
            decode_timedelta=False,
            create_default_indexes=False,
        )
        for path in paths
    ]


def open_datasets(
    paths: list[str], group: str, chunks: dict | None = None
) -> list[xr.Dataset]:
    """Open a given group in input Zarr datasets as "raw" xarray.Dataset objects.

    Fast-version (disable most decoding and index creation).
    """
    return [
        xr.open_zarr(
            path,
            group=group,
            chunks=chunks,
            decode_cf=True,
            decode_times=False,
            decode_timedelta=False,
            create_default_indexes=False,
        )
        for path in paths
    ]


def _get_stac_proj_crs(attrs: dict[str, Any]) -> pyproj.CRS | None:
    crss = []
    if "proj:epsg" in attrs:
        # STAC projection extension version < 2.0
        crss.append(pyproj.CRS.from_epsg(attrs["proj:epsg"]))
    if "proj:code" in attrs:
        crss.append(pyproj.CRS.from_user_input(attrs["proj:code"]))
    if "proj:wkt2" in attrs:
        crss.append(pyproj.CRS.from_user_input(attrs["proj:wkt2"]))
    if "proj:projjson" in attrs:
        crss.append(pyproj.CRS.from_user_input(attrs["proj:projjson"]))

    if len(crss):
        if len(crss) > 1 and not all(crs == crss[0] for crs in crss[1:]):
            raise ValueError("found conficting CRS in metadata")
        return crss[0]
    else:
        return None


def get_wgs84_polygon_from_stac(xr_obj: xr.DataTree | xr.Dataset) -> shapely.Polygon:
    geom_dict = xr_obj.attrs["stac_discovery"]["geometry"]
    return shapely.geometry.shape(geom_dict)


def get_proj_polygon_from_stac(xr_obj: xr.DataTree | xr.Dataset) -> shapely.Polygon:
    bbox = xr_obj.attrs["stac_discovery"]["properties"].get("proj:bbox")
    if bbox is not None:
        return shapely.geometry.box(*bbox)
    else:
        return shapely.Polygon()


def get_crs_from_stac(xr_obj: xr.DataTree | xr.Dataset) -> pyproj.CRS:
    crs = _get_stac_proj_crs(xr_obj.attrs["stac_discovery"]["properties"])
    if crs is None:
        # Return WGS84 as default if no CRS is found
        # TODO: what should be the default CRS? Undefined (sphere)?
        crs = pyproj.CRS.from_epsg(4326)
    return crs


def extract_spatial_info_stac(ds: xr.Dataset) -> dict | None:
    """Extract spatial information from a Zarr group using STAC conventions.

    Assume that spatial information is represented as STAC (projection)
    attributes in data variables (EOPF Zarr groups).

    Assume x/y spatial dimensions.

    Returns None if no spatial information is detected.

    """
    # TODO: check consistent STAC attrs in each variables?
    # (assume they are consistent for now)
    var0 = next(iter(ds.data_vars.values()))

    crs = _get_stac_proj_crs(var0.attrs)
    if crs is None:
        return None

    spatial_dims = {"x", "y"}
    spatial_arrays = [
        name
        for name, var in ds.variables.items()
        if spatial_dims.intersection(var.dims) and name not in spatial_dims
    ]

    transform = affine.Affine(*var0.attrs["proj:transform"])

    return {
        "crs": [crs],
        "transform": [transform],
        "is_rectilinear": transform.is_rectilinear,
        "spatial_dimensions": {"x": ds.sizes["x"], "y": ds.sizes["y"]},
        "spatial_coordinates": ["x", "y"],
        "spatial_attrs": [],
        "spatial_arrays": spatial_arrays,
        "spatial_var_attrs": [name for name in var0.attrs if name.startswith("proj:")],
    }


def extract_spatial_info_cf(ds: xr.Dataset) -> dict | None:
    """Extract spatial information from a Zarr group using CF conventions.

    Only longitude-latitude is supported for now (WGS84 is assumed). The order
    (longitude, latitude) is used here.

    Returns None if no spatial information is detected.

    """
    std_lonlat = ("longitude", "latitude")
    lonlat_coords = {}
    lonlat_dims = set()
    is_rectilinear = True

    for name, var in ds.coords.items():
        std_name = var.attrs.get("standard_name")
        if std_name in std_lonlat:
            lonlat_coords[std_name] = name
            lonlat_dims.update(var.dims)
            if len(var.dims) > 1:
                is_rectilinear = False

    if lonlat_coords:
        lonlat_coords_ = [lonlat_coords.get(name) for name in std_lonlat]
        spatial_arrays = [
            name
            for name, var in ds.variables.items()
            if set(var.dims).intersection(lonlat_dims) and name not in lonlat_coords_
        ]

        return {
            # TODO: EPSG generic spherical model? Or allow something other?
            "crs": [pyproj.CRS.from_epsg(4326)],
            "transform": None,
            "is_rectilinear": is_rectilinear,
            "spatial_dimensions": {dim: ds.sizes[dim] for dim in lonlat_dims},
            "spatial_coordinates": [c for c in lonlat_coords_ if c is not None],
            "spatial_attrs": [],
            "spatial_arrays": spatial_arrays,
            "spatial_var_attrs": [],
        }
    else:
        return None


def reproject_to_common_crs(
    polygons: list[shapely.Polygon],
    from_crs: list[pyproj.CRS],
    to_crs: pyproj.CRS,
) -> list[shapely.Polygon]:
    """Re-project polygons each to a single, common target CRS."""

    assert len(from_crs) == len(polygons)

    transformers: dict[pyproj.CRS, pyproj.Transformer] = {
        crs: pyproj.Transformer.from_crs(crs, to_crs, always_xy=True)
        for crs in set(from_crs)
    }

    out_polygons: list[shapely.Polygon] = []
    for poly, crs in zip(polygons, from_crs):
        tr = transformers[crs]
        out_polygons.append(shapely.transform(poly, tr.transform, interleaved=False))

    return out_polygons


def reproject_to_multiple_crs(
    polygon: shapely.Polygon,
    from_crs: pyproj.CRS,
    to_crs: list[pyproj.CRS],
) -> list[shapely.Polygon]:
    """Re-project a polygon to multiple target CRSs."""

    transformers: dict[pyproj.CRS, pyproj.Transformer] = {
        crs: pyproj.Transformer.from_crs(from_crs, crs, always_xy=True)
        for crs in set(to_crs)
    }

    out_polygons: list[shapely.Polygon] = []
    for crs in to_crs:
        tr = transformers[crs]
        out_polygons.append(shapely.transform(polygon, tr.transform, interleaved=False))

    return out_polygons


def compute_output_extent_latlon(
    input_extents: list[shapely.Polygon],
    input_crss: list[pyproj.CRS],
) -> shapely.Polygon:
    """Compute the spatial extent of the output HEALPix dataset
    as the union of the spatial extents of the input Zarr datasets.

    Returns a polygon with lat-lon WGS84 coordinates.
    """
    # find the most common CRS
    crs_common, _ = Counter(input_crss).most_common(1)[0]

    # TODO: ensure that the common CRS is a projected one
    # (shapely uses a planar geometry engine for unions)

    # re-project all input extents into the common CRS
    input_extents_ = reproject_to_common_crs(input_extents, input_crss, crs_common)

    # compute union and convert to lat-lon WGS84
    output_extent_ = shapely.union_all(input_extents_)

    tr = pyproj.Transformer.from_crs(
        crs_common, pyproj.CRS.from_epsg(4326), always_xy=True
    )
    output_extent = shapely.transform(output_extent_, tr.transform, interleaved=False)

    return output_extent


def compute_output_chunk_info(
    output_extent: shapely.Polygon,
    chunk_level: int,
    ellipsoid_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    chunk_cell_ids, _, chunk_is_full = healpix_geo.nested.polygon_coverage(
        shapely.coordinates.get_coordinates(output_extent),
        chunk_level,
        ellipsoid=ellipsoid_name.upper(),
        flat=True,
    )

    return chunk_cell_ids, chunk_is_full


def create_raster_index(
    ds: xr.Dataset, transform: affine.Affine, x_dim: str, y_dim: str
):
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
