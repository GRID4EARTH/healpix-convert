"""Utility methods used internally for converting datasets onto HEALPix."""

from collections import Counter

import affine
import healpix_geo
import numpy as np
import pyproj
import rasterix
import shapely
import xarray as xr


def open_datatrees(paths: list[str]) -> list[xr.DataTree]:
    """Open input Zarr datasets as "raw" xarray.DataTree objects.

    Fast-version (disable most decoding and index creation).
    """
    return [
        xr.open_datatree(
            path,
            engine="zarr",
            chunks={},
            decode_cf=False,
            decode_times=False,
            decode_timedelta=False,
            create_default_indexes=False,
        )
        for path in paths
    ]


def get_wgs84_polygon_from_stac(xr_obj: xr.DataTree | xr.Dataset) -> shapely.Polygon:
    geom_dict = xr_obj.attrs["stac_discovery"]["geometry"]
    return shapely.geometry.shape(geom_dict)


def get_proj_polygon_from_stac(xr_obj: xr.DataTree | xr.Dataset) -> shapely.Polygon:
    bbox = xr_obj.attrs["stac_discovery"]["properties"]["proj:bbox"]
    return shapely.geometry.box(*bbox)


def get_crs_from_stac(xr_obj: xr.DataTree | xr.Dataset) -> pyproj.CRS:
    crs_epsg = xr_obj.attrs["stac_discovery"]["properties"]["proj:epsg"]
    return pyproj.CRS.from_user_input(crs_epsg)


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

    if "proj:epsg" not in var0.attrs:
        return None

    spatial_arrays = [
        name for name, var in ds.data_vars.items() if "proj:epsg" in var0.attrs
    ]

    return {
        "crs": [pyproj.CRS.from_epsg(var0.attrs["proj:epsg"])],
        "transform": [affine.Affine(*var0.attrs["proj:transform"])],
        "spatial_dimensions": {"x": ds.sizes["x"], "y": ds.sizes["y"]},
        "spatial_coordinates": ["x", "y"],
        "spatial_attrs": [],
        "spatial_arrays": spatial_arrays,
        "spatial_var_attrs": [a.startswith("proj:") for a in var0.attrs],
    }


def extract_spatial_info_cf(ds: xr.Dataset) -> dict | None:
    """Extract spatial information from a Zarr group using CF conventions.

    Only latitude-longitude is supported for now (implicit sphere model as
    assumed by CF).

    Returns None if no spatial information is detected.

    """
    latlon_coords = set()
    latlon_dims = set()

    for name, var in ds.coords.items():
        if var.attrs.get("standard_name", "") in ("latitude", "longitude"):
            latlon_coords.add(name)
            latlon_dims.update(var.dims)

    if latlon_coords:
        spatial_arrays = [
            name
            for name, var in ds.data_vars.items()
            if set(var.dims).intersection(latlon_dims)
        ]

        return {
            # TODO: EPSG generic spherical model? Or allow something other?
            "crs": [pyproj.CRS.from_epsg(6404)],
            "transform": None,
            "spatial_dimensions": {dim: ds.sizes[dim] for dim in latlon_dims},
            "spatial_coordinates": list(latlon_coords),
            "spatial_arrays": [],
            "spatial_attrs": spatial_arrays,
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
