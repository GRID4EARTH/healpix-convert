import pyproj
import xarray as xr

CRSLike = int | str | pyproj.CRS


def ensure_crs(crs_like: CRSLike) -> pyproj.CRS:
    crs: pyproj.CRS

    if not isinstance(crs_like, pyproj.CRS):
        crs = pyproj.CRS.from_user_input(crs_like)
    else:
        crs = crs_like

    return crs


def create_transformer(source_crs: CRSLike, target_crs: CRSLike) -> pyproj.Transformer:
    return pyproj.Transformer.from_crs(
        ensure_crs(source_crs), ensure_crs(target_crs), always_xy=True
    )


def maybe_convert(ds: xr.Dataset, transformer: pyproj.Transformer) -> xr.Dataset:
    if not {"x", "y"}.issubset(ds.dims):
        # no spatial dims
        return ds

    xx, yy = xr.broadcast(ds["x"].drop_indexes("x"), ds["y"].drop_indexes("y"))

    axis_names = [ax.abbrev.lower() for ax in transformer.target_crs.axis_info]
    axis_attrs = transformer.target_crs.cs_to_cf()

    converted = xr.apply_ufunc(
        transformer.transform, xx, yy, output_core_dims=[[]] * len(axis_names)
    )

    new_coords = {
        name: coord.assign_attrs(attrs)
        for name, coord, attrs in zip(axis_names, converted, axis_attrs)
    }

    return ds.assign_coords(new_coords)
