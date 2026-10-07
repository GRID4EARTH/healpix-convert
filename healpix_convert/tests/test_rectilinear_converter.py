from pathlib import PurePath

import dask.array as da
import healpix_geo
import numpy as np
import pytest
import xarray as xr
import zarr

from healpix_convert.core.conversion_models import (
    ConvertStagingCache,
    InputGroupSpatialInfo,
    OutputGroupInfo,
)
from healpix_convert.core.utils import extract_spatial_info_cf
from healpix_convert.healpix_converters import DenseChunkConverter
from healpix_convert.settings.common import ConvertSettings


def make_converter(monkeypatch, datasets, cell, chunk_level=3, level=5, buffer=100_000):
    path = PurePath("measurements/land_cover")
    grid = {"indexing_scheme": "nested", "ellipsoid": {"name": "wgs84"}}
    settings = ConvertSettings.from_dict(
        {
            "group_settings": {
                str(path): {
                    "healpix": {**grid, "refinement_level": level},
                    "chunk": {
                        "method": "healpix_cell_dense",
                        "healpix": {**grid, "refinement_level": chunk_level},
                        "chunk_buffer_width": buffer,
                    },
                    "resampler": {"name": "nearest"},
                }
            }
        }
    )
    cache = ConvertStagingCache(
        input_datatrees={
            str(i): xr.DataTree.from_dict({str(path): ds})
            for i, ds in enumerate(datasets)
        },
        input_spatial_groups={
            path: InputGroupSpatialInfo(**extract_spatial_info_cf(datasets[0]))
        },
        output_groups=OutputGroupInfo(io_path_map={path: path}),
    )
    monkeypatch.setattr(
        "healpix_convert.core.utils.compute_output_chunk_info",
        lambda *args: (np.array([cell], dtype=np.uint64), np.array([True])),
    )
    return DenseChunkConverter(
        path, cache=cache, settings=settings, output_store=zarr.storage.MemoryStore()
    )


def grid(lon, lat):
    values = np.arange(len(lat) * len(lon)).reshape(len(lat), len(lon))
    return xr.Dataset(
        {"class_id": (("row", "column"), da.from_array(values, chunks=(16, 16)))},
        coords={
            "longitude": ("column", lon, {"standard_name": "longitude"}),
            "latitude": ("row", lat, {"standard_name": "latitude"}),
        },
    )


@pytest.mark.parametrize(
    "position", [(0, 0), (179.9, 0), (-179.9, 50), (45, 89.9), (225, -89.9)]
)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("positive_lon", [False, True])
def test_selection_contains_cell_and_nearest_halo(
    monkeypatch, position, reverse, positive_lon
):
    lon = np.arange(0.5, 360, 2) if positive_lon else np.arange(-179.5, 180, 2)
    lat = np.arange(-89.5, 90, 2)
    if reverse:
        lon, lat = lon[::-1], lat[::-1]
    ds = grid(lon, lat)
    cell = healpix_geo.nested.lonlat_to_healpix(*position, 3, ellipsoid="WGS84")[0]
    converter = make_converter(monkeypatch, [ds], cell, buffer=300_000)
    selected = converter.query_input_points(cell)
    assert selected is not None
    assert isinstance(selected.class_id.data, da.Array)
    assert selected.class_id.size < ds.class_id.size
    xx, yy = np.meshgrid(lon, lat)
    parent_ids = healpix_geo.nested.lonlat_to_healpix(
        xx.ravel(), yy.ravel(), 3, ellipsoid="WGS84"
    )
    expected = ds.class_id.values.ravel()[parent_ids == cell]
    assert np.isin(expected, selected.class_id.values).all()
    # Every child centre's globally nearest source must survive window selection,
    # including sources outside the parent cell at seams and near the poles.
    children = healpix_geo.nested.zoom_to(cell, 3, 5).ravel()
    outlon, outlat = healpix_geo.nested.healpix_to_lonlat(
        children, 5, ellipsoid="WGS84"
    )
    import pyproj

    transform = pyproj.Transformer.from_crs(4326, 4978, always_xy=True)
    xyz = np.array(transform.transform(xx.ravel(), yy.ravel(), np.zeros(xx.size))).T
    target = np.array(transform.transform(outlon, outlat, np.zeros(outlon.size))).T
    nearest = ((target[:, None] - xyz[None, :]) ** 2).sum(axis=-1).argmin(axis=1)
    assert np.isin(ds.class_id.values.ravel()[nearest], selected.class_id.values).all()


def test_empty_and_multiple_inputs(monkeypatch):
    cell = healpix_geo.nested.lonlat_to_healpix(0, 0, 3, ellipsoid="WGS84")[0]
    far = grid([120, 121], [60, 61])
    near = grid([-1, 0, 1], [-1, 0, 1])
    converter = make_converter(monkeypatch, [far], cell)
    assert converter.query_input_points(cell) is None
    converter.datasets = [far, near, near]
    selected = converter.query_input_points(cell)
    assert selected is not None
    converter.datasets = [near]
    single = converter.query_input_points(cell)
    assert selected.sizes["points"] == 2 * single.sizes["points"]
    np.testing.assert_array_equal(selected.class_id, np.tile(single.class_id, 2))


def test_nearest_level15_preserves_classes_and_metadata(monkeypatch):
    cell = healpix_geo.nested.lonlat_to_healpix(2, 48, 12, ellipsoid="WGS84")[0]
    lon = np.arange(1.9, 2.1, 1 / 360)
    lat = np.arange(48.1, 47.9, -1 / 360)
    ds = grid(lon, lat)
    classes = np.array([0, 10, 40, 190, 220], dtype=np.uint8)
    ds["class_id"] = (("row", "column"), classes[ds.class_id.values % len(classes)])
    converter = make_converter(
        monkeypatch, [ds], cell, chunk_level=12, level=15, buffer=1000
    )
    converter.convert(0)
    actual = converter.output_arrays["class_id"][:]
    assert actual.dtype == np.uint8
    assert np.isin(actual, classes).all()
    assert 0 in actual  # no-data is retained, not filled from adjacent classes
    assert np.unique(actual).size > 1
    cell_ids = converter.output_arrays["cell_ids"][:]
    np.testing.assert_array_equal(
        cell_ids, healpix_geo.nested.zoom_to(cell, 12, 15).ravel()
    )
    outlon, outlat = healpix_geo.nested.healpix_to_lonlat(
        cell_ids, 15, ellipsoid="WGS84"
    )
    import pyproj

    transform = pyproj.Transformer.from_crs(4326, 4978, always_xy=True)
    xx, yy = np.meshgrid(lon, lat)
    xyz = np.array(transform.transform(xx.ravel(), yy.ravel(), np.zeros(xx.size))).T
    target = np.array(transform.transform(outlon, outlat, np.zeros(outlon.size))).T
    nearest = ((target[:, None] - xyz[None, :]) ** 2).sum(axis=-1).argmin(axis=1)
    np.testing.assert_array_equal(actual, ds.class_id.values.ravel()[nearest])
    metadata = converter.output_group.attrs["dggs"]
    assert metadata["ellipsoid"]["name"] == "wgs84"
    assert metadata["indexing_scheme"] == "nested"
    assert metadata["refinement_level"] == 15
