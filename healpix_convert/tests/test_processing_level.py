"""The ``processing_level`` chunk setting: a chunk resampled by processing cells."""

from __future__ import annotations

from pathlib import PurePath

import dask.array as da
import healpix_geo
import numpy as np
import pydantic
import pyproj
import pytest
import structlog.testing
import xarray as xr
import zarr

from healpix_convert.convert import _map_processing_cells
from healpix_convert.core import utils
from healpix_convert.core.conversion_models import (
    ConvertStagingCache,
    InputGroupSpatialInfo,
    OutputGroupInfo,
)
from healpix_convert.core.utils import extract_spatial_info_cf
from healpix_convert.healpix_converters import DenseChunkConverter
from healpix_convert.settings.common import ConvertSettings

CHUNK_LEVEL = 5
LEVEL = 9
# the Gaussian kernel of the PSF resampler (metres): a level 9 cell is ~ 14 km wide
SIGMA_M = 6_000
# more than 8 kernel widths, narrower than a processing cell of level 7 (~ 56 km)
WIDE_BUFFER = 50_000
GRID = {"indexing_scheme": "nested", "ellipsoid": {"name": "wgs84"}}
PATH = PurePath("measurements/field")


def make_settings(
    processing_level=None, buffer=WIDE_BUFFER, resampler="psf", chunk_level=CHUNK_LEVEL
):
    chunk = {
        "method": "healpix_cell_dense",
        "healpix": {**GRID, "refinement_level": chunk_level},
        "chunk_buffer_width": buffer,
    }
    if processing_level is not None:
        chunk["processing_level"] = processing_level
    resampler_settings = {"name": resampler}
    if resampler == "psf":
        resampler_settings["init_params"] = {"sigma_m": SIGMA_M}
    return ConvertSettings.from_dict(
        {
            "group_settings": {
                str(PATH): {
                    "healpix": {**GRID, "refinement_level": LEVEL},
                    "chunk": chunk,
                    "resampler": resampler_settings,
                }
            }
        }
    )


def make_input(cell, rectilinear=True):
    """A smooth field on a regular grid around a chunk cell (0.05 degrees)."""
    lon0, lat0 = healpix_geo.nested.healpix_to_lonlat(
        cell, CHUNK_LEVEL, ellipsoid="WGS84"
    )
    lon0, lat0 = float(lon0[0]), float(lat0[0])
    lon = lon0 + np.arange(-2.5, 2.5, 0.05)
    lat = lat0 + np.arange(-2.5, 2.5, 0.05)
    xx, yy = np.meshgrid(lon, lat)
    values = (np.sin(xx / 1.3) * np.cos(yy / 1.7) + 0.01 * (xx + yy)).astype("float32")
    if rectilinear:
        coords = {
            "lon": ("lon", lon, {"standard_name": "longitude"}),
            "lat": ("lat", lat, {"standard_name": "latitude"}),
        }
        dims = ("lat", "lon")
    else:
        # a curvilinear (2-D coordinates) swath, slightly rotated
        rot = np.deg2rad(10)
        xr_, yr_ = xx - lon0, yy - lat0
        xx, yy = lon0 + np.cos(rot) * xr_ - np.sin(rot) * yr_, lat0 + (
            np.sin(rot) * xr_ + np.cos(rot) * yr_
        )
        coords = {
            "lon": (("row", "col"), xx, {"standard_name": "longitude"}),
            "lat": (("row", "col"), yy, {"standard_name": "latitude"}),
        }
        dims = ("row", "col")
    return xr.Dataset(
        {"field": (dims, da.from_array(values, chunks=64))}, coords=coords
    )


def make_converter(monkeypatch, ds, cell, settings, store=None):
    cache = ConvertStagingCache(
        input_datatrees={"0": xr.DataTree.from_dict({str(PATH): ds})},
        input_spatial_groups={
            PATH: InputGroupSpatialInfo(**extract_spatial_info_cf(ds))
        },
        output_groups=OutputGroupInfo(io_path_map={PATH: PATH}),
    )
    monkeypatch.setattr(
        "healpix_convert.core.utils.compute_output_chunk_info",
        lambda *args: (
            np.atleast_1d(np.asarray(cell, dtype=np.uint64)),
            np.ones(np.size(cell), dtype=bool),
        ),
    )
    return DenseChunkConverter(
        PATH,
        cache=cache,
        settings=settings,
        output_store=zarr.storage.MemoryStore() if store is None else store,
    )


@pytest.fixture(scope="module")
def cell():
    return healpix_geo.nested.lonlat_to_healpix(
        3.0, 47.0, CHUNK_LEVEL, ellipsoid="WGS84"
    )[0]


@pytest.fixture(params=[True, False], ids=["rectilinear", "curvilinear"])
def ds(request, cell):
    return make_input(cell, rectilinear=request.param)


def convert(monkeypatch, ds, cell, **kwargs):
    converter = make_converter(monkeypatch, ds, cell, make_settings(**kwargs))
    converter.convert(0)
    return converter


def arrays(converter):
    return {name: converter.output_arrays[name][:] for name in ("cell_ids", "field")}


def test_default_is_one_processing_cell(monkeypatch, ds, cell):
    converter = make_converter(monkeypatch, ds, cell, make_settings())
    assert converter.units_per_chunk == 1
    assert converter.unit_level == CHUNK_LEVEL
    # a chunk is one processing cell: the same selection, resampling and write
    # as the code before the setting existed
    unit_cell_id, cell_ids = converter.unit_cell_ids(0, 0)
    assert unit_cell_id == cell
    np.testing.assert_array_equal(
        cell_ids, healpix_geo.nested.zoom_to(cell, CHUNK_LEVEL, LEVEL).ravel()
    )
    values = converter.resample_unit(0, 0)
    converter.write_chunk(0, [values])
    out = arrays(converter)
    np.testing.assert_array_equal(out["cell_ids"], cell_ids)
    assert out["field"].dtype == np.float64
    assert np.isfinite(out["field"]).all()
    np.testing.assert_array_equal(out["field"], values["field"])


@pytest.mark.parametrize("extra_levels", [1, 2])
def test_wide_buffer_equals_single_solve(monkeypatch, ds, cell, extra_levels):
    reference = convert(monkeypatch, ds, cell)
    split = convert(monkeypatch, ds, cell, processing_level=CHUNK_LEVEL + extra_levels)
    assert split.units_per_chunk == 4**extra_levels
    a, b = arrays(reference), arrays(split)
    np.testing.assert_array_equal(a["cell_ids"], b["cell_ids"])
    assert a["field"].shape == b["field"].shape
    assert a["field"].dtype == b["field"].dtype
    # the PSF solve is iterative, in float32: the results agree to ~1e-4
    np.testing.assert_allclose(a["field"], b["field"], rtol=1e-3, atol=2e-4)
    assert np.isfinite(b["field"]).all()


def test_no_buffer_differs_only_near_processing_cell_edges(monkeypatch, cell):
    ds = make_input(cell)
    level = CHUNK_LEVEL + 1
    reference = arrays(convert(monkeypatch, ds, cell, buffer=0))
    split = arrays(convert(monkeypatch, ds, cell, buffer=0, processing_level=level))
    np.testing.assert_array_equal(reference["cell_ids"], split["cell_ids"])
    diff = np.abs(reference["field"] - split["field"])
    # the kernel is cut at the edge of a processing cell: values change there,
    # as they do at the edge of a chunk
    assert diff.max() > 1e-4

    # distance (in output cells of the group) to the edge of the processing cell
    ids = reference["cell_ids"]
    unit = ids >> (2 * (LEVEL - level))
    neighbours = healpix_geo.nested.kth_neighbourhood(ids, LEVEL, ring=2)
    near_edge = ((neighbours >> (2 * (LEVEL - level))) != unit[:, None]).any(axis=1)
    assert diff[~near_edge].max() < 1e-5
    assert diff[near_edge].max() > 1e-4


def test_processing_cells_without_input_stay_fill_value(monkeypatch, cell):
    ds = make_input(cell)
    # keep the input of the first processing cells only
    converter = make_converter(
        monkeypatch,
        ds.isel(lat=slice(0, 25)),
        cell,
        make_settings(processing_level=CHUNK_LEVEL + 1, buffer=0),
    )
    values = [converter.resample_unit(0, i) for i in range(converter.units_per_chunk)]
    assert any(v is None for v in values) and any(v is not None for v in values)
    converter.write_chunk(0, values)
    out = arrays(converter)
    size = out["cell_ids"].size // converter.units_per_chunk
    fill = converter.output_arrays["field"].fill_value
    for i, v in enumerate(values):
        part = out["field"][i * size : (i + 1) * size]
        if v is None:
            np.testing.assert_array_equal(part, np.full_like(part, fill))
        else:
            np.testing.assert_array_equal(part, v["field"])


def test_selection_of_a_processing_cell_is_buffered_around_it(monkeypatch, ds, cell):
    narrow = make_converter(
        monkeypatch, ds, cell, make_settings(processing_level=CHUNK_LEVEL + 2, buffer=0)
    )
    wide = make_converter(
        monkeypatch,
        ds,
        cell,
        make_settings(processing_level=CHUNK_LEVEL + 2, buffer=5_000),
    )
    # a chunk with the same buffer: a processing cell (5 km buffer, ring level 9,
    # finer than the processing level) selects less than the whole chunk. With a
    # wide buffer this does not hold (see the ring cases below)
    chunk = make_converter(monkeypatch, ds, cell, make_settings(buffer=5_000))
    unit_cell_id, _ = narrow.unit_cell_ids(0, 5)
    n_unit = narrow.query_input_points(unit_cell_id).sizes["points"]
    n_buffered = wide.query_input_points(unit_cell_id).sizes["points"]
    n_chunk = chunk.query_input_points(cell).sizes["points"]
    assert n_unit < n_buffered < n_chunk


def test_nearest_resampler_is_supported(monkeypatch, cell):
    ds = make_input(cell)
    a = arrays(convert(monkeypatch, ds, cell, resampler="nearest", buffer=100_000))
    b = arrays(
        convert(
            monkeypatch,
            ds,
            cell,
            resampler="nearest",
            buffer=100_000,
            processing_level=CHUNK_LEVEL + 1,
        )
    )
    np.testing.assert_array_equal(a["cell_ids"], b["cell_ids"])
    np.testing.assert_array_equal(a["field"], b["field"])


def test_out_of_bounds_processing_cell(monkeypatch, ds, cell):
    converter = make_converter(
        monkeypatch, ds, cell, make_settings(processing_level=CHUNK_LEVEL + 1)
    )
    with pytest.raises(ValueError, match="out of bounds"):
        converter.resample_unit(0, 4)
    with pytest.raises(ValueError, match="expected 4 processing cells"):
        converter.write_chunk(0, [None])


@pytest.mark.parametrize("level", [CHUNK_LEVEL - 1, 0])
def test_processing_level_coarser_than_chunk_level_is_rejected(level):
    with pytest.raises(pydantic.ValidationError, match="coarser than the chunk level"):
        make_settings(processing_level=level)


def test_processing_level_finer_than_output_level_is_rejected():
    with pytest.raises(
        pydantic.ValidationError, match="finer than the refinement level"
    ):
        make_settings(processing_level=LEVEL + 1, chunk_level=CHUNK_LEVEL + 3)
    make_settings(processing_level=LEVEL)


def test_processing_level_limit_on_processing_cells_per_chunk():
    # at most 4 levels finer than the chunk level: 256 processing cells
    settings = make_settings(processing_level=CHUNK_LEVEL + 4)
    assert settings.group_settings[PATH].chunk.processing_level == CHUNK_LEVEL + 4
    with pytest.raises(pydantic.ValidationError, match="at most 4 levels finer"):
        make_settings(processing_level=CHUNK_LEVEL + 5, chunk_level=CHUNK_LEVEL)


def test_processing_level_is_rejected_for_other_chunk_methods():
    with pytest.raises(pydantic.ValidationError, match="processing_level"):
        ConvertSettings.from_dict(
            {
                "group_settings": {
                    str(PATH): {
                        "healpix": {**GRID, "refinement_level": LEVEL},
                        "chunk": {
                            "method": "healpix_cell_sparse",
                            "healpix": {**GRID, "refinement_level": CHUNK_LEVEL},
                            "chunk_size": 1024,
                            "processing_level": CHUNK_LEVEL + 1,
                        },
                    }
                }
            }
        )


class SynchronousClient:
    """Runs the tasks at once, in this process (no scheduler, no port)."""

    def __init__(self):
        self.calls = []

    def map(self, func, iterable, **kwargs):
        self.calls.append(("map", kwargs))
        return [func(item) for item in iterable]

    def submit(self, func, *args, **kwargs):
        self.calls.append(("submit", kwargs))
        return func(*args)


def test_map_processing_cells_writes_the_same_chunk(monkeypatch, ds, cell):
    reference = arrays(convert(monkeypatch, ds, cell, processing_level=CHUNK_LEVEL + 1))
    converter = make_converter(
        monkeypatch, ds, cell, make_settings(processing_level=CHUNK_LEVEL + 1)
    )
    client = SynchronousClient()
    resources = {"GPU": 1}
    futures = _map_processing_cells(client, converter, [0], 4, resources)
    assert futures == [None]
    # one resampling task per processing cell, one write task per chunk; keys
    # must not be shared between groups
    assert [kind for kind, _ in client.calls] == ["map", "submit"]
    assert all(kw["pure"] is False for _, kw in client.calls)
    assert all(kw["resources"] == resources for _, kw in client.calls)
    assert len({kw["priority"] for _, kw in client.calls}) == 1
    out = arrays(converter)
    np.testing.assert_array_equal(out["cell_ids"], reference["cell_ids"])
    np.testing.assert_array_equal(out["field"], reference["field"])


def test_distributed_client(monkeypatch, cell, tmp_path):
    distributed = pytest.importorskip("distributed")
    # no HTTP server (it binds a port), in-process transport
    monkeypatch.setattr(
        "distributed.node.ServerNode.start_http_server", lambda *args, **kw: None
    )
    try:
        cluster = distributed.LocalCluster(
            n_workers=1,
            threads_per_worker=2,
            processes=False,
            protocol="inproc://",
            dashboard_address=None,
        )
    except (OSError, RuntimeError) as exc:  # e.g. local ports cannot be bound
        pytest.skip(f"cannot start a local dask cluster: {exc}")
    ds = make_input(cell)
    reference = arrays(convert(monkeypatch, ds, cell, processing_level=CHUNK_LEVEL + 1))
    # the tasks run on a copy of the converter (it is pickled with the task): the
    # output must be a store shared by copies, not a MemoryStore
    converter = make_converter(
        monkeypatch,
        ds,
        cell,
        make_settings(processing_level=CHUNK_LEVEL + 1),
        store=zarr.storage.LocalStore(tmp_path / "out.zarr"),
    )
    with cluster, distributed.Client(cluster) as client:
        client.gather(_map_processing_cells(client, converter, [0], 4, {}))
    out = arrays(converter)
    np.testing.assert_array_equal(out["cell_ids"], reference["cell_ids"])
    np.testing.assert_array_equal(out["field"], reference["field"])


# (processing level, buffer in metres, ring level relative to the processing
# level). The ring level is finer than, equal to and coarser than the processing
# level; the last four are buffers wider than a processing cell (~ 14 km at
# level 9, ~ 28 km at level 8)
FINER, EQUAL, COARSER = "finer", "equal", "coarser"
RING_CASES = [
    (9, 1_000, FINER),
    (9, 6_000, EQUAL),
    (9, 9_000, COARSER),
    (9, 20_000, COARSER),
    (9, WIDE_BUFFER, COARSER),
    (8, 120_000, COARSER),
]


def ring_relation(buffer, unit_level):
    ring_level = utils.chunk_buffer_ring_level(buffer)
    if ring_level > unit_level:
        return FINER
    return EQUAL if ring_level == unit_level else COARSER


@pytest.mark.parametrize("unit_level, buffer, relation", RING_CASES)
def test_wide_buffer_selects_every_point_within_it(
    monkeypatch, cell, unit_level, buffer, relation
):
    """Every input point within the buffer of a processing cell is selected, also
    when the buffer is wider than the cell (the ring is then made of cells coarser
    than the cell)."""
    assert ring_relation(buffer, unit_level) == relation
    ds = make_input(cell, rectilinear=False)
    converter = make_converter(
        monkeypatch,
        ds,
        cell,
        make_settings(processing_level=unit_level, buffer=buffer),
    )
    lon = ds["lon"].values.ravel()
    lat = ds["lat"].values.ravel()
    geod = pyproj.Geod(ellps="WGS84")
    ring_level = utils.chunk_buffer_ring_level(buffer)
    # the processing cells of the chunk, one inside and one at the edge
    first = int(cell) * 4 ** (unit_level - CHUNK_LEVEL)
    for unit_id in (first, first + 4 ** (unit_level - CHUNK_LEVEL) - 1):
        vlon, vlat = healpix_geo.nested.vertices(
            np.array([unit_id], dtype=np.uint64), unit_level, "WGS84", step=8
        )
        vlon, vlat = vlon.ravel(), vlat.ravel()
        inside = (
            healpix_geo.nested.lonlat_to_healpix(lon, lat, unit_level, "WGS84")
            == unit_id
        )
        # distance of every input point to the nearest sampled edge point
        dist = np.full(lon.shape, np.inf)
        for x, y in zip(vlon, vlat):
            _, _, d = geod.inv(np.full(lon.shape, x), np.full(lon.shape, y), lon, lat)
            dist = np.minimum(dist, d)
        # 0.95: the edge is sampled, the distance to it is slightly overestimated
        expected = inside | (dist <= 0.95 * buffer)

        selected = converter._query_chunk_input_points_latlon_curvilinear(
            np.uint64(unit_id)
        )
        got = np.zeros(lon.shape, dtype=bool)
        if selected is not None:
            keys = set(
                zip(selected["lon"].values.tolist(), selected["lat"].values.tolist())
            )
            got = np.array([(x, y) in keys for x, y in zip(lon.tolist(), lat.tolist())])
        assert expected.any()
        assert got[expected].all(), (
            f"{(expected & ~got).sum()} of {expected.sum()} points within {buffer} m "
            f"of the processing cell {unit_id} are not selected (level {unit_level}, "
            f"ring level {ring_level})"
        )


@pytest.mark.parametrize("unit_level, buffer, relation", RING_CASES)
def test_warning_if_the_buffer_ring_is_coarser_than_the_processing_cells(
    monkeypatch, cell, unit_level, buffer, relation
):
    """One warning when the converter is created, not one per task."""
    assert ring_relation(buffer, unit_level) == relation
    ds = make_input(cell, rectilinear=False)
    with structlog.testing.capture_logs() as logs:
        converter = make_converter(
            monkeypatch,
            ds,
            cell,
            make_settings(processing_level=unit_level, buffer=buffer),
        )
        for unit_index in range(2):
            unit_cell_id, _ = converter.unit_cell_ids(0, unit_index)
            converter.query_input_points(unit_cell_id)
    warnings = [log for log in logs if log["log_level"] == "warning"]
    assert len(warnings) == (1 if relation == COARSER else 0)
    if warnings:
        assert "coarser than the processing cells" in warnings[0]["event"]


def test_no_warning_without_processing_level_or_without_buffer(monkeypatch, cell):
    ds = make_input(cell, rectilinear=False)
    with structlog.testing.capture_logs() as logs:
        make_converter(monkeypatch, ds, cell, make_settings(buffer=WIDE_BUFFER))
        make_converter(
            monkeypatch, ds, cell, make_settings(processing_level=9, buffer=0)
        )
    assert not [log for log in logs if log["log_level"] == "warning"]


def test_second_chunk_lands_in_the_second_slice(monkeypatch, ds, cell):
    # two chunk cells with the same parent
    cells = np.array([cell & ~np.uint64(3), (cell & ~np.uint64(3)) + np.uint64(1)])
    settings = make_settings(processing_level=CHUNK_LEVEL + 1, buffer=0)
    converter = make_converter(monkeypatch, ds, cells, settings)
    assert converter.chunk_cell_ids.size == 2
    size = converter.cell_dim_chunk_size
    for i in (1, 0):
        converter.convert(i)
    out = arrays(converter)
    assert out["field"].shape == (2 * size,)
    for i in (0, 1):
        single = make_converter(monkeypatch, ds, cells[i], settings)
        single.convert(0)
        expected = arrays(single)
        assert expected["cell_ids"][0] == out["cell_ids"][i * size]
        part = slice(i * size, (i + 1) * size)
        np.testing.assert_array_equal(out["cell_ids"][part], expected["cell_ids"])
        np.testing.assert_array_equal(out["field"][part], expected["field"])
    assert np.isfinite(out["field"][:size]).any()
    assert np.isfinite(out["field"][size:]).any()

    # the same with the task graph helper: two write tasks, in chunk order, with
    # the same priority for the tasks of a chunk, higher for the first chunk
    client = SynchronousClient()
    other = make_converter(monkeypatch, ds, cells, settings)
    _map_processing_cells(client, other, [0, 1], 4, None)
    assert [kind for kind, _ in client.calls] == ["map", "submit", "map", "submit"]
    priorities = [kw["priority"] for _, kw in client.calls]
    assert priorities[0] == priorities[1] > priorities[2] == priorities[3]
    got = arrays(other)
    np.testing.assert_array_equal(got["field"], out["field"])
    np.testing.assert_array_equal(got["cell_ids"], out["cell_ids"])


def test_nan_of_the_resampler_becomes_the_fill_value(monkeypatch, cell):
    # an integer output has a fill value that is not NaN: NaN of the resampler
    # (cells with no input within reach) must be replaced by it
    ds = make_input(cell)
    ds = ds.assign(field=(1000 + 100 * ds.field).astype("int16"))
    converter = make_converter(
        monkeypatch,
        ds.isel(lat=slice(0, 25)),
        cell,
        make_settings(processing_level=CHUNK_LEVEL + 1, buffer=0, resampler="nearest"),
    )
    zarray = converter.output_arrays["field"]
    assert zarray.dtype == np.int16 and zarray.fill_value == 0
    converter.convert(0)
    out = converter.output_arrays["field"][:]
    # real values are all > 0, so zeros are cells without data
    assert (out >= 0).all()
    assert (out == 0).any() and (out > 0).any()
