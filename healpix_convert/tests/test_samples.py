"""Binning of 1-D samples onto sparse HEALPix cells (``healpix_convert.samples``)."""

from __future__ import annotations

import healpix_geo
import numpy as np
import pytest
import xarray as xr

from healpix_convert.samples import (
    bin_samples,
    split_by_parent,
    write_cell_dataset,
)

LEVEL = 10
NAMES = {"lon": "lon", "lat": "lat"}


def _samples(n=40, nz=6, seed=0):
    """A track of n samples with a vertical axis; several samples share a cell."""
    rng = np.random.default_rng(seed)
    lat = np.linspace(10.0, 10.2, n)  # ~22 km: with level 10 (6.4 km) ~ 12 samples a cell
    lon = np.linspace(20.0, 20.3, n)
    ds = xr.Dataset(
        {
            "lon": ("t", lon),
            "lat": ("t", lat),
            "temperature": (("t", "z"), rng.normal(280, 5, (n, nz)).astype("f4"), {"units": "K"}),
            "kind": (("t", "z"), rng.integers(0, 4, (n, nz)).astype("i1")),
            "time": ("t", np.datetime64("2026-01-01") + np.arange(n) * np.timedelta64(1, "s")),
            "height": (("t", "z"), np.tile(np.arange(nz, dtype="f4"), (n, 1))),
            "classes": ("c", np.array(["a", "b"])),
            "scalar": ((), np.float32(3.0)),
        },
        attrs={"title": "demo"},
    )
    return ds


METHODS = {"temperature": "mean", "kind": "mode", "time": "first", "height": "first"}


def _cells(ds):
    return healpix_geo.nested.lonlat_to_healpix(
        ds.lon.values, ds.lat.values, LEVEL, ellipsoid="WGS84"
    )


def test_counts_add_up_and_cells_are_sorted():
    ds = _samples()
    out = bin_samples(ds, sample_dim="t", level=LEVEL, methods=METHODS, lon="lon", lat="lat")
    expected, counts = np.unique(_cells(ds), return_counts=True)
    np.testing.assert_array_equal(out.cell_ids.values, expected)
    np.testing.assert_array_equal(out.sample_count.values, counts)
    assert out.sample_count.sum() == 40 == out.attrs["n_samples_binned"] == out.attrs["n_samples_input"]
    assert out.cell_ids.dtype == np.uint64 and out.sample_count.dtype == np.int32
    assert (np.diff(out.cell_ids.values.astype("int64")) > 0).all()
    assert out.sizes["cells"] < 40  # several samples share a cell


def test_dims_dtypes_attrs_and_carried_variables():
    ds = _samples()
    out = bin_samples(ds, sample_dim="t", level=LEVEL, methods=METHODS, lon="lon", lat="lat")
    assert out.temperature.dims == ("cells", "z")
    assert out.sizes["z"] == 6
    assert out.temperature.dtype == np.float32
    assert out.kind.dtype == np.int8
    assert out.time.dtype == ds.time.dtype
    assert out.temperature.attrs == {"units": "K"}
    assert out.attrs["title"] == "demo"
    # no sample dimension: copied
    np.testing.assert_array_equal(out.classes.values, ds.classes.values)
    assert float(out.scalar) == 3.0
    # the position variables are not binned unless asked
    assert "lon" not in out and "lat" not in out


def test_mean_first_and_mode_values():
    ds = _samples()
    out = bin_samples(ds, sample_dim="t", level=LEVEL, methods=METHODS, lon="lon", lat="lat")
    cells = _cells(ds)
    for i, cell in enumerate(out.cell_ids.values):
        members = np.nonzero(cells == cell)[0]
        np.testing.assert_allclose(
            out.temperature.values[i], ds.temperature.values[members].mean(axis=0), rtol=1e-6
        )
        assert out.time.values[i] == ds.time.values[members[0]]  # the first sample
        for z in range(6):
            values, freq = np.unique(ds.kind.values[members, z], return_counts=True)
            assert freq[values == out.kind.values[i, z]][0] == freq.max()


def test_mean_ignores_nan_and_keeps_all_nan_cells_nan():
    ds = xr.Dataset(
        {
            "lon": ("t", [0.0, 0.0, 50.0]),
            "lat": ("t", [0.0, 0.0, 50.0]),
            "v": (("t", "z"), np.array([[1.0, np.nan], [3.0, np.nan], [7.0, 8.0]], dtype="f4")),
        }
    )
    out = bin_samples(ds, sample_dim="t", level=8, methods={"v": "mean"}, lon="lon", lat="lat")
    by_count = {int(c): row for c, row in zip(out.sample_count.values, out.v.values)}
    np.testing.assert_array_equal(by_count[2][0], 2.0)
    assert np.isnan(by_count[2][1])  # only NaN in that cell: stays NaN
    np.testing.assert_array_equal(by_count[1], [7.0, 8.0])


@pytest.mark.parametrize(
    "method,expected",
    [("min", 1.0), ("max", 5.0), ("sum", 9.0), ("last", 5.0), ("first", 1.0), ("rms", np.sqrt(35 / 3))],
)
def test_other_methods(method, expected):
    ds = xr.Dataset(
        {"lon": ("t", [1.0, 1.0, 1.0]), "lat": ("t", [1.0, 1.0, 1.0]), "v": ("t", [1.0, 3.0, 5.0])}
    )
    out = bin_samples(ds, sample_dim="t", level=5, methods={"v": method}, lon="lon", lat="lat")
    assert out.v.values.tolist() == pytest.approx([expected])


def test_mode_ties_pick_the_first_and_nan_is_not_counted():
    ds = xr.Dataset(
        {
            "lon": ("t", [0.0] * 4),
            "lat": ("t", [0.0] * 4),
            "a": ("t", np.array([3, 2, 2, 3], dtype="i2")),
            "b": ("t", np.array([np.nan, 5.0, np.nan, np.nan])),
        }
    )
    out = bin_samples(ds, sample_dim="t", level=5, methods={"a": "mode", "b": "mode"}, lon="lon", lat="lat")
    assert out.a.values.tolist() == [3]  # 3 and 2 tie: the first value met
    assert out.b.values.tolist() == [5.0]


def test_mode_with_many_samples_in_one_cell():
    n = 100
    values = np.where(np.arange(n) % 3 == 0, 7, 9).astype("i1")  # 9 is the mode
    ds = xr.Dataset({"lon": ("t", np.zeros(n)), "lat": ("t", np.zeros(n)), "v": ("t", values)})
    out = bin_samples(ds, sample_dim="t", level=3, methods={"v": "mode"}, lon="lon", lat="lat")
    assert out.v.values.tolist() == [9]


def test_invalid_positions_are_not_binned_and_are_counted():
    ds = xr.Dataset(
        {
            "lon": ("t", [0.0, np.nan, 1.0, 2.0]),
            "lat": ("t", [0.0, 0.0, np.nan, 95.0]),
            "v": ("t", [1.0, 2.0, 3.0, 4.0]),
        }
    )
    out = bin_samples(ds, sample_dim="t", level=6, methods={"v": "mean"}, lon="lon", lat="lat")
    assert out.attrs["n_samples_input"] == 4 and out.attrs["n_samples_binned"] == 1
    assert out.sample_count.sum() == 1


def test_variable_without_method_is_an_error_unless_a_default_is_given():
    ds = _samples()
    methods = {k: v for k, v in METHODS.items() if k != "height"}
    with pytest.raises(ValueError, match="'height'.*no method"):
        bin_samples(ds, sample_dim="t", level=LEVEL, methods=methods, lon="lon", lat="lat")
    out = bin_samples(
        ds, sample_dim="t", level=LEVEL, methods=methods, default_method="first", lon="lon", lat="lat"
    )
    assert "height" in out
    out = bin_samples(
        ds, sample_dim="t", level=LEVEL, methods={**methods, "height": "drop"}, lon="lon", lat="lat"
    )
    assert "height" not in out


@pytest.mark.parametrize(
    "methods,match",
    [
        ({"temperature": "median"}, "unknown method"),
        ({"nothing": "mean"}, "does not have"),
        ({"time": "mean"}, "only be reduced"),
    ],
)
def test_bad_methods(methods, match):
    with pytest.raises(ValueError, match=match):
        bin_samples(
            _samples(),
            sample_dim="t",
            level=LEVEL,
            methods={**METHODS, **methods},
            lon="lon",
            lat="lat",
        )


def test_a_method_for_a_variable_without_the_sample_dimension_is_ignored():
    out = bin_samples(
        _samples(),
        sample_dim="t",
        level=LEVEL,
        methods={**METHODS, "classes": "mean"},
        lon="lon",
        lat="lat",
    )
    assert out.classes.values.tolist() == ["a", "b"]


def test_time_min_max():
    ds = _samples()
    out = bin_samples(
        ds,
        sample_dim="t",
        level=LEVEL,
        methods={**METHODS, "time": "max"},
        lon="lon",
        lat="lat",
    )
    cells = _cells(ds)
    for i, cell in enumerate(out.cell_ids.values):
        assert out.time.values[i] == ds.time.values[cells == cell].max()


def test_split_by_parent_keeps_every_cell_once():
    ds = _samples(n=400)
    ds["lon"] = ("t", np.linspace(20.0, 24.0, 400))
    ds["lat"] = ("t", np.linspace(10.0, 14.0, 400))
    out = bin_samples(ds, sample_dim="t", level=LEVEL, methods=METHODS, lon="lon", lat="lat")
    tiles = split_by_parent(out, level=LEVEL, parent_level=5)
    assert len(tiles) > 1
    assert sum(t.sizes["cells"] for t in tiles.values()) == out.sizes["cells"]
    assert sum(int(t.sample_count.sum()) for t in tiles.values()) == 400
    for parent, tile in tiles.items():
        assert (tile.cell_ids.values >> np.uint64(2 * (LEVEL - 5)) == parent).all()
        assert tile.sizes["z"] == 6
        np.testing.assert_array_equal(tile.classes.values, out.classes.values)
    assert list(tiles) == sorted(tiles)
    with pytest.raises(ValueError, match="parent_level"):
        split_by_parent(out, level=LEVEL, parent_level=LEVEL + 1)


def test_write_cell_dataset_roundtrip_and_metadata(tmp_path):
    ds = _samples()
    out = bin_samples(ds, sample_dim="t", level=LEVEL, methods=METHODS, lon="lon", lat="lat")
    path = str(tmp_path / "tile.zarr")
    write_cell_dataset(out, path, level=LEVEL, group="samples", root_attrs={"origin": "test"})

    back = xr.open_dataset(path, engine="zarr", group="samples", consolidated=True)
    assert back.temperature.dims == ("z", "cells")  # the cell dimension last, as the converters
    np.testing.assert_array_equal(back.cell_ids.values, out.cell_ids.values)
    np.testing.assert_array_equal(back.sample_count.values, out.sample_count.values)
    np.testing.assert_allclose(back.temperature.values.T, out.temperature.values)
    assert back.attrs["dggs"]["refinement_level"] == LEVEL
    assert back.attrs["dggs"]["indexing_scheme"] == "nested"
    assert back.attrs["dggs"]["ellipsoid"]["name"] == "WGS84"
    assert back.attrs["dggs"]["coordinate"] == "cell_ids"
    assert back.attrs["dggs"]["spatial_dimension"] == "cells"
    assert back.attrs["zarr_conventions"][0]["name"] == "dggs"
    assert back.cell_ids.attrs["standard_name"] == "healpix_index"

    import zarr

    root = zarr.open_group(path, mode="r")
    assert root.attrs["origin"] == "test"
    assert root.attrs["Conventions"].startswith("CF-")


def test_write_cell_dataset_cell_axis_first_and_chunks(tmp_path):
    ds = _samples()
    out = bin_samples(ds, sample_dim="t", level=LEVEL, methods=METHODS, lon="lon", lat="lat")
    path = str(tmp_path / "tile.zarr")
    write_cell_dataset(out, path, level=LEVEL, cell_axis="first", chunks=2, ellipsoid="sphere")
    back = xr.open_zarr(path, consolidated=True)
    assert back.temperature.dims == ("cells", "z")
    assert "ellipsoid" not in back.attrs["dggs"]
    assert back.temperature.encoding["chunks"][0] == 2


def test_dtypes_casts_the_reduced_values():
    ds = xr.Dataset(
        {"lon": ("t", [0.0, 0.0]), "lat": ("t", [0.0, 0.0]), "p": ("t", np.array([10, 21], dtype="i1"))}
    )
    out = bin_samples(
        ds, sample_dim="t", level=5, methods={"p": "mean"}, dtypes={"p": "float32"}, lon="lon", lat="lat"
    )
    assert out.p.dtype == np.float32 and out.p.values.tolist() == [15.5]
    out = bin_samples(ds, sample_dim="t", level=5, methods={"p": "mean"}, lon="lon", lat="lat")
    assert out.p.dtype == np.float64


def _one_cell(**variables):
    n = len(next(iter(variables.values())))
    return xr.Dataset({"lon": ("t", np.zeros(n)), "lat": ("t", np.zeros(n)), **{k: ("t", np.asarray(v)) for k, v in variables.items()}})


@pytest.mark.parametrize("method,expected", [("first", 2.0), ("last", 4.0)])
def test_first_and_last_skip_nan(method, expected):
    ds = _one_cell(v=[np.nan, 2.0, 3.0, 4.0, np.nan])
    out = bin_samples(ds, sample_dim="t", level=5, methods={"v": method}, lon="lon", lat="lat")
    assert out.v.values.tolist() == [expected]
    all_nan = _one_cell(v=[np.nan, np.nan])
    out = bin_samples(all_nan, sample_dim="t", level=5, methods={"v": method}, lon="lon", lat="lat")
    assert np.isnan(out.v.values).all()


def test_first_skips_nat_per_column():
    t = np.array(["NaT", "2026-01-02", "2026-01-03"], dtype="datetime64[ns]")
    out = bin_samples(_one_cell(time=t), sample_dim="t", level=5, methods={"time": "first"}, lon="lon", lat="lat")
    assert out.time.values[0] == t[1]


def test_fill_values_are_left_out_of_every_reduction():
    ds = _one_cell(
        a=np.array([-127, 5, 7, 5], dtype="i1"),
        b=np.array([-127, 5, 9], dtype="i1")[:3].tolist() + [-127],
    )
    ds["b"] = ds["b"].astype("i1")
    fills = {"a": -127, "b": -127}
    out = bin_samples(ds, sample_dim="t", level=5, methods={"a": "first", "b": "max"}, fill_values=fills, lon="lon", lat="lat")
    assert out.a.dtype == np.int8 and out.a.values.tolist() == [5]
    assert out.b.values.tolist() == [9]
    out = bin_samples(ds, sample_dim="t", level=5, methods={"a": "mean", "b": "mode"}, fill_values=fills, lon="lon", lat="lat")
    assert out.a.dtype == np.float64 and out.a.values.tolist() == pytest.approx([17 / 3])
    assert out.b.dtype == np.int8 and out.b.values.tolist() == [5]
    # a cell with nothing valid: the fill value, not 0 and not NaN
    only_fill = _one_cell(a=np.array([-127, -127], dtype="i1"))
    out = bin_samples(only_fill, sample_dim="t", level=5, methods={"a": "first"}, fill_values={"a": -127}, lon="lon", lat="lat")
    assert out.a.values.tolist() == [-127] and out.a.dtype == np.int8
    out = bin_samples(only_fill, sample_dim="t", level=5, methods={"a": "mean"}, fill_values={"a": -127}, lon="lon", lat="lat")
    assert np.isnan(out.a.values).all()


def test_valid_positions_is_the_rule_of_bin_samples():
    from healpix_convert.samples import valid_positions

    lon = np.array([0.0, np.nan, 1.0, 2.0, 400.0])
    lat = np.array([0.0, 0.0, np.nan, 95.0, 0.0])
    ds = xr.Dataset({"lon": ("t", lon), "lat": ("t", lat), "v": ("t", np.arange(5.0))})
    out = bin_samples(ds, sample_dim="t", level=6, methods={"v": "mean"}, lon="lon", lat="lat")
    assert valid_positions(lon, lat).tolist() == [True, False, False, False, False]
    assert int(out.sample_count.sum()) == int(valid_positions(lon, lat).sum())


def test_no_valid_sample_gives_an_empty_dataset():
    ds = xr.Dataset(
        {
            "lon": ("t", [np.nan, np.nan]),
            "lat": ("t", [np.nan, np.nan]),
            "v": (("t", "z"), np.ones((2, 4), dtype="f4")),
            "k": ("t", np.ones(2, dtype="i1")),
            "s": ((), 3),
        }
    )
    out = bin_samples(ds, sample_dim="t", level=6, methods={"v": "mean", "k": "mode"}, lon="lon", lat="lat")
    assert out.sizes["cells"] == 0 and out.sizes["z"] == 4
    assert out.sample_count.shape == (0,) and out.attrs["n_samples_binned"] == 0
    assert out.v.dtype == np.float32 and out.k.dtype == np.int8
    assert int(out.s) == 3


def test_chunks_zero_is_refused(tmp_path):
    out = bin_samples(_samples(), sample_dim="t", level=LEVEL, methods=METHODS, lon="lon", lat="lat")
    with pytest.raises(ValueError, match="chunks"):
        write_cell_dataset(out, str(tmp_path / "x.zarr"), level=LEVEL, chunks=0)


def test_float_to_integer_cast_rounds_and_handles_nan():
    ds = _one_cell(p=np.array([1.0, 2.0, np.nan]))
    out = bin_samples(ds, sample_dim="t", level=5, methods={"p": "mean"}, dtypes={"p": "i2"}, lon="lon", lat="lat")
    assert out.p.values.tolist() == [2] and out.p.dtype == np.int16  # 1.5 rounds to 2
    nan_only = _one_cell(p=np.array([np.nan, np.nan]))
    with pytest.raises(ValueError, match="NaN cannot be cast"):
        bin_samples(nan_only, sample_dim="t", level=5, methods={"p": "mean"}, dtypes={"p": "i2"}, lon="lon", lat="lat")
    out = bin_samples(nan_only, sample_dim="t", level=5, methods={"p": "mean"}, dtypes={"p": "i2"}, fill_values={"p": -9}, lon="lon", lat="lat")
    assert out.p.values.tolist() == [-9]
