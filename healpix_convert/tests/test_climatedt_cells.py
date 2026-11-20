"""ClimateDT generic building blocks: GRIB / CoverageJSON readers, sphere → WGS84, cell stores, Polytope."""

import json
from pathlib import Path

import healpix_geo
import numpy as np
import pytest
import zarr

pytest.importorskip("eccodes")
pytest.importorskip("healpix_resample")

from healpix_convert.converters.climatedt import (  # noqa: E402
    SphereToEllipsoid,
    check_grib_params,
    polytope_retrieve,
    read_covjson_points,
    read_healpix_grib,
    write_cell_store,
)

G4E = Path("/Users/todaka/python/git/G4E")
OCEAN = G4E / "pr-drafts/climatedt-ocean-l10-test-2026-11-16/data"
REAL_OCEAN_GRIB = OCEAN / "full3_20260101.grib"
REAL_COVJSON = OCEAN / "bbox_20260101.json"
REAL_ATMOS_GRIB = (
    G4E / "pr-drafts/production-2026-11-18/climatedt/origin-data/g2_ifsnemo_2t_high.grib"
)

LEVEL = 4  # Nside 16, 3072 cells: fast
NSIDE = 2**LEVEL
NCELL = 12 * NSIDE**2
MISSING = 9999.0


# ── synthetic inputs ─────────────────────────────────────────────────────────


def sphere_lonlat(level: int = LEVEL):
    cells = np.arange(12 * 4**level, dtype="uint64")
    lon, lat = healpix_geo.nested.healpix_to_lonlat(cells, level, ellipsoid="sphere")
    return cells, np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)


def smooth_field(lon, lat):
    return 280.0 + 20.0 * np.cos(np.deg2rad(lat)) + 5.0 * np.sin(np.deg2rad(lon))


def grib_message(
    values,
    *,
    param_id,
    date=20260101,
    time=0,
    avg=False,
    layer=None,
    nside=NSIDE,
    ordering="nested",
    step=(0, 24),
    raw=None,
):
    """One HEALPix GRIB2 message (NaN in ``values`` → bitmap, as in the real files)."""
    import eccodes as ec

    h = ec.codes_grib_new_from_samples("GRIB2")
    try:
        ec.codes_set(h, "gridType", "healpix")
        ec.codes_set(h, "Nside", nside)
        ec.codes_set(h, "orderingConvention", ordering)
        ec.codes_set(h, "dataDate", date)
        ec.codes_set(h, "dataTime", time)
        ec.codes_set(h, "paramId", param_id)
        if avg:
            ec.codes_set(h, "productDefinitionTemplateNumber", 8)
            ec.codes_set(h, "typeOfStatisticalProcessing", 0)
            ec.codes_set(h, "forecastTime", step[0])
            ec.codes_set(h, "lengthOfTimeRange", step[1] - step[0])
            ec.codes_set(h, "indicatorOfUnitForTimeRange", 1)
            ec.codes_set(h, "endStep", step[1])
        for key, value in (raw or {}).items():  # raw keys, in order (eccodes is order-sensitive)
            ec.codes_set(h, key, value)
        if layer is not None:
            ec.codes_set(h, "typeOfFirstFixedSurface", 168)  # oceanModelLayer
            ec.codes_set(h, "level", layer)
        else:
            ec.codes_set(h, "level", 2)
        values = np.asarray(values, dtype=float)
        ec.codes_set(h, "numberOfDataPoints", values.size)
        if np.isnan(values).any():
            ec.codes_set(h, "bitmapPresent", 1)
            ec.codes_set(h, "missingValue", MISSING)
            values = np.where(np.isnan(values), MISSING, values)
        ec.codes_set_values(h, values)
        return ec.codes_get_message(h)
    finally:
        ec.codes_release(h)


def write_grib(path, messages):
    path.write_bytes(b"".join(messages))
    return path


def ocean_field(layer: int):
    _, lon, lat = sphere_lonlat()
    f = smooth_field(lon, lat) + layer
    f[lat > 30.0] = np.nan  # land
    return f


@pytest.fixture
def ocean_grib(tmp_path):
    return write_grib(
        tmp_path / "ocean.grib",
        [grib_message(ocean_field(k), param_id=263501, avg=True, layer=k) for k in (1, 2, 3)],
    )


@pytest.fixture
def atmos_grib(tmp_path):
    _, lon, lat = sphere_lonlat()
    f = smooth_field(lon, lat)
    return write_grib(
        tmp_path / "atmos.grib",
        [
            grib_message(f, param_id=167, time=0),
            grib_message(f + 1.0, param_id=167, time=600),
        ],
    )


# ── Polytope ─────────────────────────────────────────────────────────────────


class FakeClient:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail
        self.final = None

    def retrieve(self, collection, request, path):
        self.calls.append((collection, request, path))
        assert path.endswith(".part")
        assert self.final is not None and not self.final.exists()
        Path(path).write_bytes(b"GRIB-ish")
        if self.fail:
            raise RuntimeError("boom")


def test_polytope_retrieve_writes_part_then_renames(tmp_path):
    out = tmp_path / "sub" / "f.grib"
    client = FakeClient()
    client.final = out
    request = {"param": "167"}
    res = polytope_retrieve(request, "host.example", out, client=client)
    assert res == out and out.read_bytes() == b"GRIB-ish"
    assert list((tmp_path / "sub").glob("*.part")) == []
    (path,) = [c[2] for c in client.calls]
    assert path.startswith(str(out) + ".") and path.endswith(".part") and path != str(out) + ".part"
    assert client.calls == [("destination-earth", request, path)]


def test_polytope_retrieve_temp_names_are_unique(tmp_path):
    out = tmp_path / "f.grib"
    paths = []
    for _ in range(2):
        client = FakeClient()
        client.final = out
        polytope_retrieve({}, "h", out, client=client)
        paths.append(client.calls[0][2])
        out.unlink()
    assert paths[0] != paths[1]
    # another worker's partial file is left alone
    other = tmp_path / "f.grib.1.deadbeef.part"
    other.write_bytes(b"x")
    client = FakeClient()
    client.final = out
    polytope_retrieve({}, "h", out, client=client)
    assert other.exists()


def test_polytope_retrieve_collection_and_failure(tmp_path):
    out = tmp_path / "f.grib"
    client = FakeClient(fail=True)
    client.final = out
    with pytest.raises(RuntimeError, match="boom"):
        polytope_retrieve({}, "h", out, collection="other", client=client)
    assert client.calls[0][0] == "other"
    assert not out.exists() and list(tmp_path.glob("*.part")) == []


# ── GRIB ─────────────────────────────────────────────────────────────────────


def test_read_grib_ocean_layers_mask_and_times(ocean_grib):
    ds = read_healpix_grib(ocean_grib)
    assert ds.attrs == {
        "healpix_nside": NSIDE,
        "healpix_level": LEVEL,
        "healpix_ordering": "nested",
        "healpix_ellipsoid": "sphere",
    }
    v = ds["avg_thetao"]
    assert v.dims == ("time", "layer", "cells")
    assert v.shape == (1, 3, NCELL) and v.dtype == np.float32
    np.testing.assert_array_equal(ds["layer"].values, [1, 2, 3])
    assert ds["layer"].dtype.kind == "i"
    # bitmap → NaN, same mask in every layer, values in nested order
    for k in range(3):
        expected = ocean_field(k + 1).astype(np.float32)
        np.testing.assert_array_equal(np.isnan(v.values[0, k]), np.isnan(expected))
        np.testing.assert_allclose(v.values[0, k], expected, equal_nan=True, rtol=1e-6)
    assert np.isnan(v.values).any() and np.isfinite(v.values).any()
    # averaging period [start, end)
    assert ds["time"].values[0] == np.datetime64("2026-01-01T00:00", "ns")
    assert ds["time_bounds"].dims == ("time", "bnds")
    np.testing.assert_array_equal(
        ds["time_bounds"].values[0],
        np.array(["2026-01-01T00:00", "2026-01-02T00:00"], dtype="datetime64[ns]"),
    )
    assert v.attrs["paramId"] == 263501
    assert v.attrs["units"] == "K"
    assert v.attrs["stepType"] == "avg"
    assert v.attrs["typeOfLevel"] == "oceanModelLayer"
    assert "GRIB_stepRange" not in v.attrs
    assert ds["step_range"].dims == ("time",) and ds["step_range"].values[0] == "0-24"
    assert v.attrs["long_name"]


def test_read_grib_instant_times_and_param_filter(atmos_grib, ocean_grib, tmp_path):
    ds = read_healpix_grib(atmos_grib)
    v = ds["2t"]
    assert v.dims == ("time", "cells") and v.shape == (2, NCELL)
    np.testing.assert_array_equal(
        ds["time"].values,
        np.array(["2026-01-01T00:00", "2026-01-01T06:00"], dtype="datetime64[ns]"),
    )
    assert "time_bounds" not in ds and "layer" not in ds
    assert not np.isnan(v.values).any()
    np.testing.assert_allclose(v.values[1] - v.values[0], 1.0, rtol=1e-4)
    # several files, one parameter kept
    both = read_healpix_grib([atmos_grib, ocean_grib], param_ids=[263501])
    assert list(both.data_vars) == ["avg_thetao"]


def test_read_grib_refuses_ring_ordering(tmp_path):
    _, lon, lat = sphere_lonlat()
    p = write_grib(
        tmp_path / "ring.grib",
        [grib_message(smooth_field(lon, lat), param_id=167, ordering="ring")],
    )
    with pytest.raises(ValueError, match="nested"):
        read_healpix_grib(p)


def test_check_grib_params(ocean_grib, atmos_grib):
    info = check_grib_params([ocean_grib], [263501])
    assert info["messages"] == 3 and info["param_ids"] == {263501: 3}
    assert info["dates"] == ["2026-01-01T00:00"]
    assert info["grid"] == {
        "gridType": "healpix",
        "Nside": NSIDE,
        "orderingConvention": "nested",
    }
    info = check_grib_params(atmos_grib, [167])
    assert info["messages"] == 2 and len(info["dates"]) == 2
    with pytest.raises(ValueError, match="unexpected paramId"):
        check_grib_params([ocean_grib], [167])
    with pytest.raises(ValueError, match="263501"):
        check_grib_params([atmos_grib], [167, 263501])
    with pytest.raises(ValueError, match="unexpected paramId"):
        check_grib_params([ocean_grib, atmos_grib], [263501])


def test_check_grib_params_missing(atmos_grib):
    with pytest.raises(ValueError, match="not found"):
        check_grib_params([atmos_grib], [167, 235055])


def _flat(v=0.0):
    return np.full(NCELL, v)


def test_read_grib_refuses_two_ends_for_one_start(tmp_path):
    p = write_grib(
        tmp_path / "ends.grib",
        [
            grib_message(_flat(1), param_id=167, avg=True, step=(0, 24)),
            grib_message(_flat(2), param_id=235055, avg=True, step=(0, 48)),
        ],
    )
    with pytest.raises(ValueError, match="two averaging periods"):
        read_healpix_grib(p)


def test_read_grib_refuses_one_name_two_param_ids(tmp_path):
    # paramIds 141 and 228141 are both the shortName "sd"
    p = write_grib(
        tmp_path / "names.grib",
        [grib_message(_flat(1), param_id=141), grib_message(_flat(2), param_id=228141, time=600)],
    )
    with pytest.raises(ValueError, match="two paramIds"):
        read_healpix_grib(p)


def test_read_grib_times_selects_before_decoding(tmp_path, monkeypatch):
    import eccodes as ec

    p = write_grib(
        tmp_path / "t.grib",
        [grib_message(_flat(k), param_id=167, time=100 * k) for k in range(4)],
    )
    decoded = []
    real = ec.codes_get_values
    monkeypatch.setattr(ec, "codes_get_values", lambda h: decoded.append(1) or real(h))
    ds = read_healpix_grib(p, times=[np.datetime64("2026-01-01T02:00")])
    assert len(decoded) == 1  # one message decoded, once
    assert ds["2t"].shape == (1, NCELL)
    np.testing.assert_allclose(ds["2t"].values, 2.0)
    assert ds["time"].values[0] == np.datetime64("2026-01-01T02:00", "ns")
    with pytest.raises(ValueError, match="not in the GRIB"):
        read_healpix_grib(p, times=[np.datetime64("2026-03-01T00:00")])
    # a bitmap message is decoded once too
    decoded.clear()
    pm = write_grib(tmp_path / "m.grib", [grib_message(ocean_field(1), param_id=263501, layer=1)])
    ds = read_healpix_grib(pm)
    assert len(decoded) == 1 and np.isnan(ds["avg_thetao"].values).any()


def test_read_grib_monthly_means_in_hours(tmp_path):
    p = write_grib(
        tmp_path / "month_h.grib",
        [
            grib_message(_flat(1), param_id=167, avg=True, date=20260101, step=(0, 744)),
            grib_message(_flat(2), param_id=167, avg=True, date=20260101, step=(744, 1416)),
        ],
    )
    ds = read_healpix_grib(p)
    np.testing.assert_array_equal(
        ds["time"].values, np.array(["2026-01-01", "2026-02-01"], dtype="datetime64[ns]")
    )
    np.testing.assert_array_equal(
        ds["time_bounds"].values,
        np.array(
            [["2026-01-01", "2026-02-01"], ["2026-02-01", "2026-03-01"]], dtype="datetime64[ns]"
        ),
    )
    # one stepRange per time, not the first message's for all
    assert list(ds["step_range"].values) == ["0-744", "744-1416"]
    assert "GRIB_stepRange" not in ds["avg_2t"].attrs
    info = check_grib_params(p, [228004])
    assert info["dates"] == ["2026-01-01T00:00", "2026-02-01T00:00"]


def test_read_grib_monthly_mean_in_months(tmp_path):
    month = {
        "indicatorOfUnitOfTimeRange": 3,
        "forecastTime": 1,
        "indicatorOfUnitForTimeRange": 3,
        "lengthOfTimeRange": 1,
        "yearOfEndOfOverallTimeInterval": 2026,
        "monthOfEndOfOverallTimeInterval": 3,
        "dayOfEndOfOverallTimeInterval": 1,
    }
    p = write_grib(
        tmp_path / "month_m.grib",
        [grib_message(_flat(1), param_id=167, avg=True, date=20260101, raw=month)],
    )
    ds = read_healpix_grib(p)
    # eccodes reports 720 hours here; the calendar month is used instead
    assert ds["time"].values[0] == np.datetime64("2026-02-01", "ns")
    np.testing.assert_array_equal(
        ds["time_bounds"].values[0], np.array(["2026-02-01", "2026-03-01"], dtype="datetime64[ns]")
    )


def test_read_grib_monthly_mean_must_start_on_the_first(tmp_path):
    p = write_grib(
        tmp_path / "month_bad.grib",
        [grib_message(_flat(1), param_id=167, avg=True, date=20260115, step=(0, 744))],
    )
    with pytest.raises(ValueError, match="first day of a month"):
        read_healpix_grib(p)


def test_read_grib_refuses_unsupported_step_units(tmp_path):
    p = write_grib(
        tmp_path / "years.grib",
        [grib_message(_flat(1), param_id=167, avg=True, raw={"indicatorOfUnitOfTimeRange": 4})],
    )
    with pytest.raises(ValueError, match="stepUnits"):
        read_healpix_grib(p)


def test_check_grib_params_refuses_ring_ordering(tmp_path):
    p = write_grib(tmp_path / "ring.grib", [grib_message(_flat(1), param_id=167, ordering="ring")])
    with pytest.raises(ValueError, match="nested"):
        check_grib_params(p, [167])


# ── CoverageJSON ─────────────────────────────────────────────────────────────


def covjson(points, *, levels=(1, 2), name="avg_thetao", extra=None):
    """A Polytope-feature-like CoverageJSON; ``points`` = [(lat, lon)], one value per (level, point)."""
    tuples = [[lat, lon, lev] for lev in levels for lat, lon in points]
    values = [float(100 * lev + i) for lev in levels for i in range(len(points))]
    cov = {
        "type": "Coverage",
        "mars:metadata": {"Forecast date": "2026-01-01T00:00:00Z", "levtype": "o3d"},
        "domain": {
            "type": "Domain",
            "axes": {
                "t": {"values": ["2026-01-01T00:00:00Z"]},
                "composite": {
                    "dataType": "tuple",
                    "coordinates": ["latitude", "longitude", "levelist"],
                    "values": tuples,
                },
            },
        },
        "ranges": {
            name: {
                "type": "NdArray",
                "dataType": "float",
                "shape": [len(values)],
                "axisNames": [name],
                "values": values,
            }
        },
    }
    doc = {
        "type": "CoverageCollection",
        "domainType": "MultiPoint",
        "coverages": [cov],
        "parameters": {
            name: {
                "type": "Parameter",
                "unit": {"symbol": "K"},
                "observedProperty": {"id": name, "label": {"en": "Time-mean temperature"}},
            }
        },
    }
    if extra:
        doc.update(extra)
    return doc


def test_read_covjson_cells_from_lonlat(tmp_path):
    cells, lon, lat = sphere_lonlat()
    pick = np.array([700, 1500, 40, 2999])  # unsorted on purpose
    pts = [(lat[i], lon[i]) for i in pick]
    p = tmp_path / "f.json"
    p.write_text(json.dumps(covjson(pts)))
    ds = read_covjson_points(p, level=LEVEL)
    assert ds["cell_ids"].dtype == np.uint64
    np.testing.assert_array_equal(ds["cell_ids"].values, np.sort(pick))
    v = ds["avg_thetao"]
    assert v.dims == ("time", "layer", "cells") and v.shape == (1, 2, 4)
    np.testing.assert_array_equal(ds["layer"].values, [1, 2])
    assert ds["time"].values[0] == np.datetime64("2026-01-01T00:00", "ns")
    # values follow their point, not the input order
    order = np.argsort(pick)
    np.testing.assert_array_equal(v.values[0, 0], order + 100)
    np.testing.assert_array_equal(v.values[0, 1], order + 200)
    assert v.attrs["units"] == "K"
    assert v.attrs["long_name"] == "Time-mean temperature"
    assert ds.attrs["healpix_level"] == LEVEL and ds.attrs["healpix_ellipsoid"] == "sphere"


def test_read_covjson_refuses_two_points_in_one_cell(tmp_path):
    _, lon, lat = sphere_lonlat()
    i = 1234
    # two distinct points inside the same cell
    pts = [(lat[i], lon[i]), (lat[i] + 0.01, lon[i] + 0.01)]
    p = tmp_path / "dup.json"
    p.write_text(json.dumps(covjson(pts, levels=(1,))))
    with pytest.raises(ValueError, match="same HEALPix cell"):
        read_covjson_points(p, level=LEVEL)
    # the same point twice
    p.write_text(json.dumps(covjson([pts[0], pts[0]], levels=(1,))))
    with pytest.raises(ValueError, match="same HEALPix cell"):
        read_covjson_points(p, level=LEVEL)


def test_read_covjson_same_cell_in_two_layers_is_fine(tmp_path):
    _, lon, lat = sphere_lonlat()
    p = tmp_path / "ok.json"
    p.write_text(json.dumps(covjson([(lat[5], lon[5])], levels=(1, 2, 3))))
    ds = read_covjson_points(p, level=LEVEL)
    assert ds["avg_thetao"].shape == (1, 3, 1)


def test_read_covjson_checks_shape_and_axis_names(tmp_path):
    _, lon, lat = sphere_lonlat()
    doc = covjson([(lat[5], lon[5]), (lat[9], lon[9])], levels=(1,))
    rng = doc["coverages"][0]["ranges"]["avg_thetao"]
    p = tmp_path / "s.json"
    # a matching (t, points) layout is accepted
    rng.update(shape=[1, 2], axisNames=["t", "composite"])
    p.write_text(json.dumps(doc))
    assert read_covjson_points(p, level=LEVEL)["avg_thetao"].shape == (1, 1, 2)
    # points first, time second: not the assumed order
    rng.update(shape=[2, 1], axisNames=["composite", "t"])
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="shape"):
        read_covjson_points(p, level=LEVEL)
    rng.update(shape=[1, 2], axisNames=["composite", "t"])
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="axisNames"):
        read_covjson_points(p, level=LEVEL)


# ── sphere → ellipsoid ───────────────────────────────────────────────────────


@pytest.mark.parametrize("method", ["psf", "conservative"])
def test_constant_field_stays_constant(method):
    cells, _, _ = sphere_lonlat()
    s = SphereToEllipsoid(LEVEL, method=method, clip=None)
    ids, vals = s.resample(cells, np.full(cells.size, 7.5))
    assert vals.dtype == np.float32 and ids.dtype == np.uint64
    assert ids.size > 0.9 * cells.size
    np.testing.assert_allclose(vals, 7.5, rtol=1e-3 if method == "psf" else 1e-5)
    # clipped to the input range: exactly constant
    ids2, vals2 = SphereToEllipsoid(LEVEL, method=method).resample(cells, np.full(cells.size, 7.5))
    np.testing.assert_array_equal(vals2, np.float32(7.5))


@pytest.mark.parametrize("method", ["psf", "conservative"])
def test_masked_inputs_stay_out(method):
    cells, lon, lat = sphere_lonlat()
    field = np.ones(cells.size)
    field[lat > 30.0] = np.nan
    s = SphereToEllipsoid(LEVEL, method=method, clip=None)
    ids, vals = s.resample(cells, field)
    # no zero leaks in from the masked points: the field is 1 wherever it is defined
    np.testing.assert_allclose(vals, 1.0, rtol=1e-3 if method == "psf" else 1e-5)
    lon_o, lat_o = healpix_geo.nested.healpix_to_lonlat(ids, LEVEL, ellipsoid="WGS84")
    assert np.asarray(lat_o).max() < 36.0  # nothing far inside the masked area
    assert ids.size < cells.size
    # a real field: stays inside the input range with the default clip
    f = smooth_field(lon, lat)
    f[lat > 30.0] = np.nan
    _, v2 = SphereToEllipsoid(LEVEL, method=method).resample(cells, f)
    assert np.nanmin(f) - 1e-4 <= v2.min() and v2.max() <= np.nanmax(f) + 1e-4


def test_one_operator_per_mask():
    cells, lon, lat = sphere_lonlat()
    s = SphereToEllipsoid(LEVEL)
    f1 = smooth_field(lon, lat)
    f1[lat > 30.0] = np.nan
    s.resample(cells, f1)
    s.resample(cells, f1 + 1.0)  # same mask, other values
    assert len(s._operators) == 1
    f2 = smooth_field(lon, lat)
    f2[lat > 40.0] = np.nan
    s.resample(cells, f2)
    assert len(s._operators) == 2
    # the operator is reachable by its valid cells
    op = s.operator(cells[np.isfinite(f1)])
    assert op is s.operator(cells[np.isfinite(f1)])
    # nothing valid: empty result, no operator
    ids, vals = s.resample(cells, np.full(cells.size, np.nan))
    assert ids.size == 0 and vals.size == 0 and len(s._operators) == 2


def test_psf_matches_healpix_resample_directly():
    import healpix_resample

    cells, lon, lat = sphere_lonlat()
    f = smooth_field(lon, lat)
    ids, vals = SphereToEllipsoid(LEVEL).resample(cells, f)
    res = healpix_resample.PSFResampler(
        lon_deg=lon, lat_deg=lat, level=LEVEL, ellipsoid="WGS84", threshold=0.5
    ).resample(f, lam=0.0)
    expected = np.clip(np.asarray(res.cell_data), f.min(), f.max()).astype(np.float32)
    np.testing.assert_array_equal(ids, np.asarray(res.cell_ids).astype(np.uint64))
    np.testing.assert_array_equal(vals, expected)


def test_describe():
    d = SphereToEllipsoid(LEVEL).describe()
    assert d["method"] == "psf"
    assert d["init_params"] == {"threshold": 0.5}
    assert d["resample_params"] == {"lam": 0.0}
    assert d["clip"] == "input_range"
    assert d["source_grid"] == f"healpix nested sphere L{LEVEL}"
    assert d["target_grid"] == f"healpix nested WGS84 L{LEVEL}"
    assert d["healpix_resample"]
    c = SphereToEllipsoid(LEVEL, "conservative", clip=None).describe()
    assert c["method"] == "conservative" and c["clip"] is None
    json.dumps(c)
    with pytest.raises(ValueError):
        SphereToEllipsoid(LEVEL, "nearest")


# ── cell store ───────────────────────────────────────────────────────────────

DGGS = {
    "name": "healpix",
    "indexing_scheme": "nested",
    "ellipsoid": "WGS84",
    "spatial_dimension": "cells",
    "coordinate": "cell_ids",
    "compression": "none",
}


def test_write_cell_store_layout(tmp_path):
    pl, lv, cell = 1, 3, 5
    n = 4 ** (lv - pl)
    rng = np.random.default_rng(0)
    a = rng.random(n).astype(np.float32)
    a[:4] = np.nan  # all four children of the first coarse cell
    a[5] = np.nan  # one child of the second
    b = rng.random((2, n)).astype(np.float32)
    b[1, 4:8] = np.nan
    out = tmp_path / "c.zarr"
    stac = {"type": "Feature", "id": "c", "properties": {"datetime": "2026-01-01T00:00:00Z"}}
    res = write_cell_store(
        out,
        parent_cell=cell,
        parent_level=pl,
        level=lv,
        group="measurements",
        variables={"a": a, "b": b},
        var_attrs={"a": {"units": "K", "paramId": np.int64(167)}, "b": {"units": "m"}},
        root_attrs={"grid4earth:version": "p-test"},
        stac_item=stac,
        layer=np.array([1, 2], dtype="int32"),
        chunk_level=2,
        coarsest_level=1,
    )
    assert res == {
        "variables": ["a", "b"],
        "levels": [3, 2, 1],
        "n_valid": {"a": n - 5, "b": 2 * n - 4},
    }
    root = zarr.open_group(str(out), mode="r")
    assert root.metadata.zarr_format == 3
    assert root.metadata.consolidated_metadata is not None
    assert root.attrs["grid4earth:version"] == "p-test"
    assert root.attrs["stac_discovery"] == stac
    g = root["measurements"]
    assert g.metadata.consolidated_metadata is not None
    assert g.attrs["multiscales"] == {
        "layout": [
            {"asset": "3", "dggs": dict(DGGS, refinement_level=3)},
            {"asset": "2", "dggs": dict(DGGS, refinement_level=2)},
            {"asset": "1", "dggs": dict(DGGS, refinement_level=1)},
        ]
    }
    for L, nL in ((3, n), (2, n // 4), (1, n // 16)):
        lg = g[str(L)]
        assert lg.attrs["dggs"] == dict(DGGS, refinement_level=L)
        chunk = 4 ** (L - 2) if L >= 2 else nL  # the coarsest level is one chunk
        arr_a, arr_b = lg["a"], lg["b"]
        assert arr_a.shape == (nL,) and arr_a.dtype == np.float32
        assert arr_a.chunks == (chunk,)
        assert arr_a.metadata.dimension_names == ("cells",)
        assert np.isnan(arr_a.fill_value)
        assert arr_b.shape == (2, nL) and arr_b.chunks == (1, chunk)
        assert arr_b.metadata.dimension_names == ("layer", "cells")
        assert arr_a.attrs["units"] == "K" and arr_a.attrs["paramId"] == 167
        ids = lg["cell_ids"]
        assert ids.dtype == np.uint64 and ids.metadata.dimension_names == ("cells",)
        assert nL % ids.chunks[0] == 0 and nL % chunk == 0
        np.testing.assert_array_equal(ids[:], cell * nL + np.arange(nL, dtype=np.uint64))
        lay = lg["layer"]
        np.testing.assert_array_equal(lay[:], [1, 2])
        assert lay.metadata.dimension_names == ("layer",)
    np.testing.assert_array_equal(g["3/a"][:], a)
    np.testing.assert_array_equal(g["3/b"][:], b)
    # coarse level: NaN-mean of the four children, NaN only where all four are NaN
    with np.errstate(all="ignore"):
        ca = np.nanmean(a.reshape(-1, 4), axis=1)
        cb = np.nanmean(b.reshape(2, -1, 4), axis=2)
    np.testing.assert_allclose(g["2/a"][:], ca, rtol=1e-6)
    np.testing.assert_allclose(g["2/b"][:], cb, rtol=1e-6)
    assert np.isnan(g["2/a"][0])  # all four children NaN
    assert not np.isnan(g["2/a"][1])  # one NaN child of four: still a mean
    assert g["1/a"].chunks == (1,)


def test_write_cell_store_coarsest_level_is_one_chunk(tmp_path):
    kw = dict(
        parent_cell=0,
        parent_level=1,
        level=3,
        group="g",
        variables={"x": np.zeros(16, dtype=np.float32)},
        var_attrs={},
        root_attrs={},
        stac_item={},
    )
    # chunk_level 2: the chunks of level 2 hold one cell each, 4 chunks in the level
    with pytest.raises(ValueError, match="more than one chunk"):
        write_cell_store(tmp_path / "a.zarr", chunk_level=2, coarsest_level=2, **kw)
    with pytest.raises(ValueError, match="more than one chunk"):
        write_cell_store(tmp_path / "b.zarr", chunk_level=2, coarsest_level=3, **kw)
    # a single chunk: coarsest above the chunk level, or the default chunk level
    write_cell_store(tmp_path / "c.zarr", chunk_level=2, coarsest_level=1, **kw)
    write_cell_store(tmp_path / "d.zarr", coarsest_level=2, **kw)


def test_write_cell_store_defaults_one_level_one_chunk(tmp_path):
    pl, lv = 2, 4
    n = 4 ** (lv - pl)
    out = tmp_path / "d.zarr"
    res = write_cell_store(
        out,
        parent_cell=9,
        parent_level=pl,
        level=lv,
        group="g",
        variables={"x": np.arange(n, dtype=np.float32)},
        var_attrs={},
        root_attrs={},
        stac_item={"id": "d"},
    )
    assert res["levels"] == [4]
    g = zarr.open_group(str(out), mode="r")["g"]
    assert [e["asset"] for e in g.attrs["multiscales"]["layout"]] == ["4"]
    assert list(g.keys()) == ["4"]
    assert g["4/x"].chunks == (n,) and "layer" not in g["4"]
    assert g["4/cell_ids"][0] == 9 * n


def test_write_cell_store_rejects_bad_shapes(tmp_path):
    kw = dict(
        parent_cell=0,
        parent_level=1,
        level=2,
        group="g",
        var_attrs={},
        root_attrs={},
        stac_item={},
    )
    with pytest.raises(ValueError, match="cells"):
        write_cell_store(tmp_path / "e.zarr", variables={"x": np.zeros(5)}, **kw)
    with pytest.raises(ValueError, match="layer"):
        write_cell_store(tmp_path / "e.zarr", variables={"x": np.zeros((2, 4))}, **kw)


def test_cell_store_of_resampled_data_round_trip(tmp_path):
    """Reader → resampler → cell store for one parent cell of a masked 3-layer field."""
    cells, lon, lat = sphere_lonlat()
    layers = np.stack([ocean_field(k) for k in (1, 2, 3)])
    s = SphereToEllipsoid(LEVEL)
    full = np.full(layers.shape, np.nan, dtype=np.float32)
    for k in range(3):
        ids, vals = s.resample(cells, layers[k])
        full[k, ids.astype(int)] = vals
    pl, cell = 1, 8
    n = 4 ** (LEVEL - pl)
    part = full[:, cell * n : (cell + 1) * n]
    out = tmp_path / "r.zarr"
    write_cell_store(
        out,
        parent_cell=cell,
        parent_level=pl,
        level=LEVEL,
        group="ocean",
        variables={"avg_thetao": part},
        var_attrs={},
        root_attrs={},
        stac_item={},
        layer=np.array([1, 2, 3]),
    )
    got = zarr.open_group(str(out), mode="r")["ocean/4/avg_thetao"][:]
    np.testing.assert_array_equal(got, part)


# ── real files (skipped when absent) ─────────────────────────────────────────


@pytest.mark.skipif(not REAL_OCEAN_GRIB.exists(), reason="real ocean GRIB not on this machine")
def test_real_ocean_grib():
    info = check_grib_params([REAL_OCEAN_GRIB], [263501])
    assert info["messages"] == 3 and info["grid"]["Nside"] == 1024
    ds = read_healpix_grib(REAL_OCEAN_GRIB)
    v = ds["avg_thetao"]
    assert v.dims == ("time", "layer", "cells") and v.shape == (1, 3, 12 * 1024**2)
    np.testing.assert_array_equal(ds["layer"].values, [1, 2, 3])
    assert int(np.isnan(v.values[0, 0]).sum()) == 12 * 1024**2 - 8917842
    assert ds["time"].values[0] == np.datetime64("2026-01-01T00:00", "ns")
    assert ds["time_bounds"].values[0, 1] == np.datetime64("2026-01-02T00:00", "ns")
    assert ds.attrs["healpix_level"] == 10


@pytest.mark.skipif(not REAL_ATMOS_GRIB.exists(), reason="real atmosphere GRIB not on this machine")
def test_real_atmosphere_grib():
    ds = read_healpix_grib(REAL_ATMOS_GRIB, param_ids=[167])
    v = ds["2t"]
    assert v.shape[-1] == 12 * 1024**2 and not np.isnan(v.values).any()
    assert 150 < float(v.min()) and float(v.max()) < 350


@pytest.mark.skipif(
    not (REAL_COVJSON.exists() and REAL_OCEAN_GRIB.exists()),
    reason="real CoverageJSON / GRIB not on this machine",
)
def test_real_covjson_matches_the_grib_cells():
    cj = read_covjson_points(REAL_COVJSON, level=10)
    assert cj["avg_thetao"].dims == ("time", "layer", "cells")
    assert cj["avg_thetao"].shape[1] == 3
    assert cj.sizes["cells"] > 1000
    ids = cj["cell_ids"].values
    assert (np.diff(ids.astype(np.int64)) > 0).all()
    grib = read_healpix_grib(REAL_OCEAN_GRIB)["avg_thetao"].values
    # the Polytope feature returns the field's own points: same values at the same cell ids
    np.testing.assert_allclose(
        cj["avg_thetao"].values[0, 0], grib[0, 0, ids.astype(int)], rtol=1e-5
    )


def test_operator_is_reused_for_the_same_cells_in_another_order():
    cells, lon, lat = sphere_lonlat()
    f = smooth_field(lon, lat)
    f[lat > 30.0] = np.nan
    s = SphereToEllipsoid(LEVEL)
    ids1, v1 = s.resample(cells, f)
    perm = np.random.default_rng(1).permutation(cells.size)
    ids2, v2 = s.resample(cells[perm], f[perm])
    assert len(s._operators) == 1
    np.testing.assert_array_equal(ids1, ids2)
    np.testing.assert_array_equal(v1, v2)
    s.clear()
    assert not s._operators and not s._norms


def test_conservative_drops_cells_with_negligible_weight(monkeypatch):
    from healpix_convert.converters import climatedt

    cells, lon, lat = sphere_lonlat()
    s = SphereToEllipsoid(LEVEL, method="conservative")
    f = smooth_field(lon, lat)
    ids_all, _ = s.resample(cells, f)
    _, den = next(iter(s._norms.values()))
    assert (den > climatedt.CONSERVATIVE_MIN_WEIGHT * den.max()).all() or ids_all.size < den.size
    # a tiny weight in one cell is dropped, the others stay
    ids0, den0 = next(iter(s._norms.values()))
    den1 = den0.copy()
    den1[0] = 1e-9 * den0.max()
    key = next(iter(s._norms))
    s._norms[key] = (ids0, den1)
    ids, _ = s.resample(cells, f)
    assert ids.size == ids0.size - 1 and ids0[0] not in ids
