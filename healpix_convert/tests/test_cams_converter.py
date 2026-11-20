"""CAMSConverter: dataset, variable table, several times, the forecast-file layout, clipping.

The NetCDF files are written at the cache path (nothing is downloaded), as NetCDF-3 so that the
test needs no netCDF4 library; nearest-neighbour binning keeps the tests fast. Every value says
which variable and time it belongs to.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("requests")
pytest.importorskip("scipy")
xr = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")

from healpix_convert.converters.cams import CAMSConverter, _one_time_axis  # noqa: E402
from healpix_convert.settings.cams import (  # noqa: E402
    ADS_DATASET,
    ADS_FORECAST_DATASET,
    CAMS_ALL_VARIABLE_META,
    CAMS_VARIABLE_META,
)

N6 = 12 * 4**6
LAT = np.arange(90.0, -90.1, -3.0)  # a coarse regular grid
LON = np.arange(0.0, 360.0, 3.0)


def _value(k: int, t: int) -> float:
    return 0.01 * (k + 1) + 0.1 * t


def _fields(names, n_times):
    out = {}
    for k, name in enumerate(names):
        v = np.full((n_times, LAT.size, LON.size), 0.0)
        for t in range(n_times):
            v[t] = _value(k, t)
        if name == "z":  # negative below sea level: must not be clipped at 0
            v[:] = np.linspace(-500.0, 50000.0, LAT.size)[None, :, None]
        out[name] = v
    return out


def _eac4_file(path, names, times):
    data = _fields(names, len(times))
    ds = xr.Dataset(
        {n: (("valid_time", "latitude", "longitude"), v) for n, v in data.items()},
        coords={"valid_time": pd.to_datetime(times), "latitude": LAT, "longitude": LON},
    )
    ds.to_netcdf(path, engine="scipy")


def _forecast_file(path, names, refs, periods_h):
    n = len(refs) * len(periods_h)
    data = _fields(names, n)
    shape = (len(refs), len(periods_h), LAT.size, LON.size)
    valid = np.array(
        [[np.datetime64(r) + np.timedelta64(p, "h") for p in periods_h] for r in refs],
        dtype="datetime64[ns]",
    )
    ds = xr.Dataset(
        {
            n_: (
                ("forecast_reference_time", "forecast_period", "latitude", "longitude"),
                v.reshape(shape),
            )
            for n_, v in data.items()
        },
        coords={
            "forecast_reference_time": pd.to_datetime(refs),
            "forecast_period": np.array(periods_h, dtype="int64"),
            "valid_time": (("forecast_reference_time", "forecast_period"), valid),
            "latitude": LAT,
            "longitude": LON,
        },
    )
    ds.to_netcdf(path, engine="scipy")


def _convert(tmp_path, write, **kwargs):
    conv = CAMSConverter(local_dir=tmp_path, method="nn", **kwargs)
    write(conv._nc_path())
    out = str(tmp_path / "out.zarr")
    result = conv.prepare(out)
    for i in range(result.n_chunks):
        conv.convert_group(i)
    conv.consolidate(out)
    return conv, out, result


def test_default_unchanged(tmp_path) -> None:
    names = list(CAMS_VARIABLE_META)
    conv, out, result = _convert(
        tmp_path,
        lambda p: _eac4_file(p, names, ["2024-01-01T00:00"]),
        date="2024-01-01",
    )
    assert conv._nc_path().name == "cams_20240101T00.nc"  # the name of earlier versions
    g = zarr.open_group(out, mode="r")["measurements/aod"]
    assert sorted(g.array_keys()) == sorted([*names, "cell_ids", "crs", "number"])
    assert g["aod550"].shape == (1, N6) and g["aod550"].chunks == (1, 256)
    np.testing.assert_allclose(g["aod550"][0, ::999], _value(0, 0), rtol=1e-6)
    props = zarr.open_group(out, mode="r").attrs["stac_discovery"]["properties"]
    assert props["source_dataset"] == ADS_DATASET
    assert props["description"].startswith("CAMS EAC4 reanalysis aerosol optical depth")
    assert props["source_requests"]["inputs"]["time"] == "00:00"
    assert props["source_grid"] == f"{ADS_DATASET} regular 3°"
    assert props["resampling:parameters"]["resampler"] == "nearest-neighbour binning"
    assert "healpix-convert" in props["processing:software"]


@pytest.mark.parametrize("date", ["2026-08-12", "2026-01-15"])
def test_forecast_analyses_of_a_day(tmp_path, date) -> None:
    """The global analyses (lead time 0 at 00 and 12 UTC) with every field of the ESA file."""
    names = list(CAMS_ALL_VARIABLE_META)
    refs = [f"{date}T00:00", f"{date}T12:00"]
    conv, out, result = _convert(
        tmp_path,
        lambda p: _forecast_file(p, names, refs, [0]),
        date=date,
        time="00:00/12:00",
        dataset=ADS_FORECAST_DATASET,
        request={"type": "forecast", "leadtime_hour": "0"},
        variables=CAMS_ALL_VARIABLE_META,
    )
    assert [t.hour for t in result.times] == [0, 12]
    tree = xr.open_datatree(out, engine="zarr", consolidated=True)
    ds = tree["measurements/aod"].to_dataset()
    assert [str(t)[:16] for t in ds.time.values] == [f"{date}T00:00", f"{date}T12:00"]
    for k, name in enumerate(names):
        assert ds[name].shape == (2, N6)
        if name != "z":
            np.testing.assert_allclose(
                ds[name].isel(cells=123).values, [_value(k, 0), _value(k, 1)], rtol=1e-6
            )
    z = ds["z"].values
    assert np.nanmin(z) < 0 and np.isfinite(z).all()  # not clipped at 0, no hole
    assert (
        ds["aod1240"].attrs.get("valid_min") == 0.0 and "valid_min" not in ds["z"].attrs
    )
    props = tree.attrs["stac_discovery"]["properties"]
    assert props["source_dataset"] == ADS_FORECAST_DATASET
    req = props["source_requests"]
    assert (
        req["dataset"] == ADS_FORECAST_DATASET and req["inputs"]["leadtime_hour"] == "0"
    )
    assert req["inputs"]["time"] == ["00:00", "12:00"]
    assert "surface_geopotential" in req["inputs"]["variable"]
    assert conv._nc_path().name.startswith(f"cams_{date.replace('-', '')}T00-12n2_v")


def test_one_time_axis_from_reference_and_period() -> None:
    refs = ["2026-08-12T00:00", "2026-08-12T12:00"]
    ds = xr.Dataset(
        {
            "a": (
                ("forecast_reference_time", "forecast_period", "x"),
                np.arange(8.0).reshape(2, 2, 2),
            )
        },
        coords={
            "forecast_reference_time": pd.to_datetime(refs),
            "forecast_period": np.array([0, 3]),
        },
    )
    out = _one_time_axis(ds)
    assert out["a"].dims == ("valid_time", "x")
    assert [str(t)[11:16] for t in out.valid_time.values] == [
        "00:00",
        "03:00",
        "12:00",
        "15:00",
    ]
    np.testing.assert_array_equal(out["a"].values[:, 0], [0, 2, 4, 6])
    one = xr.Dataset(
        {"a": (("x",), [1.0, 2.0])},
        coords={
            "forecast_reference_time": pd.Timestamp("2026-01-15T12:00"),
            "forecast_period": 3,
        },
    )
    out1 = _one_time_axis(one)
    assert (
        out1["a"].dims == ("valid_time", "x")
        and str(out1.valid_time.values[0])[:16] == "2026-01-15T15:00"
    )
    eac4 = xr.Dataset(
        {"a": (("valid_time",), [1.0])},
        coords={"valid_time": pd.to_datetime(["2024-01-01"])},
    )
    assert _one_time_axis(eac4) is eac4


def test_missing_variable_refused(tmp_path) -> None:
    with pytest.raises(
        ValueError, match=r"requested but not in the file: \['aod469', .*'z'\]"
    ):
        _convert(
            tmp_path,
            lambda p: _eac4_file(p, list(CAMS_VARIABLE_META), ["2024-01-01T00:00"]),
            date="2024-01-01",
            variables=CAMS_ALL_VARIABLE_META | {},
        )


def test_requests_and_cache_names() -> None:
    eac4 = CAMSConverter(date="2025-05-20", time="09:00")
    assert eac4.requests() == {
        "dataset": ADS_DATASET,
        "inputs": {
            "variable": [v[0] for v in CAMS_VARIABLE_META.values()],
            "date": "2025-05-20",
            "time": "09:00",
            "format": "netcdf",
        },
    }
    other = CAMSConverter(
        date="2025-05-20", time="09:00", variables=CAMS_ALL_VARIABLE_META
    )
    assert other._nc_path().name != eac4._nc_path().name
    with pytest.raises(ValueError, match="no variable"):
        CAMSConverter(date="2025-05-20", variables={})
