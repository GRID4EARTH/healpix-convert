"""ClimateDT converter: chunked writes of the sphere → WGS84 resampled fields."""

import healpix_geo
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

pytest.importorskip("cfgrib")

from healpix_convert.converters import climatedt  # noqa: E402
from healpix_convert.converters.climatedt import ClimateDTConverter  # noqa: E402

LEVEL = climatedt._CHILD_LEVEL
CHUNK_SIZE = climatedt._CHUNK_SIZE
# two chunks far apart: with the old bug, the second one got the values of
# the first cells of the sphere
CHUNKS = (0, 100)


def _native_dataset(n_times: int = 2, masked: bool = False) -> xr.Dataset:
    """A ClimateDT-like dataset: one value per native (spherical) cell."""
    cells = np.arange(12 * 4**LEVEL, dtype="uint64")
    lon, lat = healpix_geo.nested.healpix_to_lonlat(cells, LEVEL, ellipsoid="sphere")
    lon = np.asarray(lon, dtype="float64")
    lat = np.asarray(lat, dtype="float64")
    field = _field(lon, lat)
    data = np.stack([field + t for t in range(n_times)])
    if masked:
        # like an ocean field over land
        data[:, lat > 30.0] = np.nan
    times = pd.date_range("2020-01-02T01:00", periods=n_times, freq="h")
    return xr.Dataset(
        {"t2m": (("valid_time", "values"), data)},
        coords={
            "valid_time": ("valid_time", times.values),
            "latitude": ("values", lat),
            "longitude": ("values", lon),
        },
    )


def _converter(tmp_path, masked: bool = False) -> tuple[ClimateDTConverter, str]:
    conv = ClimateDTConverter(date="20200102", time="0100", local_dir=tmp_path)
    # skip the Polytope download and the GRIB reading
    conv._grib_path().touch()
    conv._ds = _native_dataset(masked=masked)
    output = str(tmp_path / "out.zarr")
    conv.prepare(output_path=output)
    return conv, output


def _chunk_cells(chunk_index: int) -> np.ndarray:
    return np.arange(chunk_index * CHUNK_SIZE, (chunk_index + 1) * CHUNK_SIZE)


def _field(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    return 280.0 + 20.0 * np.cos(np.deg2rad(lat)) + 5.0 * np.sin(np.deg2rad(lon))


def _wgs84_lonlat(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon, lat = healpix_geo.nested.healpix_to_lonlat(
        cells.astype("uint64"), LEVEL, ellipsoid="WGS84"
    )
    return np.asarray(lon), np.asarray(lat)


def test_chunks_hold_their_own_cells(tmp_path) -> None:
    conv, output = _converter(tmp_path)
    for idx in CHUNKS:
        conv.convert_group(idx)

    grp = zarr.open_group(output, mode="r")["measurements/sfc"]
    t2m = grp["t2m"][...]
    for t in range(t2m.shape[0]):
        vals = conv._ds["t2m"].isel(valid_time=t).values
        res = conv._resampler(np.isfinite(vals)).resample(vals, lam=0.0)
        expected = np.full(12 * 4**LEVEL, np.nan, dtype="float32")
        expected[res.cell_ids] = res.cell_data.astype("float32")
        expected = np.clip(expected, vals.min(), vals.max())
        for idx in CHUNKS:
            cells = _chunk_cells(idx)
            np.testing.assert_array_equal(t2m[t, cells], expected[cells])
            # the resampled map stays close to the input field
            assert not np.isnan(t2m[t, cells]).any()
            np.testing.assert_allclose(t2m[t, cells], vals[cells], atol=0.5)
    for idx in CHUNKS:
        np.testing.assert_array_equal(grp["cell_ids"][_chunk_cells(idx)], _chunk_cells(idx))


def test_values_are_moved_to_wgs84(tmp_path) -> None:
    conv, output = _converter(tmp_path)
    # a steep north-south gradient (1 K per 0.1 degree), where the spherical
    # and the WGS84 cell centres are furthest apart (about 45 degrees)
    conv._ds["t2m"] = 10.0 * conv._ds["latitude"].expand_dims(valid_time=conv._ds["valid_time"])
    cell = healpix_geo.nested.lonlat_to_healpix(
        np.array([10.0]), np.array([45.0]), LEVEL, ellipsoid="WGS84"
    )
    chunk_index = int(cell[0]) // CHUNK_SIZE
    conv.convert_group(chunk_index)

    cells = _chunk_cells(chunk_index)
    t2m = zarr.open_group(output, mode="r")["measurements/sfc/t2m"][0, cells]
    copy = conv._ds["t2m"].isel(valid_time=0).values[cells]
    truth = 10.0 * _wgs84_lonlat(cells)[1]
    # a plain copy of the spherical values is about 1 K off; the resampled
    # values are much closer to the field at the WGS84 cell centres
    assert np.abs(copy - truth).mean() > 0.5
    assert np.abs(t2m - truth).mean() < 0.2 * np.abs(copy - truth).mean()


def test_masked_cells_stay_nan(tmp_path) -> None:
    conv, output = _converter(tmp_path, masked=True)
    for idx in range(12 * 4**climatedt._CHUNK_LEVEL):
        conv.convert_group(idx)

    t2m = zarr.open_group(output, mode="r")["measurements/sfc/t2m"][0]
    vals = conv._ds["t2m"].isel(valid_time=0).values
    # no zero leaking in from the masked points, no NaN far from the mask
    assert np.nanmin(t2m) >= np.nanmin(vals) - 1e-3
    assert np.nanmax(t2m) <= np.nanmax(vals) + 1e-3
    _, lat = _wgs84_lonlat(np.arange(t2m.size))
    assert np.isnan(t2m[lat > 31.0]).all()
    assert not np.isnan(t2m[lat < 29.0]).any()


def test_resamples_once_per_timestep(tmp_path, monkeypatch) -> None:
    conv, _ = _converter(tmp_path)
    resampler = conv._resampler(np.isfinite(conv._ds["t2m"].isel(valid_time=0).values))
    calls = []
    resample = resampler.resample

    def counting_resample(*args, **kwargs):
        calls.append(1)
        return resample(*args, **kwargs)

    monkeypatch.setattr(resampler, "resample", counting_resample)
    for idx in CHUNKS:
        conv.convert_group(idx)

    # one variable, two timesteps: once per (variable, timestep), not per chunk
    assert len(calls) == 2


def test_output_declares_wgs84(tmp_path) -> None:
    _, output = _converter(tmp_path)
    attrs = dict(zarr.open_group(output, mode="r")["measurements/sfc"].attrs)
    assert "wgs84" in str(attrs).lower()


def test_extra_dimension_is_refused(tmp_path) -> None:
    conv, _ = _converter(tmp_path)
    conv._ds["t2m"] = conv._ds["t2m"].expand_dims(level=[1, 2], axis=1)
    with pytest.raises(NotImplementedError, match="level"):
        conv.convert_group(0)
