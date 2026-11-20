"""ERA5Converter with variable tables passed in, pressure levels and provenance.

As in test_era5_times.py, the GRIB files are written from the ecCodes samples
at the cache paths (nothing is downloaded) and the PSF resampling is replaced
by a constant map holding the first source value, so each output value tells
which variable, time and level it came from.
"""

import numpy as np
import pytest

pytest.importorskip("cfgrib")
eccodes = pytest.importorskip("eccodes")
xr = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")

from healpix_convert.converters import era5  # noqa: E402
from healpix_convert.converters.era5 import (  # noqa: E402
    ENDA_GROUP,
    OPER_GROUP,
    PL_SOURCE,
    ERA5Converter,
)
from healpix_convert.settings.era5 import (  # noqa: E402
    ERA5_ALL_VARIABLE_META,
    ERA5_ENDA_VARIABLE_META,
    ERA5_OPER_VARIABLE_META,
    ERA5_PL_VARIABLE_META,
)

N_CELLS = 12 * 4**7
LEVELS = [1000, 950, 925, 900, 850, 800, 700]  # as the ESA ADF_ECMWF
HOURS = [0, 9, 10, 23]
VERSION = {"grid4earth:version": "p2026-11-18"}


def _param(meta_entry) -> int:
    return int(meta_entry[0].split(".")[0])


def _value(param: int, hour: int, level_index: int = 0) -> float:
    return 1000.0 * param + 10.0 * hour + level_index


def _expver(hour: int) -> str:
    # the last hours of a recent day may still be ERA5T
    return "0005" if hour >= 10 else "0001"


def _write_grib(path, date, params, hours, levels=None, nan_at=None):
    """Surface fields (levels None) or pressure-level fields, levels written in
    ascending order on purpose: the converter must reorder them. ``nan_at``
    (param, level) gets missing values (a bitmap) on its first half of points."""
    with open(path, "wb") as f:
        for hour in hours:
            for level in sorted(levels) if levels else [None]:
                for param in params:
                    gid = eccodes.codes_grib_new_from_samples("reduced_gg_pl_32_grib1")
                    eccodes.codes_set(gid, "localDefinitionNumber", 1)
                    eccodes.codes_set_string(gid, "marsClass", "ea")
                    eccodes.codes_set_string(gid, "marsType", "an")
                    eccodes.codes_set_string(gid, "expver", _expver(hour))
                    eccodes.codes_set(gid, "table2Version", 128)
                    eccodes.codes_set(gid, "indicatorOfParameter", param)
                    if level is None:
                        eccodes.codes_set_string(gid, "typeOfLevel", "surface")
                        eccodes.codes_set(gid, "level", 0)
                        index = 0
                    else:
                        eccodes.codes_set_string(gid, "typeOfLevel", "isobaricInhPa")
                        eccodes.codes_set(gid, "level", level)
                        index = LEVELS.index(level)
                    eccodes.codes_set(gid, "dataDate", int(date.replace("-", "")))
                    eccodes.codes_set(gid, "dataTime", hour * 100)
                    size = eccodes.codes_get_size(gid, "values")
                    values = np.full(size, _value(param, hour, index))
                    if nan_at == (param, level):
                        eccodes.codes_set(gid, "bitmapPresent", 1)
                        eccodes.codes_set(gid, "missingValue", 9999)
                        values[: size // 2] = 9999
                    eccodes.codes_set_values(gid, values)
                    eccodes.codes_write(gid, f)
                    eccodes.codes_release(gid)


@pytest.fixture
def no_psf(monkeypatch):
    """Replace the PSF resampling by a map of the first source value."""
    monkeypatch.setattr(era5, "_get_resampler", lambda lon, lat: None)

    def full_map(self, group, var, t, ds, nr, t_dim):
        value = ds[var].isel({t_dim: t}).values.ravel()[0]
        return np.full(N_CELLS, value, dtype=np.float32)

    monkeypatch.setattr(ERA5Converter, "_full_map", full_map)


def _convert(
    tmp_path,
    date,
    pl_levels_in_grib=LEVELS,
    pl_params=None,
    pl_hours=HOURS,
    nan_at=None,
    **kwargs,
):
    converter = ERA5Converter(
        date=date,
        time="/".join(f"{h:02d}:00:00" for h in HOURS),
        time_chunk="all",
        local_dir=tmp_path,
        **kwargs,
    )
    paths = converter.grib_paths()
    if ENDA_GROUP in paths:
        _write_grib(
            paths[ENDA_GROUP],
            date,
            [_param(v) for v in converter.enda_variables.values()],
            [h for h in HOURS if h % 3 == 0],
        )
    if OPER_GROUP in paths:
        _write_grib(
            paths[OPER_GROUP],
            date,
            [_param(v) for v in converter.oper_variables.values()],
            HOURS,
        )
    if PL_SOURCE in paths:
        _write_grib(
            paths[PL_SOURCE],
            date,
            pl_params or [_param(v) for v in converter.pl_variables.values()],
            pl_hours,
            levels=pl_levels_in_grib,
            nan_at=nan_at,
        )
    out = str(tmp_path / "out.zarr")
    result = converter.prepare(out)
    for i in range(result.n_chunks):
        converter.convert_group(i)
    converter.consolidate(out)
    return converter, out, result


def _oper_only(**extra):
    """D2 / D3 of production p2026-11-18: no enda, all surface fields from oper,
    the pressure levels of the ESA ADF; ``extra`` overrides any of them."""
    kwargs = dict(
        enda_variables={},
        oper_variables=ERA5_ALL_VARIABLE_META,
        pressure_levels=LEVELS,
        root_attributes=VERSION,
    )
    kwargs.update(extra)
    return kwargs


@pytest.mark.usefixtures("no_psf")
@pytest.mark.parametrize("date", ["2026-08-12", "2026-01-15"])
def test_oper_only_with_pressure_levels(tmp_path, date) -> None:
    converter, out, result = _convert(tmp_path, date, **_oper_only())

    root = zarr.open_group(out, mode="r")
    assert ENDA_GROUP not in root  # no EDA requested, no enda group
    assert result.ds_enda is None
    assert root.attrs["grid4earth:version"] == "p2026-11-18"

    oper = root[OPER_GROUP]
    for var in ERA5_ALL_VARIABLE_META:  # the 15 surface fields, all hourly
        assert oper[var].shape == (len(HOURS), N_CELLS)
    np.testing.assert_array_equal(oper["isobaric"][:], LEVELS)
    assert dict(oper["isobaric"].attrs)["units"] == "hPa"
    for var in ERA5_PL_VARIABLE_META:
        arr = oper[f"{var}_isobaric"]
        assert arr.shape == (len(HOURS), len(LEVELS), N_CELLS)
        assert arr.chunks == (len(HOURS), 1, 1024)
        assert arr.metadata.dimension_names == ("time", "isobaric", "cells")

    # every value comes from its own variable, time and level
    param_r = _param(ERA5_PL_VARIABLE_META["r"])
    for t, hour in enumerate(HOURS):
        for k in range(len(LEVELS)):
            np.testing.assert_array_equal(
                oper["r_isobaric"][t, k, ::4096], _value(param_r, hour, k)
            )
    np.testing.assert_array_equal(
        oper["t2m"][:, 777], [_value(167, h) for h in HOURS]
    )  # t2m from the oper stream (D3: no enda)

    tree = xr.open_datatree(out, engine="zarr", consolidated=True)
    ds = tree[OPER_GROUP].to_dataset()
    assert str(ds.time.values[0])[:10] == date
    picked = ds.q_isobaric.sel(isobaric=850, time=f"{date}T09:00")
    np.testing.assert_array_equal(
        picked.isel(cells=0).values,
        _value(_param(ERA5_PL_VARIABLE_META["q"]), 9, LEVELS.index(850)),
    )

    # expver per time: read from the GRIB, not from the request
    assert oper.attrs["source_expver"] == [_expver(h) for h in HOURS]

    props = tree.attrs["stac_discovery"]["properties"]
    assert set(props["source_requests"]) == {OPER_GROUP, PL_SOURCE}
    pl_request = props["source_requests"][PL_SOURCE]
    assert pl_request["levtype"] == "pl"
    assert pl_request["levelist"] == "/".join(str(lev) for lev in LEVELS)
    assert pl_request["param"] == "130.128/133.128/157.128"
    assert props["source_expver"] == ["0001", "0005"]
    assert props["source_grid"] == "Gaussian N320"
    assert "t, q, r on 1000, 950" in props["description"]
    params = props["resampling:parameters"]
    assert params["ellipsoid"] == "WGS84"
    assert params["threshold"] == 0.5 and params["lam"] == 0.0
    assert params["ensemble_member"] is None
    assert "healpix-convert" in props["processing:software"]
    assert "processing:datetime" in props


@pytest.mark.usefixtures("no_psf")
def test_default_tables_unchanged(tmp_path) -> None:
    """Without the new arguments: enda + oper groups, no pressure levels."""
    converter, out, result = _convert(tmp_path, "2026-08-12")

    assert PL_SOURCE not in converter.grib_paths()
    root = zarr.open_group(out, mode="r")
    assert set(root[ENDA_GROUP].array_keys()) >= set(ERA5_ENDA_VARIABLE_META)
    assert set(root[OPER_GROUP].array_keys()) >= set(ERA5_OPER_VARIABLE_META)
    assert "isobaric" not in root[OPER_GROUP]
    assert "grid4earth:version" not in root.attrs
    props = root.attrs["stac_discovery"]["properties"]
    assert props["description"].startswith(
        "ERA5 reanalysis surface fields (enda + oper"
    )
    assert props["source_grid"] == "Gaussian N256/N320"
    assert props["resampling:parameters"]["ensemble_member"] == 0
    assert set(props["source_requests"]) == {ENDA_GROUP, OPER_GROUP}


def test_missing_pressure_level_refused(tmp_path, no_psf) -> None:
    with pytest.raises(ValueError, match=r"pressure levels \[925\] hPa missing"):
        _convert(
            tmp_path,
            "2026-08-12",
            pl_levels_in_grib=[lev for lev in LEVELS if lev != 925],
            **_oper_only(),
        )


def test_requests_without_enda_need_no_eda_time() -> None:
    converter = ERA5Converter(
        date="2026-01-15", time="10:00:00", **_oper_only()
    )  # 10 UTC has no EDA analysis, but no EDA is asked for
    requests = converter.requests()
    assert ENDA_GROUP not in requests
    assert requests[OPER_GROUP]["param"].split("/")[0] == "167.128"  # t2m first


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(root_attributes={"stac_discovery": {}}), "root_attributes"),
        (dict(root_attributes={"Conventions": "x"}), "root_attributes"),
        (dict(pressure_levels=[850, 850]), "distinct"),
        (dict(pressure_levels=[0]), "distinct"),
        (dict(pressure_levels=[850], pl_variables={}), "without pl_variables"),
        (dict(enda_variables={}, oper_variables={}), "no variable"),
        (
            dict(pressure_levels=[850], oper_variables={"t_isobaric": ("1", "K", "x")}),
            "clash",
        ),
    ],
)
def test_arguments_checked(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        ERA5Converter(date="2026-08-12", **kwargs)


class _MeanPSF:
    """Stand-in for PSFResampler: every output cell gets the mean of the source
    values it is given, so a map tells which field, time and level (and which
    valid points) went in."""

    def __init__(self, lon_deg=None, lat_deg=None, **kwargs):
        self.n_points = None if lon_deg is None else len(lon_deg)

    def resample(self, values, lam=0.0):
        if self.n_points is not None:
            assert len(values) == self.n_points
        return type(
            "Result",
            (),
            {
                "cell_ids": np.arange(N_CELLS),
                "cell_data": np.full(N_CELLS, values.mean()),
            },
        )


def test_real_full_map_levels_and_masks(tmp_path, monkeypatch) -> None:
    """_full_map itself runs: the cache key must keep the levels apart, and a
    level with missing values goes through the masked resampler."""
    monkeypatch.setattr(era5, "_get_resampler", lambda lon, lat: _MeanPSF())
    monkeypatch.setattr(era5, "PSFResampler", _MeanPSF)
    param_r = _param(ERA5_PL_VARIABLE_META["r"])
    _, out, _ = _convert(tmp_path, "2026-08-12", nan_at=(param_r, 700), **_oper_only())

    oper = zarr.open_group(out, mode="r")[OPER_GROUP]
    param_t = _param(ERA5_PL_VARIABLE_META["t"])
    for k in range(len(LEVELS)):
        np.testing.assert_allclose(
            oper["t_isobaric"][1, k, ::997], _value(param_t, HOURS[1], k), rtol=1e-6
        )
    # the masked level keeps the value of its valid points (all equal here)
    k700 = LEVELS.index(700)
    np.testing.assert_allclose(
        oper["r_isobaric"][2, k700, ::997], _value(param_r, HOURS[2], k700), rtol=1e-6
    )
    assert np.isfinite(oper["r_isobaric"][:]).all()


@pytest.mark.usefixtures("no_psf")
def test_pressure_levels_only(tmp_path) -> None:
    """No surface table: the oper group holds only the pressure levels."""
    converter, out, result = _convert(
        tmp_path, "2026-01-15", **_oper_only(oper_variables={})
    )

    assert set(converter.grib_paths()) == {PL_SOURCE}
    oper = zarr.open_group(out, mode="r")[OPER_GROUP]
    assert {"t_isobaric", "q_isobaric", "r_isobaric", "isobaric"} <= set(
        oper.array_keys()
    )
    assert "t2m" not in oper
    assert oper.attrs["source_expver"] == [_expver(h) for h in HOURS]
    assert len(result.times[OPER_GROUP]) == len(HOURS)


def test_missing_pressure_level_variable_refused(tmp_path, no_psf) -> None:
    params = [_param(ERA5_PL_VARIABLE_META[v]) for v in ("t", "q")]  # no r
    with pytest.raises(ValueError, match=r"requested but not in the GRIB: \['r'\]"):
        _convert(tmp_path, "2026-08-12", pl_params=params, **_oper_only())


def test_pressure_level_times_must_match_oper(tmp_path, no_psf) -> None:
    with pytest.raises(ValueError, match="pressure-level times differ"):
        _convert(tmp_path, "2026-08-12", pl_hours=HOURS[:-1], **_oper_only())


def test_cache_names_follow_the_tables() -> None:
    default = ERA5Converter(date="2026-08-12").grib_paths()
    assert default[OPER_GROUP].name == "era5_oper_20260812T00_N320.grib"  # as before
    assert default[ENDA_GROUP].name == "era5_enda_20260812T00_N256.grib"

    other = ERA5Converter(date="2026-08-12", **_oper_only()).grib_paths()
    assert other[OPER_GROUP].name != default[OPER_GROUP].name  # another table
    assert other[PL_SOURCE].name.startswith(
        "era5_pl_20260812T00_1000-950-925-900-850-800-700"
    )
    fewer = ERA5Converter(
        date="2026-08-12", **_oper_only(pl_variables={"t": ERA5_PL_VARIABLE_META["t"]})
    ).grib_paths()
    assert fewer[PL_SOURCE].name != other[PL_SOURCE].name
