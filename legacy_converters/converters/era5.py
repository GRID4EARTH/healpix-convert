"""
legacy_converters/converters/era5.py
======================================
ERA5 legacy converter.

Downloads ERA5 from CDS via two MARS requests (enda N256 + oper N320),
projects onto HEALPix level 7 using PSFResampler, and writes an
EOPF-compliant zarr.

Usage
-----
    from legacy_converters.converters.era5 import ERA5Converter
    converter = ERA5Converter(date="2025-01-01")
    result    = converter.prepare(output_path="out.zarr")
    for idx in range(result.n_chunks):
        converter.convert_group(idx)
    converter.consolidate("out.zarr")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import healpix_geo
import numpy as np
import xarray as xr
import zarr
from healpix_resample import PSFResampler

from legacy_converters.core.healpix_conventions import DGGSZarrConvention, Healpix
from legacy_converters.core.stac import StacItem
from legacy_converters.settings.era5 import (
    CDS_DATASET,
    CDS_URL,
    DEFAULT_NUMBERS,
    ERA5_ENDA_GRID,
    ERA5_ENDA_VARIABLE_META,
    ERA5_OPER_GRID,
    ERA5_OPER_VARIABLE_META,
)

log = logging.getLogger(__name__)

# ── HEALPix constants ─────────────────────────────────────────────────────────
_CHILD_LEVEL = 7
_CHUNK_LEVEL = 2
_N_CHILD = 12 * 4**_CHILD_LEVEL  # 196,608
_N_PARENT = 12 * 4**_CHUNK_LEVEL  # 192
_CHUNK_SIZE = 4 ** (_CHILD_LEVEL - _CHUNK_LEVEL)  # 1,024

_resampler_cache: dict = {}


def _get_resampler(
    lon: np.ndarray, lat: np.ndarray, level: int = _CHILD_LEVEL, threshold: float = 0.5
) -> PSFResampler:
    key = (id(lon), level)
    if key not in _resampler_cache:
        log.info(f"Building PSFResampler level={level} on {len(lon):,} pts...")
        _resampler_cache[key] = PSFResampler(
            lon_deg=lon,
            lat_deg=lat,
            level=level,
            threshold=threshold,
            verbose=False,
        )
        log.info("PSFResampler ready.")
    return _resampler_cache[key]


@dataclass
class ERA5PrepareResult:
    output_path: str
    n_chunks: int
    n_times: int
    start_dt: datetime
    end_dt: datetime
    grib_enda: Path
    grib_oper: Path
    ds_enda: xr.Dataset
    ds_oper: xr.Dataset
    nr_enda: PSFResampler
    nr_oper: PSFResampler
    lon_enda: np.ndarray
    lat_enda: np.ndarray
    lon_oper: np.ndarray
    lat_oper: np.ndarray
    vars_enda: list[str]
    vars_oper: list[str]


class ERA5Converter:
    """
    Converter for ERA5 reanalysis surface fields to EOPF-DGGS HEALPix zarr.

    Parameters
    ----------
    date      : "YYYY-MM-DD"
    time      : "HH:MM:SS"
    step      : MARS step (default "0")
    numbers   : ensemble member numbers (default 0-9)
    local_dir : directory for local GRIB cache
    """

    def __init__(
        self,
        date: str,
        time: str = "00:00:00",
        step: str = "0",
        numbers: list[int] = None,
        local_dir: Path = Path("."),
    ):
        self.date = date
        self.time = time
        self.step = step
        self.numbers = numbers if numbers is not None else DEFAULT_NUMBERS
        self.local_dir = Path(local_dir)
        self._result: ERA5PrepareResult | None = None
        self._map_cache: dict = {}
        self._sub_resampler_cache: dict = {}

    def prepare(
        self, output_path: str, force_download: bool = False
    ) -> ERA5PrepareResult:
        """Download ERA5 GRIBs (if needed), build PSFResamplers, init zarr."""
        import cdsapi
        import cfgrib
        import pandas as pd

        self._map_cache = {}
        self._sub_resampler_cache = {}

        grib_enda = self._grib_path("enda", ERA5_ENDA_GRID)
        grib_oper = self._grib_path("oper", ERA5_OPER_GRID)

        if not grib_enda.exists() or force_download:
            self._download_enda(grib_enda, cdsapi.Client(url=CDS_URL))
        if not grib_oper.exists() or force_download:
            self._download_oper(grib_oper, cdsapi.Client(url=CDS_URL))

        log.info("Loading cfgrib enda...")
        ds_enda = xr.merge(
            cfgrib.open_datasets(str(grib_enda)), join="override", compat="override"
        )
        log.info("Loading cfgrib oper...")
        ds_oper = xr.merge(
            cfgrib.open_datasets(str(grib_oper)), join="override", compat="override"
        )

        # cfgrib names some oper fields by their native shortName, which differs
        # from the keys used in ERA5_OPER_VARIABLE_META. Rename so the variables
        # are recognised (otherwise mslp/sic are silently dropped).
        ds_oper = ds_oper.rename(
            {
                k: v
                for k, v in {
                    "msl": "mslp",
                    "siconc": "sic",
                }.items()
                if k in ds_oper.data_vars and v not in ds_oper.data_vars
            }
        )

        lon_enda, lat_enda = self._extract_lonlat(ds_enda)
        lon_oper, lat_oper = self._extract_lonlat(ds_oper)

        nr_enda = _get_resampler(lon_enda, lat_enda)
        nr_oper = _get_resampler(lon_oper, lat_oper)

        vt = ds_enda.get("valid_time", ds_enda.get("time", None))
        if vt is not None:
            times = [
                pd.Timestamp(t).tz_localize(None).to_pydatetime().replace(tzinfo=UTC)
                for t in np.atleast_1d(vt.values).ravel()[:1]
            ]
        else:
            hh = int(self.time[:2])
            times = [
                datetime(
                    int(self.date[:4]),
                    int(self.date[5:7]),
                    int(self.date[8:]),
                    hh,
                    tzinfo=UTC,
                )
            ]

        vars_enda = [v for v in ERA5_ENDA_VARIABLE_META if v in ds_enda.data_vars]
        vars_oper = [v for v in ERA5_OPER_VARIABLE_META if v in ds_oper.data_vars]
        start_dt, end_dt = min(times), max(times)
        n_times = len(times)
        n_chunks = _N_PARENT

        self._init_zarr(output_path, n_times, vars_enda, vars_oper, start_dt, end_dt)

        self._result = ERA5PrepareResult(
            output_path=output_path,
            n_chunks=n_chunks,
            n_times=n_times,
            start_dt=start_dt,
            end_dt=end_dt,
            grib_enda=grib_enda,
            grib_oper=grib_oper,
            ds_enda=ds_enda,
            ds_oper=ds_oper,
            nr_enda=nr_enda,
            nr_oper=nr_oper,
            lon_enda=lon_enda,
            lat_enda=lat_enda,
            lon_oper=lon_oper,
            lat_oper=lat_oper,
            vars_enda=vars_enda,
            vars_oper=vars_oper,
        )
        log.info(f"ERA5 prepare done: {n_chunks} chunks, {n_times} timestep(s)")
        return self._result

    def convert_group(self, chunk_index: int) -> None:
        """Project one spatial chunk to HEALPix and write to zarr.

        The full-globe PSF resampling is computed once per variable and cached,
        then sliced for each chunk. This avoids recomputing the global resample
        for every one of the spatial chunks.
        """
        assert self._result is not None, "call prepare() first"
        r = self._result

        cell_ids = healpix_geo.nested.zoom_to(
            chunk_index,
            depth=_CHUNK_LEVEL,
            new_depth=_CHILD_LEVEL,
        )[0]
        start = chunk_index * _CHUNK_SIZE
        chunk_sl = slice(start, start + _CHUNK_SIZE)

        root = zarr.open_group(r.output_path, mode="a")

        for group, ds, nr, variables, meta in [
            (
                "measurements/enda",
                r.ds_enda,
                r.nr_enda,
                r.vars_enda,
                ERA5_ENDA_VARIABLE_META,
            ),
            (
                "measurements/oper",
                r.ds_oper,
                r.nr_oper,
                r.vars_oper,
                ERA5_OPER_VARIABLE_META,
            ),
        ]:
            grp = root[group]
            t_dim = next(
                (d for d in ["valid_time", "time", "number", "step"] if d in ds.dims),
                None,
            )

            for var in variables:
                zarr_a = grp[var]
                for t in range(r.n_times):
                    full_map = self._full_map(group, var, t, ds, nr, t_dim)
                    zarr_a[t, chunk_sl] = full_map[cell_ids]

            grp["cell_ids"][chunk_sl] = cell_ids

        log.debug(f"Chunk {chunk_index + 1}/{_N_PARENT} written.")

    def _full_map(
        self,
        group: str,
        var: str,
        t: int,
        ds: xr.Dataset,
        nr: PSFResampler,
        t_dim: str | None,
    ) -> np.ndarray:
        """Return the full-globe HEALPix map for one variable/timestep.

        Computed once via the PSFResampler and cached for reuse across all
        spatial chunks. The PSF is run raw (lam=0, no smoothing) for maximum
        fidelity. Land-masked fields (NaN in the source, e.g. SST/sea-ice over
        land) are resampled from their valid points only, so masked output cells
        stay NaN instead of poisoning the global solve. The result is clipped to
        the source value range [min, max]: this removes only the few Gibbs-ringing
        cells that overshoot beyond what physically exists in the source, never
        alters in-range values, and preserves the sign of signed fields.
        """
        key = (group, var, t)
        full_map = self._map_cache.get(key)
        if full_map is not None:
            return full_map

        nside = 2**_CHILD_LEVEL
        da = ds[var]
        vals = (
            da.isel({t_dim: t}).values.ravel().astype(np.float64)
            if t_dim
            else da.values.ravel().astype(np.float64)
        )
        full_map = np.full(12 * nside**2, np.nan, dtype=np.float32)

        valid = np.isfinite(vals)
        n_valid = int(valid.sum())
        if n_valid == 0:
            self._map_cache[key] = full_map
            return full_map

        if n_valid == vals.size:
            res = nr.resample(vals, lam=0.0)
        else:
            # source has NaN (land mask): resample valid points only
            sub = self._valid_resampler(group, var, valid)
            res = sub.resample(vals[valid], lam=0.0)
        full_map[res.cell_ids] = res.cell_data.astype(np.float32)

        # Clip to the source range: removes deconvolution overshoot only, never
        # extrapolates beyond values present in the source (NaN cells stay NaN).
        src_lo = float(np.nanmin(vals[valid]))
        src_hi = float(np.nanmax(vals[valid]))
        full_map = np.clip(full_map, src_lo, src_hi)

        self._map_cache[key] = full_map
        return full_map

    def _group_lonlat(self, group: str) -> tuple[np.ndarray, np.ndarray]:
        r = self._result
        if group.endswith("enda"):
            return r.lon_enda, r.lat_enda
        return r.lon_oper, r.lat_oper

    def _valid_resampler(self, group: str, var: str, valid: np.ndarray) -> PSFResampler:
        """PSFResampler built on the valid (non-NaN) source points of a field."""
        k = (group, var)
        sub = self._sub_resampler_cache.get(k)
        if sub is None:
            lon, lat = self._group_lonlat(group)
            log.info(
                f"Building masked PSFResampler for {group}/{var} "
                f"on {int(valid.sum()):,}/{valid.size:,} valid pts..."
            )
            sub = PSFResampler(
                lon_deg=lon[valid],
                lat_deg=lat[valid],
                level=_CHILD_LEVEL,
                threshold=0.5,
                verbose=False,
            )
            self._sub_resampler_cache[k] = sub
        return sub

    def consolidate(
        self,
        output_path: str | None = None,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        """Inject STAC metadata and consolidate zarr."""
        assert self._result is not None
        path = output_path or self._result.output_path
        if storage_options:
            root = zarr.open_group(path, mode="a", storage_options=storage_options)
        else:
            root = zarr.open_group(path, mode="a")
        root.attrs["stac_discovery"] = self._build_stac(path).model_dump()
        zarr.consolidate_metadata(root.store)
        log.info(f"ERA5 zarr consolidated: {path}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _grib_path(self, stream: str, grid: str) -> Path:
        tag = self.date.replace("-", "")
        return self.local_dir / f"era5_{stream}_{tag}T{self.time[:2]}_{grid}.grib"

    def _download_enda(self, path: Path, client: Any) -> None:
        log.info(f"Downloading ERA5 enda → {path}")
        client.retrieve(
            CDS_DATASET,
            {
                "class": "ea",
                "date": self.date,
                "expver": "1",
                "levtype": "sfc",
                "param": "/".join(v[0] for v in ERA5_ENDA_VARIABLE_META.values()),
                "step": self.step,
                "stream": "enda",
                "time": self.time,
                "type": "an",
                "grid": ERA5_ENDA_GRID,
                "number": "/".join(str(n) for n in self.numbers),
            },
            str(path),
        )

    def _download_oper(self, path: Path, client: Any) -> None:
        log.info(f"Downloading ERA5 oper → {path}")
        client.retrieve(
            CDS_DATASET,
            {
                "class": "ea",
                "date": self.date,
                "expver": "1",
                "levtype": "sfc",
                "param": "/".join(v[0] for v in ERA5_OPER_VARIABLE_META.values()),
                "step": self.step,
                "stream": "oper",
                "time": self.time,
                "type": "an",
                "grid": ERA5_OPER_GRID,
            },
            str(path),
        )

    @staticmethod
    def _extract_lonlat(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
        if "values" in ds.dims:
            return (
                ds.longitude.values.ravel().astype(float),
                ds.latitude.values.ravel().astype(float),
            )
        lat2d, lon2d = np.meshgrid(
            ds.latitude.values, ds.longitude.values, indexing="ij"
        )
        return lon2d.ravel().astype(float), lat2d.ravel().astype(float)

    def _init_zarr(
        self,
        output_path: str,
        n_times: int,
        vars_enda: list,
        vars_oper: list,
        start_dt: datetime,
        end_dt: datetime,
    ) -> None:
        root = zarr.open_group(output_path, mode="w")

        healpix_model = Healpix(
            refinement_level=_CHILD_LEVEL,
            indexing_scheme="nested",
            ellipsoid={"name": "wgs84"},
        )
        dggs_convention = DGGSZarrConvention().model_dump()

        for group, variables, meta in [
            ("measurements/enda", vars_enda, ERA5_ENDA_VARIABLE_META),
            ("measurements/oper", vars_oper, ERA5_OPER_VARIABLE_META),
        ]:
            grp = root.require_group(group)
            grp.attrs["zarr_conventions"] = [dggs_convention]
            grp.attrs["dggs"] = healpix_model.model_dump()
            grp.create_array(
                "cell_ids",
                shape=(_N_CHILD,),
                dtype=np.int64,
                chunks=(_CHUNK_SIZE,),
                dimension_names=("cells",),
            )
            for var in variables:
                _, unit, long_name = meta[var]
                grp.create_array(
                    var,
                    shape=(n_times, _N_CHILD),
                    dtype=np.float32,
                    chunks=(1, _CHUNK_SIZE),
                    fill_value=np.nan,
                    dimension_names=("time", "cells"),
                    attributes={"units": unit, "long_name": long_name},
                )
        zarr.consolidate_metadata(root.store)
        log.info(f"ERA5 zarr skeleton initialised: {output_path}")

    def _build_stac(self, output_path: str) -> StacItem:
        assert self._result is not None
        import shapely

        r = self._result

        # The STAC id must match the product name on disk: derive it from the
        # output path rather than a fresh timestamp (which would not match the
        # filename's creation stamp set at prepare() time).
        name = Path(str(output_path)).name
        product_id = name[:-5] if name.endswith(".zarr") else name

        return StacItem(
            stac_version="1.1.0",
            stac_extensions=[
                "https://stac-extensions.github.io/product/v1.0.0/schema.json"
            ],
            id=product_id,
            bbox=[-180.0, -90.0, 180.0, 90.0],
            geometry=shapely.geometry.mapping(shapely.box(-180, -90, 180, 90)),
            properties={
                "datetime": r.start_dt.isoformat(),
                "start_datetime": r.start_dt.isoformat(),
                "end_datetime": r.end_dt.isoformat(),
                "created": datetime.now(UTC).isoformat(),
                "product:type": "ADF_ECMWA",
                "description": (
                    f"ERA5 reanalysis surface fields (enda + oper streams) on "
                    f"HEALPix NESTED level {_CHILD_LEVEL} (~50 km/pixel)"
                ),
                "healpix:level": _CHILD_LEVEL,
                "healpix:nside": 2**_CHILD_LEVEL,
                "healpix:ordering": "NESTED",
                "source_grid": f"Gaussian {ERA5_ENDA_GRID}/{ERA5_OPER_GRID}",
                "source_dataset": CDS_DATASET,
                "resampling:method": (f"PSFResampler(level={_CHILD_LEVEL})"),
                "Conventions": "CF-1.9",
            },
            links=[],
            assets={},
        )
