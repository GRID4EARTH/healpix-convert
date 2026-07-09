"""
legacy_converters/converters/cams.py
======================================
CAMS EAC4 legacy converter.

Downloads CAMS reanalysis AOD from the ADS REST API, projects onto HEALPix
level 6, and writes an EOPF-compliant zarr.

Resampling method (``method=`` argument, default ``"psf"``)
  - ``"psf"``: PSFResampler (same engine as ERA5), tuned for the regular CAMS
    grid with ``threshold=0.01`` (full cell coverage, 0 NaN) and ``lam=5.0``
    (Tikhonov damping, removes the Gibbs ringing that otherwise yields negative
    AOD on a regular grid). A final ``clip(0, None)`` is the physical safety net.
  - ``"nn"``: nearest-neighbour binning — exact, artifact-free and faster; the
    ~1% of empty equatorial pixels are filled from HEALPix ring neighbours.
  Both give 0 NaN / 0 negative at level 6; they correlate > 0.98. See
  ``legacy_converters/settings/cams.py`` for the rationale.

Usage
-----
    from legacy_converters.converters.cams import CAMSConverter

    converter = CAMSConverter(date="2025-01-01")           # method="psf"
    result    = converter.prepare(output_path="out.zarr")
    for idx in range(result.n_chunks):
        converter.convert_group(idx)
    converter.consolidate("out.zarr")
"""

from __future__ import annotations

import logging
import os
import time as time_lib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import healpix_geo
import healpy as hp
import numpy as np
import requests
import xarray as xr
import zarr.api.synchronous as zarr
from healpix_resample import PSFResampler

from legacy_converters.core.healpix_conventions import (
    DGGSZarrConvention,
    Healpix,
    write_cf_grid_mapping,
)
from legacy_converters.core.stac import StacItem
from legacy_converters.settings.cams import (
    ADS_DATASET,
    ADS_URL,
    CAMS_PSF_LAM,
    CAMS_PSF_THRESHOLD,
    CAMS_VARIABLE_META,
)

log = logging.getLogger(__name__)

# ── HEALPix constants ─────────────────────────────────────────────────────────
from legacy_converters.settings.cams import CAMS_CONVERT_SETTINGS as _S

_g = _S["group_settings"]["measurements/aod"]
_CHILD_LEVEL = _g["healpix"]["refinement_level"]  # 6
_CHUNK_LEVEL = _g["chunk"]["healpix"]["refinement_level"]  # 2
_N_CHILD = 12 * 4**_CHILD_LEVEL  # 49,152
_N_PARENT = 12 * 4**_CHUNK_LEVEL  # 192
_CHUNK_SIZE = 4 ** (_CHILD_LEVEL - _CHUNK_LEVEL)  # 256


def _nn_project(
    values: np.ndarray, lon: np.ndarray, lat: np.ndarray, level: int
) -> np.ndarray:
    """Nearest-neighbour binning from regular lat/lon → HEALPix NESTED.
    Fills the few empty equatorial pixels using HEALPix ring neighbours.
    Guaranteed zero NaN output.
    """
    nside = 2**level
    n_pix = 12 * nside**2
    theta = np.deg2rad(90.0 - lat)
    phi = np.deg2rad(lon % 360.0)
    pix = hp.ang2pix(nside, theta, phi, nest=True)

    sums = np.zeros(n_pix, np.float64)
    cnts = np.zeros(n_pix, np.int32)
    np.add.at(sums, pix, values)
    np.add.at(cnts, pix, 1)
    with np.errstate(invalid="ignore"):
        result = np.where(cnts > 0, sums / cnts, np.nan).astype(np.float32)

    r_ring = hp.reorder(result, n2r=True)
    for p in np.where(np.isnan(r_ring))[0]:
        nb = hp.get_all_neighbours(nside, p)
        v = nb[(nb >= 0) & ~np.isnan(r_ring[nb])]
        if len(v):
            r_ring[p] = float(np.nanmean(r_ring[v]))
    return hp.reorder(r_ring, r2n=True)


@dataclass
class CAMSPrepareResult:
    output_path: str
    n_chunks: int
    n_times: int
    start_dt: datetime
    end_dt: datetime
    nc_path: Path


class CAMSConverter:
    """
    Converter for CAMS EAC4 aerosol reanalysis to EOPF-DGGS HEALPix zarr.

    Parameters
    ----------
    date          : "YYYY-MM-DD"
    time          : "HH:MM"
    local_dir     : directory for local NetCDF cache
    method        : "psf" (default) or "nn" — resampling onto HEALPix
    psf_threshold : PSFResampler cell-coverage threshold (only for method="psf")
    psf_lam       : Tikhonov damping passed to resample() (only for method="psf")
    """

    settings = CAMS_VARIABLE_META

    def __init__(
        self,
        date: str,
        time: str = "00:00",
        local_dir: Path = Path("."),
        method: str = "psf",
        psf_threshold: float = CAMS_PSF_THRESHOLD,
        psf_lam: float = CAMS_PSF_LAM,
    ):
        if method not in ("psf", "nn"):
            raise ValueError(f"method must be 'psf' or 'nn', got {method!r}")
        self.date = date
        self.time = time
        self.local_dir = Path(local_dir)
        self.method = method
        self.psf_threshold = psf_threshold
        self.psf_lam = psf_lam
        self._ds: xr.Dataset | None = None
        self._lon: np.ndarray | None = None
        self._lat: np.ndarray | None = None
        self._result: CAMSPrepareResult | None = None
        self._resampler: PSFResampler | None = None
        self._map_cache: dict = {}  # (var, t) -> full-globe HEALPix map

    # ── Step 1: prepare ───────────────────────────────────────────────────────

    def prepare(
        self, output_path: str, force_download: bool = False
    ) -> CAMSPrepareResult:
        """Download CAMS NetCDF (if needed) and initialise the zarr skeleton."""
        nc_path = self._nc_path()

        if not nc_path.exists() or force_download:
            self._download(nc_path)
        else:
            log.info(f"CAMS already cached: {nc_path}")

        self._load_dataset(nc_path)
        start_dt, end_dt = self._get_time_range()

        n_times = self._ds.sizes.get("valid_time", self._ds.sizes.get("time", 1))
        n_chunks = _N_PARENT

        self._init_zarr(output_path, n_times, start_dt, end_dt)
        self._result = CAMSPrepareResult(
            output_path=output_path,
            n_chunks=n_chunks,
            n_times=n_times,
            start_dt=start_dt,
            end_dt=end_dt,
            nc_path=nc_path,
        )
        log.info(f"CAMS prepare done: {n_chunks} chunks, {n_times} timestep(s)")
        return self._result

    # ── Step 2: convert_group ─────────────────────────────────────────────────

    def convert_group(self, chunk_index: int) -> None:
        assert self._result is not None, "call prepare() first"
        self._load_dataset(self._result.nc_path)

        cell_ids = healpix_geo.nested.zoom_to(
            chunk_index,
            depth=_CHUNK_LEVEL,
            new_depth=_CHILD_LEVEL,
        )[0]
        start = chunk_index * _CHUNK_SIZE
        chunk_sl = slice(start, start + _CHUNK_SIZE)
        t_dim = "valid_time" if "valid_time" in self._ds.dims else "time"
        n_t = self._result.n_times
        root = zarr.open_group(self._result.output_path, mode="a")
        grp = root["measurements/aod"]

        for var in CAMS_VARIABLE_META:
            if var not in self._ds.data_vars:
                continue
            zarr_a = grp[var]
            for t in range(n_t):
                full_map = self._full_map(var, t, t_dim)
                zarr_a[t, chunk_sl] = full_map[cell_ids]

        grp["cell_ids"][chunk_sl] = cell_ids
        log.debug(f"Chunk {chunk_index + 1}/{_N_PARENT} written.")

    def _get_resampler(self) -> PSFResampler:
        """Build the PSFResampler once on the source grid and reuse it."""
        if self._resampler is None:
            log.info(
                f"Building PSFResampler level={_CHILD_LEVEL} "
                f"threshold={self.psf_threshold} on {self._lon.size:,} points..."
            )
            self._resampler = PSFResampler(
                lon_deg=self._lon,
                lat_deg=self._lat,
                level=_CHILD_LEVEL,
                threshold=self.psf_threshold,
                verbose=False,
            )
        return self._resampler

    def _full_map(self, var: str, t: int, t_dim: str) -> np.ndarray:
        """Full-globe HEALPix map for one variable/timestep.

        Computed once per (var, t) and cached, so the global resampling is not
        repeated for each of the 192 spatial chunks. Output is clipped to >= 0
        (AOD is non-negative) regardless of the method.
        """
        key = (var, t)
        full_map = self._map_cache.get(key)
        if full_map is None:
            da = self._ds[var]
            vals = (
                da.isel({t_dim: t}).values.ravel().astype(np.float64)
                if t_dim in da.dims
                else da.values.ravel().astype(np.float64)
            )
            if self.method == "psf":
                full_map = np.full(_N_CHILD, np.nan, dtype=np.float32)
                res = self._get_resampler().resample(vals, lam=self.psf_lam)
                full_map[res.cell_ids] = res.cell_data.astype(np.float32)
            else:  # "nn"
                full_map = _nn_project(vals, self._lon, self._lat, _CHILD_LEVEL)
            full_map = np.clip(full_map, 0.0, None).astype(np.float32)
            self._map_cache[key] = full_map
        return full_map

    # ── Step 3: consolidate ───────────────────────────────────────────────────

    def consolidate(
        self,
        output_path: str | None = None,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        assert self._result is not None
        path = output_path or self._result.output_path
        stac = self._build_stac(path)
        if storage_options:
            root = zarr.open_group(path, mode="a", storage_options=storage_options)
        else:
            root = zarr.open_group(path, mode="a")
        root.attrs["stac_discovery"] = stac.model_dump()
        zarr.consolidate_metadata(root.store)
        log.info(f"CAMS zarr consolidated: {path}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _nc_path(self) -> Path:
        tag = self.date.replace("-", "")
        return self.local_dir / f"cams_{tag}T{self.time[:2].replace(':', '')}.nc"

    def _ads_key(self) -> str:
        rc = os.path.expanduser("~/.cdsapirc")
        with open(rc) as f:
            return next(
                line.split(":", 1)[1].strip() for line in f if line.startswith("key:")
            )

    def _download(self, nc_path: Path) -> None:
        """Download CAMS EAC4 via ADS REST API (bypasses cdsapi to avoid grid injection)."""
        log.info(f"Downloading CAMS EAC4 → {nc_path}")
        key = self._ads_key()
        headers = {"PRIVATE-TOKEN": key, "Content-Type": "application/json"}
        base = f"{ADS_URL.rstrip('/')}/retrieve/v1"

        r = requests.post(
            f"{base}/processes/{ADS_DATASET}/execution",
            headers=headers,
            json={
                "inputs": {
                    "variable": [v[0] for v in CAMS_VARIABLE_META.values()],
                    "date": self.date,
                    "time": self.time[:5],
                    "format": "netcdf",
                }
            },
            timeout=30,
        )
        r.raise_for_status()
        job_id = r.json()["jobID"]
        log.info(f"  job: {job_id}")

        for _ in range(120):
            j = requests.get(
                f"{base}/jobs/{job_id}", headers=headers, timeout=10
            ).json()
            log.info(f"  status: {j['status']}")
            if j["status"] == "successful":
                break
            if j["status"] == "failed":
                raise RuntimeError(f"CAMS job {job_id} failed")
            time_lib.sleep(15)

        dl_url = requests.get(
            f"{base}/jobs/{job_id}/results", headers=headers, timeout=10
        ).json()["asset"]["value"]["href"]
        with requests.get(dl_url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(nc_path, "wb") as fout:
                for chunk in resp.iter_content(1024 * 1024):
                    fout.write(chunk)
        log.info("CAMS download complete.")

    def _load_dataset(self, nc_path: Path) -> None:
        if self._ds is None:
            self._ds = xr.open_dataset(nc_path, engine="netcdf4")
            lat2d, lon2d = np.meshgrid(
                self._ds.latitude.values, self._ds.longitude.values, indexing="ij"
            )
            self._lon = lon2d.ravel().astype(np.float64)
            self._lat = lat2d.ravel().astype(np.float64)

    def _get_time_range(self) -> tuple[datetime, datetime]:
        import pandas as pd

        coord = "valid_time" if "valid_time" in self._ds.coords else "time"
        ts = [
            pd.Timestamp(t).tz_localize(None).to_pydatetime()
            for t in np.atleast_1d(self._ds[coord].values).ravel()
        ]
        return (
            min(ts).replace(tzinfo=UTC),
            max(ts).replace(tzinfo=UTC),
        )

    def _init_zarr(
        self, output_path: str, n_times: int, start_dt: datetime, end_dt: datetime
    ) -> None:
        healpix_model = Healpix(
            refinement_level=_CHILD_LEVEL,
            indexing_scheme="nested",
            ellipsoid={"name": "wgs84"},
        )
        dggs_convention = DGGSZarrConvention().model_dump()

        root = zarr.open_group(output_path, mode="w")
        grp = root.require_group("measurements/aod")
        grp.attrs["zarr_conventions"] = [dggs_convention]
        grp.attrs["dggs"] = healpix_model.model_dump()

        grp.create_array(
            "cell_ids",
            shape=(_N_CHILD,),
            dtype=np.int64,
            chunks=(_CHUNK_SIZE,),
            dimension_names=("cells",),
        )
        arr = grp.create_array(
            "number",
            shape=(n_times,),
            dtype=np.int32,
            chunks=(n_times,),
            dimension_names=("number",),
        )
        arr[:] = np.arange(n_times)

        for var, (_, unit, long_name) in CAMS_VARIABLE_META.items():
            grp.create_array(
                var,
                shape=(n_times, _N_CHILD),
                dtype=np.float32,
                chunks=(1, _CHUNK_SIZE),
                fill_value=np.nan,
                dimension_names=("time", "cells"),
                attributes={"units": unit, "long_name": long_name, "valid_min": 0.0},
            )
        # CF 1.13 HEALPix grid mapping alongside the DGGS-Zarr convention
        write_cf_grid_mapping(grp, healpix_model, CAMS_VARIABLE_META)
        zarr.consolidate_metadata(root.store)
        log.info(f"CAMS zarr skeleton initialised: {output_path}")

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
                "product:type": "ADF_CAMSA",
                "description": (
                    f"CAMS EAC4 reanalysis aerosol optical depth on "
                    f"HEALPix NESTED level {_CHILD_LEVEL} (~100 km/pixel)"
                ),
                "healpix:level": _CHILD_LEVEL,
                "healpix:nside": 2**_CHILD_LEVEL,
                "healpix:ordering": "NESTED",
                "source_grid": f"{ADS_DATASET} regular 0.75°",
                "source_dataset": ADS_DATASET,
                "resampling:method": (
                    f"PSFResampler(threshold={self.psf_threshold}, lam={self.psf_lam})"
                    if self.method == "psf"
                    else "nearest-neighbour binning"
                ),
                "Conventions": "CF-1.9",
            },
            links=[],
            assets={},
        )
