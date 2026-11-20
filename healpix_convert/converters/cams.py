"""
healpix_convert/converters/cams.py
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
    ~1% of empty equatorial pixels are filled from their HEALPix neighbours.
  Both give 0 NaN / 0 negative at level 6; they correlate > 0.98. See
  ``healpix_convert/settings/cams.py`` for the rationale.

Usage
-----
    from healpix_convert.converters.cams import CAMSConverter

    converter = CAMSConverter(date="2025-01-01")           # method="psf"
    result    = converter.prepare(output_path="out.zarr")
    for idx in range(result.n_chunks):
        converter.convert_group(idx)
    converter.consolidate("out.zarr")

The dataset, the variable table and the extra request fields are arguments, e.g. a whole day
of the global analyses (lead time 0 at 00 and 12 UTC) with every field of CAMS_ALL_VARIABLE_META::

    CAMSConverter(
        date="2026-08-12", time="00:00/12:00",
        dataset=ADS_FORECAST_DATASET,
        request={"type": "forecast", "leadtime_hour": "0"},
        variables=CAMS_ALL_VARIABLE_META,
    )
"""

from __future__ import annotations

import hashlib
import logging
import os
import time as time_lib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import healpix_geo
import numpy as np
import requests
import xarray as xr
import zarr.api.synchronous as zarr
from healpix_resample import PSFResampler

from healpix_convert.core.healpix_conventions import Healpix
from healpix_convert.core.metadata import (
    CF_GRID_MAPPING_VARIABLE,
    get_cf_cell_id_attrs,
    get_cf_data_variable_attrs,
    get_cf_grid_mapping_attrs,
    get_healpix_group_attrs,
    write_root_conventions,
)
from healpix_convert.core.stac import STAC_PROCESSING_SCHEMA_URL, StacItem
from healpix_convert.settings.cams import (
    ADS_DATASET,
    ADS_URL,
    CAMS_PSF_LAM,
    CAMS_PSF_THRESHOLD,
    CAMS_VARIABLE_META,
)
from healpix_convert.settings.common import MetadataSettings

log = logging.getLogger(__name__)

# ── HEALPix constants ─────────────────────────────────────────────────────────
from healpix_convert.settings.cams import CAMS_CONVERT_SETTINGS as _S

_g = _S["group_settings"]["measurements/aod"]
_CHILD_LEVEL = _g["healpix"]["refinement_level"]  # 6
_CHUNK_LEVEL = _g["chunk"]["healpix"]["refinement_level"]  # 2
_N_CHILD = 12 * 4**_CHILD_LEVEL  # 49,152
_N_PARENT = 12 * 4**_CHUNK_LEVEL  # 192
_CHUNK_SIZE = 4 ** (_CHILD_LEVEL - _CHUNK_LEVEL)  # 256


def _nn_project(
    values: np.ndarray, lon: np.ndarray, lat: np.ndarray, level: int
) -> np.ndarray:
    """Nearest-neighbour binning from regular lat/lon → HEALPix NESTED (WGS84).
    Fills the few empty equatorial pixels using their HEALPix neighbours.
    Guaranteed zero NaN output.
    """
    n_pix = 12 * 4**level
    pix = healpix_geo.nested.lonlat_to_healpix(
        np.asarray(lon, dtype=np.float64),
        np.asarray(lat, dtype=np.float64),
        level,
        ellipsoid="WGS84",
    ).astype(np.int64)

    sums = np.zeros(n_pix, np.float64)
    cnts = np.zeros(n_pix, np.int32)
    np.add.at(sums, pix, values)
    np.add.at(cnts, pix, 1)
    with np.errstate(invalid="ignore"):
        result = np.where(cnts > 0, sums / cnts, np.nan).astype(np.float32)

    empty = np.flatnonzero(np.isnan(result))
    if empty.size:
        # all 8 neighbours of each empty pixel (-1 where there is none)
        neighbours = healpix_geo.nested.neighbours(empty.astype(np.uint64), level)
        for p, nb in zip(empty, neighbours):
            nb = nb[nb >= 0]
            v = nb[~np.isnan(result[nb])]
            if len(v):
                result[p] = float(np.mean(result[v]))
    return result


@dataclass
class CAMSPrepareResult:
    output_path: str
    n_chunks: int
    n_times: int
    start_dt: datetime
    end_dt: datetime
    nc_path: Path
    #: the analysis / valid times, in the order of the time axis
    times: list[datetime] = field(default_factory=list)


def _mars_list(value: str) -> list[str]:
    return [item for item in str(value).split("/") if item]


def _nonnegative(var: str, meta: tuple[str, str, str]) -> bool:
    """AOD fields (dimensionless optical depths) are clipped at 0; other fields to their source range."""
    return "optical_depth" in meta[0] or var.endswith("aod550") or var.startswith("aod")


class CAMSConverter:
    """
    Converter for CAMS EAC4 aerosol reanalysis to EOPF-DGGS HEALPix zarr.

    Parameters
    ----------
    date          : "YYYY-MM-DD"
    time          : "HH:MM", or several as "HH:MM/HH:MM/..." (one store, a time axis)
    local_dir     : directory for local NetCDF cache
    method        : "psf" (default) or "nn" — resampling onto HEALPix
    psf_threshold : PSFResampler cell-coverage threshold (only for method="psf")
    psf_lam       : Tikhonov damping passed to resample() (only for method="psf")
    dataset       : ADS dataset, default ``cams-global-reanalysis-eac4`` (EAC4);
                    ``ADS_FORECAST_DATASET`` for dates after the end of EAC4
    request       : more fields of the ADS request, e.g. ``{"type": "forecast",
                    "leadtime_hour": "0"}`` for the analyses of the forecast dataset
    variables     : ``{name: (ADS variable name, unit, long name)}``, default
                    CAMS_VARIABLE_META (the six 550 nm AOD fields)

    AOD fields are clipped at 0, other fields (``z``) to the range of their source
    values. With several times the group gets a CF ``time`` coordinate.
    The ADS key is read from ``~/.adsapirc`` if it exists, else from ``~/.cdsapirc``
    (one ECMWF account for both).
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
        metadata: MetadataSettings | None = None,
        dataset: str = ADS_DATASET,
        request: Mapping[str, Any] | None = None,
        variables: Mapping[str, tuple[str, str, str]] | None = None,
    ):
        if method not in ("psf", "nn"):
            raise ValueError(f"method must be 'psf' or 'nn', got {method!r}")
        self.metadata = metadata if metadata is not None else MetadataSettings()
        self.date = date
        self.time = time
        self.times = _mars_list(time)
        if not self.times:
            raise ValueError(f"no time in {time!r}")
        self.dataset = dataset
        self.request = dict(request or {})
        self.variables = dict(CAMS_VARIABLE_META if variables is None else variables)
        if not self.variables:
            raise ValueError("no variable to convert")
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
        missing = [v for v in self.variables if v not in self._ds.data_vars]
        if missing:
            raise ValueError(
                f"{nc_path.name}: requested but not in the file: {missing}"
            )
        start_dt, end_dt = self._get_time_range()
        times = self._times()

        n_times = self._ds.sizes.get("valid_time", self._ds.sizes.get("time", 1))
        n_chunks = _N_PARENT

        self._init_zarr(output_path, n_times, start_dt, end_dt, times=times)
        self._result = CAMSPrepareResult(
            output_path=output_path,
            n_chunks=n_chunks,
            n_times=n_times,
            start_dt=start_dt,
            end_dt=end_dt,
            nc_path=nc_path,
            times=times,
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

        for var in self.variables:
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
        repeated for each of the 192 spatial chunks. AOD output is clipped to >= 0
        (AOD is non-negative) regardless of the method; other fields (the surface
        geopotential) to the range of their source values.
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
            if _nonnegative(var, self.variables[var]):
                full_map = np.clip(full_map, 0.0, None).astype(np.float32)
            else:
                full_map = np.clip(full_map, np.nanmin(vals), np.nanmax(vals)).astype(
                    np.float32
                )
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
        """The NetCDF cache file; the default dataset, table and one time keep
        the names of earlier versions."""
        tag = self.date.replace("-", "")
        if len(self.times) == 1:
            tag += f"T{self.times[0][:2]}"
        else:
            tag += f"T{self.times[0][:2]}-{self.times[-1][:2]}n{len(self.times)}"
        default = (
            self.dataset == ADS_DATASET
            and not self.request
            and self.variables == dict(CAMS_VARIABLE_META)
            and len(self.times) == 1
        )
        if not default:
            key = "|".join(
                [self.dataset, "/".join(self.times), repr(sorted(self.request.items()))]
                + [v[0] for v in self.variables.values()]
            )
            tag += "_v" + hashlib.sha1(key.encode()).hexdigest()[:6]
        return self.local_dir / f"cams_{tag}.nc"

    def requests(self) -> dict[str, Any]:
        """The ADS request (dataset and inputs), as sent by the download."""
        return {
            "dataset": self.dataset,
            "inputs": {
                "variable": [v[0] for v in self.variables.values()],
                "date": self.date,
                "time": [t[:5] for t in self.times]
                if len(self.times) > 1
                else self.times[0][:5],
                "format": "netcdf",
                **self.request,
            },
        }

    def _ads_key(self) -> str:
        for rc in ("~/.adsapirc", "~/.cdsapirc"):
            path = os.path.expanduser(rc)
            if os.path.exists(path):
                with open(path) as f:
                    return next(
                        line.split(":", 1)[1].strip()
                        for line in f
                        if line.startswith("key:")
                    )
        raise FileNotFoundError("no ADS key: ~/.adsapirc or ~/.cdsapirc")

    def _download(self, nc_path: Path) -> None:
        """Download CAMS EAC4 via ADS REST API (bypasses cdsapi to avoid grid injection)."""
        log.info(f"Downloading CAMS {self.dataset} → {nc_path}")
        key = self._ads_key()
        headers = {"PRIVATE-TOKEN": key, "Content-Type": "application/json"}
        base = f"{ADS_URL.rstrip('/')}/retrieve/v1"

        req = self.requests()
        r = requests.post(
            f"{base}/processes/{req['dataset']}/execution",
            headers=headers,
            json={"inputs": req["inputs"]},
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
            # a partial file is never taken for the cache: written aside, then renamed
            part = nc_path.with_name(nc_path.name + ".part")
            with open(part, "wb") as fout:
                for chunk in resp.iter_content(1024 * 1024):
                    fout.write(chunk)
            part.replace(nc_path)
        log.info("CAMS download complete.")

    def _load_dataset(self, nc_path: Path) -> None:
        if self._ds is None:
            import importlib.util

            # the ADS sends NetCDF-4: netCDF4 when installed, else what xarray finds
            engine = "netcdf4" if importlib.util.find_spec("netCDF4") else None
            self._ds = _one_time_axis(xr.open_dataset(nc_path, engine=engine))
            lat2d, lon2d = np.meshgrid(
                self._ds.latitude.values, self._ds.longitude.values, indexing="ij"
            )
            self._lon = lon2d.ravel().astype(np.float64)
            self._lat = lat2d.ravel().astype(np.float64)

    def _times(self) -> list[datetime]:
        import pandas as pd

        coord = "valid_time" if "valid_time" in self._ds.coords else "time"
        return [
            pd.Timestamp(t).tz_localize(None).to_pydatetime().replace(tzinfo=UTC)
            for t in np.atleast_1d(self._ds[coord].values).ravel()
        ]

    def _grid_step(self) -> str:
        lat = np.unique(np.round(np.asarray(self._ds.latitude.values, dtype=float), 6))
        return f"{float(np.median(np.diff(lat))):g}" if lat.size > 1 else "?"

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
        self,
        output_path: str,
        n_times: int,
        start_dt: datetime,
        end_dt: datetime,
        times: list[datetime] | None = None,
    ) -> None:
        healpix_model = Healpix(
            refinement_level=_CHILD_LEVEL,
            indexing_scheme="nested",
            ellipsoid={"name": "wgs84"},
        )
        root = zarr.open_group(output_path, mode="w")
        write_root_conventions(root, self.metadata)

        grp = root.require_group("measurements/aod")
        grp.attrs.update(get_healpix_group_attrs(healpix_model, self.metadata))

        grp.create_array(
            "cell_ids",
            shape=(_N_CHILD,),
            dtype=np.int64,
            chunks=(_CHUNK_SIZE,),
            dimension_names=("cells",),
            attributes=get_cf_cell_id_attrs(self.metadata),
        )

        # CF HEALPix grid mapping variable, alongside the DGGS-Zarr convention
        grid_mapping_attrs = get_cf_grid_mapping_attrs(healpix_model, self.metadata)
        if grid_mapping_attrs is not None:
            crs = grp.create_array(
                CF_GRID_MAPPING_VARIABLE,
                shape=(),
                dtype=np.int8,
                attributes=grid_mapping_attrs,
            )
            crs[...] = 0

        arr = grp.create_array(
            "number",
            shape=(n_times,),
            dtype=np.int32,
            chunks=(n_times,),
            dimension_names=("number",),
        )
        arr[:] = np.arange(n_times)

        if times and len(times) > 1:
            # the times in hours since 00 UTC of the first day (CF)
            epoch = times[0].replace(hour=0, minute=0, second=0, microsecond=0)
            time_arr = grp.create_array(
                "time",
                shape=(len(times),),
                dtype=np.int64,
                chunks=(len(times),),
                dimension_names=("time",),
                attributes={
                    "standard_name": "time",
                    "axis": "T",
                    "units": f"hours since {epoch:%Y-%m-%d %H:%M:%S}",
                    "calendar": "proleptic_gregorian",
                },
            )
            time_arr[:] = [(t - epoch) // timedelta(hours=1) for t in times]

        cf_variable_attrs = get_cf_data_variable_attrs(healpix_model, self.metadata)

        for var, (_, unit, long_name) in self.variables.items():
            grp.create_array(
                var,
                shape=(n_times, _N_CHILD),
                dtype=np.float32,
                chunks=(1, _CHUNK_SIZE),
                fill_value=np.nan,
                dimension_names=("time", "cells"),
                attributes={
                    "units": unit,
                    "long_name": long_name,
                    **(
                        {"valid_min": 0.0}
                        if _nonnegative(var, self.variables[var])
                        else {}
                    ),
                    **cf_variable_attrs,
                },
            )
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
        now = datetime.now(UTC).isoformat()

        return StacItem(
            stac_version="1.1.0",
            stac_extensions=[
                "https://stac-extensions.github.io/product/v1.0.0/schema.json",
                STAC_PROCESSING_SCHEMA_URL,
            ],
            id=product_id,
            bbox=[-180.0, -90.0, 180.0, 90.0],
            geometry=shapely.geometry.mapping(shapely.box(-180, -90, 180, 90)),
            properties={
                "datetime": r.start_dt.isoformat(),
                "start_datetime": r.start_dt.isoformat(),
                "end_datetime": r.end_dt.isoformat(),
                "created": now,
                "product:type": "ADF_CAMSA",
                "description": (
                    f"CAMS EAC4 reanalysis aerosol optical depth on "
                    f"HEALPix NESTED level {_CHILD_LEVEL} (~100 km/pixel)"
                    if self.dataset == ADS_DATASET
                    and self.variables == dict(CAMS_VARIABLE_META)
                    else f"CAMS {self.dataset} ({', '.join(self.variables)}) on "
                    f"HEALPix NESTED level {_CHILD_LEVEL} (~100 km/pixel)"
                ),
                "healpix:level": _CHILD_LEVEL,
                "healpix:nside": 2**_CHILD_LEVEL,
                "healpix:ordering": "NESTED",
                "source_grid": f"{self.dataset} regular {self._grid_step()}°",
                "source_dataset": self.dataset,
                "source_requests": self.requests(),
                "resampling:method": (
                    f"PSFResampler(threshold={self.psf_threshold}, lam={self.psf_lam})"
                    if self.method == "psf"
                    else "nearest-neighbour binning"
                ),
                "resampling:parameters": {
                    "resampler": (
                        "healpix_resample.PSFResampler"
                        if self.method == "psf"
                        else "nearest-neighbour binning"
                    ),
                    "level": _CHILD_LEVEL,
                    "indexing_scheme": "nested",
                    "ellipsoid": "WGS84",
                    "threshold": self.psf_threshold if self.method == "psf" else None,
                    "lam": self.psf_lam if self.method == "psf" else None,
                    "chunk_level": _CHUNK_LEVEL,
                    "clip": "AOD at 0; other fields to the source range of each field and time",
                },
                "processing:datetime": now,
                "processing:software": _software_versions(),
                "processing:lineage": (
                    "Resampled onto HEALPix by healpix-convert CAMSConverter from the "
                    f"ADS dataset {self.dataset} (request in source_requests)"
                ),
            },
            links=[],
            assets={},
        )


def _one_time_axis(ds: xr.Dataset) -> xr.Dataset:
    """A NetCDF of the forecast dataset has two time dimensions (forecast_reference_time,
    forecast_period); give it one ``valid_time`` axis like the EAC4 files (valid time =
    reference time + period, in that order)."""
    import pandas as pd

    dims = [d for d in ("forecast_reference_time", "forecast_period") if d in ds.dims]
    if not dims:
        if "valid_time" in ds.dims or "time" in ds.dims:
            return ds
        # one time: reference time and period (or the valid time) as scalar coordinates
        if "valid_time" in ds.coords:
            return ds.expand_dims("valid_time")
        if "forecast_reference_time" in ds.coords and "forecast_period" in ds.coords:
            ref = pd.Timestamp(ds["forecast_reference_time"].values)
            per = ds["forecast_period"].values
            per = (
                pd.Timedelta(hours=float(per))
                if np.issubdtype(np.asarray(per).dtype, np.number)
                else pd.Timedelta(per)
            )
            ds = ds.drop_vars(["forecast_reference_time", "forecast_period"])
            return ds.expand_dims(valid_time=[np.datetime64(ref + per, "ns")])
        return ds
    if "valid_time" in ds.coords and set(ds["valid_time"].dims) == set(dims):
        valid = ds["valid_time"].transpose(*dims).values.ravel()
    else:
        ref = pd.to_datetime(np.atleast_1d(ds["forecast_reference_time"].values))
        per = (
            pd.to_timedelta(np.atleast_1d(ds["forecast_period"].values), unit="h")
            if np.issubdtype(np.asarray(ds["forecast_period"].values).dtype, np.number)
            else pd.to_timedelta(np.atleast_1d(ds["forecast_period"].values))
        )
        valid = np.array([r + p for r in ref for p in per], dtype="datetime64[ns]")
    ds = ds.drop_vars([c for c in ("valid_time",) if c in ds.coords])
    ds = ds.stack(valid_time=dims, create_index=False)
    ds = ds.drop_vars([d for d in dims if d in ds.coords])
    return ds.assign_coords(valid_time=("valid_time", valid)).transpose(
        "valid_time", ...
    )


def _software_versions() -> dict[str, str]:
    from healpix_convert.converters.era5 import _software_versions as versions

    return versions()
