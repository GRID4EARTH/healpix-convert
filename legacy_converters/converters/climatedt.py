"""
legacy_converters/converters/climatedt.py
==========================================
ClimateDT (IFS-NEMO) legacy converter.

ClimateDT data is natively on HEALPix level 7 NESTED — no regridding needed.
This converter downloads via Polytope, reads the GRIB (already on HEALPix),
and packs it directly into an EOPF-compliant zarr dataset.

Optional sphere → WGS84 ellipsoid correction (PSFResampler) can be enabled
for rigorous treatment of the coordinate system difference. It adds ~5-10 s
per timestep but is usually not needed at level 7 precision.

Usage
-----
    from legacy_converters.converters.climatedt import ClimateDTConverter

    # Surface fields
    converter = ClimateDTConverter(date="20200102", time="0100")
    result    = converter.prepare(output_path="out.zarr")
    for idx in range(result.n_chunks):
        converter.convert_group(idx)
    converter.consolidate("out.zarr")

    # Ocean 3D temperature (depth levels 1-5)
    converter = ClimateDTConverter(
        date="20200102", time="0000",
        params="263501", levtype="o3d", levelist="1/2/3/4/5"
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cfgrib
import healpix_geo
import healpix_resample
import numpy as np
import xarray as xr
import zarr.api.synchronous as zarr

from legacy_converters.core.healpix_conventions import (
    DGGSZarrConvention,
    Healpix,
    write_cf_grid_mapping,
)
from legacy_converters.core.stac import StacItem
from legacy_converters.settings.climatedt import (
    CDT_CLASS,
    CDT_DATASET,
    CDT_DEFAULT_SFC_PARAMS,
    CDT_EXPERIMENT,
    CDT_EXPVER,
    CDT_GENERATION,
    CDT_MODEL,
    CDT_OCEAN_VARIABLE_META,
    CDT_RESOLUTION,
    CDT_SFC_VARIABLE_META,
    CDT_STREAM,
    CDT_TYPE,
    POLYTOPE_ADDRESS,
    POLYTOPE_COLLECTION,
)

log = logging.getLogger(__name__)

# ── HEALPix constants ─────────────────────────────────────────────────────────
from legacy_converters.settings.climatedt import CDT_SFC_CONVERT_SETTINGS as _S

_g = _S["group_settings"]["measurements/sfc"]
_CHILD_LEVEL = _g["healpix"]["refinement_level"]  # 7
_CHUNK_LEVEL = _g["chunk"]["healpix"]["refinement_level"]  # 2
_N_CHILD = 12 * 4**_CHILD_LEVEL  # 196,608
_N_PARENT = 12 * 4**_CHUNK_LEVEL  # 192
_CHUNK_SIZE = 4 ** (_CHILD_LEVEL - _CHUNK_LEVEL)  # 1,024


@dataclass
class CDTPrepareResult:
    output_path: str
    n_chunks: int
    n_times: int
    start_dt: datetime
    end_dt: datetime
    grib_path: Path
    is_ocean: bool
    var_meta: dict  # {var: (param_id, unit, long_name)}


class ClimateDTConverter:
    """
    Converter for ClimateDT IFS-NEMO output to EOPF-DGGS HEALPix zarr.

    Data is natively on HEALPix level 7 — direct copy, no reprojection.

    Parameters
    ----------
    date                : "YYYYMMDD"
    time                : "HHMM" (e.g. "0100")
    params              : MARS param ids (default surface: "134/165/166/167/168")
    levtype             : "sfc" | "o3d"
    levelist            : depth levels for ocean 3D (e.g. "1/2/3/4/5")
    activity            : ScenarioMIP | ...
    ellipsoid_correction: if True, apply PSF sphere→WGS84 correction (~5-10 s/timestep)
    local_dir           : cache directory
    """

    def __init__(
        self,
        date: str,
        time: str = "0000",
        params: str = CDT_DEFAULT_SFC_PARAMS,
        levtype: str = "sfc",
        levelist: str | None = None,
        activity: str = "ScenarioMIP",
        experiment: str = CDT_EXPERIMENT,
        expver: str = CDT_EXPVER,
        generation: str = CDT_GENERATION,
        model: str = CDT_MODEL,
        resolution: str = CDT_RESOLUTION,
        realization: str = "1",
        ellipsoid_correction: bool = False,
        local_dir: Path = Path("."),
    ):
        self.date = date
        self.time = time
        self.params = params
        self.levtype = levtype
        self.levelist = levelist
        self.activity = activity
        self.experiment = experiment
        self.expver = expver
        self.generation = generation
        self.model = model
        self.resolution = resolution
        self.realization = realization
        self.ellipsoid_correction = ellipsoid_correction
        self.local_dir = Path(local_dir)

        self._ds: xr.Dataset | None = None
        self._nr: healpix_resample.PSFResampler | None = None
        self._result: CDTPrepareResult | None = None

    # ── Step 1: prepare ───────────────────────────────────────────────────────

    def prepare(
        self, output_path: str, force_download: bool = False
    ) -> CDTPrepareResult:
        """Download ClimateDT GRIB (if needed) and initialise the zarr skeleton."""
        grib_path = self._grib_path()

        if not grib_path.exists() or force_download:
            self._download(grib_path)
        else:
            log.info(f"ClimateDT already cached: {grib_path}")

        self._load_dataset(grib_path)
        start_dt, end_dt = self._get_time_range()

        is_ocean = self.levtype == "o3d"
        var_meta = CDT_OCEAN_VARIABLE_META if is_ocean else CDT_SFC_VARIABLE_META
        # keep only vars present in the dataset
        var_meta = {k: v for k, v in var_meta.items() if k in self._ds.data_vars}

        n_times = max(
            self._ds.sizes.get("valid_time", self._ds.sizes.get("time", 1)), 1
        )
        n_chunks = _N_PARENT

        self._init_zarr(output_path, n_times, start_dt, end_dt, var_meta, is_ocean)

        self._result = CDTPrepareResult(
            output_path=output_path,
            n_chunks=n_chunks,
            n_times=n_times,
            start_dt=start_dt,
            end_dt=end_dt,
            grib_path=grib_path,
            is_ocean=is_ocean,
            var_meta=var_meta,
        )
        log.info(f"ClimateDT prepare done: {n_chunks} chunks, {n_times} timestep(s)")
        return self._result

    # ── Step 2: convert_group ─────────────────────────────────────────────────

    def convert_group(self, chunk_index: int) -> None:
        """Copy (or correct) ClimateDT data for one HEALPix parent cell."""
        assert self._result is not None, "call prepare() first"
        self._load_dataset(self._result.grib_path)

        cell_ids = healpix_geo.nested.zoom_to(
            chunk_index,
            depth=_CHUNK_LEVEL,
            new_depth=_CHILD_LEVEL,
        )[0]
        start = chunk_index * _CHUNK_SIZE
        chunk_sl = slice(start, start + _CHUNK_SIZE)
        r = self._result
        root = zarr.open_group(r.output_path, mode="a")
        t_dim = next((d for d in ["valid_time", "time"] if d in self._ds.dims), None)
        grp_name = "measurements/ocean" if r.is_ocean else "measurements/sfc"
        grp = root[grp_name]

        for var in r.var_meta:
            if var not in self._ds.data_vars:
                continue
            da = self._ds[var]
            zarr_a = grp[var]

            for t in range(r.n_times):
                vals = (
                    da.isel({t_dim: t}).values.ravel().astype(np.float32)
                    if t_dim
                    else da.values.ravel().astype(np.float32)
                )

                if self.ellipsoid_correction:
                    # Sphere → WGS84 ellipsoid via PSFResampler
                    if self._nr is None:
                        self._build_resampler()
                    res = self._nr.resample(vals.astype(np.float64), lam=0.0)
                    full_map = np.full(_N_CHILD, np.nan, dtype=np.float32)
                    full_map[res.cell_ids] = res.cell_data.astype(np.float32)
                    zarr_a[t, chunk_sl] = full_map[
                        cell_ids.astype(int) - cell_ids[0].astype(int)
                    ]
                else:
                    # Direct copy: data is already on the target HEALPix grid
                    zarr_a[t, chunk_sl] = vals[cell_ids.astype(int)]

        grp["cell_ids"][chunk_sl] = cell_ids
        log.debug(f"Chunk {chunk_index + 1}/{_N_PARENT} written.")

    # ── Step 3: consolidate ───────────────────────────────────────────────────

    def consolidate(
        self,
        output_path: str | None = None,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        assert self._result is not None
        path = output_path or self._result.output_path
        stac = self._build_stac()
        if storage_options:
            root = zarr.open_group(path, mode="a", storage_options=storage_options)
        else:
            root = zarr.open_group(path, mode="a")
        root.attrs["stac_discovery"] = stac.model_dump()
        zarr.consolidate_metadata(root.store)
        log.info(f"ClimateDT zarr consolidated: {path}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _grib_path(self) -> Path:
        tag = f"cdt_{self.date}T{self.time}_{self.levtype}"
        return self.local_dir / f"{tag}.grib"

    def _download(self, grib_path: Path) -> None:
        try:
            import polytope.api as polytope
        except ImportError:
            raise ImportError(
                "Install polytope: pip install polytope-client destinelab"
            )
        log.info(f"Downloading ClimateDT → {grib_path}")
        client = polytope.Client(address=POLYTOPE_ADDRESS)
        request = {
            "activity": self.activity,
            "class": CDT_CLASS,
            "dataset": CDT_DATASET,
            "date": self.date,
            "experiment": self.experiment,
            "expver": self.expver,
            "generation": self.generation,
            "levtype": self.levtype,
            "model": self.model,
            "param": self.params,
            "realization": self.realization,
            "resolution": self.resolution,
            "stream": CDT_STREAM,
            "time": self.time,
            "type": CDT_TYPE,
        }
        if self.levelist:
            request["levelist"] = self.levelist
        client.retrieve(POLYTOPE_COLLECTION, request, str(grib_path))
        log.info("ClimateDT download complete.")

    def _load_dataset(self, grib_path: Path) -> None:
        if self._ds is None:
            log.info(f"Loading ClimateDT GRIB: {grib_path}")
            self._ds = xr.merge(
                cfgrib.open_datasets(str(grib_path)),
                join="override",
                compat="override",
            )
            log.info(f"  {dict(self._ds.sizes)}  vars: {list(self._ds.data_vars)}")

    def _build_resampler(self) -> None:
        """Build PSFResampler from native pixel centres (sphere → WGS84 correction)."""
        log.info("Building PSFResampler for sphere→WGS84 ellipsoid correction …")
        self._nr = healpix_resample.PSFResampler(
            lon_deg=self._ds.longitude.values.ravel().astype(float),
            lat_deg=self._ds.latitude.values.ravel().astype(float),
            level=_CHILD_LEVEL,
            threshold=0.5,
            verbose=False,
            ellipsoid="WGS84",
        )

    def _get_time_range(self) -> tuple[datetime, datetime]:
        import pandas as pd

        vt = self._ds.get("valid_time", self._ds.get("time", None))
        if vt is not None:
            ts = [
                pd.Timestamp(t).tz_localize(None).to_pydatetime().replace(tzinfo=UTC)
                for t in np.atleast_1d(vt.values).ravel()
            ]
        else:
            dt = datetime.strptime(f"{self.date}{self.time[:4]}", "%Y%m%d%H%M")
            ts = [dt.replace(tzinfo=UTC)]
        return min(ts), max(ts)

    def _init_zarr(
        self,
        output_path: str,
        n_times: int,
        start_dt: datetime,
        end_dt: datetime,
        var_meta: dict,
        is_ocean: bool,
    ) -> None:
        healpix_model = Healpix(
            refinement_level=_CHILD_LEVEL,
            indexing_scheme="nested",
            ellipsoid={"name": "wgs84"},
        )
        dggs_convention = DGGSZarrConvention().model_dump()
        grp_name = "measurements/ocean" if is_ocean else "measurements/sfc"

        root = zarr.open_group(output_path, mode="w")
        grp = root.require_group(grp_name)
        grp.attrs["zarr_conventions"] = [dggs_convention]
        grp.attrs["dggs"] = healpix_model.model_dump()

        grp.create_array(
            "cell_ids",
            shape=(_N_CHILD,),
            dtype=np.int64,
            chunks=(_CHUNK_SIZE,),
            dimension_names=("cells",),
        )

        grp.create_array(
            "time",
            shape=(n_times,),
            dtype="datetime64[ns]",
            chunks=(n_times,),
            dimension_names=("time",),
        )

        for var, (_, unit, long_name) in var_meta.items():
            grp.create_array(
                var,
                shape=(n_times, _N_CHILD),
                dtype=np.float32,
                chunks=(1, _CHUNK_SIZE),
                fill_value=np.nan,
                dimension_names=("time", "cells"),
                attributes={"units": unit, "long_name": long_name},
            )

        # CF 1.13 HEALPix grid mapping alongside the DGGS-Zarr convention
        write_cf_grid_mapping(grp, healpix_model, var_meta)

        zarr.consolidate_metadata(root.store)
        log.info(f"ClimateDT zarr skeleton initialised: {output_path}")

    def _build_stac(self) -> StacItem:
        assert self._result is not None
        import shapely

        r = self._result
        vs = r.start_dt.strftime("%Y%m%dT%H%M%S")
        ve = r.end_dt.strftime("%Y%m%dT%H%M%S")
        cr = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        pt = "ADF_CLDTO" if r.is_ocean else "ADF_CLMDT"

        return StacItem(
            stac_version="1.1.0",
            stac_extensions=[
                "https://stac-extensions.github.io/product/v1.0.0/schema.json"
            ],
            id=f"S00__{pt}_{vs}_{ve}_{cr}",
            bbox=[-180.0, -90.0, 180.0, 90.0],
            geometry=shapely.geometry.mapping(shapely.box(-180, -90, 180, 90)),
            properties={
                "datetime": r.start_dt.isoformat(),
                "start_datetime": r.start_dt.isoformat(),
                "end_datetime": r.end_dt.isoformat(),
                "created": datetime.now(UTC).isoformat(),
                "product:type": pt,
                "description": (
                    f"ClimateDT IFS-NEMO {'ocean 3D' if r.is_ocean else 'surface'} "
                    f"fields on native HEALPix NESTED level {_CHILD_LEVEL}"
                ),
                "healpix:level": _CHILD_LEVEL,
                "healpix:nside": 2**_CHILD_LEVEL,
                "healpix:ordering": "NESTED",
                "source_grid": f"Native HEALPix L{_CHILD_LEVEL} (IFS-NEMO, DestinE)",
                "source_dataset": CDT_DATASET,
                "Conventions": "CF-1.9",
                "ellipsoid_correction": self.ellipsoid_correction,
            },
            links=[],
            assets={},
        )
