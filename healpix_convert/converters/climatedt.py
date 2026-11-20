"""
healpix_convert/converters/climatedt.py
==========================================
ClimateDT (IFS-NEMO) legacy converter.

ClimateDT data is natively on HEALPix level 7 NESTED, on the sphere. This
converter downloads via Polytope, reads the GRIB and resamples it onto the
level 7 NESTED grid on the WGS84 ellipsoid (the grid the output declares),
then packs it into an EOPF-compliant zarr dataset. The two grids are up to
~14 km apart (at 45°), so the values cannot be copied as they are.

The resampler is set in ``CDT_SFC_CONVERT_SETTINGS`` (PSF for now, as for ERA5).

Usage
-----
    from healpix_convert.converters.climatedt import ClimateDTConverter

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

import hashlib
import json
import logging
import os
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

from healpix_convert.core.healpix_conventions import Healpix
from healpix_convert.core.metadata import (
    CF_GRID_MAPPING_VARIABLE,
    get_cf_cell_id_attrs,
    get_cf_data_variable_attrs,
    get_cf_grid_mapping_attrs,
    get_healpix_group_attrs,
    write_root_conventions,
)
from healpix_convert.core.stac import StacItem
from healpix_convert.settings.climatedt import (
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
from healpix_convert.settings.common import MetadataSettings

log = logging.getLogger(__name__)

# ── HEALPix constants ─────────────────────────────────────────────────────────
from healpix_convert.settings.climatedt import CDT_SFC_CONVERT_SETTINGS as _S

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

    Data is natively on HEALPix level 7 on the sphere; it is resampled once
    per variable/timestep onto the same level on the WGS84 ellipsoid.

    Parameters
    ----------
    date                : "YYYYMMDD"
    time                : "HHMM" (e.g. "0100")
    params              : MARS param ids (default surface: "134/165/166/167/168")
    levtype             : "sfc" | "o3d"
    levelist            : depth levels for ocean 3D (e.g. "1/2/3/4/5")
    activity            : ScenarioMIP | ...
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
        local_dir: Path = Path("."),
        metadata: MetadataSettings | None = None,
    ):
        self.metadata = metadata if metadata is not None else MetadataSettings()
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
        self.local_dir = Path(local_dir)

        self._ds: xr.Dataset | None = None
        self._s2e = SphereToEllipsoid(
            _CHILD_LEVEL,
            method="psf",
            init_params=_g["resampler"]["init_params"],
            resample_params=_g["resampler"]["resample_params"],
        )
        self._map_cache: dict = {}  # (var, t) -> full-sphere WGS84 map
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
        """Write the WGS84 ClimateDT data of one HEALPix parent cell."""
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
            zarr_a = grp[var]
            for t in range(r.n_times):
                full_map = self._wgs84_map(var, t, t_dim)
                zarr_a[(t, *(slice(None),) * (zarr_a.ndim - 2), chunk_sl)] = full_map[
                    ..., cell_ids.astype(int)
                ].reshape(zarr_a.shape[1:-1] + (_CHUNK_SIZE,))

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

    def _wgs84_map(self, var: str, t: int, t_dim: str | None) -> np.ndarray:
        """Full-sphere WGS84 map for one variable/timestep.

        Shape ``(_N_CHILD,)``, or ``(n_layer, _N_CHILD)`` when the variable has
        a dimension besides time and cells (one map per layer, masked per
        layer). Computed once per (var, t) and cached, so the global
        resampling is not repeated for each of the 192 spatial chunks. Masked
        fields (NaN in the source, e.g. ocean fields over land) are resampled
        from their valid points only, so the masked cells stay NaN. As for
        ERA5, the result is clipped to the source range, which removes the PSF
        overshoot only.
        """
        key = (var, t)
        full_map = self._map_cache.get(key)
        if full_map is not None:
            return full_map

        da = self._ds[var]
        if t_dim:
            da = da.isel({t_dim: t})
        extra_shape = da.shape[:-1]  # the cells are the last dimension
        vals = da.values.reshape(-1, _N_CHILD).astype(np.float64)
        cells = np.arange(_N_CHILD, dtype=np.uint64)
        full_map = np.full((vals.shape[0], _N_CHILD), np.nan, dtype=np.float32)
        for i, layer in enumerate(vals):
            if np.isfinite(layer).any():
                out_ids, out_vals = self._s2e.resample(cells, layer)
                full_map[i, out_ids.astype(np.int64)] = out_vals
        full_map = full_map.reshape(extra_shape + (_N_CHILD,))

        self._map_cache[key] = full_map
        return full_map

    def _resampler(self, valid: np.ndarray) -> healpix_resample.PSFResampler:
        """Sphere → WGS84 resampler on the valid native points (built once per mask)."""
        cells = np.arange(_N_CHILD, dtype=np.uint64)[valid]
        return self._s2e.operator(cells)

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
        grp_name = "measurements/ocean" if is_ocean else "measurements/sfc"

        root = zarr.open_group(output_path, mode="w")
        write_root_conventions(root, self.metadata)

        grp = root.require_group(grp_name)
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

        grp.create_array(
            "time",
            shape=(n_times,),
            dtype="datetime64[ns]",
            chunks=(n_times,),
            dimension_names=("time",),
        )

        cf_variable_attrs = get_cf_data_variable_attrs(healpix_model, self.metadata)

        ds = self._ds if self._ds is not None else xr.Dataset()
        t_dim = next((d for d in ["valid_time", "time"] if d in ds.dims), None)
        made_dims: set[str] = set()
        for var, (_, unit, long_name) in var_meta.items():
            # dimensions besides time and cells (e.g. ocean layers); the cells come last
            extra = (
                [d for d in ds[var].dims if d != t_dim][:-1] if var in ds.data_vars else []
            )
            sizes = {d: ds.sizes[d] for d in extra}
            for d in extra:
                if d not in made_dims:
                    made_dims.add(d)
                    coord = ds[d].values if d in ds.coords else np.arange(sizes[d])
                    c = grp.create_array(
                        d,
                        shape=coord.shape,
                        dtype=coord.dtype,
                        chunks=coord.shape,
                        dimension_names=(d,),
                    )
                    c[...] = coord
            grp.create_array(
                var,
                shape=(n_times, *sizes.values(), _N_CHILD),
                dtype=np.float32,
                chunks=(1, *sizes.values(), _CHUNK_SIZE),
                fill_value=np.nan,
                dimension_names=("time", *extra, "cells"),
                attributes={
                    "units": unit,
                    "long_name": long_name,
                    **cf_variable_attrs,
                },
            )

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
            },
            links=[],
            assets={},
        )


# ══════════════════════════════════════════════════════════════════════════════
# Generic building blocks (no dataset settings, no default host, no default
# parameter list): Polytope retrieve, HEALPix GRIB / CoverageJSON readers,
# sphere → WGS84 resampling and the per-parent-cell store writer.
# ══════════════════════════════════════════════════════════════════════════════

_LAYER_TYPES = ("oceanModelLayer", "oceanModel")
# GRIB stepUnits code → numpy timedelta unit
_STEP_UNITS = {
    0: "m",
    1: "h",
    2: "D",
    13: "s",
}
_STEP_UNIT_MULT = {10: ("h", 3), 11: ("h", 6), 12: ("h", 12)}


def _as_paths(paths) -> list[Path]:
    if isinstance(paths, (str, os.PathLike)):
        return [Path(paths)]
    return [Path(p) for p in paths]


def polytope_retrieve(
    request: dict,
    address: str,
    out_path,
    *,
    collection: str = "destination-earth",
    client=None,
) -> Path:
    """One Polytope retrieve of ``request`` to ``out_path``.

    The data is written to ``<out_path>.part`` and renamed when complete, so a
    file under its final name is always whole. No retry here (the caller
    retries). ``client`` is any object with ``.retrieve(collection, request,
    path)``; the default is ``polytope.api.Client(address=address)``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_name(out_path.name + ".part")
    if client is None:
        try:
            import polytope.api as polytope
        except ImportError:
            raise ImportError(
                "Install polytope: pip install polytope-client destinelab"
            )
        client = polytope.Client(address=address)
    part.unlink(missing_ok=True)
    try:
        client.retrieve(collection, request, str(part))
        if not part.exists() or part.stat().st_size == 0:
            raise RuntimeError(f"Polytope returned no data for {request}")
        os.replace(part, out_path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return out_path


def _grib_messages(path: Path):
    """Yield eccodes handles of every message of a GRIB file (released after use)."""
    import eccodes as ec

    with open(path, "rb") as fh:
        while True:
            h = ec.codes_grib_new_from_file(fh)
            if h is None:
                break
            try:
                yield h
            finally:
                ec.codes_release(h)


def _grib_times(h) -> tuple[np.datetime64, np.datetime64]:
    """(start, end) of the message: the averaging period for stepType avg, else (valid, valid)."""
    import eccodes as ec

    def stamp(date_key: str, time_key: str) -> np.datetime64:
        d, t = int(ec.codes_get(h, date_key)), int(ec.codes_get(h, time_key))
        return np.datetime64(
            f"{d // 10000:04d}-{d // 100 % 100:02d}-{d % 100:02d}"
            f"T{t // 100:02d}:{t % 100:02d}",
            "s",
        )

    valid = stamp("validityDate", "validityTime")
    if ec.codes_get(h, "stepType") != "avg":
        return valid, valid
    unit = int(ec.codes_get(h, "stepUnits"))
    if unit in _STEP_UNITS:
        mult, u = 1, _STEP_UNITS[unit]
    elif unit in _STEP_UNIT_MULT:
        u, mult = _STEP_UNIT_MULT[unit]
    else:
        raise ValueError(f"unsupported GRIB stepUnits {unit}")
    start = stamp("dataDate", "dataTime") + np.timedelta64(
        int(ec.codes_get(h, "startStep")) * mult, u
    )
    return start, valid


def _grib_grid(h) -> dict:
    import eccodes as ec

    grid = {"gridType": ec.codes_get(h, "gridType")}
    if grid["gridType"] != "healpix":
        raise ValueError(f"not a HEALPix GRIB message: gridType {grid['gridType']}")
    grid["Nside"] = int(ec.codes_get(h, "Nside"))
    grid["orderingConvention"] = ec.codes_get(h, "orderingConvention")
    return grid


def check_grib_params(paths, expected_param_ids) -> dict:
    """Read every GRIB message of ``paths`` and check its paramId.

    Raises ``ValueError`` if a paramId is not in ``expected_param_ids`` or if
    one expected paramId is absent. Returns ``{"messages": n, "param_ids":
    {pid: n}, "dates": [...], "grid": {"gridType", "Nside",
    "orderingConvention"}}``; ``dates`` are the sorted ISO start times of the
    messages (the averaging period start for stepType avg).
    """
    import eccodes as ec

    expected = {int(p) for p in expected_param_ids}
    counts: dict[int, int] = {}
    dates: set[str] = set()
    grid: dict | None = None
    n = 0
    for path in _as_paths(paths):
        for h in _grib_messages(path):
            n += 1
            pid = int(ec.codes_get(h, "paramId"))
            counts[pid] = counts.get(pid, 0) + 1
            dates.add(str(_grib_times(h)[0].astype("datetime64[m]")))
            g = _grib_grid(h)
            if grid is None:
                grid = g
            elif g != grid:
                raise ValueError(f"{path}: grid {g} differs from {grid}")
    unexpected = sorted(set(counts) - expected)
    if unexpected:
        raise ValueError(
            f"unexpected paramId {unexpected} in the GRIB (expected {sorted(expected)})"
        )
    missing = sorted(expected - set(counts))
    if missing:
        raise ValueError(
            f"paramId {missing} not found in the GRIB (found {sorted(counts)})"
        )
    return {
        "messages": n,
        "param_ids": dict(sorted(counts.items())),
        "dates": sorted(dates),
        "grid": grid or {},
    }


def _healpix_attrs(level: int) -> dict:
    return {
        "healpix_nside": 2**level,
        "healpix_level": level,
        "healpix_ordering": "nested",
        "healpix_ellipsoid": "sphere",
    }


def read_healpix_grib(paths, *, param_ids=None) -> xr.Dataset:
    """HEALPix GRIB (nested) → ``xr.Dataset`` on the full sphere.

    Dimensions are ``("time", "cells")``, or ``("time", "layer", "cells")``
    for ocean model layers, per variable; ``cells`` holds ``12 * Nside**2``
    values in nested order on the sphere. The variable name is the GRIB
    shortName; missing values (bitmap) are NaN; values are float32.
    ``time`` is the start of the averaging period for stepType avg (with
    ``time_bounds`` [start, end) on dims ``("time", "bnds")``), else the valid
    time; ``layer`` is the GRIB ``level``. ``param_ids`` keeps only those
    messages. Read with eccodes directly (cfgrib cannot read HEALPix grids).
    """
    import eccodes as ec

    wanted = None if param_ids is None else {int(p) for p in param_ids}
    paths = _as_paths(paths)

    # pass 1: the metadata of every message
    msgs: list[dict] = []
    grid: dict | None = None
    for path in paths:
        for i, h in enumerate(_grib_messages(path)):
            pid = int(ec.codes_get(h, "paramId"))
            if wanted is not None and pid not in wanted:
                continue
            g = _grib_grid(h)
            if g["orderingConvention"] != "nested":
                raise ValueError(f"{path}: ordering {g['orderingConvention']} (nested only)")
            if grid is None:
                grid = g
            elif g != grid:
                raise ValueError(f"{path}: grid {g} differs from {grid}")
            start, end = _grib_times(h)
            level_type = ec.codes_get(h, "typeOfLevel")
            msgs.append(
                {
                    "path": path,
                    "index": i,
                    "name": ec.codes_get(h, "shortName"),
                    "start": start,
                    "end": end,
                    "layer": (
                        int(ec.codes_get(h, "level")) if level_type in _LAYER_TYPES else None
                    ),
                    "attrs": {
                        "units": ec.codes_get(h, "units"),
                        "long_name": ec.codes_get(h, "name"),
                        "paramId": pid,
                        "typeOfLevel": level_type,
                        "stepType": ec.codes_get(h, "stepType"),
                        "GRIB_stepRange": ec.codes_get(h, "stepRange"),
                    },
                }
            )
    if not msgs or grid is None:
        raise ValueError(f"no GRIB message to read in {[str(p) for p in paths]}")
    nside = grid["Nside"]
    level = int(round(np.log2(nside)))
    if 2**level != nside:
        raise ValueError(f"Nside {nside} is not a power of two")
    ncell = 12 * nside**2

    times = sorted({m["start"] for m in msgs})
    t_index = {t: i for i, t in enumerate(times)}
    layers = sorted({m["layer"] for m in msgs if m["layer"] is not None})
    l_index = {v: i for i, v in enumerate(layers)}
    ends: dict[np.datetime64, np.datetime64] = {}
    attrs: dict[str, dict] = {}
    has_layer: dict[str, bool] = {}
    for m in msgs:
        attrs.setdefault(m["name"], m["attrs"])
        if has_layer.setdefault(m["name"], m["layer"] is not None) != (
            m["layer"] is not None
        ):
            raise ValueError(f"{m['name']}: messages with and without a layer")
        if m["attrs"]["stepType"] == "avg":
            ends[m["start"]] = m["end"]

    data: dict[str, np.ndarray] = {}
    for name, lay in has_layer.items():
        shape = (len(times), len(layers), ncell) if lay else (len(times), ncell)
        data[name] = np.full(shape, np.nan, dtype=np.float32)
    seen: set[tuple] = set()
    wanted_idx: dict[Path, dict[int, dict]] = {}
    for m in msgs:
        wanted_idx.setdefault(m["path"], {})[m["index"]] = m

    # pass 2: the values
    for path in paths:
        sel = wanted_idx.get(path, {})
        for i, h in enumerate(_grib_messages(path)):
            m = sel.get(i)
            if m is None:
                continue
            key = (m["name"], m["start"], m["layer"])
            if key in seen:
                raise ValueError(f"duplicate GRIB message {key}")
            seen.add(key)
            if int(ec.codes_get(h, "numberOfDataPoints")) != ncell:
                raise ValueError(f"{path}: message {i} is not a full sphere field")
            vals = ec.codes_get_values(h).astype(np.float32)
            if ec.codes_get(h, "bitmapPresent"):
                vals[ec.codes_get_values(h) == ec.codes_get(h, "missingValue")] = np.nan
            t = t_index[m["start"]]
            if m["layer"] is None:
                data[m["name"]][t] = vals
            else:
                data[m["name"]][t, l_index[m["layer"]]] = vals

    time_ns = np.array(times, dtype="datetime64[ns]")
    coords: dict[str, Any] = {"time": ("time", time_ns)}
    if layers:
        coords["layer"] = ("layer", np.array(layers, dtype="int32"))
    if ends:
        nat = np.datetime64("NaT", "ns")
        coords["time_bounds"] = (
            ("time", "bnds"),
            np.array(
                [[t, ends.get(t, nat)] for t in times],
                dtype="datetime64[ns]",
            ),
        )
    variables = {
        name: (
            ("time", "layer", "cells") if has_layer[name] else ("time", "cells"),
            arr,
            attrs[name],
        )
        for name, arr in data.items()
    }
    return xr.Dataset(variables, coords=coords, attrs=_healpix_attrs(level))


def _covjson_time(value: str) -> np.datetime64:
    return np.datetime64(value.rstrip("Z"), "ns")


def read_covjson_points(path, *, level: int) -> xr.Dataset:
    """CoverageJSON as returned by the Polytope ``feature`` request → ``xr.Dataset``.

    One or more coverages with the points as a composite axis (tuples of
    latitude, longitude[, levelist]) or as separate, equally long ``latitude``
    / ``longitude`` / ``levelist`` axes; an optional ``t`` axis. Only the
    returned points are kept: dimension ``cells`` holds them, with the
    coordinate ``cell_ids`` (uint64, nested on the sphere at ``level``, sorted
    ascending). Same variable names and dimensions as
    :func:`read_healpix_grib`; missing points are NaN. With several times the
    values are taken as ``(t, point)``. Raises if two values land on the same
    (time, layer, cell).
    """
    with open(path) as fh:
        doc = json.load(fh)
    coverages = doc["coverages"] if doc.get("type") == "CoverageCollection" else [doc]
    params = doc.get("parameters", {})

    rows: dict[str, list] = {}  # name -> [(t, layer, lat, lon, value)]
    for cov in coverages:
        axes = cov["domain"]["axes"]
        meta = cov.get("mars:metadata", {})
        if "composite" in axes:
            names = axes["composite"]["coordinates"]
            tup = axes["composite"]["values"]
            col = {n: [p[i] for p in tup] for i, n in enumerate(names)}
        else:
            col = {n: axes[n]["values"] for n in ("latitude", "longitude") if n in axes}
            if "levelist" in axes:
                col["levelist"] = axes["levelist"]["values"]
        if "latitude" not in col or "longitude" not in col:
            raise ValueError(f"{path}: no latitude / longitude axes in a coverage")
        lat, lon = col["latitude"], col["longitude"]
        npts = len(lat)
        if len(lon) != npts or ("levelist" in col and len(col["levelist"]) != npts):
            raise ValueError(f"{path}: latitude / longitude / levelist axes differ in length")
        if "levelist" in col:
            layer = [int(v) for v in col["levelist"]]
        elif "levelist" in meta:
            layer = [int(meta["levelist"])] * npts
        else:
            layer = [None] * npts
        if "t" in axes:
            tvals = [_covjson_time(v) for v in axes["t"]["values"]]
        elif "Forecast date" in meta:
            tvals = [_covjson_time(meta["Forecast date"])]
        else:
            raise ValueError(f"{path}: no time axis and no Forecast date in a coverage")
        for name, rng in cov["ranges"].items():
            vals = np.array(
                [np.nan if v is None else v for v in rng["values"]], dtype=np.float64
            )
            if vals.size != npts * len(tvals):
                raise ValueError(
                    f"{path}: {name} has {vals.size} values for {npts} points x {len(tvals)} times"
                )
            vals = vals.reshape(len(tvals), npts)
            out = rows.setdefault(name, [])
            for ti, t in enumerate(tvals):
                out.extend(zip([t] * npts, layer, lat, lon, vals[ti]))
    if not rows:
        raise ValueError(f"{path}: no range in the CoverageJSON")

    all_lat = np.array([r[2] for v in rows.values() for r in v], dtype=np.float64)
    all_lon = np.array([r[3] for v in rows.values() for r in v], dtype=np.float64)
    all_ids = np.asarray(
        healpix_geo.nested.lonlat_to_healpix(all_lon, all_lat, level, ellipsoid="sphere")
    ).astype(np.uint64)
    cell_ids = np.unique(all_ids)
    times = sorted({r[0] for v in rows.values() for r in v})
    layers = sorted({r[1] for v in rows.values() for r in v if r[1] is not None})
    t_index = {t: i for i, t in enumerate(times)}
    l_index = {v: i for i, v in enumerate(layers)}

    variables = {}
    pos = 0
    for name, v in rows.items():
        ids = all_ids[pos : pos + len(v)]
        pos += len(v)
        c = np.searchsorted(cell_ids, ids)
        t = np.array([t_index[r[0]] for r in v])
        vals = np.array([r[4] for r in v], dtype=np.float32)
        lay = [r[1] for r in v]
        arr_shape = (len(times), len(cell_ids))
        if layers:
            if any(x is None for x in lay):
                raise ValueError(f"{name}: points with and without a layer")
            arr_shape = (len(times), len(layers), len(cell_ids))
            idx = (t, np.array([l_index[x] for x in lay]), c)
        else:
            idx = (t, c)
        arr = np.full(arr_shape, np.nan, dtype=np.float32)
        hits = np.zeros(arr_shape, dtype=bool)
        if hits[idx].any() or len(set(zip(*idx))) != len(v):
            raise ValueError(
                f"{name}: two points of the CoverageJSON fall on the same HEALPix cell "
                f"at level {level}"
            )
        arr[idx] = vals
        p = params.get(name, {})
        attrs = {
            "units": p.get("unit", {}).get("symbol", ""),
            "long_name": p.get("observedProperty", {}).get("label", {}).get("en", name),
        }
        variables[name] = (
            ("time", "layer", "cells") if layers else ("time", "cells"),
            arr,
            attrs,
        )
    coords: dict[str, Any] = {
        "time": ("time", np.array(times, dtype="datetime64[ns]")),
        "cell_ids": ("cells", cell_ids),
    }
    if layers:
        coords["layer"] = ("layer", np.array(layers, dtype="int32"))
    return xr.Dataset(variables, coords=coords, attrs=_healpix_attrs(level))


class SphereToEllipsoid:
    """Move a field from the HEALPix grid on the sphere to the one on the WGS84 ellipsoid.

    The two grids share the cell ids but not the cell centres (up to ~14 km
    apart at 45 degrees at level 7). Operators are built on the valid points
    only (one per distinct valid mask, cached) so that masked inputs carry no
    weight.

    Parameters
    ----------
    level       : HEALPix level of the input and of the output grid
    method      : "psf" (``healpix_resample.PSFResampler``) or "conservative"
                  (``BilinearResampler`` with ``conservative=True``, normalised
                  over the valid inputs)
    init_params : keyword arguments of the resampler constructor; default for
                  "psf": ``{"threshold": 0.5}``
    resample_params : keyword arguments of ``resample``; default for "psf":
                  ``{"lam": 0.0}``
    clip        : "input_range" clips the result to [min, max] of the valid
                  inputs (removes the PSF overshoot only), None does not clip
    """

    METHODS = ("psf", "conservative")

    def __init__(
        self,
        level: int,
        method: str = "psf",
        init_params: dict | None = None,
        resample_params: dict | None = None,
        clip: str | None = "input_range",
    ):
        if method not in self.METHODS:
            raise ValueError(f"unknown method {method!r}; use one of {self.METHODS}")
        if clip not in (None, "input_range"):
            raise ValueError(f"unknown clip {clip!r}; use None or 'input_range'")
        self.level = int(level)
        self.method = method
        if init_params is None:
            init_params = {"threshold": 0.5} if method == "psf" else {}
        if resample_params is None:
            resample_params = {"lam": 0.0} if method == "psf" else {}
        self.init_params = dict(init_params)
        self.resample_params = dict(resample_params)
        self.clip = clip
        self._operators: dict[bytes, Any] = {}
        self._norms: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _key(valid_ids: np.ndarray) -> bytes:
        return hashlib.blake2b(valid_ids.tobytes(), digest_size=16).digest()

    def operator(self, valid_ids: np.ndarray):
        """The resampler on the (sphere) cell ids ``valid_ids``, built once per set of ids."""
        valid_ids = np.ascontiguousarray(valid_ids, dtype=np.uint64)
        key = self._key(valid_ids)
        if key not in self._operators:
            log.info(
                f"Building the sphere→WGS84 {self.method} resampler on "
                f"{valid_ids.size:,} points …"
            )
            lon, lat = healpix_geo.nested.healpix_to_lonlat(
                valid_ids, self.level, ellipsoid="sphere"
            )
            cls = (
                healpix_resample.PSFResampler
                if self.method == "psf"
                else healpix_resample.BilinearResampler
            )
            self._operators[key] = cls(
                lon_deg=np.asarray(lon, dtype=float),
                lat_deg=np.asarray(lat, dtype=float),
                level=self.level,
                ellipsoid="WGS84",
                verbose=False,
                **self.init_params,
            )
        return self._operators[key]

    def resample(self, cell_ids, values) -> tuple[np.ndarray, np.ndarray]:
        """``cell_ids``: nested ids on the sphere (uint64); ``values``: 1-D, NaN = masked.

        Returns ``(cell ids on the WGS84 ellipsoid, float32 values)`` for every
        output cell the operator reaches (none if every input is masked).
        """
        cell_ids = np.asarray(cell_ids).astype(np.uint64, copy=False)
        values = np.asarray(values, dtype=np.float64)
        if cell_ids.shape != values.shape or values.ndim != 1:
            raise ValueError("cell_ids and values must be 1-D and of the same length")
        valid = np.isfinite(values)
        if not valid.any():
            return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.float32)
        v = values[valid]
        op = self.operator(cell_ids[valid])
        if self.method == "psf":
            res = op.resample(v, **self.resample_params)
            ids = np.asarray(res.cell_ids)
            out = np.asarray(res.cell_data, dtype=np.float64)
        else:
            key = self._key(np.ascontiguousarray(cell_ids[valid], dtype=np.uint64))
            if key not in self._norms:
                # the same operator applied to ones: the weight each output cell received
                res1 = op.resample(np.ones_like(v), conservative=True)
                self._norms[key] = (
                    np.asarray(res1.cell_ids),
                    np.asarray(res1.cell_data, dtype=np.float64),
                )
            ids1, den = self._norms[key]
            res = op.resample(v, conservative=True)
            ids = np.asarray(res.cell_ids)
            if not np.array_equal(ids, ids1):
                raise RuntimeError("conservative operator: output cells differ")
            num = np.asarray(res.cell_data, dtype=np.float64)
            keep = den > 0
            ids, out = ids[keep], num[keep] / den[keep]
        if self.clip == "input_range":
            out = np.clip(out, v.min(), v.max())
        return ids.astype(np.uint64), out.astype(np.float32)

    def describe(self) -> dict:
        from importlib.metadata import PackageNotFoundError, version

        try:
            hr_version = version("healpix-resample")
        except PackageNotFoundError:
            hr_version = "unknown"
        return {
            "method": self.method,
            "init_params": dict(self.init_params),
            "resample_params": dict(self.resample_params),
            "clip": self.clip,
            "source_grid": f"healpix nested sphere L{self.level}",
            "target_grid": f"healpix nested WGS84 L{self.level}",
            "healpix_resample": hr_version,
        }


def _json_safe(value):
    """numpy scalars / arrays → plain Python for zarr attributes."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_cell_store(
    out_path,
    *,
    parent_cell: int,
    parent_level: int,
    level: int,
    group: str,
    variables: dict[str, np.ndarray],
    var_attrs: dict[str, dict],
    root_attrs: dict,
    stac_item: dict,
    layer: np.ndarray | None = None,
    chunk_level: int | None = None,
    coarsest_level: int | None = None,
    storage_options: dict | None = None,
) -> dict:
    """Write the values of one HEALPix parent cell as the store the mosaic engine reads.

    Layout (zarr v3, consolidated metadata of the group and of the root)::

        root attrs               root_attrs + {"stac_discovery": stac_item}
        <group>                  attrs["multiscales"] = {"layout": [{"asset": "<L>", "dggs": {...}}, ...]}
        <group>/<L>/<var>        float32, fill NaN, dims ["cells"] (["layer", "cells"] for 2-D)
        <group>/<L>/cell_ids     uint64 = parent_cell * 4**(L - parent_level) + arange
        <group>/<L>/layer        only with ``layer``

    ``variables[name]`` holds the values of the parent cell at ``level`` in
    nested order, shape ``(4**(level - parent_level),)`` or ``(n_layer,
    4**(level - parent_level))`` (then ``layer`` is required). Levels
    ``coarsest_level`` .. ``level - 1`` (default: ``level`` only) are the NaN-mean
    of the four children. Chunks hold ``4**(L - chunk_level)`` cells (default
    ``chunk_level = parent_level``: one chunk per level) and, in 2-D arrays,
    one layer.

    Returns ``{"variables": [...], "levels": [...], "n_valid": {var: int}}``
    (``n_valid`` counts the non-NaN values at ``level``).
    """
    if chunk_level is None:
        chunk_level = parent_level
    if coarsest_level is None:
        coarsest_level = level
    if not parent_level <= chunk_level <= level:
        raise ValueError("need parent_level <= chunk_level <= level")
    if not parent_level <= coarsest_level <= level:
        raise ValueError("need parent_level <= coarsest_level <= level")
    n_fine = 4 ** (level - parent_level)
    if not 0 <= parent_cell < 12 * 4**parent_level:
        raise ValueError(f"parent_cell {parent_cell} outside level {parent_level}")
    if layer is not None:
        layer = np.asarray(layer)
    data: dict[str, np.ndarray] = {}
    for name, arr in variables.items():
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim not in (1, 2) or arr.shape[-1] != n_fine:
            raise ValueError(
                f"{name}: shape {arr.shape}, expected (..., {n_fine}) cells of the parent cell"
            )
        if arr.ndim == 2 and (layer is None or arr.shape[0] != len(layer)):
            raise ValueError(f"{name}: a 2-D array needs `layer` of the same length")
        data[name] = arr

    levels = list(range(level, coarsest_level - 1, -1))
    dggs = {
        L: {
            "name": "healpix",
            "refinement_level": L,
            "indexing_scheme": "nested",
            "ellipsoid": "WGS84",
            "spatial_dimension": "cells",
            "coordinate": "cell_ids",
            "compression": "none",
        }
        for L in levels
    }
    kwargs = {"storage_options": storage_options} if storage_options else {}
    root = zarr.open_group(
        str(out_path), mode="w", zarr_format=3, use_consolidated=False, **kwargs
    )
    root.attrs.update(_json_safe({**root_attrs, "stac_discovery": stac_item}))
    grp = root.require_group(group)
    grp.attrs["multiscales"] = {
        "layout": [{"asset": str(L), "dggs": dggs[L]} for L in levels]
    }

    current = dict(data)
    for L in levels:
        n = 4 ** (L - parent_level)
        chunk = min(n, 4 ** (L - chunk_level)) if L >= chunk_level else n
        lg = grp.create_group(str(L))
        lg.attrs["dggs"] = dggs[L]
        for name, arr in current.items():
            two_d = arr.ndim == 2
            a = lg.create_array(
                name,
                shape=arr.shape,
                chunks=(1, chunk) if two_d else (chunk,),
                dtype="float32",
                fill_value=float("nan"),
                dimension_names=["layer", "cells"] if two_d else ["cells"],
                attributes=_json_safe(var_attrs.get(name, {})),
            )
            a[...] = arr
        ids = lg.create_array(
            "cell_ids",
            shape=(n,),
            chunks=(chunk,),
            dtype="uint64",
            dimension_names=["cells"],
        )
        ids[:] = np.uint64(parent_cell) * np.uint64(n) + np.arange(n, dtype=np.uint64)
        if layer is not None:
            la = lg.create_array(
                "layer",
                shape=layer.shape,
                chunks=layer.shape,
                dtype=layer.dtype,
                dimension_names=["layer"],
            )
            la[...] = layer
        if L > coarsest_level:
            current = {
                name: _nanmean4(arr) for name, arr in current.items()
            }
    zarr.consolidate_metadata(root.store, path=group)
    zarr.consolidate_metadata(root.store)
    return {
        "variables": list(data),
        "levels": levels,
        "n_valid": {name: int(np.isfinite(arr).sum()) for name, arr in data.items()},
    }


def _nanmean4(arr: np.ndarray) -> np.ndarray:
    """NaN-mean of each group of four consecutive cells (the children in nested order)."""
    g = arr.reshape(*arr.shape[:-1], arr.shape[-1] // 4, 4)
    valid = np.isfinite(g)
    count = valid.sum(axis=-1)
    total = np.where(valid, g, 0.0).sum(axis=-1, dtype=np.float64)
    out = np.full(count.shape, np.nan, dtype=np.float32)
    np.divide(total, count, out=total, where=count > 0)
    out[count > 0] = total[count > 0].astype(np.float32)
    return out
