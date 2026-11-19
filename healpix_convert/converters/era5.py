"""
healpix_convert/converters/era5.py
======================================
ERA5 legacy converter.

Downloads ERA5 from CDS via two MARS requests (enda N256 + oper N320),
projects onto HEALPix level 7 using PSFResampler, and writes an
EOPF-compliant zarr.

Usage
-----
    from healpix_convert.converters.era5 import ERA5Converter
    converter = ERA5Converter(date="2025-01-01")
    result    = converter.prepare(output_path="out.zarr")
    for idx in range(result.n_chunks):
        converter.convert_group(idx)
    converter.consolidate("out.zarr")

Several analysis times go in one store when ``date`` and ``time`` are MARS
lists or ranges, e.g. a whole day with all its times in one chunk::

    converter = ERA5Converter(
        date="20260812",
        time="/".join(f"{h:02d}:00:00" for h in range(24)),
        time_chunk="all",
    )

The variable tables are arguments. All surface fields from the hourly oper
stream (no enda group), plus temperature, specific and relative humidity on
some pressure levels, written as ``t_isobaric`` etc. in the oper group::

    from healpix_convert.settings.era5 import ERA5_ALL_VARIABLE_META

    converter = ERA5Converter(
        date="20260812",
        time="/".join(f"{h:02d}:00:00" for h in range(24)),
        time_chunk="all",
        enda_variables={},
        oper_variables=ERA5_ALL_VARIABLE_META,
        pressure_levels=[1000, 950, 925, 900, 850, 800, 700],
        root_attributes={"project:version": "v1"},
    )
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import healpix_geo
import numpy as np
import xarray as xr
import zarr
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
from healpix_convert.core.stac import (
    STAC_PROCESSING_SCHEMA_URL,
    StacItem,
    _package_versions,
)
from healpix_convert.settings.common import MetadataSettings
from healpix_convert.settings.era5 import (
    CDS_DATASET,
    CDS_URL,
    DEFAULT_NUMBERS,
    ERA5_ENDA_GRID,
    ERA5_ENDA_VARIABLE_META,
    ERA5_OPER_GRID,
    ERA5_OPER_VARIABLE_META,
    ERA5_PL_VARIABLE_META,
)

log = logging.getLogger(__name__)

# ── HEALPix constants ─────────────────────────────────────────────────────────
_CHILD_LEVEL = 7
_CHUNK_LEVEL = 2
_N_CHILD = 12 * 4**_CHILD_LEVEL  # 196,608
_N_PARENT = 12 * 4**_CHUNK_LEVEL  # 192
_CHUNK_SIZE = 4 ** (_CHILD_LEVEL - _CHUNK_LEVEL)  # 1,024

_resampler_cache: dict = {}

ENDA_GROUP = "measurements/enda"
OPER_GROUP = "measurements/oper"
#: key of the pressure-level source in grib_paths() / dataset(); its fields are
#: written to OPER_GROUP as ``<name>_isobaric``
PL_SOURCE = "measurements/oper:isobaric"
PL_SUFFIX = "_isobaric"
PL_DIM = "isobaric"
_CFGRIB_PL_DIM = "isobaricInhPa"

#: PSFResampler settings of every field (recorded in the provenance)
_PSF_THRESHOLD = 0.5
_PSF_LAM = 0.0
#: packages recorded besides healpix-convert and healpix-resample: the GRIB
#: decoding and the cell geometry decide the values too
_EXTRA_PROVENANCE_PACKAGES = ("healpix-geo", "cfgrib", "eccodes", "xarray", "zarr")


def _mars_list(value: str) -> list[str]:
    """The items of a MARS list (``a/b/c``)."""
    return [item for item in str(value).split("/") if item]


def _enda_times(times: list[str]) -> list[str]:
    """The times at which ERA5 has EDA (stream=enda) analyses: every 3 hours."""
    return [t for t in times if int(t[:2]) % 3 == 0]


def _cache_tag(date: str, times: list[str]) -> str:
    """File-name-safe tag of a MARS date (or date list / range) and time list.

    A single date and time give the tag of earlier versions, e.g.
    ``20250101T00``; a range ``20300101/to/20300131`` becomes
    ``20300101-to-20300131``. Several times are summarised by the first and the
    last hour, their number and a hash of the list.
    """
    tag = date.replace("-", "").replace("/", "-")
    if len(times) == 1:
        return f"{tag}T{times[0][:2]}"
    digest = hashlib.sha1("/".join(times).encode()).hexdigest()[:6]
    return f"{tag}T{times[0][:2]}-{times[-1][:2]}n{len(times)}h{digest}"


def _grib_expvers(path: Path) -> dict[datetime, str]:
    """Experiment version (``expver``) of each validity time of a GRIB file.

    ``0001`` is the final ERA5, ``0005`` the preliminary ERA5T. If the messages
    of one time disagree, their versions are joined with ``/``.
    """
    import eccodes

    found: dict[datetime, set[str]] = {}
    with open(path, "rb") as f:
        while (gid := eccodes.codes_grib_new_from_file(f)) is not None:
            try:
                date = eccodes.codes_get(gid, "validityDate")
                hhmm = eccodes.codes_get(gid, "validityTime")
                try:
                    expver = eccodes.codes_get_string(gid, "expver")
                except eccodes.KeyValueNotFoundError:
                    expver = "unknown"
            finally:
                eccodes.codes_release(gid)
            t = datetime.strptime(f"{date:08d}{hhmm:04d}", "%Y%m%d%H%M").replace(
                tzinfo=UTC
            )
            found.setdefault(t, set()).add(expver)
    return {t: "/".join(sorted(v)) for t, v in found.items()}


def _software_versions() -> dict[str, str]:
    versions = _package_versions()
    for package in _EXTRA_PROVENANCE_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _get_resampler(
    lon: np.ndarray,
    lat: np.ndarray,
    level: int = _CHILD_LEVEL,
    threshold: float = _PSF_THRESHOLD,
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
    """What prepare() found. The enda fields are None when the converter has
    no enda variables; the pressure-level fields are empty without levels."""

    output_path: str
    n_chunks: int
    n_times: int
    start_dt: datetime
    end_dt: datetime
    grib_enda: Path | None
    grib_oper: Path | None
    ds_enda: xr.Dataset | None
    ds_oper: xr.Dataset | None
    nr_enda: PSFResampler | None
    nr_oper: PSFResampler | None
    lon_enda: np.ndarray | None
    lat_enda: np.ndarray | None
    lon_oper: np.ndarray | None
    lat_oper: np.ndarray | None
    vars_enda: list[str]
    vars_oper: list[str]
    #: analysis times of each group (the EDA has fewer: it is 3-hourly)
    times: dict[str, list[datetime]] = field(default_factory=dict)
    #: pressure-level source (fields written as ``<name>_isobaric`` in oper)
    grib_pl: Path | None = None
    ds_pl: xr.Dataset | None = None
    vars_pl: list[str] = field(default_factory=list)
    pressure_levels: list[int] = field(default_factory=list)
    #: expver of each analysis time of each group (0001 ERA5, 0005 ERA5T)
    expvers: dict[str, list[str]] = field(default_factory=dict)


class ERA5Converter:
    """
    Converter for ERA5 reanalysis surface fields to EOPF-DGGS HEALPix zarr.

    Parameters
    ----------
    date       : "YYYY-MM-DD", or a MARS date list or range
                 ("20260812/20260813", "20260801/to/20260831")
    time       : "HH:MM:SS", or a MARS list of times
                 ("00:00:00/06:00:00/12:00:00/18:00:00"). The EDA (enda group)
                 has analyses every 3 hours only: it gets the times among these
                 that fall on 00, 03, ..., 21 UTC.
    step       : MARS step (default "0")
    numbers    : ensemble member numbers (default 0-9); the first one is written
    local_dir  : directory for local GRIB cache (see :py:meth:`grib_paths`)
    metadata   : conventions followed by the output dataset
                 (default: all of them, see :py:class:`MetadataSettings`)
    time_chunk : number of analysis times in one zarr chunk, or "all" for all
                 the times of a group (one chunk then holds the time series of
                 1024 cells). Default 1, one chunk per time.
    enda_variables : variables of the enda group, ``{name: (GRIB param id,
                 unit, long name)}``; default ERA5_ENDA_VARIABLE_META. An empty
                 mapping writes no enda group and requests nothing from the EDA.
    oper_variables : surface variables of the oper group, same form; default
                 ERA5_OPER_VARIABLE_META.
    pressure_levels : pressure levels in hPa (e.g. ``[1000, 850, 700]``); the
                 pl_variables are requested on them (stream oper, levtype pl,
                 same grid as oper) and written to the oper group as
                 ``<name>_isobaric`` (time, isobaric, cells), in this order,
                 with an ``isobaric`` coordinate. Default none.
    pl_variables : pressure-level variables, same form; default
                 ERA5_PL_VARIABLE_META (t, q, r). Used only with pressure_levels.
    root_attributes : attributes added to the root group of the output, e.g.
                 the version of a production; ``stac_discovery`` and
                 ``Conventions`` are written by the converter and refused here.

    With several times, each group gets a ``time`` coordinate (the enda and oper
    groups may have different times).

    Memory: each process converting chunks keeps the full HEALPix map of every
    variable at every time (0.75 MiB each), e.g. about 6 GiB for a month of
    hourly analyses. For long ranges, write one store per day instead.
    """

    def __init__(
        self,
        date: str,
        time: str = "00:00:00",
        step: str = "0",
        numbers: list[int] = None,
        local_dir: Path = Path("."),
        metadata: MetadataSettings | None = None,
        time_chunk: int | Literal["all"] = 1,
        enda_variables: Mapping[str, tuple[str, str, str]] | None = None,
        oper_variables: Mapping[str, tuple[str, str, str]] | None = None,
        pressure_levels: Sequence[int] = (),
        pl_variables: Mapping[str, tuple[str, str, str]] | None = None,
        root_attributes: Mapping[str, Any] | None = None,
    ):
        if time_chunk != "all" and (not isinstance(time_chunk, int) or time_chunk < 1):
            raise ValueError(
                f"time_chunk must be a positive int or 'all', got {time_chunk!r}"
            )
        self.enda_variables = dict(
            ERA5_ENDA_VARIABLE_META if enda_variables is None else enda_variables
        )
        self.oper_variables = dict(
            ERA5_OPER_VARIABLE_META if oper_variables is None else oper_variables
        )
        self.pressure_levels = [int(level) for level in pressure_levels]
        if len(set(self.pressure_levels)) != len(self.pressure_levels) or any(
            level <= 0 for level in self.pressure_levels
        ):
            raise ValueError(
                f"pressure_levels must be distinct positive hPa values, "
                f"got {list(pressure_levels)!r}"
            )
        self.pl_variables = (
            dict(ERA5_PL_VARIABLE_META if pl_variables is None else pl_variables)
            if self.pressure_levels
            else {}
        )
        if self.pressure_levels and not self.pl_variables:
            raise ValueError("pressure_levels given without pl_variables")
        if (
            not self.oper_variables
            and not self.pl_variables
            and not self.enda_variables
        ):
            raise ValueError("no variable to convert")
        clash = {f"{v}{PL_SUFFIX}" for v in self.pl_variables} & set(
            self.oper_variables
        )
        if clash:
            raise ValueError(f"pressure-level names clash with oper variables: {clash}")
        self.root_attributes = dict(root_attributes or {})
        reserved = {"stac_discovery", "Conventions"} & set(self.root_attributes)
        if reserved:
            raise ValueError(f"root_attributes may not set {sorted(reserved)}")
        self.date = date
        self.time = time
        self.times = _mars_list(time)
        if not self.times:
            raise ValueError(f"no time in {time!r}")
        self.time_chunk = time_chunk
        self.step = step
        self.numbers = numbers if numbers is not None else DEFAULT_NUMBERS
        self.local_dir = Path(local_dir)
        self.metadata = metadata if metadata is not None else MetadataSettings()
        self._result: ERA5PrepareResult | None = None
        self._map_cache: dict = {}
        self._sub_resampler_cache: dict = {}
        self._pl_level_ds: dict[int, xr.Dataset] = {}

    def grib_paths(self) -> dict[str, Path]:
        """The local GRIB files of each group (downloaded if missing).

        GRIB files already at these paths are used as they are, so data
        downloaded beforehand can be converted by placing it there.
        """
        paths = {}
        if self.enda_variables:
            enda_times = _enda_times(self.times) or self.times
            tag = self._table_tag(self.enda_variables, ERA5_ENDA_VARIABLE_META)
            paths[ENDA_GROUP] = (
                self.local_dir
                / f"era5_enda_{_cache_tag(self.date, enda_times)}{tag}_{ERA5_ENDA_GRID}.grib"
            )
        if self.oper_variables:
            tag = self._table_tag(self.oper_variables, ERA5_OPER_VARIABLE_META)
            paths[OPER_GROUP] = (
                self.local_dir
                / f"era5_oper_{_cache_tag(self.date, self.times)}{tag}_{ERA5_OPER_GRID}.grib"
            )
        if self.pressure_levels:
            levels = "-".join(str(level) for level in self.pressure_levels)
            tag = self._table_tag(self.pl_variables, ERA5_PL_VARIABLE_META)
            paths[PL_SOURCE] = (
                self.local_dir
                / f"era5_pl_{_cache_tag(self.date, self.times)}_{levels}{tag}_{ERA5_OPER_GRID}.grib"
            )
        return paths

    def _table_tag(self, table: Mapping, default: Mapping) -> str:
        """File-name tag of a variable table (and step) that differs from the
        default, so that a cached GRIB of another table is never reused."""
        if dict(table) == dict(default) and self.step == "0":
            return ""
        key = "/".join(v[0] for v in table.values()) + f"|{self.step}"
        return f"_v{hashlib.sha1(key.encode()).hexdigest()[:6]}"

    def download(self, force: bool = False) -> dict[str, Path]:
        """Download the GRIB files from the CDS unless they are already there."""
        if self.enda_variables and not _enda_times(self.times):
            raise ValueError(
                f"none of the times {self.times} has an ERA5 EDA analysis "
                "(stream=enda, every 3 hours: 00, 03, ..., 21 UTC)"
            )
        paths = self.grib_paths()
        missing = [key for key, p in paths.items() if force or not p.exists()]
        if missing:
            import cdsapi

            for key in missing:
                log.info(f"Downloading ERA5 {key} → {paths[key]}")
                cdsapi.Client(url=CDS_URL).retrieve(
                    CDS_DATASET, self.requests()[key], str(paths[key])
                )
        return paths

    def requests(self) -> dict[str, dict[str, str]]:
        """The CDS (MARS) request of each GRIB file, as sent by download().

        ``expver`` is asked as 1 (final ERA5); for recent dates the CDS serves
        ERA5T (5) instead. The version actually received is read from the GRIB
        and recorded per time (``source_expver``).
        """
        common = {"class": "ea", "expver": "1", "step": self.step, "type": "an"}
        requests = {}
        if self.enda_variables:
            requests[ENDA_GROUP] = {
                **common,
                "date": self.date,
                "levtype": "sfc",
                "param": "/".join(v[0] for v in self.enda_variables.values()),
                "stream": "enda",
                "time": "/".join(_enda_times(self.times)),
                "grid": ERA5_ENDA_GRID,
                "number": "/".join(str(n) for n in self.numbers),
            }
        if self.oper_variables:
            requests[OPER_GROUP] = {
                **common,
                "date": self.date,
                "levtype": "sfc",
                "param": "/".join(v[0] for v in self.oper_variables.values()),
                "stream": "oper",
                "time": "/".join(self.times),
                "grid": ERA5_OPER_GRID,
            }
        if self.pressure_levels:
            requests[PL_SOURCE] = {
                **common,
                "date": self.date,
                "levtype": "pl",
                "levelist": "/".join(str(level) for level in self.pressure_levels),
                "param": "/".join(v[0] for v in self.pl_variables.values()),
                "stream": "oper",
                "time": "/".join(self.times),
                "grid": ERA5_OPER_GRID,
            }
        return requests

    def dataset(self) -> dict[str, xr.Dataset]:
        """The downloaded GRIB files opened with cfgrib, one dataset per group.

        Each dataset has a ``time`` dimension (of length 1 for a single time)
        and holds the first requested ensemble member only.
        """
        import cfgrib

        datasets = {}
        for group, path in self.grib_paths().items():
            log.info(f"Loading cfgrib {path.name}...")
            ds = xr.merge(
                cfgrib.open_datasets(str(path)), join="override", compat="override"
            )
            # cfgrib names some oper fields by their native shortName, which
            # differs from the keys used in ERA5_OPER_VARIABLE_META. Rename so the
            # variables are recognised (otherwise mslp/sic are silently dropped).
            ds = ds.rename(
                {
                    k: v
                    for k, v in {
                        "msl": "mslp",
                        "siconc": "sic",
                    }.items()
                    if k in ds.data_vars and v not in ds.data_vars
                }
            )
            if "number" in ds.dims:
                ds = ds.isel(number=0)
            if "time" not in ds.dims:
                ds = ds.expand_dims("time")
            if group == PL_SOURCE:
                ds = self._select_levels(ds, path)
            datasets[group] = ds
        return datasets

    def _select_levels(self, ds: xr.Dataset, path: Path) -> xr.Dataset:
        """The requested pressure levels of a pressure-level dataset, in order."""
        if _CFGRIB_PL_DIM not in ds.dims:
            if _CFGRIB_PL_DIM not in ds.coords:
                raise ValueError(f"{path.name}: no pressure levels found")
            ds = ds.expand_dims(_CFGRIB_PL_DIM)
        found = [int(round(float(v))) for v in ds[_CFGRIB_PL_DIM].values]
        missing = [level for level in self.pressure_levels if level not in found]
        if missing:
            raise ValueError(f"{path.name}: pressure levels {missing} hPa missing")
        order = [found.index(level) for level in self.pressure_levels]
        return ds.isel({_CFGRIB_PL_DIM: order})

    def prepare(
        self, output_path: str, force_download: bool = False
    ) -> ERA5PrepareResult:
        """Download ERA5 GRIBs (if needed), build PSFResamplers, init zarr."""
        import pandas as pd

        self._map_cache = {}
        self._sub_resampler_cache = {}

        self._pl_level_ds = {}

        paths = self.download(force=force_download)
        datasets = self.dataset()
        ds_enda = datasets.get(ENDA_GROUP)
        ds_oper = datasets.get(OPER_GROUP)
        ds_pl = datasets.get(PL_SOURCE)

        lon_enda = lat_enda = nr_enda = None
        if ds_enda is not None:
            lon_enda, lat_enda = self._extract_lonlat(ds_enda)
            nr_enda = _get_resampler(lon_enda, lat_enda)
        lon_oper = lat_oper = nr_oper = None
        # the pressure levels are on the oper grid and share its resampler
        ds_grid = ds_oper if ds_oper is not None else ds_pl
        if ds_grid is not None:
            lon_oper, lat_oper = self._extract_lonlat(ds_grid)
            nr_oper = _get_resampler(lon_oper, lat_oper)
        if ds_pl is not None and ds_oper is not None:
            lon_pl, lat_pl = self._extract_lonlat(ds_pl)
            if not (
                np.array_equal(lon_pl, lon_oper) and np.array_equal(lat_pl, lat_oper)
            ):
                raise ValueError("the pressure levels are not on the oper grid")

        def valid_times(ds: xr.Dataset) -> list[datetime]:
            return [
                pd.Timestamp(t).tz_localize(None).to_pydatetime().replace(tzinfo=UTC)
                for t in np.atleast_1d(ds["valid_time"].values).ravel()
            ]

        times = {}
        if ds_enda is not None:
            times[ENDA_GROUP] = valid_times(ds_enda)
        if ds_oper is not None:
            times[OPER_GROUP] = valid_times(ds_oper)
        if ds_pl is not None:
            pl_times = valid_times(ds_pl)
            if OPER_GROUP in times and pl_times != times[OPER_GROUP]:
                raise ValueError(
                    "the pressure-level times differ from the oper times: "
                    f"{pl_times} vs {times[OPER_GROUP]}"
                )
            times[OPER_GROUP] = pl_times
        all_times = sorted({t for group_times in times.values() for t in group_times})

        def present(meta: Mapping, ds: xr.Dataset | None, what: str) -> list[str]:
            if ds is None:
                return []
            missing = [v for v in meta if v not in ds.data_vars]
            if missing:
                # a store silently missing a requested field is worse than none
                raise ValueError(
                    f"ERA5 {what}: requested but not in the GRIB: {missing}"
                )
            return list(meta)

        vars_enda = present(self.enda_variables, ds_enda, ENDA_GROUP)
        vars_oper = present(self.oper_variables, ds_oper, OPER_GROUP)
        vars_pl = present(self.pl_variables, ds_pl, PL_SOURCE)
        if ds_pl is not None:
            for k in range(len(self.pressure_levels)):
                self._pl_level_ds[k] = ds_pl.isel({_CFGRIB_PL_DIM: k})

        # expver actually received, per time (ERA5 0001 or ERA5T 0005)
        expvers: dict[str, list[str]] = {}
        for group in times:
            sources = [group] + (
                [PL_SOURCE] if group == OPER_GROUP and ds_pl is not None else []
            )
            per_time: dict[datetime, set[str]] = {}
            for source in sources:
                if source in paths:
                    for t, v in _grib_expvers(paths[source]).items():
                        per_time.setdefault(t, set()).update(v.split("/"))
            expvers[group] = [
                "/".join(sorted(per_time.get(t, {"unknown"}))) for t in times[group]
            ]

        start_dt, end_dt = all_times[0], all_times[-1]
        n_times = len(all_times)
        n_chunks = _N_PARENT

        self._init_zarr(
            output_path,
            times,
            vars_enda,
            vars_oper,
            start_dt,
            end_dt,
            vars_pl=vars_pl,
            expvers=expvers,
        )

        self._result = ERA5PrepareResult(
            output_path=output_path,
            n_chunks=n_chunks,
            n_times=n_times,
            start_dt=start_dt,
            end_dt=end_dt,
            grib_enda=paths.get(ENDA_GROUP),
            grib_oper=paths.get(OPER_GROUP),
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
            times=times,
            grib_pl=paths.get(PL_SOURCE),
            ds_pl=ds_pl,
            vars_pl=vars_pl,
            pressure_levels=list(self.pressure_levels),
            expvers=expvers,
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

        for group, ds, nr, variables in [
            (ENDA_GROUP, r.ds_enda, r.nr_enda, r.vars_enda),
            (OPER_GROUP, r.ds_oper, r.nr_oper, r.vars_oper),
        ]:
            if group not in r.times:
                continue
            grp = root[group]

            # all the times of the chunk at once: a zarr chunk may hold several
            for var in variables:
                zarr_a = grp[var]
                zarr_a[:, chunk_sl] = np.stack(
                    [
                        self._full_map(group, var, t, ds, nr, "time")[cell_ids]
                        for t in range(len(r.times[group]))
                    ]
                )

            if group == OPER_GROUP:
                for var in r.vars_pl:
                    zarr_a = grp[f"{var}{PL_SUFFIX}"]
                    for k, level in enumerate(r.pressure_levels):
                        zarr_a[:, k, chunk_sl] = np.stack(
                            [
                                self._full_map(
                                    f"{OPER_GROUP}/{PL_DIM}/{level}",
                                    var,
                                    t,
                                    self._pl_level_ds[k],
                                    nr,
                                    "time",
                                )[cell_ids]
                                for t in range(len(r.times[group]))
                            ]
                        )

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
            res = nr.resample(vals, lam=_PSF_LAM)
        else:
            # source has NaN (land mask): resample valid points only
            sub = self._valid_resampler(group, var, valid)
            res = sub.resample(vals[valid], lam=_PSF_LAM)
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
        """PSFResampler built on the valid (non-NaN) source points of a field.

        Cached per mask, so that a field whose mask changes with time gets the
        right resampler, and fields sharing a mask (sst, sic) share it.
        """
        # keyed on the grid, not the group: the pressure levels share the oper grid
        grid = "enda" if group.endswith("enda") else "oper"
        k = (grid, hashlib.sha1(np.packbits(valid).tobytes()).hexdigest())
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
                threshold=_PSF_THRESHOLD,
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
        times: dict[str, list[datetime]],
        vars_enda: list,
        vars_oper: list,
        start_dt: datetime,
        end_dt: datetime,
        vars_pl: list | None = None,
        expvers: dict[str, list[str]] | None = None,
    ) -> None:
        vars_pl = vars_pl or []
        expvers = expvers or {}
        root = zarr.open_group(output_path, mode="w")
        write_root_conventions(root, self.metadata)
        if self.root_attributes:
            root.attrs.update(self.root_attributes)

        healpix_model = Healpix(
            refinement_level=_CHILD_LEVEL,
            indexing_scheme="nested",
            ellipsoid={"name": "wgs84"},
        )

        # a time coordinate is only needed to tell several times apart
        several_times = any(len(group_times) > 1 for group_times in times.values())

        for group, variables, meta in [
            (ENDA_GROUP, vars_enda, self.enda_variables),
            (OPER_GROUP, vars_oper, self.oper_variables),
        ]:
            if group not in times:
                continue
            n_times = len(times[group])
            time_chunk = (
                n_times if self.time_chunk == "all" else min(self.time_chunk, n_times)
            )
            grp = root.require_group(group)
            grp.attrs.update(get_healpix_group_attrs(healpix_model, self.metadata))
            if group in expvers:
                # one entry per analysis time: 0001 = ERA5, 0005 = ERA5T
                grp.attrs["source_expver"] = expvers[group]
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

            if several_times:
                # analysis times in hours since 00 UTC of the first day (CF)
                epoch = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                time_arr = grp.create_array(
                    "time",
                    shape=(n_times,),
                    dtype=np.int64,
                    chunks=(n_times,),
                    dimension_names=("time",),
                    attributes={
                        "standard_name": "time",
                        "long_name": "analysis time",
                        "axis": "T",
                        "units": f"hours since {epoch:%Y-%m-%d %H:%M:%S}",
                        "calendar": "proleptic_gregorian",
                    },
                )
                time_arr[:] = [(t - epoch) // timedelta(hours=1) for t in times[group]]

            cf_variable_attrs = get_cf_data_variable_attrs(healpix_model, self.metadata)

            for var in variables:
                _, unit, long_name = meta[var]
                grp.create_array(
                    var,
                    shape=(n_times, _N_CHILD),
                    dtype=np.float32,
                    chunks=(time_chunk, _CHUNK_SIZE),
                    fill_value=np.nan,
                    dimension_names=("time", "cells"),
                    attributes={
                        "units": unit,
                        "long_name": long_name,
                        **cf_variable_attrs,
                    },
                )

            if group == OPER_GROUP and vars_pl:
                n_levels = len(self.pressure_levels)
                level_arr = grp.create_array(
                    PL_DIM,
                    shape=(n_levels,),
                    dtype=np.float64,
                    chunks=(n_levels,),
                    dimension_names=(PL_DIM,),
                    attributes={
                        "standard_name": "air_pressure",
                        "long_name": "pressure",
                        "units": "hPa",
                        "positive": "down",
                        "axis": "Z",
                    },
                )
                level_arr[:] = self.pressure_levels
                for var in vars_pl:
                    _, unit, long_name = self.pl_variables[var]
                    grp.create_array(
                        f"{var}{PL_SUFFIX}",
                        shape=(n_times, n_levels, _N_CHILD),
                        dtype=np.float32,
                        chunks=(time_chunk, 1, _CHUNK_SIZE),
                        fill_value=np.nan,
                        dimension_names=("time", PL_DIM, "cells"),
                        attributes={
                            "units": unit,
                            "long_name": long_name,
                            **cf_variable_attrs,
                        },
                    )
        zarr.consolidate_metadata(root.store)
        log.info(f"ERA5 zarr skeleton initialised: {output_path}")

    def _build_stac(self, output_path: str) -> StacItem:
        assert self._result is not None
        import shapely

        r = self._result
        default_layout = (
            r.ds_enda is not None and r.ds_oper is not None and not r.vars_pl
        )
        if default_layout:
            description = (
                f"ERA5 reanalysis surface fields (enda + oper streams) on "
                f"HEALPix NESTED level {_CHILD_LEVEL} (~50 km/pixel)"
            )
        else:
            streams = " + ".join(
                s
                for s, ds in (("enda", r.ds_enda), ("oper", r.ds_oper))
                if ds is not None
            )
            description = (
                f"ERA5 reanalysis fields ({streams or 'oper'} stream"
                f"{'s' if ' + ' in streams else ''}"
                + (
                    f"; {', '.join(r.vars_pl)} on "
                    f"{', '.join(str(lev) for lev in r.pressure_levels)} hPa"
                    if r.vars_pl
                    else ""
                )
                + f") on HEALPix NESTED level {_CHILD_LEVEL} (~50 km/pixel)"
            )
        grids = [ERA5_ENDA_GRID] if r.ds_enda is not None else []
        if r.ds_oper is not None or r.ds_pl is not None:
            grids.append(ERA5_OPER_GRID)
        requests = self.requests()
        now = datetime.now(UTC).isoformat()
        all_expvers = sorted(
            {
                v
                for group_v in r.expvers.values()
                for item in group_v
                for v in item.split("/")
            }
        )

        # The STAC id must match the product name on disk: derive it from the
        # output path rather than a fresh timestamp (which would not match the
        # filename's creation stamp set at prepare() time).
        name = Path(str(output_path)).name
        product_id = name[:-5] if name.endswith(".zarr") else name

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
                "product:type": "ADF_ECMWA",
                "description": description,
                "healpix:level": _CHILD_LEVEL,
                "healpix:nside": 2**_CHILD_LEVEL,
                "healpix:ordering": "NESTED",
                "source_grid": f"Gaussian {'/'.join(grids)}",
                "source_dataset": CDS_DATASET,
                # the request of each GRIB file and the versions received
                "source_requests": requests,
                "source_expver": all_expvers,
                "resampling:method": (f"PSFResampler(level={_CHILD_LEVEL})"),
                "resampling:parameters": {
                    "resampler": "healpix_resample.PSFResampler",
                    "level": _CHILD_LEVEL,
                    "indexing_scheme": "nested",
                    "ellipsoid": "WGS84",
                    "threshold": _PSF_THRESHOLD,
                    "lam": _PSF_LAM,
                    "chunk_level": _CHUNK_LEVEL,
                    "clip": "to the source minimum and maximum of each field and time",
                    "masked_fields": (
                        "resampled from their valid source points only, one "
                        "resampler per mask; cells without valid points stay NaN"
                    ),
                    "ensemble_member": (
                        int(self.numbers[0]) if r.ds_enda is not None else None
                    ),
                },
                "processing:datetime": now,
                "processing:software": _software_versions(),
                "processing:lineage": (
                    "Resampled onto HEALPix by healpix-convert ERA5Converter from "
                    f"the CDS dataset {CDS_DATASET} (requests in source_requests)"
                ),
            },
            links=[],
            assets={},
        )
