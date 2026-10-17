"""
Multi-resolution (pyramid) copy of a converted HEALPix dataset.

This is an optional stage that runs after :py:func:`convert_group_to_healpix`:
it reads the HEALPix level groups of a converted dataset and writes a new
dataset where every level, from the finest native level down to a given
coarsest level, holds all the variables that can be shown at that level.
Viewers such as Gridlook can then pick the level from the zoom.

"""

from __future__ import annotations

import copy
import fnmatch
from collections.abc import Iterator, Sequence
from pathlib import PurePath
from typing import Any, Literal

import numpy as np
import structlog
import zarr
from zarr.codecs import BytesCodec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.registry import get_codec_class

from healpix_convert.core.healpix_conventions import DGGSZarrConvention, Healpix
from healpix_convert.core.multiscales_conventions import (
    HealpixMultiscales,
    MultiscalesZarrConvention,
)

log = structlog.get_logger()

#: Name of the STAC item property that records the resampling method.
#: (provisional name, to be reviewed)
STAC_RESAMPLING_PROPERTY = "multires:resampling_method"

#: Cells read at once from an input array (when larger than its chunks).
_READ_SIZE = 1 << 20

#: Lossless counterpart of a lossy serializer, for ``lossless_derived_levels``:
#: same scaling, so the derived levels keep the digital-number lattice.
_LOSSLESS_SERIALIZERS = {"hcc-sentinel2-pyramid": "hcc-sentinel2"}
_LOSSLESS_KEYS = ("quantification_value", "add_offset", "effort")


class _Block:
    """Sorted cells of one refinement level within one block, with the
    variables defined on them.

    ``sums`` and ``counts`` hold, per variable, the sum and the number of the
    valid cells each cell derives from (at the level the variable was read
    from), so that coarsening gives the mean of all the valid descendants (a
    block mean, like the provider's resampled copies) and not the mean of the
    children's means. They are computed from the values when missing."""

    def __init__(
        self,
        cells: np.ndarray,
        values: dict[str, np.ndarray],
        sums: dict[str, np.ndarray] | None = None,
        counts: dict[str, np.ndarray] | None = None,
    ):
        self.cells = cells
        self.values = values
        self.sums = {} if sums is None else sums
        self.counts = {} if counts is None else counts

    def accumulators(self, name: str, fill: Any) -> tuple[np.ndarray, np.ndarray]:
        """Running sum and count of the valid cells behind each cell of
        ``name``; fill values and non-finite values are not valid."""
        if name not in self.sums:
            v = self.values[name]
            valid = np.isfinite(v) & ~_fill_mask(v, fill)
            self.sums[name] = np.where(valid, v, 0).astype(np.float64)
            self.counts[name] = valid.astype(np.int64)
        return self.sums[name], self.counts[name]

    def coarsen(self, fill_values: dict[str, Any]) -> _Block:
        """Each parent cell gets the mean of the valid cells it derives from
        (the fill value when there is none)."""
        parents = self.cells >> 2
        starts = np.flatnonzero(np.r_[True, parents[1:] != parents[:-1]])
        values, sums, counts = {}, {}, {}
        for name, v in self.values.items():
            fill = fill_values[name]
            total, count = (
                np.add.reduceat(a, starts) for a in self.accumulators(name, fill)
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = total / count
            if np.issubdtype(v.dtype, np.integer):
                mean = np.rint(mean)
            mean[count == 0] = 0 if fill is None else fill
            values[name] = mean.astype(v.dtype)
            sums[name], counts[name] = total, count
        return _Block(parents[starts], values, sums, counts)

    def merge(self, other: _Block, fill_values: dict[str, Any]) -> _Block:
        """Both blocks' variables on the union of their cells. A variable of
        ``other`` replaces the same variable of ``self`` (with its own running
        sums: a resampled copy restarts the mean from its values)."""
        cells = np.union1d(self.cells, other.cells)
        values, sums, counts = {}, {}, {}
        for block in (self, other):
            idx = np.searchsorted(cells, block.cells)
            for name, v in block.values.items():
                fill = fill_values[name]
                values[name] = np.full(len(cells), fill, dtype=v.dtype)
                values[name][idx] = v
                sums[name] = np.zeros(len(cells), dtype=np.float64)
                counts[name] = np.zeros(len(cells), dtype=np.int64)
                sums[name][idx], counts[name][idx] = block.accumulators(name, fill)
        return _Block(cells, values, sums, counts)


class _LevelReader:
    """Reads the cells and variables of one input level group sequentially,
    one block (cells sharing an ancestor at the block level) at a time."""

    def __init__(
        self,
        group: zarr.Group,
        coordinate: str,
        names: list[str],
        block_shift: int,
    ):
        self.coordinate = group[coordinate]
        self.arrays = {name: group[name] for name in names}
        self.block_shift = block_shift
        self.size = self.coordinate.shape[0]
        chunk = self.coordinate.chunks[0]
        self.read_size = max(chunk, (_READ_SIZE // chunk) * chunk)
        self.position = 0
        self.cells = np.empty(0, dtype=self.coordinate.dtype)
        self.values = {
            name: np.empty(0, dtype=a.dtype) for name, a in self.arrays.items()
        }
        self.last_cell: int | None = None

    def _read_more(self) -> bool:
        if self.position >= self.size:
            return False
        stop = min(self.position + self.read_size, self.size)
        cells = self.coordinate[self.position : stop]
        previous = self.cells[-1:] if len(self.cells) else np.empty(0, cells.dtype)
        if self.last_cell is not None and len(previous) == 0:
            previous = np.array([self.last_cell], dtype=cells.dtype)
        if np.any(np.diff(np.concatenate([previous, cells]).astype(np.int64)) <= 0):
            raise ValueError(
                f"the {self.coordinate.path!r} cell ids must be sorted in ascending "
                "(nested) order and unique"
            )
        self.cells = np.concatenate([self.cells, cells])
        for name, a in self.arrays.items():
            self.values[name] = np.concatenate(
                [self.values[name], a[self.position : stop]]
            )
        self.position = stop
        return True

    def next_block(self) -> int | None:
        """Block id of the next unread cell, or None at the end."""
        while not len(self.cells):
            if not self._read_more():
                return None
        return int(self.cells[0] >> self.block_shift)

    def take(self, block: int) -> _Block | None:
        """Cells and values of the given block (None if the next unread cell
        is in a later block)."""
        if self.next_block() != block:
            return None
        while int(self.cells[-1] >> self.block_shift) == block and self._read_more():
            pass
        end = int(np.searchsorted(self.cells >> self.block_shift, block, side="right"))
        out = _Block(
            self.cells[:end], {name: v[:end] for name, v in self.values.items()}
        )
        self.last_cell = int(self.cells[end - 1])
        self.cells = self.cells[end:]
        self.values = {name: v[end:] for name, v in self.values.items()}
        return out


class _LevelWriter:
    """Appends cells and variables of one output level group, in whole chunks."""

    def __init__(
        self,
        group: zarr.Group,
        coordinate: str,
        templates: dict[str, zarr.Array],
        attrs: dict[str, dict],
        chunk_size: int,
    ):
        self.chunk_size = chunk_size
        self.names = [coordinate, *[n for n in templates if n != coordinate]]
        self.coordinate = coordinate
        self.fill_values = {}
        self.arrays = {}
        for name in self.names:
            template = templates[name]
            self.fill_values[name] = template.fill_value
            self.arrays[name] = group.create_array(
                name,
                shape=(0,),
                chunks=(chunk_size,),
                dtype=template.dtype,
                fill_value=template.fill_value,
                serializer=template.serializer,
                compressors=template.compressors,
                filters=template.filters,
                dimension_names=template.metadata.dimension_names,
                attributes=attrs[name],
            )
        self.buffers: dict[str, list[np.ndarray]] = {name: [] for name in self.names}
        self.buffered = 0
        self.size = 0

    def append(self, block: _Block):
        n = len(block.cells)
        if not n:
            return
        self.buffers[self.coordinate].append(block.cells)
        for name in self.names[1:]:
            v = block.values.get(name)
            if v is None:
                v = np.full(n, self.fill_values[name], dtype=self.arrays[name].dtype)
            self.buffers[name].append(v)
        self.buffered += n
        if self.buffered >= self.chunk_size:
            self._flush(whole_chunks=True)

    def _flush(self, whole_chunks: bool):
        n = self.buffered
        if whole_chunks:
            n = (n // self.chunk_size) * self.chunk_size
        if not n:
            return
        for name in self.names:
            data = np.concatenate(self.buffers[name])
            array = self.arrays[name]
            array.resize((self.size + n,))
            array[self.size : self.size + n] = data[:n]
            self.buffers[name] = [data[n:]]
        self.size += n
        self.buffered -= n

    def close(self) -> int:
        self._flush(whole_chunks=False)
        return self.size


def _same_encoding(a: zarr.Array, b: zarr.Array) -> bool:
    """Whether a stored chunk of ``a`` decodes to the same values in ``b``."""
    da, db = a.metadata.to_dict(), b.metadata.to_dict()
    keys = ("data_type", "fill_value", "codecs", "chunk_grid", "chunk_key_encoding")
    return all(da.get(k) == db.get(k) for k in keys)


def _copy_stored_chunk(src: zarr.Array, k: int, dst: zarr.Array, j: int) -> None:
    """Copy chunk ``k`` of ``src`` to chunk ``j`` of ``dst`` as stored, without
    decoding it (an absent chunk, i.e. all fill value, stays absent)."""
    key = f"{src.path}/{src.metadata.encode_chunk_key((k,))}"
    data = sync(src.store.get(key, prototype=default_buffer_prototype()))
    if data is not None:
        sync(dst.store.set(f"{dst.path}/{dst.metadata.encode_chunk_key((j,))}", data))


class _DenseLevelWriter:
    """Appends the cells and variables of one output level group in
    parent-aligned dense chunks: each chunk holds the ``4 ** depth`` cells of
    one cell at level ``level - depth``, in nested order, with the fill value
    where there is no data. Serializers that code a chunk as the raster of one
    parent cell (``healpix_cell_dense`` chunks, e.g. the Sentinel-2
    serializers of healpix-compress) need this layout.

    A variable whose values at this level are those of an input array with the
    same chunks and the same encoding (``raw_sources``: a native level) has its
    chunks copied as stored, without decoding and re-encoding them, so that a
    lossy serializer does not add its error a second time.

    In a floating variable, the cells without data are written as NaN rather
    than as the fill value: the Sentinel-2 serializers carry NaN as no data
    (a validity mask), while a fill value such as ``0.0`` would be coded as a
    value next to the real ones, come back as a slightly different value, and
    widen the quantisation step of the chunk."""

    def __init__(
        self,
        group: zarr.Group,
        coordinate: str,
        templates: dict[str, zarr.Array],
        attrs: dict[str, dict],
        depth: int,
        raw_sources: dict[str, tuple[zarr.Array, zarr.Array]],
        serializers: dict[str, Any] | None = None,
    ):
        self.n = 4**depth
        self.shift = 2 * depth
        self.coordinate = coordinate
        self.names = [coordinate, *[n for n in templates if n != coordinate]]
        self.fill_values = {}
        self.arrays = {}
        for name in self.names:
            template = templates[name]
            self.fill_values[name] = template.fill_value
            self.arrays[name] = group.create_array(
                name,
                shape=(0,),
                chunks=(self.n,),
                dtype=template.dtype,
                fill_value=template.fill_value,
                serializer=(serializers or {}).get(name, template.serializer),
                compressors=template.compressors,
                filters=template.filters,
                dimension_names=template.metadata.dimension_names,
                attributes=attrs[name],
            )
        # stored chunks to copy: input array, and input chunk index per chunk id
        self.raw: dict[str, tuple[zarr.Array, dict[int, int]]] = {}
        index_cache: dict[str, dict[int, int]] = {}
        for name, (source, source_cells) in raw_sources.items():
            if not _same_encoding(source, self.arrays[name]):
                continue
            key = str(source_cells.store_path)
            if key not in index_cache:
                firsts = np.asarray(source_cells[:: self.n], dtype=np.uint64)
                index_cache[key] = {
                    int(c) >> self.shift: k for k, c in enumerate(firsts.tolist())
                }
            self.raw[name] = (source, index_cache[key])
        self.current: int | None = None
        self.buffers: dict[str, np.ndarray] = {}
        self.size = 0

    def _no_data(self, name: str) -> Any:
        dtype = self.arrays[name].dtype
        return np.nan if np.issubdtype(dtype, np.floating) else self.fill_values[name]

    def _start(self, chunk_id: int):
        self.current = chunk_id
        self.buffers = {
            name: np.full(self.n, self._no_data(name), dtype=self.arrays[name].dtype)
            for name in self.names[1:]
        }

    def _flush(self):
        if self.current is None:
            return
        start, stop = self.size, self.size + self.n
        j = self.size // self.n
        for array in self.arrays.values():
            array.resize((stop,))
        first = np.uint64(self.current) << np.uint64(self.shift)
        self.arrays[self.coordinate][start:stop] = first + np.arange(
            self.n, dtype=np.uint64
        )
        for name in self.names[1:]:
            raw = self.raw.get(name)
            if raw is not None and self.current in raw[1]:
                _copy_stored_chunk(raw[0], raw[1][self.current], self.arrays[name], j)
            elif not np.all(_fill_mask(self.buffers[name], self._no_data(name))):
                # a chunk without data is not written (it reads as the fill value)
                self.arrays[name][start:stop] = self.buffers[name]
        self.size = stop
        self.current = None

    def append(self, block: _Block):
        if not len(block.cells):
            return
        ids = block.cells >> np.uint64(self.shift)
        starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
        for s, e in zip(starts, np.r_[starts[1:], len(ids)]):
            chunk_id = int(ids[s])
            if chunk_id != self.current:
                self._flush()
                self._start(chunk_id)
            first = np.uint64(chunk_id) << np.uint64(self.shift)
            pos = (block.cells[s:e] - first).astype(np.int64)
            for name in self.names[1:]:
                v = block.values.get(name)
                if v is None:
                    continue
                v = v[s:e]
                fill = self.fill_values[name]
                if np.issubdtype(v.dtype, np.floating) and not _same_fill(fill, np.nan):
                    v = np.where(_fill_mask(v, fill), np.nan, v)
                self.buffers[name][pos] = v

    def close(self) -> int:
        self._flush()
        return self.size


def _dense_chunk_level(native: dict[int, zarr.Group], coordinate: str) -> int | None:
    """The level of the cells that the input chunks hold one each of, when every
    native level is written in parent-aligned dense chunks (``healpix_cell_dense``),
    else None. Checked on the chunk sizes and on the first chunk of cell ids."""
    chunk_levels = set()
    for level, group in native.items():
        cells = group[coordinate]
        n = cells.chunks[0]
        depth = int(round(np.log(n) / np.log(4)))
        if 4**depth != n or depth > level:
            return None
        if any(a.chunks != cells.chunks for _, a in group.arrays() if a.ndim == 1):
            return None
        first = np.asarray(cells[: min(n, cells.shape[0])], dtype=np.uint64)
        if len(first) != n or int(first[0]) % n or np.any(np.diff(first) != 1):
            return None
        chunk_levels.add(level - depth)
    return chunk_levels.pop() if len(chunk_levels) == 1 else None


def _extent_level(native: dict[int, zarr.Group], coordinate: str) -> int | None:
    """The finest level at which one cell holds every input cell (e.g., 7 for
    the store of one level-7 parent cell), else None. Read from the first and
    the last cell id of each native level (dense chunks: full, sorted ranges)."""
    bounds = []
    for level, group in native.items():
        cells = group[coordinate]
        if cells.shape[0]:
            bounds.append((level, int(cells[0]), int(cells[cells.shape[0] - 1])))
    if not bounds:
        return None
    for q in range(min(level for level, _, _ in bounds), -1, -1):
        ancestors = {c >> 2 * (level - q) for level, *ends in bounds for c in ends}
        if len(ancestors) == 1:
            return q
    return None


def _serializer_name(serializer: Any) -> str:
    return serializer.to_dict()["name"]


def _lossless_serializer(serializer: Any) -> Any:
    """The lossless counterpart of a lossy serializer, with the same scaling
    (``_LOSSLESS_SERIALIZERS``); a lossless serializer is returned as is."""
    spec = serializer.to_dict()
    target = _LOSSLESS_SERIALIZERS.get(spec["name"])
    if target is None:
        return serializer
    configuration = {
        k: v for k, v in spec.get("configuration", {}).items() if k in _LOSSLESS_KEYS
    }
    codec_class = get_codec_class(target)
    return codec_class.from_dict({"name": target, "configuration": configuration})


def _codec_tag(derived_serializers: dict[str, str]) -> str:
    """The codec part of the method tag, from the serializers of the derived
    levels: ``-hcc`` when one of them is a lossy healpix-compress serializer,
    ``-hcc-lossless`` when they are all lossless ones, else nothing."""
    names = set(derived_serializers.values())
    if names & set(_LOSSLESS_SERIALIZERS):
        return "-hcc"
    if any(name.startswith("hcc-") for name in names):
        return "-hcc-lossless"
    return ""


def _native_level_groups(group: zarr.Group) -> dict[int, zarr.Group]:
    """Children of ``group`` named after their HEALPix refinement level."""
    levels = {}
    for name, child in group.groups():
        if not name.isdigit():
            continue
        dggs = child.attrs.get("dggs")
        if not isinstance(dggs, dict) or dggs.get("name", "healpix") != "healpix":
            continue
        if int(dggs.get("refinement_level", -1)) != int(name):
            raise ValueError(
                f"group {child.path!r} is named after a refinement level that "
                f"differs from its dggs refinement_level {dggs.get('refinement_level')}"
            )
        if dggs.get("indexing_scheme", "nested") != "nested":
            raise ValueError(
                f"group {child.path!r}: only the nested indexing scheme is supported"
            )
        levels[int(name)] = child
    return levels


def _find_multires_groups(root: zarr.Group) -> list[str]:
    """Groups that have at least one HEALPix level child group."""
    found = []

    def visit(group: zarr.Group):
        if _native_level_groups(group):
            found.append(group.path)
            return
        for _, child in group.groups():
            visit(child)

    visit(root)
    return found


def _cell_variables(group: zarr.Group, dggs: dict) -> tuple[list[str], list[str]]:
    """Names of the variables along the spatial dimension, and of the other
    (e.g., grid mapping) arrays of a level group."""
    dim = dggs.get("spatial_dimension", "cells")
    coordinate = dggs.get("coordinate", "cell_ids")
    cell_vars, others = [], []
    for name, array in group.arrays():
        if name == coordinate:
            continue
        dims = tuple(array.metadata.dimension_names or ())
        if dims == (dim,):
            cell_vars.append(name)
        elif dim in dims:
            raise NotImplementedError(
                f"{group.path}/{name}: only 1-D variables along {dim!r} are "
                f"supported, got dimensions {dims}"
            )
        else:
            others.append(name)
    return sorted(cell_vars), sorted(others)


def _normalize(block: _Block, group: zarr.Group, templates: dict[str, zarr.Array]):
    """Give the values read from ``group`` the dtype and fill value of the
    variable's finest level (a resampled copy may be encoded differently)."""
    for name, v in block.values.items():
        source, target = group[name], templates[name]
        for key in ("scale_factor", "add_offset"):
            if source.attrs.get(key) != target.attrs.get(key):
                raise ValueError(
                    f"{source.path!r} has {key}={source.attrs.get(key)!r} but "
                    f"{target.path!r} has {key}={target.attrs.get(key)!r}: the "
                    "copy cannot be re-encoded like the finest level"
                )
        if source.dtype == target.dtype and _same_fill(
            source.fill_value, target.fill_value
        ):
            continue
        is_fill = _fill_mask(v, source.fill_value)
        if np.issubdtype(target.dtype, np.integer) and not np.issubdtype(
            v.dtype, np.integer
        ):
            # round like ``_Block.coarsen`` does, instead of truncating
            is_fill |= ~np.isfinite(v)
            v = np.rint(np.where(is_fill, 0, v))
        v = v.astype(target.dtype)
        v[is_fill] = target.fill_value
        block.values[name] = v


def _same_fill(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b
    return bool(np.isnan(a) and np.isnan(b)) if np.isnan(a) or np.isnan(b) else a == b


def _fill_mask(v: np.ndarray, fill: Any) -> np.ndarray:
    if fill is None:
        return np.zeros(v.shape, dtype=bool)
    if np.issubdtype(v.dtype, np.floating) and np.isnan(fill):
        return np.isnan(v)
    return v == fill


def _with_refinement_level(attrs: dict, level: int) -> dict:
    attrs = copy.deepcopy(attrs)
    if "refinement_level" in attrs:
        attrs["refinement_level"] = level
    return attrs


_FLAG_ATTRIBUTES = ("flag_values", "flag_masks", "flag_meanings")


def _categorical_variables(native: dict[int, zarr.Group]) -> list[str]:
    """Names of the variables with CF flag attributes (masks, classifications),
    which a mean does not suit."""
    found = []
    for level in sorted(native, reverse=True):
        cell_vars, _ = _cell_variables(native[level], native[level].attrs["dggs"])
        for name in cell_vars:
            attrs = native[level][name].attrs
            if name not in found and any(k in attrs for k in _FLAG_ATTRIBUTES):
                found.append(name)
    return found


def _check_serializers(
    path: str, native: dict[int, zarr.Group], dense_chunks: bool | None
) -> bool:
    """Whether the group is written in parent-aligned dense chunks.

    A serializer other than the default bytes one may code a chunk as the
    raster of one parent cell (e.g., the Sentinel-2 serializers of
    healpix-compress), so it can only be carried over to dense output chunks,
    which need an input written in ``healpix_cell_dense`` chunks."""
    special = [
        (level, name, array.serializer.__class__.__name__)
        for level, group in native.items()
        for name, array in group.arrays()
        if not isinstance(array.serializer, BytesCodec)
    ]
    dense = bool(special) if dense_chunks is None else dense_chunks
    if special and not dense:
        level, name, serializer = special[0]
        raise NotImplementedError(
            f"{path}/{level}/{name} is written with the {serializer} serializer, "
            "which needs dense output chunks (dense_chunks=False was given)"
        )
    if dense:
        coordinate = native[max(native)].attrs["dggs"].get("coordinate", "cell_ids")
        if _dense_chunk_level(native, coordinate) is None:
            what = (
                f"{special[0][0]}/{special[0][1]} is written with the "
                f"{special[0][2]} serializer"
                if special
                else "dense_chunks=True was given"
            )
            raise NotImplementedError(
                f"{path}: {what}, which needs dense output chunks, but the input "
                "is not written in parent-aligned dense (healpix_cell_dense) chunks"
            )
    return dense


def _resolve_groups(root: zarr.Group, groups: str | Sequence[str] | None) -> list[str]:
    """Paths of the groups to process: by default every group that has
    HEALPix level child groups, otherwise the given paths or unix-like
    patterns matched against them."""
    candidates = _find_multires_groups(root)
    if groups is None:
        return candidates
    paths: list[str] = []
    for pattern in [groups] if isinstance(groups, str) else groups:
        pattern = pattern.strip("/")
        matches = [c for c in candidates if fnmatch.fnmatchcase(c, pattern)]
        if not matches:
            raise ValueError(
                f"no group with HEALPix level child groups matches {pattern!r} "
                f"(found: {candidates})"
            )
        paths.extend(m for m in matches if m not in paths)
    return paths


def _check_inputs(
    root: zarr.Group,
    group_paths: list[str],
    *,
    min_level: int,
    explicit: bool,
    dense_chunks: bool | None = None,
) -> tuple[dict[str, frozenset[str]], dict[str, bool]]:
    """Everything that can go wrong before the output is created. Returns the
    groups to process with, for each, the categorical variables to leave out
    (a group with nothing else is left out as a whole), and whether each is
    written in dense chunks."""
    tops: dict[str, int] = {}
    skip: dict[str, frozenset[str]] = {}
    dense: dict[str, bool] = {}
    for path in group_paths:
        native = _native_level_groups(root[path])
        if min_level > min(native):
            raise ValueError(
                f"group {path!r}: min_level {min_level} is finer than the "
                f"coarsest native level {min(native)}"
            )
        dense[path] = _check_serializers(path, native, dense_chunks)
        cell_vars = {
            name
            for level in native
            for name in _cell_variables(native[level], native[level].attrs["dggs"])[0]
        }
        categorical = _categorical_variables(native)
        if categorical and explicit:
            raise ValueError(
                f"group {path!r}: {categorical} have CF flag attributes "
                "(categorical data), which the mean does not suit"
            )
        if categorical:
            whole = len(categorical) == len(cell_vars)
            log.warning(
                f"group {path!r}: {categorical} have CF flag attributes "
                "(categorical data), which the mean does not suit; "
                + ("the group is left out" if whole else "they are left out")
            )
            if whole:
                continue
        skip[path] = frozenset(categorical)
        tops[path] = max(native)
    if not skip:
        raise ValueError("no group left to process")
    if len(set(tops.values())) > 1:
        raise ValueError(
            f"the processed groups have different finest native levels ({tops}); "
            "process them into separate datasets"
        )
    return skip, {path: dense[path] for path in skip}


def _block_level(native: dict[int, zarr.Group], coordinate: str, min_level: int) -> int:
    """Level of the blocks processed at once: about one input chunk of the
    finest level, and never finer than ``min_level``."""
    top = max(native)
    chunk = native[top][coordinate].chunks[0]
    chunk_depth = int(np.floor(np.log(chunk) / np.log(4)))
    return max(0, min(min_level, top - chunk_depth))


def _iter_blocks(readers: dict[int, _LevelReader]) -> Iterator[int]:
    while True:
        heads = [b for b in (r.next_block() for r in readers.values()) if b is not None]
        if not heads:
            return
        yield min(heads)


def _build_group(
    in_group: zarr.Group,
    out_group: zarr.Group,
    *,
    min_level: int,
    method: str,
    chunk_size: int | None,
    overwrite_resampled: bool,
    skip: frozenset[str] = frozenset(),
    dense: bool = False,
    lossless_derived_levels: bool = False,
) -> dict[str, Any]:
    native = _native_level_groups(in_group)
    top = max(native)

    dggs_top = dict(native[top].attrs["dggs"])
    coordinate = dggs_top.get("coordinate", "cell_ids")
    block_level = _block_level(native, coordinate, min_level)

    # A variable may be stored at several levels: e.g., Sentinel-2 L2A also ships
    # the 10 m bands resampled by the provider at 20 m and 60 m. By default these
    # resampled copies are kept (used at their level, and coarsened from there);
    # with ``overwrite_resampled`` they are ignored and every level below the
    # finest one is the mean of the finer level.
    native_vars: dict[int, list[str]] = {}
    resampled: dict[int, list[str]] = {}
    other_arrays: dict[str, zarr.Array] = {}
    first_level: dict[str, int] = {}
    for level in sorted(native, reverse=True):
        cell_vars, others = _cell_variables(native[level], native[level].attrs["dggs"])
        cell_vars = [name for name in cell_vars if name not in skip]
        copies = [name for name in cell_vars if name in first_level]
        if copies:
            resampled[level] = copies
            log.info(
                f"level {level}: {copies} also stored at a finer level, "
                + ("ignored" if overwrite_resampled else "kept")
            )
            if overwrite_resampled:
                cell_vars = [name for name in cell_vars if name not in first_level]
        for name in cell_vars:
            first_level.setdefault(name, level)
        native_vars[level] = cell_vars
        for name in others:
            other_arrays.setdefault(name, native[level][name])

    # the encoding of a variable is the one of its finest level
    fill_values: dict[str, Any] = {}
    templates: dict[str, zarr.Array] = {}
    for name, level in first_level.items():
        templates[name] = native[level][name]
        fill_values[name] = native[level][name].fill_value

    # level the values of each variable come from, at each output level
    origin: dict[int, dict[str, int]] = {}
    current_origin: dict[str, int] = {}
    for level in range(top, min_level - 1, -1):
        for name in native_vars.get(level, []):
            current_origin[name] = level
        origin[level] = dict(current_origin)
    coordinate_template = native[top][coordinate]
    if chunk_size is None:
        chunk_size = coordinate_template.chunks[0]
    derived_serializers: dict[str, Any] = {}
    if dense:
        # a native level keeps the input's chunks (one cell of the input chunk
        # level each); a derived level has chunks of the same level, or of the
        # smallest input chunk size when that would make them smaller, and never
        # larger than the cell that holds all the input (e.g., the store of one
        # level-7 parent cell keeps its level 12 inside that cell)
        chunk_level = _dense_chunk_level(native, coordinate)
        min_depth = min(level - chunk_level for level in native)
        extent_level = _extent_level(native, coordinate)

        def depth_at(level: int) -> int:
            depth = max(level - chunk_level, min_depth)
            if extent_level is not None:
                depth = min(depth, max(level - extent_level, 0))
            return depth

        # the derived levels take the serializer of the finest level, or its
        # lossless counterpart with ``lossless_derived_levels``
        derived_serializers = {
            name: _lossless_serializer(t.serializer)
            if lossless_derived_levels
            else t.serializer
            for name, t in templates.items()
        }

    readers = {
        level: _LevelReader(
            native[level], coordinate, native_vars[level], 2 * (level - block_level)
        )
        for level in native
    }

    # output level groups and writers
    writers: dict[int, _LevelWriter] = {}
    layout = []
    for level in range(top, min_level - 1, -1):
        names = sorted(origin[level])
        source_group = native.get(level, native[top])
        group_attrs = copy.deepcopy(dict(source_group.attrs))
        dggs = Healpix.model_validate(dggs_top).model_copy(
            update={"refinement_level": level}
        )
        group_attrs["dggs"] = dggs.model_dump()
        level_group = out_group.create_group(str(level), attributes=group_attrs)

        for name, array in other_arrays.items():
            copied = level_group.create_array(
                name,
                shape=array.shape,
                chunks=array.chunks,
                dtype=array.dtype,
                fill_value=array.fill_value,
                serializer=array.serializer,
                compressors=array.compressors,
                filters=array.filters,
                dimension_names=array.metadata.dimension_names,
                attributes=_with_refinement_level(dict(array.attrs), level),
            )
            copied[...] = array[...]

        attrs = {coordinate: dict(coordinate_template.attrs)}
        for name in names:
            # the attributes follow the values: those of the level they come from
            # (e.g., a provider's resampled copy), the encoding is the finest one
            attrs[name] = {
                **dict(native[origin[level][name]][name].attrs),
                "native_refinement_level": origin[level][name],
            }
        level_templates = {
            coordinate: coordinate_template,
            **{n: templates[n] for n in names},
        }
        if dense:
            raw_sources = {
                n: (native[level][n], native[level][coordinate])
                for n in names
                if level in native and origin[level][n] == level
            }
            writers[level] = _DenseLevelWriter(
                level_group,
                coordinate,
                level_templates,
                attrs,
                depth_at(level),
                raw_sources,
                {n: derived_serializers[n] for n in names if origin[level][n] != level},
            )
        else:
            writers[level] = _LevelWriter(
                level_group, coordinate, level_templates, attrs, chunk_size
            )

        entry: dict[str, Any] = {"asset": str(level), "dggs": dggs}
        if level < top:
            # variables native at this level are copied, the others are derived
            entry.update(
                derived_from=str(level + 1),
                transform={"scale": [2.0]},
                resampling_method=method,
            )
        layout.append(entry)

    # data, one block at a time
    fill_values[coordinate] = coordinate_template.fill_value
    for block in _iter_blocks(readers):
        current: _Block | None = None
        for level in range(top, min_level - 1, -1):
            if current is not None:
                current = current.coarsen(fill_values)
            if level in readers:
                new = readers[level].take(block)
                if new is not None:
                    _normalize(new, native[level], templates)
                    # a kept resampled copy replaces the coarsened values
                    current = (
                        new if current is None else current.merge(new, fill_values)
                    )
            if current is not None:
                writers[level].append(current)

    sizes = {level: writer.close() for level, writer in writers.items()}
    for level, size in sizes.items():
        log.info(f"level {level}: {size} cells")

    # names of the serializers of the derived levels, of the variables that have any
    derived_names = {
        name: _serializer_name(derived_serializers[name])
        for name in derived_serializers
        if any(origin[level].get(name, level) != level for level in origin)
    }
    tag = f"{method}-l{top}-l{min_level}{_codec_tag(derived_names)}"
    group_attrs = {
        k: v
        for k, v in in_group.attrs.items()
        if k not in ("zarr_conventions", "multiscales")
    }
    multiscales = HealpixMultiscales.model_validate(
        {"layout": layout, "resampling_method": method}
    )
    group_attrs.update(
        {
            "zarr_conventions": [
                MultiscalesZarrConvention().model_dump(),
                DGGSZarrConvention().model_dump(),
            ],
            "multiscales": multiscales.model_dump(),
            "multires_method": tag,
            "native_refinement_levels": {
                str(level): names for level, names in native_vars.items() if names
            },
        }
    )
    if derived_names:
        group_attrs["derived_serializers"] = derived_names
    if resampled:
        group_attrs["resampled_copies"] = {
            "levels": {str(level): names for level, names in resampled.items()},
            "handling": "overwritten_with_mean" if overwrite_resampled else "kept",
        }
    out_group.attrs.update(group_attrs)
    return {"tag": tag, "top": top, "sizes": sizes}


def create_multires_dataset(
    input_path: str,
    output_path: str,
    *,
    min_level: int,
    groups: str | Sequence[str] | None = None,
    method: Literal["mean"] = "mean",
    chunk_size: int | None = None,
    overwrite_resampled: bool = False,
    dense_chunks: bool | None = None,
    lossless_derived_levels: bool = False,
    input_storage_options: dict[str, Any] | None = None,
    output_storage_options: dict[str, Any] | None = None,
) -> str:
    """Write a multi-resolution (pyramid) copy of a converted HEALPix dataset.

    Each processed group of the input dataset must hold one child group per
    native HEALPix refinement level, named after the level (e.g.,
    ``measurements/reflectance/20``, ``.../19`` and ``.../17`` for Sentinel-2,
    where each band is stored at its native resolution). In the output, the
    group holds every level from the finest native level down to ``min_level``:

    - a variable is copied unchanged at its native level and is coarsened
      towards coarser levels, each cell being the mean of the valid cells of
      the native level (or of the resampled copy, see below) it derives from;
      fill values and non-finite values are ignored, so this is a block mean
      and not a mean of the children's means;
    - a variable never appears at levels finer than its native level (no
      upsampling);
    - the cells of a level are the union of the cells of its variables; a
      variable has its fill value where it has no data;
    - a variable stored at several levels (e.g., Sentinel-2 L2A, which also
      ships the 10 m bands resampled by the provider at 20 m and 60 m) keeps
      these resampled copies by default: at such a level the copy is used as is
      (it replaces the mean, with the fill value where the copy has no data),
      and coarser levels are derived from it. With ``overwrite_resampled=True``
      the copies are ignored and every level is the mean of the finer one.
      The group then records the copies in ``resampled_copies``.

    Every output array records its ``native_refinement_level``: the level its
    values come from. The output group carries a zarr-conventions
    ``multiscales`` layout (assets relative to the group), the method tag
    (``multires_method``, e.g. ``"mean-l20-l12"``), the input variables used
    at each level (``native_refinement_levels``) and, when the input has
    resampled copies, ``resampled_copies``
    (``{"levels": {"19": [...]}, "handling": "kept" | "overwritten_with_mean"}``).
    In the root ``stac_discovery`` attribute,
    the method tag is appended to the item ``id`` (so that several pyramids of
    one product can share a STAC collection) and the resampling method is
    recorded as the ``multires:resampling_method`` property.

    Only the processed groups (and their parents) are written to the output
    dataset. Zarr metadata is consolidated in the root group and in every
    processed group, so that the latter can be opened on its own (e.g., by
    Gridlook).

    The input is read sequentially, one block of cells sharing an ancestor at
    ``min(min_level, ~chunk level of the finest input level)`` at a time, so at
    most ``max(chunk size, 4 ** (finest level - min_level))`` cells of the
    finest level are held in memory at once (plus a running sum and count per
    variable). The inputs are checked (levels, ``min_level``, serializers,
    categorical variables) before the output is created.

    By default the output chunks hold a fixed number of cells. An input written
    with a serializer other than the default bytes one (e.g., the Sentinel-2
    serializers of healpix-compress, which code a chunk as the raster of one
    parent cell) is written in parent-aligned dense chunks instead, with the
    serializer of each variable's finest level: each chunk holds all the
    cells of one parent cell, with the fill value where there is no data. This
    needs an input written in ``healpix_cell_dense`` chunks. The native levels
    keep the input's chunks, and the variables copied unchanged there have
    their chunks copied as stored (not decoded and re-encoded, so that a lossy
    serializer does not add its error twice); a derived level has chunks of the
    same parent level, or of the smallest input chunk size when that would make
    them smaller, and never larger than the cell that holds all the input (so
    that the store of one parent cell writes no cell outside it). The derived
    levels take the serializer of the variable's finest level, which may be
    lossy (``hcc-sentinel2-pyramid``): their values are then the block means
    plus the serializer's error. ``lossless_derived_levels=True`` writes them
    with the lossless counterpart instead (``hcc-sentinel2``, same scaling).
    The group records the serializers of the derived levels in
    ``derived_serializers``, and the method tag ends with ``-hcc`` (lossy) or
    ``-hcc-lossless`` when they are healpix-compress serializers.

    Parameters
    ----------
    input_path : str
        Path to the converted HEALPix Zarr dataset.
    output_path : str
        Path to the output Zarr dataset (must not exist).
    min_level : int
        Coarsest refinement level to write. Must not be finer than the coarsest
        native level.
    groups : str or list, optional
        Group(s) to process: paths or unix-like patterns (as for
        :py:func:`create_healpix_dataset`), matched against the groups that
        have HEALPix level child groups. By default, every such group.
    method : {"mean"}
        Resampling method. Only ``"mean"`` is supported, which is not
        suited to categorical variables (masks, classifications): variables
        with CF ``flag_values`` / ``flag_masks`` / ``flag_meanings``
        attributes are left out (with a warning) when ``groups`` is not
        given, and raise an error when their group is explicitly requested.
    chunk_size : int, optional
        Number of cells per output chunk. Defaults to the chunk size of the
        cell ids of the finest native level. Not used with dense chunks.
    overwrite_resampled : bool, optional
        If True, ignore the copies of a variable stored at a coarser level than
        its finest one, and derive those levels with ``method`` instead
        (default: False, the copies are kept).
    dense_chunks : bool, optional
        Write parent-aligned dense chunks (see above). By default (None), only
        when the input uses a serializer other than the default bytes one.
    lossless_derived_levels : bool, optional
        With dense chunks, write the derived levels with the lossless
        counterpart of a lossy serializer (``hcc-sentinel2`` for
        ``hcc-sentinel2-pyramid``, same ``quantification_value`` and
        ``add_offset``), so that their values are the block means rounded to
        the digital-number lattice. Default: False, the serializer of the
        finest level, as for the native levels.
    input_storage_options, output_storage_options : dict, optional
        Storage options passed to Zarr (useful for, e.g., S3 object stores).

    Returns
    -------
    str
        The method tag, e.g. ``"mean-l20-l12"``, or ``"mean-l20-l12-hcc"``
        when the derived levels use a lossy healpix-compress serializer.

    """
    if method != "mean":
        raise ValueError(f"unsupported resampling method {method!r}")

    in_root = zarr.open_group(
        input_path, mode="r", storage_options=input_storage_options
    )
    group_paths = _resolve_groups(in_root, groups)
    if not group_paths:
        raise ValueError(f"no group with HEALPix level child groups in {input_path!r}")
    skip, dense = _check_inputs(
        in_root,
        group_paths,
        min_level=min_level,
        explicit=groups is not None,
        dense_chunks=dense_chunks,
    )
    group_paths = list(skip)
    if chunk_size is not None and any(dense.values()):
        raise ValueError("chunk_size cannot be set with dense chunks")

    out_root = zarr.open_group(
        output_path, mode="w-", zarr_format=3, storage_options=output_storage_options
    )

    tags = set()
    for path in group_paths:
        log.info(f"••• building multi-resolution group {path!r}...")
        in_group = in_root[path]
        # parent groups, with their attributes
        parts = PurePath(path).parts
        for i in range(1, len(parts)):
            parent = "/".join(parts[:i])
            if parent not in out_root:
                out_root.create_group(parent, attributes=dict(in_root[parent].attrs))
        out_group = out_root.create_group(path)
        result = _build_group(
            in_group,
            out_group,
            min_level=min_level,
            method=method,
            chunk_size=chunk_size,
            overwrite_resampled=overwrite_resampled,
            skip=skip[path],
            dense=dense[path],
            lossless_derived_levels=lossless_derived_levels,
        )
        tags.add(result["tag"])
    if len(tags) > 1:
        raise ValueError(
            f"the processed groups give different method tags ({sorted(tags)}); "
            "process them into separate datasets"
        )
    (tag,) = tags

    root_attrs = copy.deepcopy(dict(in_root.attrs))
    stac = root_attrs.get("stac_discovery")
    if isinstance(stac, dict):
        if "id" in stac:
            stac["id"] = f"{stac['id']}_{tag}"
        stac.setdefault("properties", {})[STAC_RESAMPLING_PROPERTY] = method
    out_root.attrs.update(root_attrs)

    log.info("••• consolidating output Zarr metadata...")
    for path in group_paths:
        zarr.consolidate_metadata(out_root.store, path=path)
    zarr.consolidate_metadata(out_root.store)

    log.info(f"••• done! ({tag})")
    return tag
