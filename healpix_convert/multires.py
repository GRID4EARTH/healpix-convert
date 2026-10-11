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
from collections.abc import Iterator, Sequence
from pathlib import PurePath
from typing import Any, Literal

import numpy as np
import structlog
import zarr

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


class _Block:
    """Sorted cells of one refinement level within one block, with the
    variables defined on them."""

    def __init__(self, cells: np.ndarray, values: dict[str, np.ndarray]):
        self.cells = cells
        self.values = values

    def coarsen(self, fill_values: dict[str, Any]) -> _Block:
        """Mean of the (up to 4) children of each parent cell, ignoring fill
        values and non-finite values."""
        parents = self.cells >> 2
        starts = np.flatnonzero(np.r_[True, parents[1:] != parents[:-1]])
        out = {}
        for name, v in self.values.items():
            fill = fill_values[name]
            valid = np.isfinite(v) & (v != fill) if fill is not None else np.isfinite(v)
            total = np.add.reduceat(np.where(valid, v, 0).astype(np.float64), starts)
            count = np.add.reduceat(valid.astype(np.int64), starts)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = total / count
            if np.issubdtype(v.dtype, np.integer):
                mean = np.rint(mean)
            no_data = count == 0
            mean[no_data] = 0 if fill is None else fill
            out[name] = mean.astype(v.dtype)
        return _Block(parents[starts], out)

    def merge(self, other: _Block, fill_values: dict[str, Any]) -> _Block:
        """Both blocks' variables on the union of their cells."""
        cells = np.union1d(self.cells, other.cells)
        values = {}
        for block in (self, other):
            idx = np.searchsorted(cells, block.cells)
            for name, v in block.values.items():
                full = np.full(len(cells), fill_values[name], dtype=v.dtype)
                full[idx] = v
                values[name] = full
        return _Block(cells, values)


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
        if tuple(array.metadata.dimension_names or ()) == (dim,):
            cell_vars.append(name)
        else:
            others.append(name)
    return sorted(cell_vars), sorted(others)


def _with_refinement_level(attrs: dict, level: int) -> dict:
    attrs = copy.deepcopy(attrs)
    if "refinement_level" in attrs:
        attrs["refinement_level"] = level
    return attrs


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
) -> dict[str, Any]:
    native = _native_level_groups(in_group)
    top = max(native)
    if min_level > min(native):
        raise ValueError(
            f"group {in_group.path!r}: min_level {min_level} is finer than the "
            f"coarsest native level {min(native)}"
        )

    dggs_top = dict(native[top].attrs["dggs"])
    coordinate = dggs_top.get("coordinate", "cell_ids")
    block_level = _block_level(native, coordinate, min_level)

    # variables native at each level: a variable stored at several levels (e.g.,
    # Sentinel-2 L2A also ships the 10 m bands resampled at 20 m and 60 m) is
    # taken from its finest level only, so every level is derived the same way
    native_vars: dict[int, list[str]] = {}
    other_arrays: dict[str, zarr.Array] = {}
    seen: dict[str, int] = {}
    for level in sorted(native, reverse=True):
        cell_vars, others = _cell_variables(native[level], native[level].attrs["dggs"])
        skipped = [name for name in cell_vars if name in seen]
        if skipped:
            log.info(
                f"level {level}: {skipped} also stored at a finer level, "
                "derived from there instead"
            )
        cell_vars = [name for name in cell_vars if name not in seen]
        for name in cell_vars:
            seen[name] = level
        native_vars[level] = cell_vars
        for name in others:
            other_arrays.setdefault(name, native[level][name])

    fill_values: dict[str, Any] = {}
    templates: dict[str, zarr.Array] = {}
    for level, names in native_vars.items():
        for name in names:
            templates[name] = native[level][name]
            fill_values[name] = native[level][name].fill_value
    coordinate_template = native[top][coordinate]
    if chunk_size is None:
        chunk_size = coordinate_template.chunks[0]

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
        names = [n for lv, ns in native_vars.items() if lv >= level for n in ns]
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
            attrs[name] = {
                **dict(templates[name].attrs),
                "native_refinement_level": seen[name],
            }
        writers[level] = _LevelWriter(
            level_group,
            coordinate,
            {coordinate: coordinate_template, **{n: templates[n] for n in names}},
            attrs,
            chunk_size,
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
                    current = (
                        new if current is None else current.merge(new, fill_values)
                    )
            if current is not None:
                writers[level].append(current)

    sizes = {level: writer.close() for level, writer in writers.items()}
    for level, size in sizes.items():
        log.info(f"level {level}: {size} cells")

    tag = f"{method}-l{top}-l{min_level}"
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
                str(level): names for level, names in native_vars.items()
            },
        }
    )
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
      towards coarser levels, each parent cell being the mean of its (up to 4)
      children (fill values and non-finite values are ignored);
    - a variable never appears at levels finer than its native level (no
      upsampling);
    - the cells of a level are the union of the cells of its variables; a
      variable has its fill value where it has no data;
    - a variable stored at several native levels (e.g., Sentinel-2 L2A, which
      also ships the 10 m bands resampled at 20 m and 60 m) is taken from its
      finest level only; its copies at coarser levels are not used.

    Every output array records its ``native_refinement_level``. The output
    group carries a zarr-conventions ``multiscales`` layout (assets relative
    to the group), the method tag (``multires_method``, e.g.
    ``"mean-l20-l12"``) and the native variables of each level
    (``native_refinement_levels``). In the root ``stac_discovery`` attribute,
    the method tag is appended to the item ``id`` (so that several pyramids of
    one product can share a STAC collection) and the resampling method is
    recorded as the ``multires:resampling_method`` property.

    Only the processed groups (and their parents) are written to the output
    dataset. Zarr metadata is consolidated in the root group and in every
    processed group, so that the latter can be opened on its own (e.g., by
    Gridlook).

    The input is read sequentially, one block of cells sharing an ancestor at
    about the chunk level of the finest input level at a time, so the memory
    use does not depend on the size of the dataset.

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
        Group(s) to process. By default, every group that has HEALPix level
        child groups.
    method : {"mean"}
        Resampling method. Only ``"mean"`` is supported, which is not
        suited to categorical variables (masks, classifications).
    chunk_size : int, optional
        Number of cells per output chunk. Defaults to the chunk size of the
        cell ids of the finest native level.
    input_storage_options, output_storage_options : dict, optional
        Storage options passed to Zarr (useful for, e.g., S3 object stores).

    Returns
    -------
    str
        The method tag, e.g. ``"mean-l20-l12"``.

    """
    if method != "mean":
        raise ValueError(f"unsupported resampling method {method!r}")

    in_root = zarr.open_group(
        input_path, mode="r", storage_options=input_storage_options
    )
    if groups is None:
        group_paths = _find_multires_groups(in_root)
    elif isinstance(groups, str):
        group_paths = [groups]
    else:
        group_paths = list(groups)
    group_paths = [g.strip("/") for g in group_paths]
    if not group_paths:
        raise ValueError(f"no group with HEALPix level child groups in {input_path!r}")

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
        )
        tags.add(result["tag"])

    if len(tags) != 1:
        raise ValueError(
            f"the processed groups have different finest native levels ({sorted(tags)}); "
            "process them into separate datasets"
        )
    tag = tags.pop()

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
