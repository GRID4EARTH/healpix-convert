"""Chunk and shard layout of the HEALPix arrays of a group.

One rule gives the layout of an array at any refinement level, native or
derived: the chunk is *relative* to the level of the array, the shard is
*absolute*.
"""

from __future__ import annotations


def chunk_shard_levels(
    level: int,
    chunk_level_offset: int,
    shard_level: int | None,
    coarsest_level: int = 0,
) -> tuple[int, int | None]:
    """Refinement levels of the cells that give the chunks and the shards of an
    array at the refinement level ``level``.

    A chunk holds the children of one cell of the level ``level -
    chunk_level_offset`` (``4 ** chunk_level_offset`` cells), and a shard the
    children of one cell of the level ``shard_level`` (if given). Both stop at
    ``coarsest_level``: the array is then written in a single chunk per cell of
    that level, which is the whole array for ``coarsest_level`` 0 or the
    coarsest level that the converted data covers entirely.

    A shard that would hold at most one chunk (``shard_level`` not coarser than
    the chunk level) is not written: the second value is then None and the
    array is written in plain chunks.

    """
    chunk = max(level - chunk_level_offset, coarsest_level)

    if shard_level is None:
        return chunk, None

    shard = max(shard_level, coarsest_level)

    return chunk, (shard if shard < chunk else None)
