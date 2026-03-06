import itertools

import numpy as np
from tlz.dicttoolz import merge
from tlz.itertoolz import sliding_window


def expand_chunks(chunksizes, sizes):
    def _expand(chunksize, total):
        n_chunks, last_chunksize = divmod(total, chunksize)
        last_chunk = (last_chunksize,) if last_chunksize > 0 else ()
        return (chunksize,) * n_chunks + last_chunk

    return {
        name: _expand(chunksize, sizes[name]) for name, chunksize in chunksizes.items()
    }


def chunk_regions(chunksizes):
    chunk_offsets = (
        (name, np.cumulative_sum(chunks, include_initial=True))
        for name, chunks in chunksizes.items()
    )
    chunk_ranges = list(
        [slice(int(left), int(right)) for left, right in sliding_window(2, chunks)]
        for _, chunks in chunk_offsets
    )

    for regions in itertools.product(*chunk_ranges):
        yield dict(zip(chunksizes.keys(), regions))


def merge_regions(*rgens):
    for indices in itertools.product(*rgens):
        yield merge(*indices)


def subset_affine(affine, region):
    start_x = region["x"].start
    start_y = region["y"].start

    stop_x = region["x"].stop
    stop_y = region["y"].stop

    shape = (stop_x - start_x, stop_y - start_y)

    return affine * affine.translation(start_x, start_y), shape
