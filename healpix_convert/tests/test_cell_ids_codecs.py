"""The cell ids coordinate is delta-coded before zstd (#32)."""

from __future__ import annotations

import numpy as np
import zarr

from healpix_convert.healpix_converters import CELL_IDS_CODECS


def _write(values, codecs):
    array = zarr.create(
        store=zarr.storage.MemoryStore(),
        shape=values.shape,
        chunks=values.shape,
        dtype=np.uint64,
        codecs=codecs,
    )
    array[:] = values
    return array


def test_cell_ids_round_trip_and_shrink():
    # one dense level-11 chunk at level 20: a contiguous range of 4**9 ids
    n = 4**9
    values = np.uint64(3_723_275 * n) + np.arange(n, dtype=np.uint64)

    delta = _write(values, CELL_IDS_CODECS)
    plain = _write(values, None)

    np.testing.assert_array_equal(delta[:], values)
    assert delta.nbytes_stored() * 100 < plain.nbytes_stored()
