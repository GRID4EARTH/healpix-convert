"""The output arrays of a chunk are written concurrently."""

import threading

import numpy as np
import pytest
import zarr

from healpix_convert.healpix_converters import _write_chunk_arrays


class BarrierArray:
    """Stands for an output array: each write waits until all the writes have started."""

    def __init__(self, barrier):
        self.barrier = barrier
        self.written = None

    def __setitem__(self, key, value):
        self.barrier.wait(timeout=5)
        self.written = (key, value)


class FailingArray:
    def __setitem__(self, key, value):
        raise OSError("disk full")


class TestWriteChunkArrays:
    @pytest.mark.parametrize("n_arrays", [1, 2, 4])
    def test_values(self, n_arrays):
        root = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
        arrays = [
            root.create_array(f"v{i}", shape=(2, 8), chunks=(2, 4), dtype="float64")
            for i in range(n_arrays)
        ]
        values = [np.full((2, 4), i + 0.5) for i in range(n_arrays)]

        _write_chunk_arrays(
            [(a, (..., slice(4, 8)), v) for a, v in zip(arrays, values, strict=True)]
        )

        for a, v in zip(arrays, values, strict=True):
            np.testing.assert_array_equal(a[:, 4:], v)
            np.testing.assert_array_equal(a[:, :4], 0.0)

    def test_concurrent(self):
        # every write blocks until all of them have started: this only completes
        # when the writes run at the same time (a sequential loop times out)
        barrier = threading.Barrier(4)
        arrays = [BarrierArray(barrier) for _ in range(4)]

        _write_chunk_arrays([(a, i, np.zeros(1)) for i, a in enumerate(arrays)])

        assert [a.written[0] for a in arrays] == [0, 1, 2, 3]

    def test_error_is_raised(self):
        with pytest.raises(OSError, match="disk full"):
            _write_chunk_arrays(
                [(FailingArray(), 0, np.zeros(1)), (FailingArray(), 1, np.zeros(1))]
            )
