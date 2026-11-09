"""Chunk and shard layout of the converted levels (``chunk_level_offset``)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
import zarr

from healpix_convert import create_healpix_dataset
from healpix_convert.core.layout import chunk_shard_levels
from healpix_convert.settings.common import ConvertSettings
from healpix_convert.tests.test_float_fill import STAC_ITEM


def test_chunk_shard_levels():
    # parent level 7, native level 20, derived levels 19 to 14: chunks of 4 ** 9
    # cells, shards of one level-8 cell
    layout = {
        level: chunk_shard_levels(level, 9, 8, 7) for level in range(20, 13, -1)
    }
    assert layout[20] == (11, 8)
    assert layout[19] == (10, 8)
    assert layout[18] == (9, 8)
    # the shard holds a single chunk: no sharding
    assert layout[17] == (8, None)
    # the chunk is clamped to the parent cell: a single chunk, no sharding
    assert layout[16] == (7, None)
    assert layout[14] == (7, None)
    # no shard setting
    assert chunk_shard_levels(20, 9, None, 7) == (11, None)


@pytest.mark.parametrize(
    ["offset", "expected_chunks", "expected_shards"],
    [
        pytest.param(2, (16,), (4**3,), id="sharded"),
        # the chunk cells are not finer than the chunk setting's: plain chunks
        pytest.param(3, (4**3,), None, id="single-chunk"),
        pytest.param(5, (4**3,), None, id="clamped"),
        pytest.param(None, (4**3,), None, id="default"),
    ],
)
def test_native_chunk_level_offset(tmp_path, offset, expected_chunks, expected_shards):
    lon = np.linspace(-1.0, 1.0, 16)
    lat = np.linspace(-1.0, 1.0, 16)
    ds = xr.Dataset(
        {"v": (("lat", "lon"), np.arange(1, 257).reshape(16, 16).astype("float32"))},
        coords={
            "lon": ("lon", lon, {"standard_name": "longitude", "units": "degrees_east"}),
            "lat": ("lat", lat, {"standard_name": "latitude", "units": "degrees_north"}),
        },
    )
    tree = xr.DataTree.from_dict({"/measurements": ds})
    tree.attrs["stac_discovery"] = STAC_ITEM
    input_path = str(tmp_path / "input.zarr")
    tree.to_zarr(input_path, mode="w", consolidated=True)

    healpix = {"indexing_scheme": "nested", "ellipsoid": {"name": "wgs84"}}
    group = {
        "healpix": {"refinement_level": 5, **healpix},
        # shards (conversion chunks) of one level-2 cell: 4 ** 3 cells
        "chunk": {
            "method": "healpix_cell_dense",
            "healpix": {"refinement_level": 2, **healpix},
        },
        "resampler": {"name": "nearest"},
    }
    if offset is not None:
        group["chunk_level_offset"] = offset
    settings = ConvertSettings.model_validate({"group_settings": {"measurements": group}})
    output_path = str(tmp_path / "output.zarr")
    create_healpix_dataset([input_path], settings, output_path)

    v = zarr.open_group(output_path, mode="r")["measurements/v"]
    assert v.chunks == expected_chunks
    assert v.shards == expected_shards
    # data was written
    assert np.isfinite(v[:]).any()


def test_chunk_level_offset_needs_dense_chunks():
    with pytest.raises(ValueError, match="healpix_cell_dense"):
        ConvertSettings.model_validate(
            {
                "group_settings": {
                    "measurements": {
                        "healpix": {
                            "refinement_level": 5,
                            "indexing_scheme": "nested",
                            "ellipsoid": {"name": "wgs84"},
                        },
                        "chunk": {"method": "no_chunk"},
                        "chunk_level_offset": 2,
                        "resampler": {"name": "nearest"},
                    }
                }
            }
        )
