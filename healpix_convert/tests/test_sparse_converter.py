"""The sparse chunk converter writes the input values with their dtype."""

from __future__ import annotations

import numpy as np
import xarray as xr

from healpix_convert import create_healpix_dataset
from healpix_convert.settings.common import ConvertSettings
from healpix_convert.tests.test_batch_resample import VARIABLES, _write_input

HEALPIX = {"indexing_scheme": "nested", "ellipsoid": {"name": "wgs84"}}


def test_sparse_cell_point_conversion(tmp_path):
    input_path = _write_input(tmp_path / "input.zarr", VARIABLES)
    settings = ConvertSettings.model_validate(
        {
            "group_settings": {
                "measurements": {
                    "healpix": {"refinement_level": 29, **HEALPIX},
                    "chunk": {
                        "method": "healpix_cell_sparse",
                        "healpix": {"refinement_level": 4, **HEALPIX},
                        # room for all the input points in one chunk
                        "chunk_size": 48 * 48,
                    },
                    "resampler": {"name": "cell-point"},
                }
            }
        }
    )
    output_path = str(tmp_path / "output.zarr")

    create_healpix_dataset([input_path], settings, output_path)

    expected = xr.open_datatree(input_path, engine="zarr")["measurements"].to_dataset()
    actual = xr.open_dataset(output_path, engine="zarr", group="measurements")
    # the slots of a chunk beyond its cells keep the fill values
    written_cells = actual["cell_ids"].values != np.iinfo("uint64").max
    actual = actual.isel(cells=written_cells)
    for name in VARIABLES:
        assert actual[name].dtype == expected[name].dtype
        written = actual[name].values
        values = expected[name].values
        # every input value is written, and nothing else
        for row_written, row_values in zip(
            written.reshape(-1, written.shape[-1]), values.reshape(-1, 48 * 48)
        ):
            np.testing.assert_array_equal(
                np.unique(row_written[np.isfinite(row_written)]), np.unique(row_values)
            )
