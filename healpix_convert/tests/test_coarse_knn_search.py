"""``coarse_level_delta`` in the resampler ``init_params`` reaches the neighbour
search of healpix-resample and leaves the converted values unchanged."""

from __future__ import annotations

import healpix_resample
import numpy as np
import pytest
import xarray as xr

from healpix_convert import create_healpix_dataset
from healpix_convert.settings.common import ConvertSettings

STAC_ITEM = {
    "type": "Feature",
    "stac_version": "1.1.0",
    "stac_extensions": [],
    "id": "synthetic-input",
    "bbox": [-1.0, -1.0, 1.0, 1.0],
    "geometry": {
        "type": "Polygon",
        "coordinates": [[[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]],
    },
    "properties": {"datetime": "2026-10-31T00:00:00Z"},
    "links": [],
    "assets": {},
}

RESAMPLERS = {
    "psf": {
        "name": "psf",
        "init_params": {"Npt": 9, "threshold": 0.1},
        "resample_params": {"lam": 0.1, "max_iter": 200, "tol": 1e-10},
    },
    "nearest": {"name": "nearest", "init_params": {}},
}


def _write_input(path):
    lon = np.linspace(-1.0, 1.0, 96)
    lat = np.linspace(-1.0, 1.0, 96)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    ds = xr.Dataset(
        {"a": (("lat", "lon"), (np.sin(3 * xx) * np.cos(2 * yy)).astype("float32"))},
        coords={
            "lon": ("lon", lon, {"standard_name": "longitude", "units": "degrees_east"}),
            "lat": ("lat", lat, {"standard_name": "latitude", "units": "degrees_north"}),
        },
    )
    tree = xr.DataTree.from_dict({"/measurements": ds})
    tree.attrs["stac_discovery"] = STAC_ITEM
    tree.to_zarr(path, mode="w", consolidated=True)
    return str(path)


def _settings(resampler, coarse_level_delta):
    resampler = {**resampler, "init_params": dict(resampler["init_params"])}
    if coarse_level_delta is not None:
        resampler["init_params"]["coarse_level_delta"] = coarse_level_delta
    return ConvertSettings.model_validate(
        {
            "group_settings": {
                "measurements": {
                    "healpix": {
                        "refinement_level": 9,
                        "indexing_scheme": "nested",
                        "ellipsoid": {"name": "wgs84"},
                    },
                    "chunk": {
                        "method": "healpix_cell_dense",
                        "healpix": {
                            "refinement_level": 5,
                            "indexing_scheme": "nested",
                            "ellipsoid": {"name": "wgs84"},
                        },
                    },
                    "resampler": resampler,
                }
            }
        }
    )


def _convert(tmp_path, resampler, coarse_level_delta, label):
    input_path = _write_input(tmp_path / f"{label}-input.zarr")
    output_path = str(tmp_path / f"{label}-output.zarr")
    create_healpix_dataset([input_path], _settings(resampler, coarse_level_delta), output_path)
    ds = xr.open_dataset(output_path, engine="zarr", group="measurements")
    return ds["cell_ids"].values, ds["a"].values


@pytest.mark.parametrize("name", list(RESAMPLERS))
def test_same_result_as_the_fine_search(tmp_path, monkeypatch, name):
    seen = []
    original = healpix_resample.knn._healpix_weighted_nearest

    def recording(*args, **kwargs):
        seen.append(kwargs.get("coarse_level_delta"))
        return original(*args, **kwargs)

    monkeypatch.setattr(healpix_resample.knn, "_healpix_weighted_nearest", recording)

    cells_fine, values_fine = _convert(tmp_path, RESAMPLERS[name], None, "fine")
    assert seen and set(seen) == {None}
    seen.clear()
    cells_coarse, values_coarse = _convert(tmp_path, RESAMPLERS[name], 2, "coarse")
    assert seen and set(seen) == {2}

    np.testing.assert_array_equal(cells_coarse, cells_fine)
    np.testing.assert_array_equal(values_coarse, values_fine)
