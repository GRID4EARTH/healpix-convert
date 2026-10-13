"""Variables that share resample parameters are resampled in one call per chunk."""

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
    "properties": {"datetime": "2026-10-13T00:00:00Z"},
    "links": [],
    "assets": {},
}
VARIABLES = ("a", "b", "c")


def _write_input(path, names):
    lon = np.linspace(-1.0, 1.0, 48)
    lat = np.linspace(-1.0, 1.0, 48)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    fields = {
        "a": (("lat", "lon"), (np.sin(3 * xx) * np.cos(2 * yy)).astype("float32")),
        "b": (("lat", "lon"), (xx * yy + 0.5).astype("float32")),
        # a variable with a non-spatial dimension
        "c": (
            ("time", "lat", "lon"),
            np.stack([np.cos(xx), np.sin(yy)]).astype("float32"),
        ),
    }
    ds = xr.Dataset(
        {name: fields[name] for name in names},
        coords={
            "lon": (
                "lon",
                lon,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "lat": (
                "lat",
                lat,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
        },
    )
    tree = xr.DataTree.from_dict({"/measurements": ds})
    tree.attrs["stac_discovery"] = STAC_ITEM
    tree.to_zarr(path, mode="w", consolidated=True)
    return str(path)


def _settings(lam):
    return ConvertSettings.model_validate(
        {
            "group_settings": {
                "measurements": {
                    "healpix": {
                        "refinement_level": 7,
                        "indexing_scheme": "nested",
                        "ellipsoid": {"name": "wgs84"},
                    },
                    "chunk": {
                        "method": "healpix_cell_dense",
                        "healpix": {
                            "refinement_level": 4,
                            "indexing_scheme": "nested",
                            "ellipsoid": {"name": "wgs84"},
                        },
                    },
                    "resampler": {
                        "name": "psf",
                        "init_params": {"Npt": 9, "threshold": 0.1},
                        "resample_params": {"lam": lam, "max_iter": 200, "tol": 1e-10},
                    },
                }
            }
        }
    )


def _convert(tmp_path, names, lam, label):
    input_path = _write_input(tmp_path / f"{label}-input.zarr", names)
    output_path = str(tmp_path / f"{label}-output.zarr")
    create_healpix_dataset([input_path], _settings(lam), output_path)
    ds = xr.open_dataset(output_path, engine="zarr", group="measurements")
    return {name: ds[name].values for name in names}


class TestBatchedResample:
    @pytest.mark.parametrize(
        ["lam", "expected_calls_per_chunk"],
        [
            pytest.param(0.1, 1, id="shared-params"),
            pytest.param({"a": 0.1, "b": 0.2, "c": 0.1}, 2, id="two-param-groups"),
        ],
    )
    def test_one_call_per_parameter_group(
        self, tmp_path, monkeypatch, lam, expected_calls_per_chunk
    ):
        calls = []
        original = healpix_resample.PSFResampler.resample

        def counting(self, val, **kwargs):
            calls.append(np.shape(val))
            return original(self, val, **kwargs)

        monkeypatch.setattr(healpix_resample.PSFResampler, "resample", counting)
        constructions = []
        original_init = healpix_resample.PSFResampler.__init__

        def counting_init(self, *args, **kwargs):
            constructions.append(1)
            original_init(self, *args, **kwargs)

        monkeypatch.setattr(healpix_resample.PSFResampler, "__init__", counting_init)

        _convert(tmp_path, VARIABLES, lam, "batched")

        # one resampler per chunk with input points, and as many calls on it as parameter groups
        assert len(calls) == expected_calls_per_chunk * len(constructions)
        # "a", "b" and the two rows of "c" in the same batch when the parameters are shared
        if expected_calls_per_chunk == 1:
            assert all(shape[0] == 4 for shape in calls)

    @pytest.mark.parametrize("name", VARIABLES)
    def test_same_result_as_one_variable_at_a_time(self, tmp_path, name):
        batched = _convert(tmp_path, VARIABLES, 0.1, "all")[name]
        alone = _convert(tmp_path, (name,), 0.1, name)[name]

        np.testing.assert_array_equal(np.isnan(batched), np.isnan(alone))
        valid = ~np.isnan(alone)
        # the conjugate gradient stops on a criterion shared by the rows of a batch, so
        # the rows of a batch can take a few more iterations than alone: equal to the
        # solver tolerance (tol=1e-10 here), not bit for bit
        np.testing.assert_allclose(batched[valid], alone[valid], rtol=0, atol=1e-5)
