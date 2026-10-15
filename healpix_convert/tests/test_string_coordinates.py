"""Non-spatial string coordinates are carried over to the output (#33)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
import zarr

from healpix_convert import create_healpix_dataset
from healpix_convert.healpix_converters import _unchanged_array_data
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
    "properties": {"datetime": "2026-08-18T00:00:00Z"},
    "links": [],
    "assets": {},
}


def test_object_strings_are_written_as_variable_length_strings():
    kwargs = _unchanged_array_data(np.array(["b02", "b03"], dtype=object))

    assert kwargs["dtype"] is str
    assert kwargs["data"].tolist() == ["b02", "b03"]
    assert kwargs["fill_value"] == ""

    array = zarr.create(store=zarr.storage.MemoryStore(), shape=(2,), **kwargs)
    array[:] = kwargs["data"]
    assert array[:].tolist() == ["b02", "b03"]


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(np.arange(3.0), id="float"),
        pytest.param(
            np.array(["b02", None], dtype=object), id="object-not-only-strings"
        ),
    ],
)
def test_other_arrays_are_written_unchanged(values):
    actual = _unchanged_array_data(values)

    assert actual == {"dtype": values.dtype, "data": values}


@pytest.fixture
def input_path(tmp_path):
    """A lat/lon grid with a band dimension labelled by strings, as Zarr v2.

    Depending on the numpy and xarray versions, the strings of a Zarr v2 store
    are read back as a numpy object array (how the string coordinates of an
    EOPF product reached the converter on the GRID4EARTH hub) or as StringDType.
    """
    lon = np.linspace(-1.0, 1.0, 8)
    lat = np.linspace(-1.0, 1.0, 8)

    ds = xr.Dataset(
        {
            "t2m": (("lat", "lon"), np.arange(64, dtype="float32").reshape(8, 8)),
            "gain": ("band", np.array([1.0, 2.0, 3.0])),
        },
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
            "band": ("band", np.array(["b02", "b03", "b04"], dtype=object)),
        },
    )

    tree = xr.DataTree.from_dict({"/measurements": ds})
    tree.attrs["stac_discovery"] = STAC_ITEM

    path = tmp_path / "input.zarr"
    tree.to_zarr(path, mode="w", consolidated=True, zarr_format=2)

    return str(path)


@pytest.fixture
def settings():
    return ConvertSettings.model_validate(
        {
            "group_settings": {
                "measurements": {
                    "healpix": {"refinement_level": 4, "indexing_scheme": "nested"},
                    "chunk": {"method": "no_chunk"},
                    "resampler": {"name": "nearest"},
                }
            }
        }
    )


def test_string_coordinates_survive_a_conversion(tmp_path, input_path, settings):
    output_path = str(tmp_path / "output.zarr")
    create_healpix_dataset([input_path], settings, output_path)

    band = zarr.open_group(output_path, mode="r")["measurements/band"]
    assert band[:].tolist() == ["b02", "b03", "b04"]
