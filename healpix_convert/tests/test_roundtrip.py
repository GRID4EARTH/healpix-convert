"""End-to-end conversion of a tiny synthetic dataset.

Covers what the unit tests cannot: that the metadata the writers produce
survives an actual conversion and is readable from the output.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from healpix_convert import create_healpix_dataset
from healpix_convert.core.stac import SETTINGS_EXPRESSION_FORMAT
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


@pytest.fixture
def input_path(tmp_path):
    """A tiny lat/lon dataset, written as zarr with STAC discovery metadata."""
    lon = np.linspace(-1.0, 1.0, 8)
    lat = np.linspace(-1.0, 1.0, 8)

    ds = xr.Dataset(
        {"t2m": (("lat", "lon"), np.arange(64, dtype="float32").reshape(8, 8))},
        coords={
            "lon": ("lon", lon, {"standard_name": "longitude", "units": "degrees_east"}),
            "lat": ("lat", lat, {"standard_name": "latitude", "units": "degrees_north"}),
        },
    )

    tree = xr.DataTree.from_dict({"/measurements": ds})
    tree.attrs["stac_discovery"] = STAC_ITEM

    path = tmp_path / "input.zarr"
    tree.to_zarr(path, mode="w", consolidated=True)

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


@pytest.fixture
def converted(tmp_path, input_path, settings):
    output_path = str(tmp_path / "output.zarr")
    create_healpix_dataset([input_path], settings, output_path)

    return output_path


def test_cf_metadata_survives_conversion(converted) -> None:
    ds = xr.open_dataset(
        converted, group="measurements", engine="zarr", decode_coords="all"
    )

    # the grid mapping is decoded, not merely present
    assert "crs" in ds.coords
    assert ds["crs"].attrs["grid_mapping_name"] == "healpix"
    assert ds["crs"].attrs["refinement_level"] == 4

    # and the resampling method is recorded on the data variable
    assert ds["t2m"].attrs["cell_methods"] == "area: point"


def test_conventions_declared_in_root_group_only(converted) -> None:
    tree = xr.open_datatree(converted, engine="zarr")

    assert tree.attrs["Conventions"] == "CF-1.13"
    # CF 1.13 section 2.7.2: root group only, never duplicated in children
    assert "Conventions" not in tree["measurements"].attrs


def test_dggs_convention_declared_on_the_healpix_group(converted) -> None:
    tree = xr.open_datatree(converted, engine="zarr")
    attrs = tree["measurements"].attrs

    assert [c["name"] for c in attrs["zarr_conventions"]] == ["dggs"]
    assert attrs["dggs"]["name"] == "healpix"
    assert attrs["dggs"]["refinement_level"] == 4


def test_provenance_records_the_settings(converted, settings) -> None:
    tree = xr.open_datatree(converted, engine="zarr")
    properties = tree.attrs["stac_discovery"]["properties"]

    assert properties["processing:expression"]["format"] == SETTINGS_EXPRESSION_FORMAT
    assert properties["processing:expression"]["expression"] == settings.to_dict()
    assert "healpix-convert" in properties["processing:software"]
    assert "synthetic-input" in properties["processing:lineage"] or "input.zarr" in properties["processing:lineage"]

    # the inputs stay findable: STAC records what this was derived from
    derived = [l for l in tree.attrs["stac_discovery"]["links"] if l["rel"] == "derived_from"]
    assert len(derived) == 1
    assert derived[0]["href"].endswith("input.zarr")


def test_the_data_is_actually_resampled(converted) -> None:
    ds = xr.open_dataset(converted, group="measurements", engine="zarr")

    # the output covers the input's extent, not the whole sphere: a 2x2 degree
    # input touches only a handful of cells at level 4 (~3.7 degrees across)
    assert 0 < ds.sizes["cells"] < 12 * 4**4

    # nearest-neighbour resampling reuses input values, so nothing is invented
    resampled = ds["t2m"].values
    assert np.isfinite(resampled).any()
    assert set(resampled[np.isfinite(resampled)]) <= set(np.arange(64, dtype="float32"))
