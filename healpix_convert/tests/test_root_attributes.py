"""Root attributes of the input datasets carried over to the output
(``metadata.root_attributes``), and integer class maps stored with an identity
``scale_factor`` / ``add_offset`` that stay integers (``nearest_variables``)."""

from __future__ import annotations

import numpy as np
import pytest
import shapely
import xarray as xr
import zarr
from pydantic import ValidationError

from healpix_convert import (
    ConvertSettings,
    ConvertStagingCache,
    convert_group_to_healpix,
    create_healpix_dataset,
    create_staging_cache,
    finalize_healpix_dataset,
    prepare_healpix_dataset,
)
from healpix_convert.settings.common import MetadataSettings
from healpix_convert.tests.test_batch_resample import STAC_ITEM

HEALPIX = {"indexing_scheme": "nested", "ellipsoid": {"name": "wgs84"}}
CHUNK = {
    "method": "healpix_cell_dense",
    "healpix": {"refinement_level": 1, **HEALPIX},
    "chunk_buffer_width": 0.0,
}
ROWS, COLUMNS = 24, 48
FULL_EXTENT = shapely.box(-180.0, -90.0, 180.0, 90.0)

OTHER = {"quality": "good", "bands": {"01": {"gain": 3.5, "values": [1, 2, 3]}}}
HISTORY = {"Level-1C": [{"processor": "p", "version": "1"}]}


def _write_input(path, *, item_id, lon_range, attrs, class_map=False):
    lon = np.linspace(*lon_range, COLUMNS)
    lat = np.linspace(-40.0, 40.0, ROWS)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    variables = {"a": (("rows", "columns"), np.cos(np.deg2rad(xx)) + yy / 100.0)}
    if class_map:
        classes = ((np.arange(COLUMNS)[None, :] + np.arange(ROWS)[:, None]) % 11 + 1).astype(float)
        classes[0, :] = np.nan  # no data: the stored fill value 0
        variables["scl"] = (("rows", "columns"), classes)
        variables["scaled"] = (("rows", "columns"), classes * 0.5)
    ds = xr.Dataset(
        variables,
        coords={
            "latitude": (("rows", "columns"), yy, {"standard_name": "latitude"}),
            "longitude": (("rows", "columns"), xx, {"standard_name": "longitude"}),
        },
    )
    if class_map:
        ds["scl"].encoding.update(
            dtype="uint8", _FillValue=0, scale_factor=1, add_offset=0
        )
        # a real packing: the stored integers are not the physical values
        ds["scaled"].encoding.update(
            dtype="uint8", _FillValue=0, scale_factor=0.5, add_offset=0
        )
    tree = xr.DataTree.from_dict({"/measurements": ds})
    box = shapely.box(lon_range[0], -40.0, lon_range[1], 40.0)
    tree.attrs["stac_discovery"] = STAC_ITEM | {
        "id": item_id,
        "geometry": shapely.geometry.mapping(box),
        "bbox": list(box.bounds),
    }
    tree.attrs.update(attrs)
    tree.to_zarr(path, mode="w", consolidated=True)
    return str(path)


def _settings(**kwargs) -> dict:
    group = {
        "healpix": {"refinement_level": 4, **HEALPIX},
        "chunk": CHUNK,
        "resampler": {"name": "nearest"},
        "output_dtype": "float32",
    }
    group.update(kwargs.pop("group", {}))
    return {"group_settings": {"measurements": group}, **kwargs}


def test_default_carries_nothing(tmp_path):
    src = _write_input(
        tmp_path / "in.zarr",
        item_id="p1",
        lon_range=(-60, 60),
        attrs={"other_metadata": OTHER},
    )
    out = str(tmp_path / "out.zarr")
    create_healpix_dataset([src], _settings(), out)
    attrs = zarr.open_group(out, mode="r").attrs
    assert "other_metadata" not in attrs and "stac_discovery" in attrs


def test_one_input_keyed_by_id(tmp_path):
    src = _write_input(
        tmp_path / "in.zarr",
        item_id="product-1",
        lon_range=(-60, 60),
        attrs={"other_metadata": OTHER, "processing_history": HISTORY, "unlisted": 1},
    )
    out = str(tmp_path / "out.zarr")
    settings = _settings(
        metadata={"root_attributes": ["other_metadata", "processing_history", "missing"]}
    )
    create_healpix_dataset([src], settings, out)
    attrs = zarr.open_group(out, mode="r").attrs
    assert attrs["other_metadata"] == {"product-1": OTHER}
    assert attrs["processing_history"] == {"product-1": HISTORY}
    assert "missing" not in attrs and "unlisted" not in attrs
    assert attrs["stac_discovery"]["id"]


def test_several_inputs_keyed_per_input(tmp_path):
    a = _write_input(
        tmp_path / "a.zarr", item_id="tile-A", lon_range=(-60, 0),
        attrs={"other_metadata": OTHER},
    )
    b = _write_input(
        tmp_path / "b.zarr", item_id="tile-B", lon_range=(0, 60),
        attrs={"other_metadata": OTHER | {"quality": "degraded"}},
    )
    c = _write_input(tmp_path / "c.zarr", item_id="tile-C", lon_range=(60, 120), attrs={})
    out = str(tmp_path / "out.zarr")
    create_healpix_dataset(
        [a, b, c], _settings(metadata={"root_attributes": ["other_metadata"]}), out
    )
    carried = zarr.open_group(out, mode="r").attrs["other_metadata"]
    # tile-C has no such attribute: left out, not an empty entry
    assert carried == {"tile-A": OTHER, "tile-B": OTHER | {"quality": "degraded"}}


def test_multi_stage_conversion_keeps_them(tmp_path):
    src = _write_input(
        tmp_path / "in.zarr", item_id="staged", lon_range=(-60, 60),
        attrs={"other_metadata": OTHER},
    )
    out = str(tmp_path / "out.zarr")
    raw = _settings(metadata={"root_attributes": ["other_metadata"]})
    cache = create_staging_cache([src], raw)
    prepare_healpix_dataset(cache, raw, out)
    reloaded = ConvertStagingCache.from_json(cache.to_json())
    convert_group_to_healpix("measurements", cache=reloaded, settings=raw, output_path=out)
    finalize_healpix_dataset(reloaded, raw, out)
    assert zarr.open_group(out, mode="r").attrs["other_metadata"] == {"staged": OTHER}


@pytest.mark.parametrize("names", [["stac_discovery"], ["Conventions"], ["a", "a"]])
def test_invalid_root_attributes(names):
    with pytest.raises(ValidationError, match="root_attributes"):
        MetadataSettings.model_validate({"root_attributes": names})


def test_json_schema_documents_the_setting():
    schema = ConvertSettings.model_json_schema()
    assert "root_attributes" in schema["$defs"]["MetadataSettings"]["properties"]


def test_identity_packed_class_map_stays_uint8(tmp_path):
    src = _write_input(
        tmp_path / "in.zarr", item_id="cls", lon_range=(-60, 60), attrs={}, class_map=True
    )
    out = str(tmp_path / "out.zarr")
    create_healpix_dataset(
        [src], _settings(group={"nearest_variables": ["scl", "scaled"]}), out
    )
    level = zarr.open_group(out, mode="r")["measurements"]
    scl = level["scl"]
    assert scl.dtype == np.uint8 and scl.fill_value == 0
    values = scl[:]
    assert set(np.unique(values)) <= set(range(0, 12))
    assert (values > 0).any() and (values == 0).any()
    # a packing that changes the values: physical values, floats with NaN
    scaled = level["scaled"]
    assert scaled.dtype.kind == "f" and np.isnan(scaled.fill_value)
    assert np.nanmax(scaled[:]) <= 6.0


def test_without_nearest_variables_class_map_is_unchanged(tmp_path):
    src = _write_input(
        tmp_path / "in.zarr", item_id="cls", lon_range=(-60, 60), attrs={}, class_map=True
    )
    out = str(tmp_path / "out.zarr")
    create_healpix_dataset([src], _settings(), out)
    scl = zarr.open_group(out, mode="r")["measurements"]["scl"]
    assert scl.dtype == np.float32 and np.isnan(scl.fill_value)
