"""Root attributes of the input datasets carried over to the output
(``metadata.root_attributes``)."""

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
from healpix_convert.core.metadata import write_carried_root_attributes
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

DESCRIPTION = {"quality": "good", "bands": {"01": {"gain": 3.5, "values": [1, 2, 3]}}}
HISTORY = {"step-1": [{"processor": "p", "version": "1"}]}


def _write_input(path, *, item_id, lon_range, attrs):
    lon = np.linspace(*lon_range, COLUMNS)
    lat = np.linspace(-40.0, 40.0, ROWS)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    variables = {"a": (("rows", "columns"), np.cos(np.deg2rad(xx)) + yy / 100.0)}
    ds = xr.Dataset(
        variables,
        coords={
            "latitude": (("rows", "columns"), yy, {"standard_name": "latitude"}),
            "longitude": (("rows", "columns"), xx, {"standard_name": "longitude"}),
        },
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
        attrs={"product_description": DESCRIPTION},
    )
    out = str(tmp_path / "out.zarr")
    create_healpix_dataset([src], _settings(), out)
    attrs = zarr.open_group(out, mode="r").attrs
    assert "product_description" not in attrs and "stac_discovery" in attrs


def test_one_input_keyed_by_id(tmp_path):
    src = _write_input(
        tmp_path / "in.zarr",
        item_id="product-1",
        lon_range=(-60, 60),
        attrs={"product_description": DESCRIPTION, "history": HISTORY, "unlisted": 1},
    )
    out = str(tmp_path / "out.zarr")
    settings = _settings(
        metadata={"root_attributes": ["product_description", "history", "missing"]}
    )
    create_healpix_dataset([src], settings, out)
    attrs = zarr.open_group(out, mode="r").attrs
    assert attrs["product_description"] == {"product-1": DESCRIPTION}
    assert attrs["history"] == {"product-1": HISTORY}
    assert "missing" not in attrs and "unlisted" not in attrs
    assert attrs["stac_discovery"]["id"]


def test_several_inputs_keyed_per_input(tmp_path):
    a = _write_input(
        tmp_path / "a.zarr", item_id="input-a", lon_range=(-60, 0),
        attrs={"product_description": DESCRIPTION},
    )
    b = _write_input(
        tmp_path / "b.zarr", item_id="input-b", lon_range=(0, 60),
        attrs={"product_description": DESCRIPTION | {"quality": "degraded"}},
    )
    c = _write_input(tmp_path / "c.zarr", item_id="input-c", lon_range=(60, 120), attrs={})
    out = str(tmp_path / "out.zarr")
    create_healpix_dataset(
        [a, b, c], _settings(metadata={"root_attributes": ["product_description"]}), out
    )
    carried = zarr.open_group(out, mode="r").attrs["product_description"]
    # input-c has no such attribute: left out, not an empty entry
    assert carried == {"input-a": DESCRIPTION, "input-b": DESCRIPTION | {"quality": "degraded"}}


def test_multi_stage_conversion_keeps_them(tmp_path):
    src = _write_input(
        tmp_path / "in.zarr", item_id="staged", lon_range=(-60, 60),
        attrs={"product_description": DESCRIPTION},
    )
    out = str(tmp_path / "out.zarr")
    raw = _settings(metadata={"root_attributes": ["product_description"]})
    cache = create_staging_cache([src], raw)
    prepare_healpix_dataset(cache, raw, out)
    reloaded = ConvertStagingCache.from_json(cache.to_json())
    convert_group_to_healpix("measurements", cache=reloaded, settings=raw, output_path=out)
    finalize_healpix_dataset(reloaded, raw, out)
    assert zarr.open_group(out, mode="r").attrs["product_description"] == {"staged": DESCRIPTION}


@pytest.mark.parametrize("names", [["stac_discovery"], ["Conventions"], ["zarr_conventions"], ["a", "a"], [""]])
def test_invalid_root_attributes(names):
    with pytest.raises(ValidationError, match="root_attributes"):
        MetadataSettings.model_validate({"root_attributes": names})


def test_json_schema_documents_the_setting():
    schema = ConvertSettings.model_json_schema()
    assert "root_attributes" in schema["$defs"]["MetadataSettings"]["properties"]


def _tree(attrs: dict, item_id=...) -> xr.DataTree:
    tree = xr.DataTree()
    tree.attrs.update(attrs)
    if item_id is not ...:
        tree.attrs["stac_discovery"] = {"id": item_id}
    return tree


def _carry(trees: dict, names=("p",)) -> dict:
    group = zarr.create_group(store=None)
    metadata = MetadataSettings.model_validate({"root_attributes": list(names)})
    write_carried_root_attributes(group, trees, metadata)
    return dict(group.attrs)


def test_key_fallbacks_and_duplicates():
    # no stac_discovery, or no id: keyed by the input path
    attrs = _carry({"a.zarr": _tree({"p": 1}), "b.zarr": _tree({"p": 2}, item_id=None)})
    assert attrs == {"p": {"a.zarr": 1, "b.zarr": 2}}
    # two inputs with the same id: neither is overwritten
    attrs = _carry(
        {"a.zarr": _tree({"p": 1}, "X"), "b.zarr": _tree({"p": 2}, "X")}
    )
    assert attrs == {"p": {"X": 1, "b.zarr": 2}}
    # nothing to carry
    assert _carry({"a.zarr": _tree({"q": 1}, "X")}) == {}


def test_values_are_plain_json():
    attrs = _carry(
        {"a.zarr": _tree({"p": {"v": np.float64(1.5), "w": np.arange(3), 2: "k"}}, "X")}
    )
    assert attrs == {"p": {"X": {"v": 1.5, "w": [0, 1, 2], "2": "k"}}}
