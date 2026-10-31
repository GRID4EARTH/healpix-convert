"""Control over the STAC metadata written in the output root group."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from healpix_convert import create_healpix_dataset
from healpix_convert.core.stac import STAC_PROCESSING_SCHEMA_URL
from healpix_convert.settings.common import ConvertSettings, MetadataSettings
from healpix_convert.settings.stac import STACSettings

PROJ_SCHEMA_URL = "https://stac-extensions.github.io/projection/v1.1.0/schema.json"

STAC_ITEM = {
    "type": "Feature",
    "stac_version": "1.1.0",
    "stac_extensions": [PROJ_SCHEMA_URL],
    "id": "synthetic-input",
    "bbox": [-1.0, -1.0, 1.0, 1.0],
    "geometry": {
        "type": "Polygon",
        "coordinates": [[[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]],
    },
    "properties": {
        "datetime": "2026-08-18T00:00:00Z",
        "proj:epsg": 4326,
        "internal:job_id": "run-1234",
    },
    "links": [
        {"rel": "self", "href": "s3://internal/in.json"},
        {"rel": "license", "href": "https://example.org/license"},
    ],
    "assets": {
        "data": {"href": "s3://internal/in.zarr", "type": "application/vnd+zarr"},
        "thumbnail": {"href": "s3://internal/in.png", "type": "image/png"},
    },
}


def test_stac_settings_default() -> None:
    settings = STACSettings()

    assert settings.include_conversion_settings
    assert settings.exclude_properties == []
    assert settings.override_properties == {}
    assert settings.exclude_extensions == []
    assert settings.override_extensions == []
    assert settings.exclude_links == []
    assert settings.override_links == []
    assert settings.override_assets == {}

    # the propagated metadata passes through untouched
    assert settings.filter_properties({"datetime": "x"}) == {"datetime": "x"}
    assert settings.filter_extensions([PROJ_SCHEMA_URL]) == [PROJ_SCHEMA_URL]
    assert settings.filter_links(STAC_ITEM["links"]) == STAC_ITEM["links"]

    # except the assets, which describe the input files
    assert settings.exclude_assets == [".*"]
    assert settings.filter_assets(STAC_ITEM["assets"]) == {}


def test_metadata_settings_default_stac() -> None:
    assert MetadataSettings().stac == STACSettings()


def test_exclude_properties_name_and_pattern() -> None:
    settings = STACSettings(exclude_properties=["internal:.*", "datetime"])

    assert settings.filter_properties(STAC_ITEM["properties"]) == {"proj:epsg": 4326}

    # the pattern is matched against the whole name
    assert settings.filter_properties({"my_internal:job_id": 1}) == {
        "my_internal:job_id": 1
    }


def test_override_properties_beats_exclude() -> None:
    settings = STACSettings(
        exclude_properties=["license"], override_properties={"license": "CC-BY-4.0"}
    )

    assert settings.filter_properties({"license": "proprietary"}) == {
        "license": "CC-BY-4.0"
    }


def test_override_properties_adds_missing_properties() -> None:
    settings = STACSettings(override_properties={"license": "CC-BY-4.0"})

    assert settings.filter_properties({"datetime": "x"}) == {
        "datetime": "x",
        "license": "CC-BY-4.0",
    }


def test_exclude_extensions_name_and_pattern() -> None:
    settings = STACSettings(
        exclude_extensions=["https://stac-extensions.github.io/projection/.*"]
    )

    assert settings.filter_extensions(
        [PROJ_SCHEMA_URL, STAC_PROCESSING_SCHEMA_URL]
    ) == [STAC_PROCESSING_SCHEMA_URL]


def test_override_extensions_beats_exclude() -> None:
    settings = STACSettings(
        exclude_extensions=[".*"], override_extensions=[PROJ_SCHEMA_URL]
    )

    assert settings.filter_extensions([PROJ_SCHEMA_URL]) == [PROJ_SCHEMA_URL]


def test_override_extensions_are_not_duplicated() -> None:
    settings = STACSettings(override_extensions=[PROJ_SCHEMA_URL])

    assert settings.filter_extensions([PROJ_SCHEMA_URL]) == [PROJ_SCHEMA_URL]


def test_exclude_links_by_rel() -> None:
    settings = STACSettings(exclude_links=["self|parent"])

    assert settings.filter_links(STAC_ITEM["links"]) == [
        {"rel": "license", "href": "https://example.org/license"}
    ]


def test_link_without_rel_is_kept() -> None:
    # a link with no `rel` has nothing to match on: only a pattern that
    # accepts the empty string drops it
    links = [{"href": "https://example.org"}]

    assert STACSettings(exclude_links=["self"]).filter_links(links) == links
    assert STACSettings(exclude_links=[".*"]).filter_links(links) == []


def test_override_links_beats_exclude() -> None:
    link = {"rel": "license", "href": "https://example.org/other"}
    settings = STACSettings(exclude_links=[".*"], override_links=[link])

    assert settings.filter_links(STAC_ITEM["links"]) == [link]


def test_override_links_are_not_duplicated() -> None:
    link = STAC_ITEM["links"][0]
    settings = STACSettings(override_links=[link])

    assert settings.filter_links(STAC_ITEM["links"]) == STAC_ITEM["links"]


def test_exclude_assets_key_and_pattern() -> None:
    settings = STACSettings(exclude_assets=["thumb.*"])

    assert settings.filter_assets(STAC_ITEM["assets"]) == {
        "data": STAC_ITEM["assets"]["data"]
    }


def test_assets_are_kept_with_an_empty_exclude() -> None:
    settings = STACSettings(exclude_assets=[])

    assert settings.filter_assets(STAC_ITEM["assets"]) == STAC_ITEM["assets"]


def test_override_assets_beats_exclude() -> None:
    asset = {"href": "s3://public/out.zarr", "type": "application/vnd+zarr"}
    settings = STACSettings(exclude_assets=["data"], override_assets={"data": asset})

    assert settings.filter_assets(STAC_ITEM["assets"])["data"] == asset


def test_invalid_patterns_rejected() -> None:
    for field in (
        "exclude_properties",
        "exclude_extensions",
        "exclude_links",
        "exclude_assets",
    ):
        with pytest.raises(ValidationError, match=".*invalid regular expression.*"):
            STACSettings(**{field: ["[unbalanced"]})


def test_stac_settings_from_convert_settings() -> None:
    settings = ConvertSettings.model_validate(
        {
            "metadata": {"stac": {"exclude_properties": ["internal:.*"]}},
            "group_settings": {},
        }
    )

    assert settings.metadata.stac.exclude_properties == ["internal:.*"]


def _write_input(tmp_path, name: str, item: dict) -> str:
    ds = xr.Dataset(
        {"t2m": (("lat", "lon"), np.arange(64, dtype="float32").reshape(8, 8))},
        coords={
            "lon": ("lon", np.linspace(-1.0, 1.0, 8), {"standard_name": "longitude"}),
            "lat": ("lat", np.linspace(-1.0, 1.0, 8), {"standard_name": "latitude"}),
        },
    )
    tree = xr.DataTree.from_dict({"/measurements": ds})
    tree.attrs["stac_discovery"] = item

    path = tmp_path / name
    tree.to_zarr(path, mode="w", consolidated=True)

    return str(path)


@pytest.fixture
def input_path(tmp_path):
    return _write_input(tmp_path, "input.zarr", STAC_ITEM)


def _convert(tmp_path, input_path, stac_settings: dict) -> dict:
    settings = ConvertSettings.model_validate(
        {
            "metadata": {"stac": stac_settings},
            "group_settings": {
                "measurements": {
                    "healpix": {"refinement_level": 4, "indexing_scheme": "nested"},
                    "chunk": {"method": "no_chunk"},
                    "resampler": {"name": "nearest"},
                }
            },
        }
    )
    output_path = str(tmp_path / "output.zarr")
    create_healpix_dataset([input_path], settings, output_path)

    return xr.open_datatree(output_path, engine="zarr").attrs["stac_discovery"]


def test_excluded_properties_are_not_written(tmp_path, input_path) -> None:
    # the data provider does not want the input's internal bookkeeping in the output
    stac = _convert(tmp_path, input_path, {"exclude_properties": ["internal:.*"]})

    assert "internal:job_id" not in stac["properties"]
    assert stac["properties"]["proj:epsg"] == 4326


def test_overridden_properties_are_written(tmp_path, input_path) -> None:
    stac = _convert(
        tmp_path, input_path, {"override_properties": {"license": "CC-BY-4.0"}}
    )

    assert stac["properties"]["license"] == "CC-BY-4.0"


def test_excluded_links_are_not_written(tmp_path, input_path) -> None:
    # the `self` link points at the input item, not at this output
    stac = _convert(tmp_path, input_path, {"exclude_links": ["self"]})

    rels = [link["rel"] for link in stac["links"]]
    assert "self" not in rels
    assert "license" in rels


def test_overridden_links_are_written(tmp_path, input_path) -> None:
    link = {"rel": "canonical", "href": "https://example.org/out.zarr"}
    stac = _convert(tmp_path, input_path, {"override_links": [link]})

    assert link in stac["links"]


def test_derived_from_links_keep_precedence(tmp_path, input_path) -> None:
    stac = _convert(tmp_path, input_path, {"exclude_links": [".*"]})

    assert [link["rel"] for link in stac["links"]] == ["derived_from"]


def test_input_assets_are_not_written_by_default(tmp_path, input_path) -> None:
    # the input's assets point at files the output does not hold
    stac = _convert(tmp_path, input_path, {})

    assert stac["assets"] == {}


def test_input_assets_can_be_propagated(tmp_path, input_path) -> None:
    stac = _convert(tmp_path, input_path, {"exclude_assets": []})

    assert stac["assets"] == STAC_ITEM["assets"]


def test_overridden_assets_are_written(tmp_path, input_path) -> None:
    asset = {"href": "s3://public/out.zarr", "type": "application/vnd+zarr"}
    stac = _convert(tmp_path, input_path, {"override_assets": {"data": asset}})

    assert stac["assets"] == {"data": asset}


def test_conversion_settings_recorded_by_default(tmp_path, input_path) -> None:
    stac = _convert(tmp_path, input_path, {})

    assert "processing:expression" in stac["properties"]


def test_conversion_settings_can_be_left_out(tmp_path, input_path) -> None:
    # the recorded settings are large and may name internal paths
    stac = _convert(tmp_path, input_path, {"include_conversion_settings": False})

    assert "processing:expression" not in stac["properties"]
    # the rest of the provenance is still written
    assert "processing:datetime" in stac["properties"]
    assert "processing:software" in stac["properties"]
    assert STAC_PROCESSING_SCHEMA_URL in stac["stac_extensions"]


def test_provenance_keeps_precedence(tmp_path, input_path) -> None:
    # the output must stay traceable: what healpix-convert records about the
    # conversion is not the provider's to drop or rewrite
    stac = _convert(
        tmp_path,
        input_path,
        {
            "exclude_properties": [".*"],
            "override_properties": {"processing:lineage": "made up"},
            "exclude_extensions": [".*"],
        },
    )

    assert "processing:expression" in stac["properties"]
    assert stac["properties"]["processing:lineage"].startswith("Resampled onto HEALPix")
    assert stac["stac_extensions"] == [STAC_PROCESSING_SCHEMA_URL]


def test_extensions_can_be_added(tmp_path, input_path) -> None:
    stac = _convert(
        tmp_path,
        input_path,
        {"override_extensions": ["https://example.org/ext/v1.0.0/schema.json"]},
    )

    assert stac["stac_extensions"] == [
        PROJ_SCHEMA_URL,
        "https://example.org/ext/v1.0.0/schema.json",
        STAC_PROCESSING_SCHEMA_URL,
    ]


def test_excluded_extensions_are_not_propagated(tmp_path, input_path) -> None:
    stac = _convert(
        tmp_path,
        input_path,
        {"exclude_extensions": ["https://stac-extensions.github.io/projection/.*"]},
    )

    assert stac["stac_extensions"] == [STAC_PROCESSING_SCHEMA_URL]


EO_SCHEMA_URL = "https://stac-extensions.github.io/eo/v1.1.0/schema.json"


def _convert_two_inputs(tmp_path, stac_settings: dict) -> dict:
    """Convert two inputs whose items differ, to exercise the merge."""
    second = STAC_ITEM | {
        "stac_extensions": [EO_SCHEMA_URL],
        "properties": STAC_ITEM["properties"] | {"only_second": "x"},
    }
    inputs = [
        _write_input(tmp_path, "input-a.zarr", STAC_ITEM),
        _write_input(tmp_path, "input-b.zarr", second),
    ]

    settings = ConvertSettings.model_validate(
        {
            "metadata": {"stac": stac_settings},
            "group_settings": {
                "measurements": {
                    "healpix": {"refinement_level": 4, "indexing_scheme": "nested"},
                    "chunk": {"method": "no_chunk"},
                    "resampler": {"name": "nearest"},
                }
            },
        }
    )
    output_path = str(tmp_path / "output.zarr")
    create_healpix_dataset(inputs, settings, output_path)

    return xr.open_datatree(output_path, engine="zarr").attrs["stac_discovery"]


def test_merge_strategy_reaches_the_conversion(tmp_path) -> None:
    # the settings must make it all the way into the merge of the input items
    stac = _convert_two_inputs(
        tmp_path, {"merge_extensions": "join_outer", "merge_properties": "join_inner"}
    )

    assert PROJ_SCHEMA_URL in stac["stac_extensions"]
    assert EO_SCHEMA_URL in stac["stac_extensions"]
    # join_inner drops what only the second input has
    assert "only_second" not in stac["properties"]
    assert stac["properties"]["proj:epsg"] == 4326


def test_mismatching_extensions_no_longer_fail(tmp_path) -> None:
    # they used to raise; pick_first, the default, keeps the first input's
    stac = _convert_two_inputs(tmp_path, {})

    assert stac["stac_extensions"] == [PROJ_SCHEMA_URL, STAC_PROCESSING_SCHEMA_URL]
