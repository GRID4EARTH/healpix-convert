"""Control over the STAC metadata written in the output root group."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from healpix_convert import create_healpix_dataset
from healpix_convert.core.stac import STAC_PROCESSING_SCHEMA_URL
from healpix_convert.settings.common import ConvertSettings
from healpix_convert.settings.conventions import MetadataSettings
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
    "links": [],
    "assets": {},
}


def test_stac_settings_default() -> None:
    settings = STACSettings()

    assert settings.include_conversion_settings
    assert settings.exclude_props == []
    assert settings.override_props == {}
    assert settings.exclude_extensions == []
    assert settings.override_extensions == []
    assert settings.filter_props({"datetime": "x"}) == {"datetime": "x"}
    assert settings.filter_extensions([PROJ_SCHEMA_URL]) == [PROJ_SCHEMA_URL]


def test_metadata_settings_default_stac() -> None:
    assert MetadataSettings().stac == STACSettings()


def test_exclude_props_name_and_pattern() -> None:
    settings = STACSettings(exclude_props=["internal:.*", "datetime"])

    assert settings.is_excluded_prop("internal:job_id")
    assert settings.is_excluded_prop("datetime")
    assert not settings.is_excluded_prop("proj:epsg")
    # the pattern is matched against the whole name
    assert not settings.is_excluded_prop("my_internal:job_id")

    assert settings.filter_props(STAC_ITEM["properties"]) == {"proj:epsg": 4326}


def test_override_props_beats_exclude() -> None:
    settings = STACSettings(
        exclude_props=["license"], override_props={"license": "CC-BY-4.0"}
    )

    assert settings.filter_props({"license": "proprietary"}) == {"license": "CC-BY-4.0"}


def test_override_props_adds_missing_properties() -> None:
    settings = STACSettings(override_props={"license": "CC-BY-4.0"})

    assert settings.filter_props({"datetime": "x"}) == {
        "datetime": "x",
        "license": "CC-BY-4.0",
    }


def test_exclude_extensions_name_and_pattern() -> None:
    settings = STACSettings(
        exclude_extensions=["https://stac-extensions.github.io/projection/.*"]
    )

    assert settings.is_excluded_extension(PROJ_SCHEMA_URL)
    assert not settings.is_excluded_extension(STAC_PROCESSING_SCHEMA_URL)

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


def test_invalid_patterns_rejected() -> None:
    with pytest.raises(ValidationError, match=".*invalid regular expression.*"):
        STACSettings(exclude_props=["[unbalanced"])

    with pytest.raises(ValidationError, match=".*invalid regular expression.*"):
        STACSettings(exclude_extensions=["[unbalanced"])


def test_stac_settings_from_convert_settings() -> None:
    settings = ConvertSettings.model_validate(
        {
            "metadata": {"stac": {"exclude_props": ["internal:.*"]}},
            "group_settings": {},
        }
    )

    assert settings.metadata.stac.exclude_props == ["internal:.*"]


@pytest.fixture
def input_path(tmp_path):
    ds = xr.Dataset(
        {"t2m": (("lat", "lon"), np.arange(64, dtype="float32").reshape(8, 8))},
        coords={
            "lon": ("lon", np.linspace(-1.0, 1.0, 8), {"standard_name": "longitude"}),
            "lat": ("lat", np.linspace(-1.0, 1.0, 8), {"standard_name": "latitude"}),
        },
    )
    tree = xr.DataTree.from_dict({"/measurements": ds})
    tree.attrs["stac_discovery"] = STAC_ITEM

    path = tmp_path / "input.zarr"
    tree.to_zarr(path, mode="w", consolidated=True)

    return str(path)


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


def test_excluded_props_are_not_written(tmp_path, input_path) -> None:
    # the data provider does not want the input's internal bookkeeping in the output
    stac = _convert(tmp_path, input_path, {"exclude_props": ["internal:.*"]})

    assert "internal:job_id" not in stac["properties"]
    assert stac["properties"]["proj:epsg"] == 4326


def test_overridden_props_are_written(tmp_path, input_path) -> None:
    stac = _convert(tmp_path, input_path, {"override_props": {"license": "CC-BY-4.0"}})

    assert stac["properties"]["license"] == "CC-BY-4.0"


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
            "exclude_props": [".*"],
            "override_props": {"processing:lineage": "made up"},
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
