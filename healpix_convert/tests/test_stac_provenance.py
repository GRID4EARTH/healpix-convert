from datetime import UTC, datetime

from healpix_convert.core.stac import (
    PROCESSING_EXTENSION,
    SETTINGS_EXPRESSION_FORMAT,
    derived_from_links,
    processing_properties,
)
from healpix_convert.settings.common import get_settings


def test_settings_are_recorded_verbatim() -> None:
    settings = get_settings("era5")
    properties = processing_properties(settings)

    expression = properties["processing:expression"]
    assert expression["format"] == SETTINGS_EXPRESSION_FORMAT
    assert expression["expression"] == settings.to_dict()


def test_resampler_and_its_parameters_are_recoverable() -> None:
    # the point of this metadata: two outputs resampled from one input by
    # different methods must be tellable apart
    settings = get_settings("era5")
    properties = processing_properties(settings)

    recorded = properties["processing:expression"]["expression"]
    resamplers = {
        group["resampler"]["name"] for group in recorded["group_settings"].values()
    }

    assert resamplers == {
        group.resampler.name for group in settings.group_settings.values()
    }


def test_software_versions_recorded() -> None:
    properties = processing_properties(get_settings("era5"))

    software = properties["processing:software"]
    assert "healpix-convert" in software
    assert all(isinstance(version, str) for version in software.values())


def test_processing_datetime_is_utc_and_injectable() -> None:
    stamp = datetime(2026, 8, 18, 12, 0, tzinfo=UTC)
    properties = processing_properties(get_settings("era5"), processing_datetime=stamp)

    assert properties["processing:datetime"] == "2026-08-18T12:00:00+00:00"

    # defaults to now, in UTC
    default = processing_properties(get_settings("era5"))
    assert datetime.fromisoformat(default["processing:datetime"]).tzinfo is not None


def test_lineage_names_the_inputs() -> None:
    properties = processing_properties(get_settings("era5"), input_ids=["item-b", "item-a"])

    assert "item-a" in properties["processing:lineage"]
    assert "item-b" in properties["processing:lineage"]


def test_no_lineage_without_inputs() -> None:
    assert "processing:lineage" not in processing_properties(get_settings("era5"))


def test_extension_url_is_versioned() -> None:
    assert PROCESSING_EXTENSION.startswith("https://stac-extensions.github.io/processing/v")
    assert PROCESSING_EXTENSION.endswith("/schema.json")


def test_lineage_names_the_input_locations() -> None:
    # the minutes ask for the input data's object-store location, not just ids
    properties = processing_properties(
        get_settings("era5"),
        input_ids=["item-a"],
        input_paths=["s3://grid4earth/public/era5/in.zarr"],
    )

    assert "s3://grid4earth/public/era5/in.zarr" in properties["processing:lineage"]


def test_derived_from_links() -> None:
    links = derived_from_links(["s3://bucket/a.zarr", "s3://bucket/b.zarr"])

    assert [link["rel"] for link in links] == ["derived_from", "derived_from"]
    assert links[0]["href"] == "s3://bucket/a.zarr"


def test_no_links_without_inputs() -> None:
    assert derived_from_links([]) == []
