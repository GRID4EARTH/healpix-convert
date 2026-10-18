"""The published JSON schema must keep describing ConvertSettings."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from healpix_convert.core.json_schema import (
    CONVERT_SETTINGS_SCHEMA_FILENAME,
    CONVERT_SETTINGS_SCHEMA_URL,
    JSON_SCHEMA_DIALECT,
    convert_settings_schema_url,
)
from healpix_convert.core.stac import format_stac_processing
from healpix_convert.settings.common import ConvertSettings, get_settings
from healpix_convert.settings.generate_json_schema import (
    convert_settings_json_schema,
    format_json_schema,
    main,
)

SCHEMA_FILENAME = CONVERT_SETTINGS_SCHEMA_FILENAME
PUBLISHED_SCHEMA = Path(__file__).parents[2] / SCHEMA_FILENAME


def _published() -> str:
    if not PUBLISHED_SCHEMA.exists():
        # the schema lives in the repository, not in the installed package
        pytest.skip(f"{SCHEMA_FILENAME} not found (not a source checkout)")

    return PUBLISHED_SCHEMA.read_text()


def test_published_schema_is_up_to_date() -> None:
    # regenerate with `pixi run publish-schema`
    assert _published() == format_json_schema()


def test_schema_identifies_itself() -> None:
    schema = convert_settings_json_schema()

    assert schema["$schema"] == JSON_SCHEMA_DIALECT
    assert schema["$id"] == CONVERT_SETTINGS_SCHEMA_URL
    assert schema["title"] == "ConvertSettings"


def test_schema_is_valid() -> None:
    jsonschema = pytest.importorskip("jsonschema")

    jsonschema.Draft202012Validator.check_schema(json.loads(_published()))


@pytest.mark.parametrize("name", ["era5", "cams-eac4", "climatedt-sfc"])
def test_schema_accepts_predefined_settings(name: str) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    settings = json.loads(get_settings(name).to_json())

    jsonschema.validate(settings, json.loads(_published()))


def test_schema_rejects_invalid_settings() -> None:
    jsonschema = pytest.importorskip("jsonschema")

    # the same settings that ConvertSettings itself refuses
    settings = {"group_settings": {"/measurements": {"healpix": {}}}}

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(settings, json.loads(_published()))


def test_main_writes_the_schema(tmp_path) -> None:
    path = tmp_path / SCHEMA_FILENAME

    assert main([str(path)]) == 0
    assert path.read_text() == format_json_schema()
    # the file it just wrote is up to date
    assert main([str(path), "--check"]) == 0


def test_main_check_detects_a_stale_schema(tmp_path) -> None:
    path = tmp_path / SCHEMA_FILENAME

    # missing
    assert main([str(path), "--check"]) == 1

    # out of date
    path.write_text(json.dumps({"title": "ConvertSettings"}))
    assert main([str(path), "--check"]) == 1


def test_schema_covers_the_settings_fields() -> None:
    schema = convert_settings_json_schema()

    assert set(schema["properties"]) == set(ConvertSettings.model_fields)


def test_schema_validates_the_settings_recorded_in_the_output() -> None:
    # the STAC `processing:expression` names this schema as its format: it must
    # be the one the recorded settings actually follow
    jsonschema = pytest.importorskip("jsonschema")

    expression = format_stac_processing(get_settings("era5"))["processing:expression"]

    assert expression["format"] == convert_settings_schema_url()
    jsonschema.validate(expression["expression"], json.loads(_published()))


def test_schema_url_of_a_release_is_its_tag() -> None:
    # an output must keep pointing at the schema of the version that wrote it,
    # not at a newer one: a tag never moves, the default branch does
    url = convert_settings_schema_url("2026.10.1")

    assert url.endswith(f"/refs/tags/v2026.10.1/{CONVERT_SETTINGS_SCHEMA_FILENAME}")


def test_schema_url_of_a_development_version_is_its_commit() -> None:
    url = convert_settings_schema_url("2026.10.2.dev4+g2bd10f00b.d20261018")

    assert url.endswith(f"/2bd10f00b/{CONVERT_SETTINGS_SCHEMA_FILENAME}")


@pytest.mark.parametrize("version", ["999", "unknown"])
def test_schema_url_falls_back_to_the_default_branch(version: str) -> None:
    # "999" is the setuptools-scm fallback, used outside a git checkout
    assert convert_settings_schema_url(version) == CONVERT_SETTINGS_SCHEMA_URL


def test_schema_url_of_the_installed_version_is_pinned() -> None:
    # the version this test runs against is either a release or a dev build,
    # never unknown, so the recorded URL is immutable
    assert convert_settings_schema_url() != CONVERT_SETTINGS_SCHEMA_URL


def test_schema_url_matches_the_recorded_version() -> None:
    # the schema the output points at must be the one of the version the
    # output says produced it
    properties = format_stac_processing(get_settings("era5"))
    version = properties["processing:software"]["healpix-convert"]

    assert properties["processing:expression"]["format"] == convert_settings_schema_url(
        version
    )


def test_schema_does_not_depend_on_the_pydantic_version() -> None:
    # pydantic <= 2.13 adds a `title` next to a `$ref` and 2.14 does not: the
    # published schema must not change with the version that generated it
    def refs(node: object) -> Iterator[dict]:
        if isinstance(node, list):
            for item in node:
                yield from refs(item)
        elif isinstance(node, dict):
            if "$ref" in node:
                yield node
            for value in node.values():
                yield from refs(value)

    assert not [
        node for node in refs(convert_settings_json_schema()) if "title" in node
    ]
