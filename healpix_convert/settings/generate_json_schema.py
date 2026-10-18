"""JSON schema of the conversion settings.

The schema is generated from the :class:`ConvertSettings` model and published
as ``convert-settings.schema.json`` in the root of the repository. Regenerate
it with ``pixi run publish-schema``.

"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from healpix_convert.core.json_schema import (
    CONVERT_SETTINGS_SCHEMA_FILENAME,
    CONVERT_SETTINGS_SCHEMA_URL,
    JSON_SCHEMA_DIALECT,
)
from healpix_convert.settings.common import ConvertSettings


def _without_ref_titles(node: Any) -> Any:
    """Return the schema without the ``title`` keys sitting next to a ``$ref``.

    pydantic <= 2.13 writes them and 2.14 does not, so dropping them keeps the
    published schema identical whichever version generated it. Such a title
    only repeats the name of the field it describes.

    """
    if isinstance(node, list):
        return [_without_ref_titles(item) for item in node]

    if not isinstance(node, dict):
        return node

    cleaned = {key: _without_ref_titles(value) for key, value in node.items()}
    if "$ref" in cleaned:
        cleaned.pop("title", None)

    return cleaned


def convert_settings_json_schema() -> dict[str, Any]:
    """Return the JSON schema of the conversion settings."""
    return {
        "$schema": JSON_SCHEMA_DIALECT,
        "$id": CONVERT_SETTINGS_SCHEMA_URL,
        **_without_ref_titles(ConvertSettings.model_json_schema()),
    }


def format_json_schema() -> str:
    """Return the JSON schema as it is written in the published file."""
    return json.dumps(convert_settings_json_schema(), indent=2) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish the settings JSON schema.")
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path(CONVERT_SETTINGS_SCHEMA_FILENAME),
        help=(
            "where to write the schema "
            f"(default: {CONVERT_SETTINGS_SCHEMA_FILENAME})"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write anything, fail if the published schema is out of date",
    )
    args = parser.parse_args(argv)

    schema = format_json_schema()

    if args.check:
        published = args.path.read_text() if args.path.exists() else None
        if published != schema:
            print(
                f"{args.path} does not match ConvertSettings, "
                f"run 'pixi run publish-schema'",
                file=sys.stderr,
            )
            return 1
        return 0

    args.path.write_text(schema)
    print(f"wrote {args.path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
