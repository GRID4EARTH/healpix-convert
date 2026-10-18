"""Identity of the published JSON schema of the conversion settings.

Kept apart from the schema generation (``healpix_convert.settings``) so that
``core`` can name the schema without depending on the settings models.

"""

from __future__ import annotations

import re

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

CONVERT_SETTINGS_SCHEMA_FILENAME = "convert-settings.schema.json"
"""Name of the published schema file, in the root of the repository."""

_SCHEMA_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/GRID4EARTH/healpix-convert/{ref}/"
    + CONVERT_SETTINGS_SCHEMA_FILENAME
)

CONVERT_SETTINGS_SCHEMA_URL = _SCHEMA_URL_TEMPLATE.format(ref="main")
"""URL of the schema published in the default branch, i.e. the latest one.

Used as the ``$id`` of the generated schema. Prefer
:py:func:`convert_settings_schema_url` to refer to the schema of a given
version of the package, which is immutable.
"""

#: A released version, tagged ``v<version>`` in the repository. At least two
#: components, so that setuptools-scm's `fallback_version` is not mistaken for
#: a release (it is used when the package is built outside a git checkout).
_RELEASE_RE = re.compile(r"[0-9]+(\.[0-9]+)+")

#: The commit a development version was built from, as setuptools-scm writes
#: it in the local part, e.g. "2026.10.2.dev4+g2bd10f00b.d20261018".
_COMMIT_RE = re.compile(r"\+g(?P<commit>[0-9a-f]{7,40})")


def _installed_version() -> str | None:
    import importlib.metadata

    try:
        return importlib.metadata.version("healpix-convert")
    except importlib.metadata.PackageNotFoundError:
        return None


def convert_settings_schema_url(version: str | None = None) -> str:
    """Return the URL of the schema published for a version of healpix-convert.

    The settings schema changes with the package, so a dataset must record the
    schema of the version that converted it rather than the latest one: a
    released version resolves to its (immutable) tag and a development version
    to the commit it was built from. A version that is neither resolves to
    :py:data:`CONVERT_SETTINGS_SCHEMA_URL`, the default branch.

    Parameters
    ----------
    version : str, optional
        Version of healpix-convert (the installed one by default).

    """
    if version is None:
        version = _installed_version()

    if version is not None:
        if _RELEASE_RE.fullmatch(version):
            # releases are tagged "v<version>", see CITATION.cff
            return _SCHEMA_URL_TEMPLATE.format(ref=f"refs/tags/v{version}")

        commit = _COMMIT_RE.search(version)
        if commit is not None:
            return _SCHEMA_URL_TEMPLATE.format(ref=commit["commit"])

    return CONVERT_SETTINGS_SCHEMA_URL
