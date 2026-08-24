import pytest
from pydantic import ValidationError
from pydantic.experimental.missing_sentinel import MISSING

from healpix_convert.core.healpix_conventions import (
    DGGSZarrConvention,
    Healpix,
    WGS84Ellipsoid,
)


def test_dggs_zarr_convention_values() -> None:
    c = DGGSZarrConvention()

    assert c.uuid == "7b255807-140c-42ca-97f6-7a1cfecdbc38"
    assert c.name == "dggs"
    assert (
        c.schema_url
        == "https://raw.githubusercontent.com/zarr-conventions/dggs/refs/tags/v1/schema.json"
    )
    assert c.spec_url == "https://github.com/zarr-conventions/dggs/blob/v1/README.md"
    assert c.description == "Discrete Global Grid Systems convention for zarr"


def test_wgs84_ellipsoid_values() -> None:
    e = WGS84Ellipsoid()

    assert e.name == "wgs84"
    assert e.semi_major_axis == 6378137.0
    assert e.inverse_flattening == 298.257223563


def test_healpix_default_values() -> None:
    h = Healpix(refinement_level=10)

    assert h.name == "healpix"
    assert h.indexing_scheme == "nested"
    assert h.ellipsoid is MISSING
    assert h.spatial_dimension == "cells"
    assert h.coordinate == "cell_ids"
    assert h.compression == "none"


def test_healpix_ellipsoid_dscriminator() -> None:
    fields = {"refinement_level": 10, "ellipsoid": {"name": "wgs84"}}

    h = Healpix.model_validate(fields)

    assert isinstance(h.ellipsoid, WGS84Ellipsoid)


def test_healpix_invalid_refinement_level() -> None:
    with pytest.raises(
        ValidationError, match=".*should be greater than or equal to 0.*"
    ):
        Healpix(refinement_level=-1)

    with pytest.raises(
        ValidationError, match=".*must be defined for.*indexing scheme.*"
    ):
        Healpix(refinement_level=None)

    with pytest.raises(
        ValidationError, match=".*undefined but compression is not 'none'"
    ):
        Healpix(refinement_level=None, indexing_scheme="zuniq", compression="compacted")

    with pytest.raises(
        ValidationError, match=".*coordinate.*omitted.*refinement level.*undefined"
    ):
        Healpix(refinement_level=None, indexing_scheme="zuniq", coordinate=MISSING)
