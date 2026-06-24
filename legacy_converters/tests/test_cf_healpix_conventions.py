import pytest
from pydantic import ValidationError

from legacy_converters.core.healpix_conventions import (
    CFHealpixGridMapping,
    Healpix,
    WGS84Ellipsoid,
)


def test_cf_healpix_default_fields() -> None:
    cf_hp = CFHealpixGridMapping(refinement_level=10)
    attrs = cf_hp.model_dump()

    assert attrs["grid_mapping_name"] == "healpix"
    assert attrs["indexing_scheme"] == "nested"
    assert "reference_ellipsoid_name" not in attrs
    assert "semi_major_axis" not in attrs
    assert "inverse_flattening" not in attrs


def test_cf_healpix_invalid_refinement_level() -> None:
    with pytest.raises(ValidationError):
        CFHealpixGridMapping(refinement_level=-1)

    with pytest.raises(ValidationError):
        CFHealpixGridMapping(refinement_level=30)

    with pytest.raises(
        ValidationError, match=".*must be defined for.*indexing scheme.*"
    ):
        CFHealpixGridMapping(indexing_scheme="nested")

    with pytest.raises(
        ValidationError, match=".*must be omitted for.*indexing scheme.*"
    ):
        CFHealpixGridMapping(indexing_scheme="zuniq", refinement_level=10)


def test_cf_grid_mapping_from_healpix_wgs84() -> None:
    hp = Healpix(
        refinement_level=20, indexing_scheme="nested", ellipsoid=WGS84Ellipsoid()
    )
    cf_hp = CFHealpixGridMapping.from_healpix(hp)
    attrs = cf_hp.model_dump()
    assert attrs["grid_mapping_name"] == "healpix"
    assert attrs["refinement_level"] == hp.refinement_level
    assert attrs["indexing_scheme"] == hp.indexing_scheme
    assert attrs["reference_ellipsoid_name"] == "WGS84"
    assert isinstance(hp.ellipsoid, WGS84Ellipsoid)
    assert attrs["semi_major_axis"] == hp.ellipsoid.semimajor_axis
    assert attrs["inverse_flattening"] == hp.ellipsoid.inverse_flattening


def test_cf_grid_mapping_from_healpix_sphere() -> None:
    hp = Healpix(refinement_level=10, indexing_scheme="nested")
    cf_hp = CFHealpixGridMapping.from_healpix(hp)
    attrs = cf_hp.model_dump()
    assert "reference_ellipsoid_name" not in attrs
    assert "semi_major_axis" not in attrs
    assert "inverse_flattening" not in attrs
