import numpy as np
import pytest
import zarr
from pydantic import ValidationError

from legacy_converters.core.healpix_conventions import (
    CFHealpixGridMapping,
    Healpix,
    WGS84Ellipsoid,
    write_cf_grid_mapping,
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
    assert attrs["semi_major_axis"] == hp.ellipsoid.semi_major_axis
    assert attrs["inverse_flattening"] == hp.ellipsoid.inverse_flattening


def test_cf_grid_mapping_from_healpix_sphere() -> None:
    hp = Healpix(refinement_level=10, indexing_scheme="nested")
    cf_hp = CFHealpixGridMapping.from_healpix(hp)
    attrs = cf_hp.model_dump()
    assert "reference_ellipsoid_name" not in attrs
    assert "semi_major_axis" not in attrs
    assert "inverse_flattening" not in attrs


def test_write_cf_grid_mapping(tmp_path) -> None:
    """The CF grid mapping emitted by the ERA5/CAMS/ClimateDT write path must
    match what the shared HealpixGroupConverter writes."""
    root = zarr.open_group(str(tmp_path / "out.zarr"), mode="w")
    grp = root.require_group("measurements/grp")
    grp.create_array(
        "cell_ids", shape=(12,), dtype=np.int64, dimension_names=("cells",)
    )
    for var in ("a", "b"):
        grp.create_array(
            var,
            shape=(1, 12),
            dtype=np.float32,
            dimension_names=("time", "cells"),
            attributes={"units": "1"},
        )

    healpix = Healpix(
        refinement_level=7, indexing_scheme="nested", ellipsoid=WGS84Ellipsoid()
    )
    write_cf_grid_mapping(grp, healpix, ["a", "b"])

    # cell-id coordinate promoted to a CF healpix_index coordinate
    assert grp["cell_ids"].attrs["standard_name"] == "healpix_index"
    assert grp["cell_ids"].attrs["units"] == "1"

    # scalar crs grid-mapping variable carrying the CF HEALPix attributes
    crs = grp["crs"]
    assert crs.shape == ()
    assert crs.attrs["grid_mapping_name"] == "healpix"
    assert crs.attrs["refinement_level"] == 7
    assert crs.attrs["indexing_scheme"] == "nested"
    assert crs.attrs["reference_ellipsoid_name"] == "WGS84"

    # grid_mapping reference kept alongside pre-existing attributes
    for var in ("a", "b"):
        assert grp[var].attrs["grid_mapping"] == "crs"
        assert grp[var].attrs["units"] == "1"
