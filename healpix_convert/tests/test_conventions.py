import typing

import pytest
import zarr.api.synchronous as zarr
from pydantic import ValidationError

from healpix_convert.core.conventions import (
    CF_SUPPORTED_VERSIONS,
    CFConventionSpec,
    DGGSConventionSpec,
    MetadataSettings,
    MultiscalesConventionSpec,
)
from healpix_convert.core.healpix_conventions import Healpix
from healpix_convert.core.metadata import (
    CELL_METHODS,
    CF_GRID_MAPPING_VARIABLE,
    cf_cell_methods,
    cf_cell_id_attrs,
    cf_data_variable_attrs,
    cf_grid_mapping_attrs,
    group_conventions_attrs,
    write_root_conventions,
)
from healpix_convert.core.multiscales_conventions import (
    HealpixMultiscales,
    HealpixMultiscalesLayoutItem,
)


@pytest.fixture
def healpix() -> Healpix:
    return Healpix(
        refinement_level=5, indexing_scheme="nested", ellipsoid={"name": "wgs84"}
    )


@pytest.fixture
def multiscales() -> HealpixMultiscales:
    return HealpixMultiscales(
        layout=[HealpixMultiscalesLayoutItem(asset="level5")],
    )


def test_metadata_settings_default_selects_all() -> None:
    metadata = MetadataSettings()

    assert [spec.name for spec in metadata.zarr_conventions] == [
        "dggs",
        "multiscales",
        "CF",
    ]
    assert metadata.conventions_attr == f"CF-{CF_SUPPORTED_VERSIONS[-1]}"
    assert metadata.write_cf_grid_mapping


def test_metadata_settings_shorthand_and_name_case() -> None:
    metadata = MetadataSettings.model_validate(
        {"zarr_conventions": ["dggs", "multiscales", "cf"]}
    )

    assert isinstance(metadata.get("dggs"), DGGSConventionSpec)
    assert isinstance(metadata.get("multiscales"), MultiscalesConventionSpec)
    # "cf" is normalized to the canonical "CF" spelling
    assert isinstance(metadata.get("CF"), CFConventionSpec)
    assert metadata.has("cf")


def test_metadata_settings_issue_73_example() -> None:
    """The selector accepts the JSON proposed in GRID4EARTH/healpix-convert#73."""
    metadata = MetadataSettings.model_validate(
        {
            "zarr_conventions": [
                "dggs",
                {"name": "multiscales"},
                {
                    "name": "CF",
                    "version": "1.13",
                    "configuration": {"healpix_grid_mapping": True},
                },
            ]
        }
    )

    assert metadata.conventions_attr == "CF-1.13"
    assert metadata.write_cf_grid_mapping


def test_metadata_settings_reject_duplicates() -> None:
    with pytest.raises(ValidationError, match="duplicate convention.*dggs"):
        MetadataSettings.model_validate({"zarr_conventions": ["dggs", "dggs"]})


def test_metadata_settings_reject_unknown_convention() -> None:
    with pytest.raises(ValidationError):
        MetadataSettings.model_validate({"zarr_conventions": ["eopf"]})


def test_cf_spec_reject_unsupported_version() -> None:
    with pytest.raises(ValidationError, match="unsupported CF version '1.9'"):
        CFConventionSpec.model_validate({"name": "CF", "version": "1.9"})


def test_cf_spec_supports_unknown_feature() -> None:
    with pytest.raises(ValueError, match="unknown CF feature"):
        CFConventionSpec().supports("nonexistent_feature")


def test_metadata_settings_without_cf() -> None:
    metadata = MetadataSettings.model_validate({"zarr_conventions": ["dggs"]})

    assert metadata.conventions_attr is None
    assert not metadata.write_cf_grid_mapping
    assert not metadata.has("CF")


def test_metadata_settings_cf_without_grid_mapping() -> None:
    metadata = MetadataSettings.model_validate(
        {
            "zarr_conventions": [
                {"name": "CF", "configuration": {"healpix_grid_mapping": False}}
            ]
        }
    )

    # CF is still declared, but no grid mapping variable is written
    assert metadata.conventions_attr == "CF-1.13"
    assert not metadata.write_cf_grid_mapping


def test_group_conventions_attrs_dggs_group(healpix) -> None:
    attrs = group_conventions_attrs(MetadataSettings(), declare=("dggs",), dggs=healpix)

    assert [c["name"] for c in attrs["zarr_conventions"]] == ["dggs"]
    assert attrs["dggs"] == healpix.model_dump()
    assert "multiscales" not in attrs


def test_group_conventions_attrs_multiscale_group(multiscales) -> None:
    attrs = group_conventions_attrs(
        MetadataSettings(), declare=("multiscales", "dggs"), multiscales=multiscales
    )

    # declaration order is preserved, and the DGGS convention applies to the children
    assert [c["name"] for c in attrs["zarr_conventions"]] == ["multiscales", "dggs"]
    assert attrs["multiscales"] == multiscales.model_dump()
    assert "dggs" not in attrs


def test_group_conventions_attrs_deselected(healpix, multiscales) -> None:
    metadata = MetadataSettings.model_validate({"zarr_conventions": ["CF"]})
    attrs = group_conventions_attrs(
        metadata, declare=("dggs", "multiscales"), dggs=healpix, multiscales=multiscales
    )

    assert attrs == {}


def test_write_root_conventions(tmp_path) -> None:
    group = zarr.create_group(str(tmp_path / "with_cf.zarr"))
    write_root_conventions(group, MetadataSettings())

    assert group.attrs["Conventions"] == "CF-1.13"

    other = zarr.create_group(str(tmp_path / "without_cf.zarr"))
    write_root_conventions(
        other, MetadataSettings.model_validate({"zarr_conventions": ["dggs"]})
    )

    assert "Conventions" not in other.attrs


@pytest.mark.parametrize(
    "selection",
    [
        # a mistyped configuration option
        {"name": "CF", "configuration": {"helpix_grid_mapping": True}},
        # a mistyped key of the convention itself
        {"name": "CF", "versoin": "1.13"},
        # an option passed to a convention that takes none
        {"name": "dggs", "configuration": {"compact_form": True}},
    ],
)
def test_unknown_keys_rejected(selection) -> None:
    # a typo must be a loud error rather than silently dropped, which would
    # otherwise write metadata that differs from what was asked for
    with pytest.raises(ValidationError, match=".*[Ee]xtra inputs are not permitted.*"):
        MetadataSettings.model_validate({"zarr_conventions": [selection]})


def _write_healpix_group(path, metadata, healpix):
    """Write a minimal HEALPix group the way the conversion paths do."""
    import numpy as np

    root = zarr.create_group(str(path))
    write_root_conventions(root, metadata)

    group = root.create_group(
        "measurements/t2m",
        attributes=group_conventions_attrs(
            metadata, declare=("dggs",), dggs=healpix
        ),
    )

    n_cells = 12 * 4**healpix.refinement_level
    cell_ids = group.create_array(
        "cell_ids",
        shape=(n_cells,),
        dtype=np.uint64,
        dimension_names=("cells",),
        attributes=cf_cell_id_attrs(metadata),
    )
    cell_ids[:] = np.arange(n_cells, dtype=np.uint64)

    grid_mapping_attrs = cf_grid_mapping_attrs(metadata, healpix)
    if grid_mapping_attrs is not None:
        crs = group.create_array(
            "crs", shape=(), dtype=np.int8, attributes=grid_mapping_attrs
        )
        crs[...] = 0

    t2m = group.create_array(
        "t2m",
        shape=(n_cells,),
        dtype=np.float32,
        dimension_names=("cells",),
        attributes={
            "units": "K",
            "standard_name": "air_temperature",
            **cf_data_variable_attrs(metadata, healpix),
        },
    )
    t2m[:] = np.zeros(n_cells, dtype=np.float32)

    # the conversion paths consolidate before handing the store over
    zarr.consolidate_metadata(root.store)

    return root


def test_cf_grid_mapping_is_decoded_by_xarray(tmp_path, healpix) -> None:
    # emitting spec-conformant attributes is not the point: CF-aware readers
    # must be able to act on them. Guard the whole chain, not just the strings.
    import xarray as xr

    _write_healpix_group(tmp_path / "cf.zarr", MetadataSettings(), healpix)

    ds = xr.open_dataset(
        str(tmp_path / "cf.zarr"),
        group="measurements/t2m",
        engine="zarr",
        decode_coords="all",
    )

    # xarray resolves `grid_mapping` and `coordinates`, and promotes both the
    # grid mapping variable and the cell ids to coordinates
    assert "crs" in ds.coords
    assert "cell_ids" in ds.coords
    assert ds["t2m"].encoding["grid_mapping"] == CF_GRID_MAPPING_VARIABLE

    # the HEALPix grid is described per CF 1.13, ellipsoid included
    crs_attrs = ds["crs"].attrs
    assert crs_attrs["grid_mapping_name"] == "healpix"
    assert crs_attrs["refinement_level"] == healpix.refinement_level
    assert crs_attrs["indexing_scheme"] == "nested"
    assert crs_attrs["reference_ellipsoid_name"] == "WGS84"

    assert ds["cell_ids"].attrs["standard_name"] == "healpix_index"

    # the CF version applies to the whole dataset, so it lives in the root group
    assert xr.open_datatree(
        str(tmp_path / "cf.zarr"), engine="zarr"
    ).attrs["Conventions"] == f"CF-{CF_SUPPORTED_VERSIONS[-1]}"


def test_no_cf_metadata_when_cf_deselected(tmp_path, healpix) -> None:
    import xarray as xr

    metadata = MetadataSettings.model_validate({"zarr_conventions": ["dggs"]})
    _write_healpix_group(tmp_path / "no_cf.zarr", metadata, healpix)

    ds = xr.open_dataset(
        str(tmp_path / "no_cf.zarr"),
        group="measurements/t2m",
        engine="zarr",
        decode_coords="all",
    )

    assert "crs" not in ds.variables
    assert "cell_ids" not in ds.coords
    assert "grid_mapping" not in ds["t2m"].attrs
    assert "coordinates" not in ds["t2m"].attrs
    assert "grid_mapping" not in ds["t2m"].encoding
    assert "standard_name" not in ds["cell_ids"].attrs


@pytest.mark.parametrize(
    ("resampler", "expected"),
    [
        ("nearest", "area: point"),
        ("k-nearest", "area: point"),
        ("bilinear", "area: point"),
        ("cell-point", "area: point"),
        ("psf", "area: mean (point spread function weighted)"),
    ],
)
def test_cell_methods_per_resampler(resampler, expected) -> None:
    # the resampling method is otherwise invisible in the output: two
    # byte-different datasets from one input would be indistinguishable
    attrs = cf_data_variable_attrs(
        MetadataSettings(), resampler_name=resampler
    )

    assert attrs["cell_methods"] == expected


def test_cell_methods_unknown_resampler() -> None:
    with pytest.raises(ValueError, match=".*no CF cell_methods defined.*"):
        cf_cell_methods(MetadataSettings(), "not-a-resampler")


def test_cell_methods_covers_every_resampler() -> None:
    # a new resampler must declare what its values represent
    from healpix_convert.settings.common import ResamplerSettings

    names = {
        cls.model_fields["name"].default
        for cls in typing.get_args(ResamplerSettings)
    }

    assert names == set(CELL_METHODS)


def test_no_cell_methods_when_cf_deselected() -> None:
    metadata = MetadataSettings.model_validate({"zarr_conventions": ["dggs"]})

    assert cf_cell_methods(metadata, "nearest") is None
    assert "cell_methods" not in cf_data_variable_attrs(
        metadata, resampler_name="nearest"
    )
