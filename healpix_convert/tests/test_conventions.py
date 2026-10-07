import pytest
import zarr.api.synchronous as zarr
from pydantic import ValidationError

from healpix_convert.core.healpix_conventions import Healpix
from healpix_convert.core.metadata import (
    CF_GRID_MAPPING_VARIABLE,
    get_cf_cell_id_attrs,
    get_cf_data_variable_attrs,
    get_cf_grid_mapping_attrs,
    get_healpix_group_attrs,
    get_multiscale_group_attrs,
    write_root_conventions,
)
from healpix_convert.core.multiscales_conventions import (
    HealpixMultiscales,
    HealpixMultiscalesLayoutItem,
)
from healpix_convert.settings.conventions import (
    CF_MIN_VERSION,
    CFConventionSelector,
    DGGSConventionSelector,
    MetadataSettings,
    MultiscalesConventionSelector,
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

    assert [selector.name for selector in metadata.zarr_conventions] == [
        "dggs",
        "multiscales",
        "cf",
    ]


def test_metadata_settings_shorthand_and_name_case() -> None:
    metadata = MetadataSettings.model_validate(
        {"zarr_conventions": ["dggs", "multiscales", "CF"]}
    )

    assert isinstance(metadata.get("dggs"), DGGSConventionSelector)
    assert isinstance(metadata.get("multiscales"), MultiscalesConventionSelector)
    # convention names are case-insensitive
    assert isinstance(metadata.get("cf"), CFConventionSelector)
    assert metadata.has("CF")


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

    cf = metadata.get("cf")
    assert cf is not None
    assert cf.version == "1.13"
    assert cf.configuration.healpix_grid_mapping


def test_convention_selector_omits_unset_version_and_configuration() -> None:
    # the conventions that define neither must not serialize an empty version
    # or configuration (Zarr's extension metadata model leaves both optional)
    assert DGGSConventionSelector().model_dump() == {"name": "dggs"}
    assert MultiscalesConventionSelector().model_dump() == {"name": "multiscales"}


def test_metadata_settings_reject_duplicates() -> None:
    with pytest.raises(ValidationError, match="duplicate convention.*dggs"):
        MetadataSettings.model_validate({"zarr_conventions": ["dggs", "dggs"]})


def test_metadata_settings_reject_unknown_convention() -> None:
    with pytest.raises(ValidationError):
        MetadataSettings.model_validate({"zarr_conventions": ["eopf"]})


def test_cf_selector_reject_older_version() -> None:
    with pytest.raises(ValidationError, match="unsupported CF version '1.9'"):
        CFConventionSelector.model_validate({"name": "cf", "version": "1.9"})


def test_cf_selector_accepts_newer_version() -> None:
    # newer CF versions keep the HEALPix grid mapping, so they need no change here
    assert CFConventionSelector.model_validate({"version": "1.14"}).version == "1.14"
    assert CFConventionSelector.model_validate({"version": "2.0"}).version == "2.0"


def test_cf_selector_reject_invalid_version() -> None:
    with pytest.raises(ValidationError, match="invalid version number '1.x'"):
        CFConventionSelector.model_validate({"version": "1.x"})


def test_cf_attrs_without_cf(healpix) -> None:
    metadata = MetadataSettings.model_validate({"zarr_conventions": ["dggs"]})

    assert not metadata.has("cf")
    assert get_cf_cell_id_attrs(metadata) == {}
    assert get_cf_grid_mapping_attrs(healpix, metadata) is None
    assert get_cf_data_variable_attrs(healpix, metadata) == {}


def test_cf_attrs_without_grid_mapping(tmp_path, healpix) -> None:
    metadata = MetadataSettings.model_validate(
        {
            "zarr_conventions": [
                {"name": "cf", "configuration": {"healpix_grid_mapping": False}}
            ]
        }
    )

    # CF is still declared, but no grid mapping variable is written
    group = zarr.create_group(str(tmp_path / "cf_no_grid_mapping.zarr"))
    write_root_conventions(group, metadata)

    assert group.attrs["Conventions"] == "CF-1.13"
    assert get_cf_grid_mapping_attrs(healpix, metadata) is None
    assert get_cf_cell_id_attrs(metadata) == {}
    assert get_cf_data_variable_attrs(healpix, metadata) == {}


def test_get_healpix_group_attrs(healpix) -> None:
    attrs = get_healpix_group_attrs(healpix, MetadataSettings())

    assert [c["name"] for c in attrs["zarr_conventions"]] == ["dggs"]
    assert attrs["dggs"] == healpix.model_dump()
    assert "multiscales" not in attrs


def test_get_multiscale_group_attrs(multiscales) -> None:
    attrs = get_multiscale_group_attrs(multiscales, MetadataSettings())

    # the DGGS convention is declared as well: it applies to the children groups
    assert [c["name"] for c in attrs["zarr_conventions"]] == ["multiscales", "dggs"]
    assert attrs["multiscales"] == multiscales.model_dump()
    assert "dggs" not in attrs


def test_group_attrs_deselected(healpix, multiscales) -> None:
    metadata = MetadataSettings.model_validate({"zarr_conventions": ["cf"]})

    assert get_healpix_group_attrs(healpix, metadata) == {}
    assert get_multiscale_group_attrs(multiscales, metadata) == {}


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
        attributes=get_healpix_group_attrs(healpix, metadata),
    )

    n_cells = 12 * 4**healpix.refinement_level
    cell_ids = group.create_array(
        "cell_ids",
        shape=(n_cells,),
        dtype=np.uint64,
        dimension_names=("cells",),
        attributes=get_cf_cell_id_attrs(metadata),
    )
    cell_ids[:] = np.arange(n_cells, dtype=np.uint64)

    grid_mapping_attrs = get_cf_grid_mapping_attrs(healpix, metadata)
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
            **get_cf_data_variable_attrs(healpix, metadata),
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
    assert (
        xr.open_datatree(str(tmp_path / "cf.zarr"), engine="zarr").attrs["Conventions"]
        == f"CF-{CF_MIN_VERSION}"
    )


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
