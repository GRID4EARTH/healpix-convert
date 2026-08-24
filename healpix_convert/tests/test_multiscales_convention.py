from healpix_convert.core.multiscales_conventions import MultiscalesZarrConvention


def test_multiscales_convention_values() -> None:
    c = MultiscalesZarrConvention()

    assert c.uuid == "d35379db-88df-4056-af3a-620245f8e347"
    assert c.name == "multiscales"
    assert (
        c.schema_url
        == "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v1/schema.json"
    )
    assert (
        c.spec_url
        == "https://github.com/zarr-conventions/multiscales/blob/v1/README.md"
    )
    assert c.description == "Multiscale layout of zarr datasets"
