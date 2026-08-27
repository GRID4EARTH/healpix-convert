import numpy as np
import pytest
import zarr

from healpix_convert.core.healpix_conventions import Healpix, WGS84Ellipsoid
from healpix_convert.healpix_converters import DenseChunkConverter


@pytest.fixture
def store():
    from zarr.storage import MemoryStore

    return MemoryStore({})


@pytest.mark.xfail(reason="incomplete test, needs a lot of setup")
def test_spatial_coordinates_as_coordinates(store):
    root = zarr.open_group(store)
    grp = root.require_group("measurements/reflectance/17")
    grp.create_array(
        "cell_ids", shape=(12,), dtype=np.uint64, dimension_names=("cells",)
    )

    variable_names = ("var1", "var2", "var3")

    for name in variable_names:
        grp.create_array(
            name, shape=(12,), dtype=np.float32, dimension_names=("cells",)
        )

    healpix = Healpix(
        refinement_level=5, indexing_scheme="nested", ellipsoid=WGS84Ellipsoid()
    )

    chunk_converter = DenseChunkConverter(
        path="/measurements/reflectance/17",
        output_store=store,
    )

    actual = {
        name: array.attrs.get("coordinates")
        for name, array in chunk_converter.output_arrays.items()
        if name in variable_names
    }
    expected = {name: healpix.coordinate for name in variable_names}

    assert actual == expected
