import numpy as np
import pytest

from healpix_convert.healpix_converters import _maybe_coerce_str_array_dtype


@pytest.mark.parametrize(
    ["values", "expected_kind"],
    [
        pytest.param(np.array(["b02", "b03"], dtype=object), "T", id="object-strings"),
        pytest.param(np.array(["b02", None], dtype=object), "O", id="object-mixed"),
        pytest.param(np.arange(2.0), "f", id="float"),
    ],
)
def test_maybe_coerce_str_array_dtype(values, expected_kind):
    actual = _maybe_coerce_str_array_dtype(values)

    assert actual.dtype.kind == expected_kind
    assert actual.tolist() == values.tolist()
