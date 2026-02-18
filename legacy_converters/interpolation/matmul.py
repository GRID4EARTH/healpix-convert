import xarray as xr


def matmul(obj: xr.Dataset | xr.DataArray, weights: xr.DataArray, *, dims: list[str]):
    """Sparse matrix multiplication between a sparse matrix and a vector

    Parameters
    ----------
    obj : xr.Dataset or xr.DataArray
        The input data.
    weights : xr.DataArray
        The array of weights.
    dims : list of str
        The dimensions to perform the ``einsum`` over. This is necessary if
        ``obj`` has more than these dimensions.
    """

    def _matmul_variable(var, weights):
        sparse_result = xr.dot(weights.variable.astype(var.dtype), var, dim=dims)
        if hasattr(sparse_result.data, "todense"):
            dense_result = sparse_result.copy(data=sparse_result.data.todense())
        else:
            dense_result = sparse_result

        return dense_result

    new_coords = obj.coords.drop_dims(dims) | weights.coords.drop_dims(dims)

    if isinstance(obj, xr.DataArray):
        result = _matmul_variable(obj.variable, weights)
        return xr.DataArray(result, coords=new_coords)

    to_drop = [name for name, var in obj.coords.items() if set(dims).issubset(var.dims)]
    return (
        obj.drop_vars(to_drop)
        .map(lambda arr: _matmul_variable(arr.variable, weights))
        .assign_coords(new_coords)
    )
