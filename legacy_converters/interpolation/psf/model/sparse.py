import sparse
import torch


def sparse_to_torch(weights, device, format="coo"):
    if weights.ndim != 2:
        reshaped = sparse.reshape(weights, (-1, weights.shape[-1]))
    else:
        reshaped = weights

    if reshaped.format in {"csr", "gcxs"}:
        sparse_coo = reshaped.tocoo()
    elif reshaped.format == "coo":
        sparse_coo = reshaped
    else:
        raise ValueError("unknown sparse format")

    values = torch.from_numpy(sparse_coo.data).double()
    coords = torch.from_numpy(sparse_coo.coords).long()

    torch_coo = torch.sparse_coo_tensor(
        coords, values, size=sparse_coo.shape
    ).coalesce()

    formats = {
        "coo": lambda x: x,
        "csr": lambda x: x.to_sparse_csr(),
        "gcxs": lambda x: x,
    }
    converter = formats.get(format)
    if converter is None:
        raise ValueError(f"unknown sparse format: {format}")

    return converter(torch_coo).to(device)
