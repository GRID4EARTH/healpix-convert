import math

import sparse
import torch
import torch.nn as nn


def sparse_to_torch(weights, device):
    reshaped = sparse.reshape(weights, (-1, weights.shape[-1])).tocoo()

    # csr
    # values = torch.from_numpy(reshaped.data).double()
    # indptr = torch.from_numpy(reshaped.indptr).long()
    # indices = torch.from_numpy(reshaped.indices).long()
    # coo
    coords = torch.from_numpy(reshaped.coords).long()
    values = torch.from_numpy(reshaped.data).double()

    return torch.sparse_coo_tensor(coords, values, size=reshaped.shape).coalesce().to_sparse_csr().to(device)


class HealpixToUTM(nn.Module):
    """
    Linear differentiable operator applying the precomputed mapping:
        data_utm = W @ hdata
    """

    def __init__(self, weights, utm_shape, device):
        """
        Parameters
        ----------
        weights : ndarray, shape (N * M, P)
            Weights for each (utm, hp) pair. Should be normalized per UTM pixel.
        utm_shape : tuple
            (N, M) shape of the UTM grid.
        K : int
            Number of HEALPix pixels (len(hidx_sorted)).
        """
        super().__init__()

        self.utm_shape = utm_shape
        self.n_cells = weights.shape[-1]

        self.register_buffer("weights", sparse_to_torch(weights, device))

    @property
    def n_utm_pixels(self):
        return math.prod(self.utm_shape)

    def forward(self, healpix_data):
        """
        Parameters
        ----------
        hdata : Tensor, shape (K,)
            Values on HEALPix pixels, ordered like hidx_sorted.

        Returns
        -------
        data_utm : Tensor, shape (N * M,)
            Reconstructed UTM image.
        """
        return torch.matmul(self.weights, healpix_data)


def interpolate_to_healpix(
    gaussian_weights,
    utm_values,
    initial_values,
    *,
    n_iter=500,
    device="cpu",
):
    model = HealpixToUTM(gaussian_weights, utm_shape=utm_values.shape, device=device)

    target_utm = torch.from_numpy(utm_values.ravel()).to(device)
    param = torch.nn.Parameter(torch.from_numpy(initial_values).double().to(device))

    optimizer = torch.optim.Adam([param], lr=1e-2)

    for iteration in range(n_iter):
        optimizer.zero_grad()

        # healpix → UTM
        prediction_utm = model(param)

        loss = torch.nansum((prediction_utm - target_utm) ** 2)
        loss.backward()

        optimizer.step()

        if iteration % 50 == 0:
            print(f"Iteration {iteration:04d}, loss = {loss.item():.6e}")

    optimized_data = param.detach().cpu().numpy().copy()

    return optimized_data
