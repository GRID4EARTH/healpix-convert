import math

import torch
import torch.nn as nn

from legacy_converters.interpolation.psf.model.sparse import sparse_to_torch


class HealpixToUTM(nn.Module):
    """
    Linear differentiable operator applying the precomputed mapping:
        data_utm = W @ hdata
    """

    def __init__(self, weights, utm_shape, device, format):
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

        self.register_buffer("weights", sparse_to_torch(weights, device, format=format))

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
    learning_rate=1e-3,
    device="cpu",
    format="coo",
):
    model = HealpixToUTM(
        gaussian_weights, utm_shape=utm_values.shape, device=device, format=format
    )

    target_utm = torch.from_numpy(utm_values.ravel()).to(device)
    param = torch.nn.Parameter(torch.from_numpy(initial_values).double().to(device))

    optimizer = torch.optim.Adam([param], lr=learning_rate)

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
