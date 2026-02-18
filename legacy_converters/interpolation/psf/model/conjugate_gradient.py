from collections.abc import Callable

import torch

from legacy_converters.interpolation.psf.model.sparse import sparse_to_torch


@torch.no_grad()
def conjugate_gradient(
    A_mv: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: torch.Tensor,
    max_iter: int = 200,
    tolerance: float = 1e-7,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x = x0.clone()

    r = b - A_mv(x)

    b_norm = torch.linalg.norm(b)
    if b_norm == 0:
        return x, {
            "residual_norms": torch.tensor([0.0], device=b.device, dtype=b.dtype),
            "iters": torch.tensor(0, device=b.device),
        }

    p = r.clone()

    rs_old = torch.dot(r, r)
    residual_norms = [torch.sqrt(rs_old)]

    for iteration in range(max_iter):
        Ap = A_mv(p)
        denom = torch.dot(p, Ap)
        if torch.abs(denom) < 1e-30:
            # breakdown, shouldn't happen for SPD unless numerical issues
            break

        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)

        residual_norms.append(torch.sqrt(rs_new))

        if residual_norms[-1] <= tolerance * b_norm:
            rs_old = rs_new
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    info = {
        "residual_norms": torch.stack(residual_norms),
        "iters": torch.tensor(len(residual_norms) - 1, device=b.device),
    }
    return x, info


@torch.no_grad()
def least_squares_conjugate_gradient(
    weights: torch.Tensor,
    y: torch.Tensor,
    x0: torch.Tensor,
    max_iter: int = 200,
    tolerance: float = 1e-7,
    damp: float = 0.0,
):
    r"""
    Solve $min \| M x - y \|_2^2$ using conjugate gradients on normal equations:
    $$
    (M^T M + \lambda I) x = M^T y
    $$
    without explicitly computing $M^T M$.

    Parameters
    ----------
    x : numpy.ndarray
        The solution.
    """
    weights_transposed = weights.T.coalesce().to_sparse_csr()
    weights_ = weights.coalesce().to_sparse_csr()
    b = weights_transposed @ y

    def A_mv(v):
        return weights_transposed @ (weights_ @ v) + damp * v

    x, info = conjugate_gradient(
        A_mv, b=b, x0=x0, max_iter=max_iter, tolerance=tolerance
    )

    return x


def interpolate_to_healpix(
    weights,
    utm_values,
    initial_values,
    *,
    max_iter: int = 200,
    tolerance: 1e-7,
    device: str = "cpu",
    format: str = "coo",
):
    weights = sparse_to_torch(weights, device, format=format)
    target_utm = torch.from_numpy(utm_values.ravel()).to(device)
    initial_values = torch.from_numpy(initial_values.ravel()).to(device)

    optimized_data = least_squares_conjugate_gradient(
        weights,
        target_utm,
        initial_values,
        max_iter=max_iter,
        tolerance=tolerance,
    )

    return optimized_data.detach().cpu().numpy().copy()
