from legacy_converters.interpolation.psf.model import adam, conjugate_gradient


def interpolate_to_healpix(
    weights,
    utm_values,
    initial_values,
    *,
    optimizer: str = "adam",
    device: str = "cpu",
    format: str = "coo",
    **additional_params,
):
    optimizers = {
        "adam": adam.interpolate_to_healpix,
        "cg": conjugate_gradient.interpolate_to_healpix,
    }

    optimizer_func = optimizers.get(optimizer)
    if optimizer_func is None:
        raise ValueError(
            f"unknown optimizer: {optimizer!r}. Choose one of {', '.join(optimizers)}"
        )

    return optimizer_func(
        weights,
        utm_values,
        initial_values,
        device=device,
        format=format,
        **additional_params,
    )
