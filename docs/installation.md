# Installation

## Requirements

- Python **≥ 3.11**

## Install from PyPI

::::{tab-set}

:::{tab-item} pip

```bash
pip install healpix-convert
```

:::

:::{tab-item} uv

```bash
uv add healpix-convert
```

:::

::::

`healpix-convert` is not available on conda-forge yet.

### Dataset converters

The ERA5, CAMS and ClimateDT converters need extra packages to download and
read their input data. Install them with the matching extra:

```bash
pip install "healpix-convert[era5]"       # cdsapi, cfgrib
pip install "healpix-convert[cams]"       # requests
pip install "healpix-convert[climatedt]"  # cfgrib, polytope-client, destinelab
```

### PyTorch

`healpix-convert` depends on [PyTorch](https://pytorch.org), which the
resampling methods of
[healpix-resample](https://grid4earth.eu/healpix-resample/) run on. On x86-64
Linux, pip installs the CUDA build of PyTorch: with its NVIDIA libraries, that
is about 3 GB to download. For a CPU-only environment, install PyTorch from its
CPU index first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install healpix-convert
```

The [PyTorch installation guide](https://pytorch.org/get-started/locally/)
lists the other builds.

## Development install

Clone the repository:

```bash
git clone https://github.com/GRID4EARTH/healpix-convert
cd healpix-convert
```

If you have [pixi](https://pixi.prefix.dev) installed, create the complete
development environment with:

```bash
pixi install -e dev
```

Otherwise, install the package in editable mode in an isolated environment of
your choice:

```bash
python -m pip install -e .
```

:::{note}
`pixi.toml` declares the `linux-64` and `win-64` platforms with CUDA 12. On a
Linux or Windows machine without an NVIDIA driver, pixi then refuses to
install the environments ("Unsatisfied requirements: `__cuda >= 12.0`"). Set
`CONDA_OVERRIDE_CUDA=12.0` to work around it, like the CI does:

```bash
CONDA_OVERRIDE_CUDA=12.0 pixi install -e dev
```

:::

### Run the tests

```bash
pixi run -e dev tests
```

### Build the documentation

```bash
pixi run -e docs build-docs
```

The HTML pages are written in `docs/_build/html`. Use
`pixi run -e docs autobuild-docs` for a live preview that rebuilds on every
change.

## Verify the installation

```python
import healpix_convert

print(healpix_convert.__version__)
```
