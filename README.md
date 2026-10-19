# healpix-convert

This package allows converting various datasets into the
[HEALPix](https://healpix.sourceforge.io/)-based DGGS format proposed by
**GRID4EARTH**.

It takes input data stored in Zarr format (one or more EOPF-compliant Zarr
datasets), then reprojects and resamples it onto a HEALPix grid, outputting a
new Zarr dataset compliant with the DGGS Zarr and CF-Healpix conventions.

**Note: this repository is currently under heavy development. API is not stable yet.**

Documentation: <https://grid4earth.eu/healpix-convert/>

## Supported data products

- ERA5
- ClimateDT
- CAMS

## Key features

- **Multiple resampling methods** — nearest, bilinear, k-nearest neighbors, PSF, cell-point
- **Multi-stage workflow** — first prepare the output dataset, then run data conversion in parallel jobs
- **HEALPix pyramids** — generate coarser refinement levels in a converted dataset, for viewers that pick the level from the zoom
- **Dask-native** — parallel conversion out of the box with a distributed Dask cluster
- **Cloud-ready** — read from and write to S3 object stores
- **STAC metadata** — input STAC discovery attributes are propagated and merged in the output
- **Zarr conventions** — [Zarr conventions](https://github.com/zarr-conventions) are used for structuring metadata in the output dataset

## Installation

### From PyPI

```sh
$ python -m pip install healpix-convert
```

The ERA5, CAMS and ClimateDT converters need extra packages to download and
read their input data. Install them with the matching extra:

```sh
$ python -m pip install "healpix-convert[era5]"       # cdsapi, cfgrib
$ python -m pip install "healpix-convert[cams]"       # requests
$ python -m pip install "healpix-convert[climatedt]"  # cfgrib, polytope-client, destinelab
```

healpix-convert depends on PyTorch (through healpix-resample). On x86-64 Linux,
pip installs the CUDA build of PyTorch: with its NVIDIA libraries, that is
about 3 GB to download. For a CPU-only environment, install PyTorch from its
CPU index first:

```sh
$ python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
$ python -m pip install healpix-convert
```

There is no conda-forge package yet.

### From source (development)

First clone this repository:

```sh
$ git clone https://github.com/GRID4EARTH/healpix-convert
$ cd healpix-convert
```

Then install it, e.g., using pip:

```sh
$ python -m pip install .
```

Note: we recommend installing it in an isolated development environment.

If you have [pixi](https://pixi.prefix.dev) installed, you can install a
complete development environment by simply using the command below:

```sh
$ pixi install -e dev
```

## Basic usage

See the [quickstart](https://grid4earth.eu/healpix-convert/tutorials/quickstart.html)
in the documentation, and the examples in the `notebooks` folder.
