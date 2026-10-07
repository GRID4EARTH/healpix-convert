# healpix-convert

This package allows converting various datasets into the
[HEALPix](https://healpix.sourceforge.io/)-based DGGS format proposed by
**GRID4EARTH**.

It takes input data stored in Zarr format (one or more EOPF-compliant Zarr
datasets), then reprojects and resamples it onto a HEALPix grid, outputting a
new Zarr dataset compliant with the DGGS Zarr and CF-Healpix conventions.

**Note: this repository is currently under heavy development. API is not stable yet.**

## Supported data products

- ERA5
- ClimateDT
- CAMS

## Key features

- **Multiple resampling methods** — nearest, bilinear, k-nearest neighbors, PSF, cell-point
- **Multi-stage workflow** — first prepare the output dataset, then run data conversion in parallel jobs
- **Dask-native** — parallel conversion out of the box with a distributed Dask cluster
- **Cloud-ready** — read from and write to S3 object stores
- **STAC metadata** — input STAC discovery attributes are propagated and merged in the output
- **Zarr conventions** — [Zarr conventions](https://github.com/zarr-conventions) are used for structuring metadata in the output dataset

## Installation

### From PyPI or conda-forge

There's no package published on PyPI and/or conda-forge yet.

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

See examples in the ``notebooks`` folder.
