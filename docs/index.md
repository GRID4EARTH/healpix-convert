# healpix-convert

`healpix-convert` converts various datasets into the
[HEALPix](https://healpix.sourceforge.io/)-based DGGS format proposed by
**GRID4EARTH**.

It takes input data stored in Zarr format (one or more EOPF-compliant Zarr
datasets), then reprojects and resamples it onto a HEALPix grid, outputting a
new Zarr dataset compliant with the DGGS Zarr and CF-HEALPix conventions.

:::{warning}
This package is under heavy development. The API is not stable yet.
:::

## Supported data products

- ERA5
- ClimateDT
- CAMS

## Key features

- **Multiple resampling methods** — nearest, bilinear, k-nearest neighbors,
  PSF, cell-point (provided by
  [healpix-resample](https://grid4earth.eu/healpix-resample/))
- **Multi-stage workflow** — first prepare the output dataset, then run data
  conversion in parallel jobs
- **HEALPix pyramids** — generate coarser refinement levels in a converted
  dataset, for viewers that pick the level from the zoom
- **Dask-native** — parallel conversion out of the box with a distributed Dask
  cluster
- **Cloud-ready** — read from and write to S3 object stores
- **STAC metadata** — input STAC discovery attributes are propagated and merged
  in the output
- **Zarr conventions** — [Zarr conventions](https://github.com/zarr-conventions)
  are used for structuring metadata in the output dataset

```{toctree}
---
maxdepth: 2
caption: User guide
hidden: true
---
installation
user-guide/index
tutorials/index
```

```{toctree}
---
maxdepth: 2
caption: Reference
hidden: true
---
api
terminology
whats-new
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Quickstart
:link: tutorials/quickstart
:link-type: doc
The conversion workflow step by step: settings, staging cache, output
preparation and conversion.
:::

:::{grid-item-card} User guide
:link: user-guide/index
:link-type: doc
Conversion settings, parallel conversion with Dask, S3 object stores and the
metadata written in the output dataset.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
Example notebooks converting ERA5, CAMS, ClimateDT and ESA CCI land cover data.
:::

:::{grid-item-card} API Reference
:link: api
:link-type: doc
Complete documentation for every public function and class.
:::

::::
