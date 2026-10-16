# User guide

How to configure and run conversions with `healpix-convert`. Start with the
{doc}`../tutorials/quickstart` for an overview of the conversion workflow.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Conversion settings
:link: settings
:link-type: doc
Output HEALPix grid, chunking, resampling methods and the other options of
each group.
:::

:::{grid-item-card} Parallel conversion with Dask
:link: dask
:link-type: doc
Convert the chunks of a group in parallel on a Dask distributed cluster.
:::

:::{grid-item-card} S3 object stores
:link: s3
:link-type: doc
Read the input datasets from and write the output dataset to S3.
:::

:::{grid-item-card} Output metadata
:link: metadata
:link-type: doc
DGGS Zarr, Multiscales and CF conventions, and the STAC metadata of the
output dataset.
:::

::::

```{toctree}
:maxdepth: 1

settings
dask
s3
metadata
```
