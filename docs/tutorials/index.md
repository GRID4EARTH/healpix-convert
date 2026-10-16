# Tutorials

The tutorials are the notebooks of the
[`notebooks`](https://github.com/GRID4EARTH/healpix-convert/tree/main/notebooks)
folder of the repository. They are shown here without being run: running them
needs access to the input data (CDS, ADS or Polytope credentials, or a local
copy of the ESA CCI land cover map) and large downloads.

The ERA5, CAMS and ClimateDT notebooks use the product converters of
`healpix_convert.converters`, which download the data and convert it in three
steps (`prepare`, `convert_group` for each chunk, `consolidate`). The ESA CCI
land cover notebook uses the generic conversion functions described in the
{doc}`quickstart`.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Quickstart
:link: quickstart
:link-type: doc
The conversion workflow step by step: settings, staging cache, output
preparation and conversion in parallel jobs.
:::

:::{grid-item-card} ERA5
:link: notebooks/era5_converter
:link-type: doc
ERA5 reanalysis surface fields (Gaussian grids) to HEALPix level 7, downloaded
from the Copernicus Climate Data Store.
:::

:::{grid-item-card} CAMS
:link: notebooks/cams_converter
:link-type: doc
CAMS EAC4 aerosol reanalysis (regular 0.75° grid) to HEALPix level 6,
downloaded from the Atmosphere Data Store.
:::

:::{grid-item-card} ClimateDT
:link: notebooks/climatedt_converter
:link-type: doc
Destination Earth ClimateDT output, natively on HEALPix level 7, retrieved
with Polytope.
:::

:::{grid-item-card} ESA CCI land cover
:link: notebooks/esacci_land_cover_converter
:link-type: doc
A categorical land cover map (GeoTIFF) to WGS84 HEALPix level 15 with
nearest-neighbor resampling.
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

quickstart
notebooks/era5_converter
notebooks/cams_converter
notebooks/climatedt_converter
notebooks/esacci_land_cover_converter
```
