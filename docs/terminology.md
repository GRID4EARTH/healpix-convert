# Terminology

## HEALPix grid

```{glossary}
HEALPix
  Hierarchical Equal Area isoLatitude Pixelization. Divides the sphere (or an
  ellipsoid) into equal-area cells identified by ``(refinement level, cell id)``.

DGGS
  Discrete Global Grid System: a hierarchy of grids covering the whole globe.
  HEALPix is the DGGS used by GRID4EARTH.

Refinement level
  HEALPix resolution parameter. At level ``k`` there are ``12 × 4^k`` cells.
  Range: 0 (12 cells) to 29.

Cell ID
  Integer identifier of a HEALPix cell. Its meaning depends on the indexing
  scheme (*nested*, *ring* or *zuniq*).

Nested scheme
  HEALPix indexing where the ids of the children cells are derived from the id
  of their parent cell. Required for chunking the output along HEALPix cells.

Ring scheme
  HEALPix indexing ordered by iso-latitude rings.

Zuniq scheme
  Unique identifier encoding both the refinement level and the cell id in a
  single integer.

WGS84
  World Geodetic System 1984: the reference ellipsoid of GPS and of the
  EPSG:4326 coordinate reference system.
```

## Conversion

```{glossary}
Spatial group
  A group of the input Zarr datasets with spatial data (projected x / y or
  longitude / latitude coordinates), converted to HEALPix.

Staging cache
  The information parsed from the input datasets and the conversion settings
  (spatial information, groups to convert, output extent), reused by all the
  conversion steps. See {py:class}`~healpix_convert.ConvertStagingCache`.

Chunk cell
  A HEALPix cell at a coarser refinement level than the output data, whose
  children cells form one chunk of the output arrays.

Chunk buffer
  Width added around a chunk cell when selecting the input data points to
  resample onto the cells of that chunk.

Multiscale group
  A group of the input Zarr dataset(s) holding several children "scale" groups,
  each converted at its own HEALPix refinement level (see
  `multiscale_settings` in {doc}`user-guide/settings`).

Pyramid multiscale group
  A group of the output Zarr dataset holding the levels converted from the
  input data (native levels) and the coarser levels generated from them
  (pyramid levels). It is the converted multiscale group itself, or a sibling
  of it. See {py:func}`~healpix_convert.generate_pyramids`.

Pyramid level group
  A group of a pyramid multiscale group holding one generated level, named
  after its refinement level, sibling of the native level groups.

Native level
  A refinement level of a multiscale group that is converted from the input
  data, as opposed to the coarser levels generated from it.

Parent cell
  The HEALPix cell of a coarser refinement level that contains a given cell
  (its ancestor in the *nested* scheme). Each cell of a generated level of a
  pyramid aggregates all the converted cells it is the ancestor of.

Resampler
  The method that computes the values of the output HEALPix cells from the
  input data points (nearest, bilinear, k-nearest neighbors, PSF, cell-point),
  provided by healpix-resample.
```

## Metadata

```{glossary}
EOPF
  ESA's Earth Observation Processing Framework, whose Zarr products are the
  main input of healpix-convert.

STAC
  SpatioTemporal Asset Catalog. The ``stac_discovery`` attribute of the input
  and output datasets is a STAC item.

Zarr conventions
  Community conventions for structuring metadata in Zarr datasets
  (https://github.com/zarr-conventions). healpix-convert writes the DGGS and
  Multiscales conventions.

CF conventions
  Climate and Forecast metadata conventions. Version 1.13 defines a HEALPix
  grid mapping.
```
