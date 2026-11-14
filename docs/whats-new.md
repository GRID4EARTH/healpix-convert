# What's new

## Unreleased

- A multiscale group converted from its own variables can be converted at
  several levels, each with settings of its own: the settings of the group give
  one level, and settings under `<group>/<level>` the others. Before, only the
  first level group got the variables and the others were converted from an
  empty group. The `drop_variables` of all the levels apply to the group, and
  the levels must agree on its `coordinates_group`. See
  {doc}`user-guide/pyramids`.
- New group settings `drop_variables` (variables of the input group that are
  not written, e.g. the projected distances `x` / `y` of a swath grid) and
  `nearest_variables` (variables written as the value of the nearest input point
  of each cell, in their own data type, at the converted level only, e.g. the
  row / column indices of a swath pixel or the time of its row). Before, every
  variable of a spatial group was resampled with the group's resampler and cast
  to `output_dtype`: indices came out as interpolated floats and a time in
  nanoseconds stored as float32 lost about 100 s. Without the settings nothing
  changes. See {doc}`user-guide/settings`.
- A group without latitude / longitude coordinates of its own can take those of
  another group of the input with the new `coordinates_group` group setting,
  e.g. quality flags stored next to the measurements of the same image grid, or
  angles on the tie-point grid of another group. Before, such a group was not
  spatial and was left out of the output. See {doc}`user-guide/settings`.
- A multiscale group can be converted from its own variables: an input group
  that holds its variables directly (a single resolution, no child group per
  resolution) gets its settings under its own path and is listed as a
  multiscale group, and its variables are converted into a level group named
  after the refinement level (`measurements` at level 15 gives
  `measurements/15`), with the derived levels of its pyramid next to it. Before,
  these settings were refused ("multiscale group ... has no child group to
  convert to HEALPix"). See {doc}`user-guide/pyramids`.

## v2026.11.0 (2026-11-09)

- The ClimateDT converter now always resamples the native (spherical) HEALPix
  level 7 fields onto the level 7 grid on the WGS84 ellipsoid, the grid its
  output declares; the two grids are up to about 14 km apart. Before, it copied
  the spherical values by default, and with `ellipsoid_correction=True` it
  wrote the values of the first cells of the sphere into every chunk and
  resampled the whole sphere again for each chunk. The `ellipsoid_correction`
  parameter is removed. The resampler is PSF for every variable for now, with
  the ERA5 settings (`threshold=0.5`, `lam=0`, clipped to the source range),
  run once per variable and timestep; masked (NaN) source cells stay NaN. A
  variable with a dimension besides time and cells (e.g. ocean depth levels)
  raises `NotImplementedError`; before, only its first level was written.
  Reported by Anne Fouilloux.
- The pyramids can be written with a `sharding_indexed` serializer in the
  `codecs` pyramid setting: the inner chunks are cut to the chunks of the
  coarsest derived levels where those hold fewer cells. Before, the conversion
  failed at the end of the pyramid stage with a `ValueError`.
- The cell ids of the derived levels are written with the codecs of the cell ids
  of the converted level (e.g. a delta filter set by the
  `healpix_coordinate_codecs` group setting), sharded like them. Before, they
  had Zarr's default codecs, and took far more space than the converted ones.
- New `"grid-bilinear"` resampler for input groups on a rectilinear lon/lat
  grid, converted by dense chunks: each cell is interpolated from the 4 grid
  nodes around its centre (healpix-resample's `GridBilinearResampler`), instead
  of the inverse-distance mean of the nearby input points of `"bilinear"`. It
  is exact for a field linear in longitude and latitude, and much faster and
  lighter on large grids. Inputs side by side at the same latitudes (e.g. the
  coarse meteorology grids of neighbouring tiles) are interpolated each on its
  own and averaged where they overlap. It is only available with a
  healpix-resample version that provides `GridBilinearResampler`.
- New `chunk_level_offset` and `shard_level` pyramid settings, and a
  `chunk_level_offset` group setting: one layout rule for every level of a
  group. A chunk holds `4 ** chunk_level_offset` cells (the children of one
  cell of the level `level - chunk_level_offset`) and a shard the chunks of one
  cell of the level `shard_level`, with no shard where it would hold a single
  chunk. Before, the chunks of a derived level followed the chunks of the
  converted levels, whose shards they could be as large as. With
  `chunk_level_offset`, `codecs` are the codecs of the chunks inside the shards
  and must not contain a `sharding_indexed` codec of their own. Without these
  settings the layout is unchanged.
- New `finalize_healpix_dataset` conversion stage, the last one, run once every
  group of the output dataset has been written: it consolidates the Zarr
  metadata of the output dataset. No other stage writes consolidated metadata
  anymore (`prepare_healpix_dataset` and `generate_pyramids` did): such a copy
  of the metadata of a whole hierarchy is stale as soon as a group or an array
  is written next to it. A conversion run in one call with
  `create_healpix_dataset` runs that stage as well.
- New `consolidate` metadata settings, which select the groups the consolidated
  Zarr metadata is written for: the root group (`root_group`), each group
  converted to HEALPix (`healpix_groups`), each multiscale group
  (`multiscale_groups`), each group holding a derived level of a pyramid
  (`derived_level_groups`) and any other group of the output dataset
  (`groups`). All four flags are set by default; before, only the root group
  was consolidated. A `groups` path that names no group of the output dataset
  is reported with a warning when the staging cache is created, and skipped
  when the metadata is consolidated.
- Minimum versions raised to healpix-resample 2026.10.2 (its `batch_size`
  option, used by the `"dem-glo30"` settings), healpix-geo 0.4.1 and, for
  development, healpix-plot 2026.7.0.
- New `output_dtype` group setting, the floating-point data type of the resampled
  data variables (e.g. `"float16"`): before, the `"bilinear"` and `"psf"`
  resamplers always wrote `float64`. The derived levels of a pyramid follow the
  data type of the converted level.
- New `"bitwise_or"` reduction for the pyramids (the `reduce` pyramid setting):
  a bit is set in a coarser cell if it is set in any of its cells, for bit
  flags (integer variables where each bit is a flag of its own).
- The input datasets of one group may now have spatial dimensions of different
  sizes, e.g. grids of different spacing next to each other: their points are
  resampled together. Before, such inputs were refused as "incompatible spatial
  info"; they still need the same spatial dimension and coordinate names.
- Projected input datasets whose x or y coordinates run the other way than
  usual (increasing y, i.e. a bottom-up raster, or decreasing x) are now
  converted: the input points of a chunk are selected with slice bounds that
  follow the signs of the transform. Before, the selection was empty for every
  chunk of such a dataset.
- A chunk whose (buffered) polygon only grazes the extent of a projected input
  dataset, with no pixel centre in the overlap, no longer gives the resampler
  zero points: that dataset is skipped for the chunk, and a chunk left without
  input is not written ({issue}`23`). The conversion of a tile stopped there
  with `nearest` and with healpix-resample < 2026.10.1 for `psf`.
- New `"mode"` reduction for the pyramids (the `reduce` pyramid setting): the
  most frequent value of the cells of a block, for categorical variables such
  as masks and classifications.
- New `codecs` pyramid setting: the Zarr codecs of the aggregated variables of
  the derived levels (Zarr's defaults if not set), e.g. a serializer of
  `healpix-compress` coding one HEALPix cell per chunk. It is checked when the
  settings are created, like the `codecs` group setting.
- The ERA5 converter writes several analysis times in one store: `date` and
  `time` take MARS lists and ranges (`date="20260801/to/20260831"`,
  `time="00:00:00/06:00:00/12:00:00/18:00:00"`), each group gets a `time`
  coordinate, and the new `time_chunk` option sets how many times a chunk
  holds (`"all"`: the time series of 1024 cells in one chunk). The enda group
  gets the 3-hourly times only, the times of the ERA5 EDA. New public
  `download()`, `dataset()` and `grib_paths()` methods give the GRIB files
  without writing a store; a single date and time give the same store and
  cache file names as before.
- The chunks of a derived level always hold all the children of one HEALPix
  cell: on a partial spatial extent, their cells are no longer coarser than the
  coarsest level the converted cells cover entirely. Before, the chunks of the
  derived levels whose chunk cells were partly covered were cut along the array
  and mixed the cells of several HEALPix cells.
- New `extra_dimensions` group setting, to align several input datasets along
  a non-spatial dimension whose labels differ between them (e.g. the
  `detector` dimension of the Sentinel-2 `conditions/geometry` group): union,
  intersection, same labels, or fixed labels for every conversion
  ({issue}`17`, {issue}`54`). Input datasets that differ along a non-spatial
  dimension without such a setting are now refused with a message pointing to
  it; before, inputs of the same size were concatenated position by position
  even when their labels differed.
- Opt-in GPU path: `HEALPIX_CONVERT_DEVICE` (e.g. `cuda`) is the torch device
  of the resamplers whose settings give none, and the lon/lat of transverse
  Mercator inputs (EPSG method 9807, metres) are computed on it; they differ
  from pyproj's in the last bits, so the results are not bit-identical to a
  run without the variable. `HEALPIX_CONVERT_GPU_SLOTS=n` limits how many
  processes use the device at once, for the resamplers and the lon/lat
  computation alike ({pull}`100`).
- New `"dem-glo30"` settings and example notebook for the Copernicus DEM
  GLO-30: bilinear interpolation of the grid nodes (`"grid-bilinear"`) to
  level 18 in float32, sharded (one shard per level-6 cell, inner chunks of
  one level-8 cell), a pyramid of block means down to level 8,
  and the licence of the data in the output STAC item. The notebook reads the
  heights from Earth Data Hub or from the CDSE COG tiles (`SOURCE`). Based on
  the GLO-30 notebook by Pablo Richard.
- The multiscales metadata describes each derived level of a pyramid: the
  converted level it is aggregated from (`derived_from`), the transformation
  from it (`transform`, whose `scale` is how much longer a cell is) and the
  method its cells are aggregated with
  (`resampling_method`, also given for the whole group).
- A derived level whose variables are aggregated from different native levels
  is described by one layout item per array, whose assets are the paths of the
  arrays (e.g. `"1/a"` aggregated from `"3/a"` and `"1/b"` from `"2/b"`)
  instead of the paths of the level groups. Before, such a level was described
  by a single item naming one of those native levels, which misdescribed the
  variables aggregated from the others (`derived_from`) and how much their
  cells were aggregated (`transform`). The Multiscales Zarr convention asks
  for the same referencing style throughout a layout, so the whole multiscale
  group is then described array by array, the native levels included.
- The HEALPix grid of a level is no longer added to its multiscales layout
  items by default (`add_healpix_positioning`): it repeats the whole grid in
  every item of that level, which one item per array makes particularly
  verbose, while the attributes of a level group already describe its grid and
  the `asset` of an item names its refinement level.
- Neighbour links store: with `HEALPIX_CONVERT_NEIGHBOUR_STORE` set to a URL
  (storage options as JSON in `HEALPIX_CONVERT_NEIGHBOUR_STORE_OPTIONS`), the
  links of the neighbour searches are kept there by geometry, and later
  resamplers on the same geometry are built from them without a new search.
  The key of a resampler covers the files of the imported healpix-resample
  package, not only its version string, and the torch and healpix-geo
  versions, so links found by other code are never reused. It does not cover
  the device of the search: machines with and without a GPU sharing a store
  all use the links of the first search. healpix-resample 2026.11.1 or later
  is required ({pull}`99`).
- New `asset_type` multiscale setting: what a layout item refers to, a level
  group (`"group"`) or one array of a level (`"array"`). With `"auto"` (the
  default), the level groups are the assets where they describe every level
  and the arrays otherwise; `"group"` rejects a pyramid whose derived level
  would aggregate its arrays from different native levels.
- New `generate_pyramids` conversion stage, which aggregates the converted
  data onto the coarser HEALPix refinement levels (the derived levels) given by
  the new `pyramid_groups` multiscale settings. One group per derived level,
  and the multiscales metadata describing the pyramid, are created with the
  rest of the structure of the output dataset by `prepare_healpix_dataset`; the
  arrays of those groups are written with the aggregated data, and the
  converted levels are never rewritten. The groups must be converted with the
  `healpix_cell_dense` chunk method: the cells of a coarser cell are then
  consecutive cells of the converted arrays, so the levels are aggregated by
  position without reading any cell id.
- `generate_pyramids(in_memory=True)` reads the output dataset once, computes
  the mean pyramids in numpy and writes the derived levels at once; the
  arrays are decoded and encoded in a pool of processes
  (`HEALPIX_CONVERT_PYRAMID_PROCESSES`). Opt-in; same results as the default
  path up to the order of the sums ({pull}`98`).
- The `asset` of a multiscales layout item is now a path relative to the group
  holding the metadata (e.g. `"7"` instead of `"measurements/7"`), as the
  Multiscales Zarr convention asks.
- Resamplers can be reused between the groups on the same input grid: with
  `HEALPIX_CONVERT_RESAMPLER_CACHE` set to a directory private to the job, the
  resamplers of the methods in `HEALPIX_CONVERT_RESAMPLER_CACHE_METHODS`
  (default `"nearest"`) are pickled there and loaded by later groups with the
  same input points, output cells and parameters. Off by default ({pull}`94`).
- The attributes of the output variables can be set from the group settings
  (`metadata.override_attrs`), and not only excluded ({issue}`36`).
- The STAC properties, extensions, links and assets propagated to the output
  dataset can be excluded and set from the conversion settings
  (`metadata.stac`). The STAC metadata written by `healpix-convert` itself
  keeps precedence, except for the recorded conversion settings, which
  `metadata.stac.include_conversion_settings` leaves out of the output. The
  input's assets are no longer propagated by default: they point at the input
  files, which the output does not hold.
- With several input datasets, the STAC properties, extensions, links and
  assets of the output can be merged as the intersection of the input ones,
  and the extensions and links also as their union
  (`metadata.stac.merge_properties`, `merge_extensions`, `merge_links` and
  `merge_assets`). They were only ever taken from the first input dataset,
  which stays the default -- except for the extensions, which used to fail the
  conversion when they differed across the inputs.
- The JSON schema of the conversion settings is generated from
  {py:class}`~healpix_convert.ConvertSettings` and published as
  `convert-settings.schema.json` in the root of the repository. The STAC
  `processing:expression` property of the output records it as the `format` of
  the settings, pinned to the tag or the commit of the `healpix-convert`
  version that produced the dataset ({issue}`19`).
- healpix-resample 2026.10.3 or later is required. Its release changes the
  results of the `"bicubic"` resampler and of the samples tied with the
  `Npt`-th nearest cell; the neighbour search accepts `coarse_level_delta` in
  the resampler `init_params` ({pull}`92`).
- The pyramid stage no longer fails on an output dataset written with storage
  options, e.g. the credentials of an object store ({pull}`90`).
- The `"mean"` and `"sum"` reductions of the pyramids sum `float16` variables
  in `float32`: a block whose `float16` sum was above 65504 (e.g. 64 cells of
  1100.0) got an infinite mean ({issue}`102`). A `"sum"` above 65504 is still
  infinite: it is written as `float16`.

## v2026.10.1 (2026-10-16)

- The CAMS converter's nearest-neighbour method now assigns the input points
  to cells with healpix-geo on the WGS84 ellipsoid that the output declares,
  instead of healpy on the sphere. Some points move to a neighbouring cell,
  and healpy is no longer needed ({pull}`50`).
- The dependencies of the ERA5, CAMS and ClimateDT converters are the `era5`,
  `cams` and `climatedt` extras, and the README (the PyPI page) shows how to
  install from PyPI ({pull}`48`).
- The variables of a chunk are resampled in one batched call ({pull}`31`) and
  the output arrays of a chunk are written concurrently ({pull}`39`).
- The group codecs are checked when the settings are created ({pull}`27`).
- The cell ids can be delta-coded before compression ({pull}`42`, {pull}`47`).
- Object string arrays are written as variable-length strings ({pull}`40`).
- The affine transform derived from x / y coordinates points at the top-left
  corner of the pixels ({pull}`24`).
- A warning is emitted when a variable with valid input values comes back
  all-NaN ({pull}`41`).
- Sparse chunks are written with the data type of the input variable
  ({pull}`46`).
- Documentation site, built with Sphinx and published on GitHub Pages.

## v2026.10.0 (2026-10-08)

- The spatial coordinate is promoted to an xarray coordinate ({pull}`1`).
- The input's description of its variables is kept ({pull}`2`).
- The conventions followed by the output dataset can be selected ({pull}`3`),
  also by the ERA5 / CAMS / ClimateDT converters ({pull}`6`).
- The output records how it was produced, as STAC processing metadata
  ({pull}`5`).
- `CITATION.cff` ({pull}`7`).
- Example notebook converting the ESA CCI land cover map ({pull}`10`).
- Input attributes can be excluded from the output ({pull}`14`).

See the
[release notes](https://github.com/GRID4EARTH/healpix-convert/releases/tag/v2026.10.0).

## v2026.09.0 (2026-09-08)

First release on PyPI.
