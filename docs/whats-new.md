# What's new

## Unreleased

- New `"mode"` reduction for the pyramids (the `reduce` pyramid setting): the
  most frequent value of the cells of a block, for categorical variables such
  as masks and classifications.
- The multiscales metadata describes each derived level of a pyramid: the
  converted level it is aggregated from (`derived_from`), the transformation
  from it (`transform`, whose `scale` is how much longer a cell is) and the
  method its cells are aggregated with
  (`resampling_method`, also given for the whole group).
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
- The `asset` of a multiscales layout item is now a path relative to the group
  holding the metadata (e.g. `"7"` instead of `"measurements/7"`), as the
  Multiscales Zarr convention asks.
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
