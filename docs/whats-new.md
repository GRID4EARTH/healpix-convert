# What's new

## Unreleased

- New `generate_pyramids` conversion stage, which aggregates the converted
  data onto the coarser HEALPix refinement levels given by the new
  `pyramid_groups` multiscale settings. One
  group per generated level, and the multiscales metadata describing the
  pyramid, are created with the rest of the structure of the output dataset by
  `prepare_healpix_dataset`; the arrays of those groups are written with the
  aggregated data, and the converted levels are never rewritten. The groups must be converted with the
  `healpix_cell_dense` chunk method: the cells of a coarser cell are then
  consecutive cells of the converted arrays, so the levels are aggregated by
  position without reading any cell id.
- The `asset` of a multiscales layout item is now a path relative to the group
  holding the metadata (e.g. `"7"` instead of `"measurements/7"`), as the
  Multiscales Zarr convention asks.
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
