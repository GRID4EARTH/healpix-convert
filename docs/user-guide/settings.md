# Conversion settings

:::{note}
This page is a first draft. Each setting is also documented in the source code
of `healpix_convert.settings`.
:::

The conversion settings are a {py:class}`~healpix_convert.ConvertSettings`
object (a [Pydantic](https://docs.pydantic.dev) model), validated when it is
created. Create it from a dictionary or a JSON string, or start from the
predefined settings of a supported product:

```python
import healpix_convert as hc

settings = hc.ConvertSettings.from_dict({"group_settings": {...}})
settings = hc.ConvertSettings.from_json(json_string)
settings = hc.get_settings("era5")  # or "cams-eac4", "climatedt-sfc"
```

`to_dict()` and `to_json()` give the settings back, e.g., to store them or to
pass them to other jobs. The settings are also recorded in the metadata of the
output dataset (see {doc}`metadata`).

## Top-level settings

- `group_settings`: the conversion settings of each group to convert, keyed by
  the path of the group in the input datasets (e.g.,
  `"measurements/land_cover"`). Every group with spatial data must have
  settings, unless it is excluded.
- `exclude_groups`: the paths of the input groups left out of the output
  dataset.
- `multiscale_settings`: the input groups (`groups`) converted as multiscale
  groups. Their children groups are converted at different refinement levels
  and written as `<group>/<refinement level>` in the output, with metadata
  following the Multiscales Zarr convention. `add_healpix_positioning`
  (default `True`) adds the HEALPix grid of each level to that metadata.
- `metadata`: the conventions followed by the output dataset (see
  {doc}`metadata`).

## Group settings

Each entry of `group_settings` has the following fields.

### `healpix`: output HEALPix grid

| Field               | Default      | Description                                                    |
| ------------------- | ------------ | -------------------------------------------------------------- |
| `refinement_level`  | (required)   | HEALPix refinement level, from 0 to 29 (`None` with `"zuniq"`) |
| `indexing_scheme`   | `"nested"`   | `"nested"`, `"ring"` or `"zuniq"`                              |
| `ellipsoid`         | (sphere)     | `{"name": "wgs84"}` for the WGS84 ellipsoid                    |
| `spatial_dimension` | `"cells"`    | Name of the HEALPix cell dimension                             |
| `coordinate`        | `"cell_ids"` | Name of the cell ids coordinate                                |
| `compression`       | `"none"`     | Cell ids compression: `"compacted"` or `"ranges"`              |

### `chunk`: chunks along the HEALPix cell dimension

`method` selects how the output arrays are chunked along the HEALPix cell
dimension:

- `"no_chunk"`: a single chunk, the whole group is resampled at once.
- `"healpix_cell_dense"`: one chunk per HEALPix cell of a coarser level
  (`healpix`, with the nested indexing scheme), densely filled with all its
  children cells at the output refinement level: $4^{(l - l_c)}$ cells per
  chunk, where $l$ is the output refinement level and $l_c$ the chunk level.
  `chunk_buffer_width` (default: 60) widens each chunk when selecting the
  input data to resample: in CRS units for projected coordinates, in metres
  for longitude / latitude coordinates.
- `"healpix_cell_sparse"`: one chunk per HEALPix cell of a coarser level
  (`healpix`), holding only the cells that contain input data points (e.g.,
  with the `"cell-point"` resampler), up to `chunk_size` cells per chunk.

Chunks are the unit of parallelism: each one can be converted independently
(see {doc}`dask` and the {doc}`../tutorials/quickstart`).

:::{note}
The `"healpix_cell_dense"` and `"healpix_cell_sparse"` methods currently need
an `ellipsoid` in both the output and the chunk grids (`healpix`), e.g.,
`{"name": "wgs84"}`. Only `"no_chunk"` works on the sphere.
:::

### `resampler`: resampling method

`name` selects the resampling method of
[healpix-resample](https://grid4earth.eu/healpix-resample/):

| `name`         | healpix-resample class | Method                                    |
| -------------- | ---------------------- | ----------------------------------------- |
| `"nearest"`    | `NearestResampler`     | Nearest neighbor (k = 1)                  |
| `"bilinear"`   | `BilinearResampler`    | Bilinear interpolation (k = 4)            |
| `"k-nearest"`  | `KNeighborsResampler`  | k-nearest neighbors interpolation         |
| `"psf"`        | `PSFResampler`         | Point spread function (PSF) interpolation |
| `"cell-point"` | `CellPointResampler`   | Cell points (refinement level 29 only)    |

`init_params` are passed to the constructor of the resampler class and
`resample_params` to its `resample()` method. Each value of `resample_params`
is either used for all the data variables of the group, or given per variable
as a dictionary keyed by variable name.

With the `"bilinear"` and `"psf"` methods, the resampled variables are written
as `float64`. The other methods keep the data type of the input variables.

### Other group settings

- `metadata`: `exclude_attrs` lists the names (or regular expressions matching
  the whole name) of the input attributes not copied to the output.
- `codecs`: the Zarr codecs of the resampled data variables (Zarr's defaults
  if not set). Each codec must be registered with Zarr where the settings are
  created.
- `healpix_coordinate_codecs`: the Zarr codecs of the cell ids coordinate
  (Zarr's defaults if not set).

## Changing the settings of all groups

{py:meth}`~healpix_convert.ConvertSettings.filter_and_replace` applies a
function to the settings of each group (given as dictionaries) and returns new,
validated settings:

```python
def set_level(group, group_settings):
    group_settings["healpix"]["refinement_level"] = 8
    return group_settings


settings = settings.filter_and_replace(set_level)
```
