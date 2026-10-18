# Output metadata

:::{note}
This page is a first draft.
:::

The output dataset describes its HEALPix grid with the conventions selected in
the `metadata` settings (all of them by default), and keeps the STAC metadata
of the input datasets.

## Selecting the conventions

```python
settings = hc.ConvertSettings.from_dict(
    {
        "metadata": {
            "zarr_conventions": [
                "dggs",
                "multiscales",
                {
                    "name": "cf",
                    "version": "1.13",
                    "configuration": {"healpix_grid_mapping": True},
                },
            ]
        },
        "group_settings": {...},
    }
)
```

Each item of `zarr_conventions` is a convention name or a mapping with a
`name` and, optionally, a `version` and a `configuration`. The conventions
left out are not written.

## DGGS Zarr convention

Each HEALPix group lists the
[DGGS Zarr convention](https://github.com/zarr-conventions/dggs) in its
`zarr_conventions` attribute, and describes its grid in a `dggs` attribute:

```python
{
    "name": "healpix",
    "refinement_level": 7,
    "indexing_scheme": "nested",
    "ellipsoid": {
        "name": "wgs84",
        "semi_major_axis": 6378137.0,
        "inverse_flattening": 298.257223563,
    },
    "spatial_dimension": "cells",
    "coordinate": "cell_ids",
    "compression": "none",
}
```

## Multiscales Zarr convention

A multiscale group (see `multiscale_settings` in {doc}`settings`) lists the
[Multiscales Zarr convention](https://github.com/zarr-conventions/multiscales)
and the DGGS Zarr convention in its `zarr_conventions` attribute. Its
`multiscales` attribute lists its children groups, one per refinement level
(`layout`), with their HEALPix grid if `add_healpix_positioning` is set.

## CF conventions

With the CF conventions (version 1.13 or newer, the first one defining the
HEALPix grid mapping):

- the root group has a `Conventions` attribute (e.g., `"CF-1.13"`);
- if `healpix_grid_mapping` is set (the default), each HEALPix group has a
  `crs` grid mapping variable (`grid_mapping_name = "healpix"`), referenced by
  the `grid_mapping` attribute of the data variables, and the cell ids
  coordinate has `standard_name = "healpix_index"`.

## Variable attributes

The attributes of the input variables (`units`, `long_name`, etc.) are copied
to the resampled variables, except the ones matching the `exclude_attrs`
patterns of the group's `metadata` settings, and the ones set by its
`override_attrs` patterns (see {doc}`settings`).

## STAC metadata

The root group of the output dataset has a `stac_discovery` attribute: a STAC
item merged from the `stac_discovery` attributes of the input datasets, with

- the geometry and bounding box of the output dataset, and an `id` derived from
  the output path;
- `derived_from` links to the input datasets;
- provenance properties following the
  [STAC processing extension](https://github.com/stac-extensions/processing):
  the date of the conversion (`processing:datetime`), the versions of
  `healpix-convert` and `healpix-resample` (`processing:software`), the
  conversion settings (`processing:expression`, whose `format` is the URL of
  the settings JSON schema of the recorded `healpix-convert` version) and the
  inputs (`processing:lineage`).

Set `metadata.stac.include_conversion_settings` to `False` to leave
`processing:expression` out (not recommended).

### Excluding and setting STAC properties and extensions

`metadata.stac` drops and sets the properties and extensions propagated from
the input items:

```python
"metadata": {
    "stac": {
        "exclude_props": ["internal:.*"],
        "override_props": {"license": "CC-BY-4.0"},
        "exclude_extensions": ["https://stac-extensions.github.io/projection/.*"],
        "override_extensions": ["https://example.org/ext/v1.0.0/schema.json"],
    }
}
```

`exclude_props` and `exclude_extensions` list exact names (property names,
extension schema URLs) or regular expressions matched against the whole name.
`override_props` sets properties and `override_extensions` lists extensions in
addition to the propagated ones. An override wins over an exclusion.

These settings apply to the propagated metadata only. The STAC metadata written
by `healpix-convert` itself keeps precedence: the `processing:*` properties and
the processing extension above are written after the settings are applied, so
the output stays traceable whatever they say. The one exception is
`include_conversion_settings`, which decides whether the settings themselves
are recorded.

:::{note}
The output item is not validated against the extensions it lists: keeping the
properties and the extensions consistent with each other is left to the data
provider.
:::
