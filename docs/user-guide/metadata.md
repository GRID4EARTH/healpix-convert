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
`multiscales` attribute lists its groups, one per refinement level (`layout`),
with their HEALPix grid if `add_healpix_positioning` is set (it is not by
default: the attributes of a level group describe its own grid). As the
convention asks, the `asset` of a layout item is a path relative to the group
holding the metadata, e.g. `"7"` for a converted level and `"pyramids/4"` for a
level derived in a child group (see {doc}`pyramids`).

The layout item of a derived level also names the converted level it is
aggregated from (`derived_from`, a path like `asset`), the transformation from
that level (`transform`) and how its cells are aggregated
(`resampling_method`, also given once for the whole group since every derived
level of a pyramid uses the same method):

```json
{
  "layout": [
    { "asset": "7" },
    {
      "asset": "6",
      "derived_from": "7",
      "transform": { "scale": [2.0] },
      "resampling_method": "mean"
    },
    {
      "asset": "4",
      "derived_from": "7",
      "transform": { "scale": [8.0] },
      "resampling_method": "mean"
    }
  ],
  "resampling_method": "mean"
}
```

The convention gives `scale` per axis, and a HEALPix group has a single one,
the cell dimension: the factor is the length of a cell relative to a cell of
the source level, `[2.0]` one level coarser and `[8.0]` three levels coarser.
Since HEALPix tessellates a 2D surface, a cell aggregates the square of that,
4 and 64 cells of the source level.

A derived level is aggregated from a converted level and never from another
derived level, so its `scale` grows with the distance to that level.

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

### Merging the input items

With several input datasets, the properties, extensions, links and assets of
the output item are merged from theirs, with the strategy chosen per kind by
`merge_properties`, `merge_extensions`, `merge_links` and `merge_assets`:

| Strategy       | Kept                                             | Available for     |
| -------------- | ------------------------------------------------ | ----------------- |
| `"pick_first"` | what the first input dataset has (the default)   | all four          |
| `"join_inner"` | what every input dataset has: their intersection | all four          |
| `"join_outer"` | what any input dataset has: their union          | extensions, links |

Properties and assets are merged by name, and `"join_inner"` keeps an entry
only when every input has it _with the same value_ — a `datetime` that differs
across the inputs does not describe the output either. They have no
`"join_outer"`: a union of mappings has to invent a rule for the keys whose
value differs across the inputs, and any such rule describes the output less
well than leaving the entry out.

Extensions and links are merged whole — two of them are the same one when they
are equal, which is all a link has to go on — so `"join_outer"` is meaningful
there, and lists them in the order they first appear.

With a single input dataset the strategies are all equivalent. The other
fields of the item are unaffected: the geometry, the bounding box and the `id`
describe the output, and `stac_version` must match across the inputs or the
conversion fails.

### Excluding and setting STAC properties, extensions, links and assets

Once merged, `metadata.stac` drops and sets what the output item propagates
from the input items:

```python
"metadata": {
    "stac": {
        "exclude_properties": ["internal:.*"],
        "override_properties": {"license": "CC-BY-4.0"},
        "exclude_extensions": ["https://stac-extensions.github.io/projection/.*"],
        "override_extensions": ["https://example.org/ext/v1.0.0/schema.json"],
        "exclude_links": ["self|parent"],
        "override_links": [
            {"rel": "canonical", "href": "https://example.org/out.zarr"}
        ],
        "override_assets": {"data": {"href": "s3://public/out.zarr"}},
    }
}
```

The four `exclude_*` settings list exact names or regular expressions matched
against the whole name. What that name is depends on the kind:

| Setting              | Matched against          | Override                               |
| -------------------- | ------------------------ | -------------------------------------- |
| `exclude_properties` | the property name        | `override_properties`: name → value    |
| `exclude_extensions` | the extension schema URL | `override_extensions`: list of URLs    |
| `exclude_links`      | the link's `rel`         | `override_links`: list of link objects |
| `exclude_assets`     | the asset key            | `override_assets`: key → asset object  |

Links have no identifier, so `rel` is what they are matched on; a link without
one matches the empty string. An override wins over an exclusion, and the
`override_*` of a list (extensions, links) appends what is not there yet rather
than replacing.

`exclude_assets` is the one setting that is not empty by default: it is
`[".*"]`, so the input's assets are not propagated. They point at the input
files, which the output does not hold, and an output item advertising files
that are not its own is worse than one advertising none. Set it to `[]` to
keep them, or describe the output's own files with `override_assets`.

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

## Consolidated metadata

Consolidating the metadata of a Zarr group copies the metadata of all its
descendants into its own metadata document, so that a reader opens the whole
hierarchy below it with a single read instead of one read per node (a lot
faster on an object store).

{py:func}`~healpix_convert.finalize_healpix_dataset`, the last stage of the
conversion, writes it, and no other stage does: such a copy is stale as soon as
a group or an array is written next to it. Converting the data does not make it
stale, though, since it only writes array chunks.

The `metadata.consolidate` settings select the groups it is written for:

```python
"metadata": {
    "consolidate": {
        "root_group": True,
        "healpix_groups": True,
        "multiscale_groups": True,
        "derived_level_groups": True,
        "groups": ["provenance"],
    }
}
```

| Setting                | Consolidates the metadata of                        |
| ---------------------- | --------------------------------------------------- |
| `root_group`           | the root group, i.e. the whole output dataset       |
| `healpix_groups`       | each group converted to HEALPix                     |
| `multiscale_groups`    | each multiscale group, i.e. all the levels it holds |
| `derived_level_groups` | each group holding a derived level of a pyramid     |
| `groups`               | the given paths, in the output dataset              |

The four flags are set by default, so every group a reader is expected to open
on its own is consolidated: the consolidated metadata of the root group can
describe much more than what such a reader needs. `groups` is empty by default,
and its paths are those of the output dataset, not of the input dataset(s), the
root group being `"/"`. A path that names no group of the output dataset is
reported with a warning when the staging cache is created (its structure is
known there, before anything is converted) and skipped when the metadata is
consolidated.

:::{note}
Consolidated metadata is not part of the Zarr v3 specification (it is an
extension that Zarr-Python writes and reads). A reader that ignores it opens
the output dataset all the same, group by group.
:::
