# HEALPix pyramids

{py:func}`~healpix_convert.generate_pyramids` is the last stage of a
conversion: it aggregates the converted data onto the coarser HEALPix
refinement levels given by the pyramid settings of each multiscale group, and
writes the arrays of their groups.

Those groups, one per derived level and named after it, and the
`multiscales` metadata describing the whole pyramid, are part of the structure
of the output dataset: {py:func}`~healpix_convert.prepare_healpix_dataset`
creates them, like it creates the converted groups. Without pyramid settings,
both stages leave the dataset alone.

```python
import healpix_convert as hc

settings = hc.ConvertSettings.from_dict(
    {
        "multiscale_settings": {
            "groups": ["measurements"],
            "pyramid_groups": {"measurements": {"derived_levels": [5, 4]}},
        },
        "group_settings": {
            "measurements/fine": {
                "healpix": {"refinement_level": 7},
                "chunk": {
                    "method": "healpix_cell_dense",
                    "healpix": {"refinement_level": 4},
                },
                ...,
            },
            "measurements/coarse": {"healpix": {"refinement_level": 6}, ...},
        },
    }
)

cache = hc.create_staging_cache(["input.zarr"], settings)
hc.prepare_healpix_dataset(cache, settings, "converted.zarr")
for group in ("measurements/fine", "measurements/coarse"):
    hc.convert_group_to_healpix(
        group, cache=cache, settings=settings, output_path="converted.zarr"
    )

hc.generate_pyramids(cache, settings, "converted.zarr")
```

{py:func}`~healpix_convert.create_healpix_dataset` does all of that in one
call, pyramids included.

```text
converted.zarr
└── measurements        (multiscale group, its layout describing the pyramid)
    ├── 7               (converted from measurements/fine)
    ├── 6               (converted from measurements/coarse)
    ├── 5               (derived, aggregated from the level 6)
    └── 4               (derived)
```

The structure of the pyramids — which levels are derived, in which groups,
with which HEALPix grid and chunks, and the multiscales metadata describing
them — is computed once, from the settings and the input datasets, when the
staging cache is created: `cache.output_groups.pyramid_multiscale_groups` holds
one entry per **pyramid multiscale group**, i.e. per group of the output
dataset holding the levels converted from the input data (the **native
levels**) and the levels derived from them (the **derived levels**). The
groups that get a pyramid, and how, are given by the `pyramid_groups`
multiscale settings (see {doc}`settings`).

## Dense chunks, and why they are required

Every child group of a pyramid multiscale group must be converted with the
`healpix_cell_dense` chunk method. The cells of such a chunk are all the
children of one HEALPix cell at a coarser (chunk) level, in ascending (nested)
order, so the cells of a coarser cell are `4 ** delta` **consecutive** cells of
the converted arrays: aggregating them is a matter of position.

Each derived level is one `Dataset.coarsen` reduction over the converted
cells:

- nothing is grouped and no cell id is read, so nothing proportional to the
  number of cells is held in memory: the ids of a densely covered 800 x 800 km
  area at the level 20 (16.6 billion cells) would be 132 GB on their own;
- the aggregation is blockwise, hence lazy: all the derived levels of all the
  pyramids are written in a single Dask computation, so a native level is read
  once and feeds every level aggregated from it. Run it on a Dask distributed
  cluster (see {doc}`dask`) to parallelize it;
- each cell of a derived level aggregates **all** the converted cells it
  contains, not the cells of the level just above it: the default `mean` is a
  block mean and not a mean of means, which differ as soon as some cells have
  no data.

Missing (NaN) values are skipped. The `reduce` pyramid setting selects another
reduction (`"min"`, `"max"`, `"sum"`): these are the reductions that give, for
a whole block of cells, the same result as aggregating all its cells at once.
The mean does not suit categorical data such as masks or classifications.

The data type and the fill value of each variable are preserved, an integer
variable being rounded instead of truncated.

A cell of a level coarser than the chunk level is covered entirely only if all
its chunk cells have been converted, which is not the case at the edges of a
partial spatial extent: the levels of `derived_levels` that are coarser than
the coarsest level whose cells are covered entirely are then rejected, and the
staging cache says which level that is. With a full-earth extent, every level
down to 0 can be derived.

## Chunks of the derived levels

The chunks of a derived level are HEALPix cells as well: a chunk holds the
`4 ** depth` cells of one HEALPix cell `depth` levels above, whose refinement
level the staging cache records next to the HEALPix grid of every level. The depth is the one of the converted chunks, so that the chunks of a
derived level hold as many cells as the ones of the level they are aggregated
from; following the chunk level of the converted data instead would divide that
number by four at every level. The depth is at least 4 (256 cells per chunk).

The chunk cells are never coarser than the coarsest level whose cells are
covered entirely: the base cells with a full-earth extent, so that the coarsest
levels hold one chunk per base cell, but the chunk level of the converted data
with a partial extent (e.g., one tile of a satellite product). A chunk then
always holds all the children of its cell, at the cost of smaller chunks for
the derived levels close to that level.

## Groups converted at several levels

The native levels of a multiscale group, e.g., the bands of a multi-resolution
product converted each at its native resolution (see `multiscale_settings` in
{doc}`settings`), are kept as they are and:

- a variable never appears at a level finer than its own native level (it is
  not upsampled);
- a variable converted at several native levels, e.g., the copies of the finest
  bands that the data provider resampled itself, is aggregated from the nearest
  native level at or finer than the derived level. Set
  `coarsen_from="finest"` to ignore those copies and always aggregate from the
  finest native level instead;
- the levels must be chunked with HEALPix cells of the same refinement level,
  so that a derived level holds the same cells whichever level its variables
  are aggregated from;
- the cells of a derived level are the union of the cells of its variables, a
  variable having its fill value where it has no data.

## Where the levels are written

The derived levels are groups of the multiscale group, next to the levels
converted from the input data:

```text
measurements        (multiscale group)
├── 7               (converted)
├── 6               (converted)
├── 5               (derived)
└── 4               (derived)
```

With a `subgroup_name`, they are written in a child group of it instead:

```text
measurements        (multiscale group)
├── 7               (converted)
├── 6               (converted)
└── pyramids
    ├── 5           (derived)
    └── 4           (derived)
```

Either way the multiscale group describes the whole pyramid in its
`multiscales` metadata, whose assets are paths relative to that group
(`"7"`, `"6"`, `"5"`, `"4"` above, or `"pyramids/5"` with a subgroup name),
and the levels converted from the input data are neither moved nor copied. They
are
not rewritten either, so the codecs they were written with, lossy ones
included, are left untouched.

`prepare_healpix_dataset(..., dry_run=True)` shows the derived level groups and
their metadata, like the rest of the structure of the output dataset, without
writing anything.
