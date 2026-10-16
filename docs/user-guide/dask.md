# Parallel conversion with Dask

:::{note}
This page is a first draft.
:::

When a [Dask distributed](https://distributed.dask.org) client is active,
{py:func}`~healpix_convert.convert_group_to_healpix` and
{py:func}`~healpix_convert.create_healpix_dataset` convert the chunks of a
group in parallel on the cluster: each chunk is one task. Without a client, the
chunks are converted one after the other.

```python
from distributed import Client

import healpix_convert as hc

client = Client()  # a local cluster, or the address of an existing scheduler

hc.convert_group_to_healpix(
    "measurements/land_cover",
    cache=cache,
    settings=settings,
    output_path=output_path,
)
```

Only groups with a chunk method other than `"no_chunk"` are converted in
parallel (see {doc}`settings`): with `"no_chunk"`, the whole group is resampled
at once in the calling process.

## Options

{py:func}`~healpix_convert.convert_group_to_healpix` has two options for the
Dask conversion:

- `resources`: the
  [worker resources](https://distributed.dask.org/en/stable/resources.html)
  needed by each chunk task, e.g., `{"GPU": 1}`, passed to
  {py:meth}`distributed.Client.map`.
- `load_input_data`: if `True`, the input data of the group is persisted in
  the distributed memory of the cluster before the conversion (or loaded in
  memory without a client). This can speed up the conversion, but needs enough
  memory for the whole input group.

## Combine with parallel jobs

`chunk_indices` limits a call to some chunks of the group: each job of a
workflow can convert its own share of the chunks, and each job can use a Dask
cluster to convert them in parallel. See the {doc}`../tutorials/quickstart`.
