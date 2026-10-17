import json

import numpy as np
import pytest
import zarr

import healpix_convert.multires as multires
from healpix_convert import create_multires_dataset

GROUP = "measurements/reflectance"

DGGS = {
    "name": "healpix",
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
DGGS_CONVENTION = {
    "uuid": "7b255807-140c-42ca-97f6-7a1cfecdbc38",
    "name": "dggs",
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/dggs/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/dggs/blob/v1/README.md",
    "description": "Discrete Global Grid Systems convention for zarr",
}

# variable name -> (native level, dtype, fill value)
VARIABLES = {
    "a": (20, "float64", 0.0),
    "b": (20, "uint16", 0),
    "c": (19, "float32", np.nan),
    "d": (17, "float64", 0.0),
}


def _descendants(parents, parent_level, level):
    shift = 2 * (level - parent_level)
    offsets = np.arange(1 << shift, dtype=np.uint64)
    return (np.asarray(parents, dtype=np.uint64)[:, None] << np.uint64(shift)) + offsets


def _native_data(rng):
    """Sparse cells at levels 20, 19 and 17 under a few level-14 parents."""
    parents = {20: [1000, 1001, 5000], 19: [1000, 5000, 7000], 17: [1000, 1001, 5000]}
    cells = {}
    for level, ps in parents.items():
        all_cells = _descendants(ps, 14, level).ravel()
        keep = rng.random(len(all_cells)) < 0.7
        cells[level] = np.sort(all_cells[keep])
    data = {}
    for name, (level, dtype, fill) in VARIABLES.items():
        n = len(cells[level])
        if np.issubdtype(np.dtype(dtype), np.integer):
            v = rng.integers(1, 10000, n).astype(dtype)
        else:
            v = rng.random(n).astype(dtype)
        v[rng.random(n) < 0.1] = fill
        if name == "a":
            v[rng.random(n) < 0.02] = np.nan
        data[name] = v
    return cells, data


def _write_input(path, cells, data, chunk=64):
    root = zarr.open_group(path, mode="w", zarr_format=3)
    root.attrs["stac_discovery"] = {"id": "S2X_TEST", "properties": {"platform": "x"}}
    root.create_group("measurements", attributes={"keep": "me"})
    group = root.create_group(GROUP, attributes={"description": "reflectance"})
    for level in sorted(cells, reverse=True):
        g = group.create_group(
            str(level),
            attributes={
                "zarr_conventions": [DGGS_CONVENTION],
                "dggs": {**DGGS, "refinement_level": level},
            },
        )
        c = g.create_array(
            "cell_ids",
            shape=cells[level].shape,
            chunks=(chunk,),
            dtype="uint64",
            fill_value=np.iinfo("uint64").max,
            dimension_names=["cells"],
            attributes={"standard_name": "healpix_index"},
        )
        c[:] = cells[level]
        for name, (native, dtype, fill) in VARIABLES.items():
            if native != level:
                continue
            a = g.create_array(
                name,
                shape=cells[level].shape,
                chunks=(chunk,),
                dtype=dtype,
                fill_value=fill,
                dimension_names=["cells"],
                attributes={"long_name": f"variable {name}", "grid_mapping": "crs"},
            )
            a[:] = data[name]
        if level == 20:
            crs = g.create_array("crs", shape=(), dtype="int8", fill_value=0)
            crs.attrs.update({"grid_mapping_name": "healpix", "refinement_level": 20})


def _is_fill(v, fill):
    return np.isnan(v) if np.isnan(fill) else v == fill


def _block_mean_reference(values: dict, from_level: int, to_level: int, dtype, fill):
    """Per level: {cell: value}, each cell being the mean of all its valid
    descendants at ``from_level`` (a block mean, not a mean of means)."""
    levels = {from_level: values}
    for level in range(from_level - 1, to_level - 1, -1):
        shift = 2 * (from_level - level)
        descendants: dict[int, list] = {}
        for cell, value in values.items():
            descendants.setdefault(cell >> shift, []).append(value)
        out = {}
        for parent, vs in descendants.items():
            valid = [
                v for v in vs if np.isfinite(v) and not _is_fill(np.float64(v), fill)
            ]
            if valid:
                mean = np.mean(valid)
                if np.issubdtype(np.dtype(dtype), np.integer):
                    mean = np.rint(mean)
                out[parent] = np.dtype(dtype).type(mean)
            else:
                out[parent] = np.dtype(dtype).type(fill)
        levels[level] = out
    return levels


def _reference(cells, data, min_level):
    """Per variable and level: {cell: value}."""
    return {
        name: _block_mean_reference(
            dict(zip(cells[native].tolist(), data[name].tolist())),
            native,
            min_level,
            dtype,
            fill,
        )
        for name, (native, dtype, fill) in VARIABLES.items()
    }


@pytest.fixture
def native(tmp_path):
    rng = np.random.default_rng(42)
    cells, data = _native_data(rng)
    path = tmp_path / "input.zarr"
    _write_input(path, cells, data)
    return path, cells, data


@pytest.mark.parametrize("read_size", [64, 1 << 20])
def test_values(native, tmp_path, monkeypatch, read_size):
    # a small read size makes blocks straddle the reads
    monkeypatch.setattr(multires, "_READ_SIZE", read_size)
    path, cells, data = native
    out = tmp_path / "out.zarr"

    tag = create_multires_dataset(str(path), str(out), min_level=12, chunk_size=50)
    assert tag == "mean-l20-l12"

    ref = _reference(cells, data, 12)
    group = zarr.open_group(out, mode="r")[GROUP]
    assert sorted(group.group_keys(), key=int) == [str(lv) for lv in range(12, 21)]

    for level in range(20, 11, -1):
        g = group[str(level)]
        out_cells = g["cell_ids"][:]
        present = [
            n for n, (native_level, *_) in VARIABLES.items() if native_level >= level
        ]
        assert sorted(
            k for k in g.array_keys() if k not in ("cell_ids", "crs")
        ) == sorted(present)

        # cells are the union of the variables' cells
        union = sorted(set().union(*(ref[n][level] for n in present)))
        np.testing.assert_array_equal(out_cells, np.array(union, dtype=np.uint64))

        for name in present:
            _, dtype, fill = VARIABLES[name]
            values = g[name][:]
            assert values.dtype == np.dtype(dtype)
            expected = np.array(
                [ref[name][level].get(c, fill) for c in out_cells.tolist()], dtype=dtype
            )
            if np.issubdtype(values.dtype, np.integer):
                np.testing.assert_array_equal(values, expected)
            else:
                np.testing.assert_allclose(values, expected, rtol=1e-6, equal_nan=True)
            assert g[name].attrs["native_refinement_level"] == VARIABLES[name][0]
            assert g[name].attrs["long_name"] == f"variable {name}"


def _add_resampled_copy(path, cells, rng, name="a", attrs=None):
    """Like Sentinel-2 L2A: ``name`` (native at 20) is also stored, resampled
    by the provider, at level 19, with another encoding (float32, NaN fill)."""
    level19 = zarr.open_group(path, mode="a")[f"{GROUP}/19"]
    copy = level19.create_array(
        name,
        shape=cells[19].shape,
        chunks=(64,),
        dtype="float32",
        fill_value=np.nan,
        dimension_names=["cells"],
        attributes={
            "long_name": f"variable {name}, resampled at level 19",
            **(attrs or {}),
        },
    )
    values = (rng.random(len(cells[19])) + 10).astype("float32")
    values[rng.random(len(values)) < 0.1] = np.nan
    copy[:] = values
    return values


def test_resampled_copies_kept(native, tmp_path):
    path, cells, data = native
    values = _add_resampled_copy(path, cells, np.random.default_rng(1))
    out = tmp_path / "out.zarr"
    create_multires_dataset(str(path), str(out), min_level=12)
    group = zarr.open_group(out, mode="r")[GROUP]

    assert group.attrs["native_refinement_levels"]["19"] == ["a", "c"]
    assert group.attrs["resampled_copies"] == {
        "levels": {"19": ["a"]},
        "handling": "kept",
    }
    assert [
        group[f"{lv}/a"].attrs["native_refinement_level"] for lv in (20, 19, 18, 12)
    ] == [20, 19, 19, 19]

    # the attributes follow the values
    assert group["20/a"].attrs["long_name"] == "variable a"
    assert group["19/a"].attrs["long_name"] == "variable a, resampled at level 19"
    assert group["12/a"].attrs["long_name"] == "variable a, resampled at level 19"

    # level 20 is untouched
    np.testing.assert_array_equal(group["20/a"][:], data["a"])
    # level 19 is the copy, re-encoded like level 20 (float64, fill 0), and
    # the fill value where the copy has no cell
    copy = {
        c: (0.0 if np.isnan(v) else float(v))
        for c, v in zip(cells[19].tolist(), values)
    }
    ref = _block_mean_reference(copy, 19, 12, "float64", 0.0)
    for level in (19, 18, 12):
        out_cells = group[f"{level}/cell_ids"][:]
        a = group[f"{level}/a"]
        assert a.dtype == np.float64 and a.fill_value == 0.0
        expected = [ref[level].get(c, 0.0) for c in out_cells.tolist()]
        np.testing.assert_allclose(a[:], expected, rtol=1e-6)


def test_resampled_copies_overwritten(native, tmp_path):
    path, cells, data = native
    _add_resampled_copy(path, cells, np.random.default_rng(1))
    out = tmp_path / "out.zarr"
    create_multires_dataset(str(path), str(out), min_level=12, overwrite_resampled=True)
    group = zarr.open_group(out, mode="r")[GROUP]

    assert group.attrs["native_refinement_levels"]["19"] == ["c"]
    assert group.attrs["resampled_copies"] == {
        "levels": {"19": ["a"]},
        "handling": "overwritten_with_mean",
    }
    assert group["19/a"].attrs["native_refinement_level"] == 20
    ref = _reference(cells, data, 12)
    out_cells = group["19/cell_ids"][:]
    expected = [ref["a"][19].get(c, 0.0) for c in out_cells.tolist()]
    np.testing.assert_allclose(group["19/a"][:], expected, equal_nan=True)


def test_native_levels_are_copied(native, tmp_path):
    path, cells, data = native
    out = tmp_path / "out.zarr"
    create_multires_dataset(str(path), str(out), min_level=12)
    src = zarr.open_group(path, mode="r")[GROUP]
    dst = zarr.open_group(out, mode="r")[GROUP]
    # level 20 holds only native variables and cells
    np.testing.assert_array_equal(dst["20/cell_ids"][:], src["20/cell_ids"][:])
    np.testing.assert_array_equal(dst["20/a"][:], src["20/a"][:])
    np.testing.assert_array_equal(dst["20/b"][:], src["20/b"][:])
    # native variables at a mixed level are copied, on the union of cells
    out_cells = dst["19/cell_ids"][:]
    idx = np.searchsorted(out_cells, src["19/cell_ids"][:])
    np.testing.assert_array_equal(dst["19/c"][:][idx], src["19/c"][:])


def test_metadata(native, tmp_path):
    path, _, _ = native
    out = tmp_path / "out.zarr"
    create_multires_dataset(str(path), str(out), min_level=12)
    root = zarr.open_group(out, mode="r")
    group = root[GROUP]

    assert root.attrs["stac_discovery"]["id"] == "S2X_TEST_mean-l20-l12"
    props = root.attrs["stac_discovery"]["properties"]
    assert props == {"platform": "x", "multires:resampling_method": "mean"}
    assert root["measurements"].attrs["keep"] == "me"

    attrs = group.attrs
    assert attrs["description"] == "reflectance"
    assert attrs["multires_method"] == "mean-l20-l12"
    assert attrs["native_refinement_levels"] == {
        "20": ["a", "b"],
        "19": ["c"],
        "17": ["d"],
    }
    assert [c["name"] for c in attrs["zarr_conventions"]] == ["multiscales", "dggs"]
    assert "resampled_copies" not in attrs

    layout = attrs["multiscales"]["layout"]
    assert attrs["multiscales"]["resampling_method"] == "mean"
    assert [e["asset"] for e in layout] == [str(lv) for lv in range(20, 11, -1)]
    assert "derived_from" not in layout[0]
    assert layout[1]["derived_from"] == "20"
    assert layout[1]["transform"] == {"scale": [2.0]}
    assert layout[1]["resampling_method"] == "mean"
    assert layout[-1]["dggs"]["refinement_level"] == 12

    for level in range(12, 21):
        g = group[str(level)]
        assert g.attrs["dggs"]["refinement_level"] == level
        assert g.attrs["dggs"]["coordinate"] == "cell_ids"
        # the grid mapping is copied to every level with its refinement level
        assert g["crs"].attrs["refinement_level"] == level


def test_consolidated_metadata(native, tmp_path):
    path, _, _ = native
    out = tmp_path / "out.zarr"
    create_multires_dataset(str(path), str(out), min_level=12)
    for zarr_json in (out / "zarr.json", out / GROUP / "zarr.json"):
        meta = json.loads(zarr_json.read_text())
        assert meta["consolidated_metadata"]["metadata"]


def test_default_groups(native, tmp_path):
    path, _, _ = native
    out = tmp_path / "out.zarr"
    create_multires_dataset(str(path), str(out), min_level=17)
    group = zarr.open_group(out, mode="r")[GROUP]
    assert sorted(group.group_keys(), key=int) == ["17", "18", "19", "20"]


def test_min_level_too_fine(native, tmp_path):
    path, _, _ = native
    with pytest.raises(ValueError, match="finer than the coarsest native level 17"):
        create_multires_dataset(str(path), str(tmp_path / "out.zarr"), min_level=18)


def test_unsorted_cells(tmp_path):
    rng = np.random.default_rng(0)
    cells, data = _native_data(rng)
    cells[20] = cells[20][::-1].copy()
    path = tmp_path / "input.zarr"
    _write_input(path, cells, data)
    with pytest.raises(ValueError, match="must be sorted"):
        create_multires_dataset(str(path), str(tmp_path / "out.zarr"), min_level=12)


def test_unsupported_method(native, tmp_path):
    path, _, _ = native
    with pytest.raises(ValueError, match="unsupported resampling method"):
        create_multires_dataset(
            str(path), str(tmp_path / "out.zarr"), min_level=12, method="nearest"
        )


def test_block_mean():
    # a level-18 cell with four 1.0 under one child and a single 9.0 under
    # another: the mean of the children's means would be 5.0
    cells = np.array([0, 1, 2, 3, 4], dtype=np.uint64)
    block = multires._Block(cells, {"v": np.array([1.0, 1.0, 1.0, 1.0, 9.0])})
    fill = {"v": np.nan}
    level19 = block.coarsen(fill)
    np.testing.assert_array_equal(level19.cells, [0, 1])
    np.testing.assert_allclose(level19.values["v"], [1.0, 9.0])
    level18 = level19.coarsen(fill)
    np.testing.assert_array_equal(level18.cells, [0])
    np.testing.assert_allclose(level18.values["v"], [2.6])


def test_integer_copy_is_rounded(native, tmp_path):
    # "b" is uint16 at level 20; a float32 copy at level 19 must be rounded,
    # not truncated, when re-encoded like level 20
    path, cells, _ = native
    values = _add_resampled_copy(path, cells, np.random.default_rng(2), name="b")
    out = tmp_path / "out.zarr"
    create_multires_dataset(str(path), str(out), min_level=12)
    b = zarr.open_group(out, mode="r")[f"{GROUP}/19/b"]
    assert b.dtype == np.uint16
    out_cells = zarr.open_group(out, mode="r")[f"{GROUP}/19/cell_ids"][:]
    idx = np.searchsorted(out_cells, cells[19])
    expected = np.where(np.isnan(values), 0, np.rint(values)).astype(np.uint16)
    np.testing.assert_array_equal(b[:][idx], expected)


def test_copy_with_other_scale_factor(native, tmp_path):
    path, cells, _ = native
    _add_resampled_copy(
        path, cells, np.random.default_rng(2), attrs={"scale_factor": 0.0001}
    )
    with pytest.raises(ValueError, match="scale_factor"):
        create_multires_dataset(str(path), str(tmp_path / "out.zarr"), min_level=12)


def test_multidimensional_variable(native, tmp_path):
    path, cells, _ = native
    level20 = zarr.open_group(path, mode="a")[f"{GROUP}/20"]
    t = level20.create_array(
        "t",
        shape=(3, len(cells[20])),
        chunks=(3, 64),
        dtype="float32",
        fill_value=np.nan,
        dimension_names=["time", "cells"],
    )
    t[:] = 1.0
    with pytest.raises(NotImplementedError, match="only 1-D variables"):
        create_multires_dataset(str(path), str(tmp_path / "out.zarr"), min_level=12)


def _add_flag_variable(path, cells, group=GROUP):
    root = zarr.open_group(path, mode="a")
    if group not in root:
        level20_attrs = dict(root[f"{GROUP}/20"].attrs)
        level20 = root.create_group(f"{group}/20", attributes=level20_attrs)
        ids = level20.create_array(
            "cell_ids",
            shape=cells[20].shape,
            chunks=(64,),
            dtype="uint64",
            fill_value=np.iinfo("uint64").max,
            dimension_names=["cells"],
        )
        ids[:] = cells[20]
    level20 = root[f"{group}/20"]
    scl = level20.create_array(
        "scl",
        shape=cells[20].shape,
        chunks=(64,),
        dtype="uint8",
        fill_value=0,
        dimension_names=["cells"],
        attributes={"flag_values": [1, 2], "flag_meanings": "water land"},
    )
    scl[:] = 2


def test_categorical_variable_skipped_by_default(native, tmp_path):
    path, cells, _ = native
    _add_flag_variable(path, cells)
    out = tmp_path / "out.zarr"
    create_multires_dataset(str(path), str(out), min_level=12)
    group = zarr.open_group(out, mode="r")[GROUP]
    assert "scl" not in group["20"]
    assert group.attrs["native_refinement_levels"]["20"] == ["a", "b"]


def test_categorical_group_skipped_by_default(native, tmp_path):
    path, cells, _ = native
    _add_flag_variable(path, cells, group="conditions/mask")
    out = tmp_path / "out.zarr"
    create_multires_dataset(str(path), str(out), min_level=12)
    root = zarr.open_group(out, mode="r")
    assert GROUP in root and "conditions" not in root
    with pytest.raises(ValueError, match="categorical"):
        create_multires_dataset(
            str(path), str(tmp_path / "out2.zarr"), min_level=12, groups="conditions/*"
        )


def test_categorical_variable_in_requested_group(native, tmp_path):
    path, cells, _ = native
    _add_flag_variable(path, cells)
    with pytest.raises(ValueError, match="categorical"):
        create_multires_dataset(
            str(path), str(tmp_path / "out.zarr"), min_level=12, groups=GROUP
        )


def test_non_default_serializer(native, tmp_path):
    path, cells, _ = native
    level20 = zarr.open_group(path, mode="a")[f"{GROUP}/20"]
    level20.create_array(
        "e",
        shape=cells[20].shape,
        chunks=(64,),
        dtype="float32",
        fill_value=np.nan,
        dimension_names=["cells"],
        serializer=zarr.codecs.ShardingCodec(chunk_shape=(16,)),
    )
    with pytest.raises(NotImplementedError, match="serializer"):
        create_multires_dataset(str(path), str(tmp_path / "out.zarr"), min_level=12)


def test_group_patterns(native, tmp_path):
    path, _, _ = native
    out = tmp_path / "out.zarr"
    create_multires_dataset(str(path), str(out), min_level=17, groups="measurements/*")
    assert GROUP in zarr.open_group(out, mode="r")
    with pytest.raises(ValueError, match="no group .* matches"):
        create_multires_dataset(
            str(path), str(tmp_path / "out2.zarr"), min_level=17, groups=["nope/*"]
        )


def test_checks_run_before_writing(native, tmp_path):
    path, _, _ = native
    out = tmp_path / "out.zarr"
    with pytest.raises(ValueError, match="finer than the coarsest native level"):
        create_multires_dataset(str(path), str(out), min_level=18)
    assert not out.exists()
