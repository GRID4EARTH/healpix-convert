"""Aligning the non-spatial dimensions of several input datasets (#17, #54)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from healpix_convert import create_healpix_dataset
from healpix_convert.healpix_converters import align_extra_dimensions
from healpix_convert.settings.common import (
    ConvertSettings,
    ExtraDimensionSettings,
    HealpixGroupSettings,
)
from healpix_convert.tests.test_batch_resample import STAC_ITEM

SPATIAL_DIMS = ["y", "x"]


def _tie_points(detectors: list[str], value: float) -> xr.Dataset:
    """A tie-point grid like Sentinel-2's ``conditions/geometry``: angles per
    detector for the detectors that see the tile, opened without default
    indexes (as the converter opens its inputs).
    """
    angles = np.full((len(detectors), 3, 3), value, dtype="f4")
    ds = xr.Dataset(
        {
            "viewing_incidence_angles": (("detector", "y", "x"), angles),
            "sun_angles": (("y", "x"), np.ones((3, 3), dtype="f4")),
        },
        coords={
            "detector": ("detector", detectors),
            "x": ("x", value + np.arange(3, dtype="f4")),
            "y": ("y", np.arange(3, dtype="f4")),
        },
    )
    return ds.drop_indexes(list(ds.indexes))


TILES = [_tie_points(["d01", "d02", "d03"], 0.0), _tie_points(["d02", "d03"], 10.0)]


class TestExtraDimensionSettings:
    @pytest.mark.parametrize(
        ["settings", "match"],
        [
            pytest.param(
                {"join": "reindex"}, "non-empty list of labels", id="reindex-without-labels"
            ),
            pytest.param(
                {"join": "reindex", "labels": ["a", "a"]}, "duplicate labels", id="duplicate-labels"
            ),
            pytest.param(
                {"join": "outer", "labels": ["a"]},
                "only used with join='reindex'",
                id="labels-without-reindex",
            ),
        ],
    )
    def test_invalid(self, settings, match):
        with pytest.raises(ValidationError, match=match):
            ExtraDimensionSettings.model_validate(settings)

    def test_cell_dimension_is_not_an_extra_dimension(self):
        with pytest.raises(ValidationError, match="HEALPix cell dimension"):
            HealpixGroupSettings.model_validate(
                {
                    "healpix": {"refinement_level": 10},
                    "chunk": {"method": "no_chunk"},
                    "resampler": {"name": "nearest"},
                    "extra_dimensions": {"cells": {"join": "outer"}},
                }
            )


class TestAlignExtraDimensions:
    @pytest.mark.parametrize(
        ["settings", "expected"],
        [
            pytest.param({"join": "outer"}, ["d01", "d02", "d03"], id="outer"),
            pytest.param({"join": "inner"}, ["d02", "d03"], id="inner"),
            pytest.param(
                {"join": "reindex", "labels": ["d01", "d02", "d03", "d04"]},
                ["d01", "d02", "d03", "d04"],
                id="reindex",
            ),
        ],
    )
    def test_labels(self, settings, expected):
        extra = {"detector": ExtraDimensionSettings.model_validate(settings)}

        aligned = align_extra_dimensions(TILES, extra, SPATIAL_DIMS)

        for ds, value in zip(aligned, [0.0, 10.0]):
            assert list(ds["detector"].values) == expected
            angles = ds["viewing_incidence_angles"]
            # an input's own detectors keep their values, the others are missing
            for label in expected:
                own = label in TILES[[0.0, 10.0].index(value)]["detector"].values
                selected = angles.sel(detector=label)
                assert (selected == value).all() if own else selected.isnull().all()
            # a variable without the dimension is left as it is
            assert (ds["sun_angles"] == 1.0).all()

    def test_exact_join_refuses_different_labels(self):
        extra = {"detector": ExtraDimensionSettings(join="exact")}

        with pytest.raises(ValueError, match="exact"):
            align_extra_dimensions(TILES, extra, SPATIAL_DIMS)

    def test_reindex_refuses_unknown_labels(self):
        extra = {"detector": ExtraDimensionSettings(join="reindex", labels=["d02", "d03"])}

        with pytest.raises(ValueError, match=r"input labels \['d01'\] are not in"):
            align_extra_dimensions(TILES, extra, SPATIAL_DIMS)

    @pytest.mark.parametrize(
        "tiles",
        [
            pytest.param(TILES, id="different-sizes"),
            # same size, different detectors: would be concatenated position by
            # position without the check
            pytest.param(
                [_tie_points(["d01", "d02"], 0.0), _tie_points(["d02", "d03"], 10.0)],
                id="same-size-different-labels",
            ),
        ],
    )
    def test_differing_inputs_without_settings(self, tiles):
        with pytest.raises(ValueError, match="set 'extra_dimensions'"):
            align_extra_dimensions(tiles, {}, SPATIAL_DIMS)

    def test_same_labels_without_settings(self):
        tiles = [_tie_points(["d01", "d02"], 0.0), _tie_points(["d01", "d02"], 10.0)]

        aligned = align_extra_dimensions(tiles, {}, SPATIAL_DIMS)

        for ds, original in zip(aligned, tiles):
            xr.testing.assert_identical(ds, original)


def _write_input(path, lon0: float, detectors: list[str]) -> str:
    """A lat/lon input with a ``detector`` dimension, 0.1 degree apart from
    its neighbour."""
    lon = lon0 + np.linspace(0.0, 0.9, 10)
    lat = np.linspace(0.0, 0.9, 10)
    values = np.stack(
        [np.full((10, 10), 10.0 * (i + 1), dtype="f4") for i in range(len(detectors))]
    )
    ds = xr.Dataset(
        {"angles": (("detector", "lat", "lon"), values)},
        coords={
            "detector": ("detector", detectors),
            "lon": ("lon", lon, {"standard_name": "longitude", "units": "degrees_east"}),
            "lat": ("lat", lat, {"standard_name": "latitude", "units": "degrees_north"}),
        },
    )
    tree = xr.DataTree.from_dict({"/geometry": ds})
    tree.attrs["stac_discovery"] = STAC_ITEM
    tree.to_zarr(path, mode="w", consolidated=True)
    return str(path)


def test_no_chunk_conversion_with_fixed_labels(tmp_path):
    """Two inputs seeing different detectors give an output with the detectors
    of the settings, in their order, whatever each input holds."""
    inputs = [
        _write_input(tmp_path / "a.zarr", 0.0, ["d02", "d01"]),
        _write_input(tmp_path / "b.zarr", 1.0, ["d03", "d02"]),
    ]
    labels = ["d01", "d02", "d03", "d04"]
    settings = ConvertSettings.model_validate(
        {
            "group_settings": {
                "geometry": {
                    "healpix": {"refinement_level": 6, "indexing_scheme": "nested"},
                    "chunk": {"method": "no_chunk"},
                    "resampler": {"name": "nearest"},
                    "extra_dimensions": {"detector": {"join": "reindex", "labels": labels}},
                }
            }
        }
    )
    output_path = str(tmp_path / "output.zarr")

    create_healpix_dataset(inputs, settings, output_path)

    ds = xr.open_dataset(output_path, engine="zarr", group="geometry")
    assert list(ds["detector"].values) == labels
    angles = ds["angles"]
    assert angles.dims == ("detector", "cells")
    # d04 is seen by no input, d02 by both
    assert angles.sel(detector="d04").isnull().all()
    assert angles.sel(detector="d02").notnull().any()
    # each detector keeps its own value (10, 20 by position in each input)
    assert set(np.unique(angles.sel(detector="d01").dropna("cells"))) == {20.0}
    assert set(np.unique(angles.sel(detector="d03").dropna("cells"))) == {10.0}
