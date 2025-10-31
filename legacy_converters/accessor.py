from __future__ import annotations

from functools import cached_property

import pyproj
import xarray as xr

from legacy_converters.crs import CRSLike, create_transformer, maybe_convert


@xr.register_datatree_accessor("grid4earth")
class ConverterAccessor:
    def __init__(self, dt: xr.DataTree):
        self._dt = dt

    def _infer_crs_code(self) -> str:
        return self._dt.attrs["other_metadata"]["horizontal_CRS_code"]

    @cached_property
    def crs(self) -> pyproj.CRS:
        """The EOPF tree's spatial CRS"""
        return pyproj.CRS.from_user_input(self._infer_crs_code())

    def convert_to(self, target_crs: CRSLike) -> xr.DataTree:
        """Attach spatial coordinates in the target CRS

        Parameters
        ----------
        target_crs : int or str or pyproj.CRS
            The target CRS, as interpreted by :py:func:`pyproj.CRS.from_user_input`.

        Returns
        -------
        xarray.DataTree
            The current object with new coordinates.
        """
        transformer = create_transformer(self.crs, target_crs)

        return xr.DataTree.from_dict(
            {
                path: maybe_convert(node.ds, transformer)
                for path, node in self._dt.subtree_with_keys
            }
        )
