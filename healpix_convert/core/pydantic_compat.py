"""Compatibility across the supported pydantic versions."""

from __future__ import annotations

try:
    # pydantic >= 2.14
    from pydantic import MISSING
except ImportError:
    # pydantic < 2.14, where the sentinel is still experimental (importing it
    # from there is deprecated since 2.14)
    from pydantic.experimental.missing_sentinel import MISSING

__all__ = ["MISSING"]
