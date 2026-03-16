import importlib.metadata

from legacy_converters import interpolation  # noqa: F401
from legacy_converters.accessor import (  # noqa: F401
    DatasetConverterAccessor,
    DataTreeConverterAccessor,
)
from legacy_converters.convert import create_healpix_dataset
from legacy_converters.settings.common import get_settings

__all__ = ["create_healpix_dataset", "get_settings"]

__version__ = importlib.metadata.version("legacy-converters")
