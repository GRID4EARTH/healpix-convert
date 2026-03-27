import importlib.metadata

from legacy_converters import interpolation  # noqa: F401
from legacy_converters.accessor import (  # noqa: F401
    DatasetConverterAccessor,
    DataTreeConverterAccessor,
)
from legacy_converters.cache import create_staging_cache
from legacy_converters.convert import (
    convert_group_to_healpix,
    create_healpix_dataset,
    prepare_healpix_dataset,
)
from legacy_converters.core.conversion_models import ConvertStagingCache
from legacy_converters.settings.common import ConvertSettings, get_settings

__all__ = [
    "ConvertSettings",
    "ConvertStagingCache",
    "convert_group_to_healpix",
    "create_healpix_dataset",
    "create_staging_cache",
    "get_settings",
    "prepare_healpix_dataset",
]

__version__ = importlib.metadata.version("legacy-converters")
