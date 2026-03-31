import pytest
from pydantic import ValidationError

from legacy_converters.settings.common import (
    ConvertSettings,
    HealpixGroupSettings,
    NearestResamplerSettings,
)


def test_convert_group_settings_resampler_discriminator() -> None:
    settings_dict = {
        "healpix": {"refinement_level": 10},
        "chunk": {"method": "no_chunk"},
        "resampler": {"name": "nearest"},
    }
    group_settings = HealpixGroupSettings.model_validate(settings_dict)

    assert isinstance(group_settings.resampler, NearestResamplerSettings)


def test_convert_group_settings_cellpoint_resampler_level() -> None:
    settings_dict = {
        "healpix": {"refinement_level": 10},
        "chunk": {"method": "no_chunk"},
        "resampler": {"name": "cell-point"},
    }

    with pytest.raises(ValueError, match=".*level must be equal to 29"):
        HealpixGroupSettings.model_validate(settings_dict)


def test_convert_settings_chunk_refinement_level():
    invalid_settings_dict = {
        "group_settings": {
            "/root/group": {
                "healpix": {"refinement_level": 11},
                "chunk": {
                    "method": "healpix_cell_dense",
                    "healpix": {"refinement_level": 16},
                },
                "resampler": {"name": "nearest"},
            }
        },
    }

    with pytest.raises(
        ValidationError, match=".*lower than the chunk refinement level.*"
    ):
        ConvertSettings.model_validate(invalid_settings_dict)
