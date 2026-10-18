from pathlib import PurePath

import pytest
from pydantic import ValidationError

from healpix_convert.settings.common import (
    ConvertSettings,
    HealpixGroupSettings,
    InputMetadataSettings,
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


def test_input_metadata_settings_default() -> None:
    settings = InputMetadataSettings()

    assert settings.exclude_attrs == []
    assert not settings.is_excluded("history")
    assert settings.filter_attrs({"units": "K"}) == {"units": "K"}
    assert settings.override_attrs == {}
    assert settings.overridden_attrs("radiance") == {}


def test_input_metadata_settings_exclude_attrs_name() -> None:
    settings = InputMetadataSettings(exclude_attrs=["history"])

    assert settings.is_excluded("history")
    # a plain name is matched exactly (not as a substring)
    assert not settings.is_excluded("history_of_appended_files")
    assert not settings.is_excluded("my_history")


def test_input_metadata_settings_exclude_attrs_pattern() -> None:
    settings = InputMetadataSettings(exclude_attrs=["_.*", "history.*"])

    assert settings.is_excluded("_io_config")
    assert settings.is_excluded("history")
    assert settings.is_excluded("history_of_appended_files")
    assert not settings.is_excluded("units")


def test_input_metadata_settings_filter_attrs() -> None:
    settings = InputMetadataSettings(exclude_attrs=["_.*"])

    attrs = {"_io_config": "whatever", "units": "K", "long_name": "temperature"}

    assert settings.filter_attrs(attrs) == {"units": "K", "long_name": "temperature"}
    # the input mapping is left untouched
    assert "_io_config" in attrs


def test_input_metadata_settings_override_attrs_name() -> None:
    settings = InputMetadataSettings(override_attrs={"radiance": {"units": "1"}})

    assert settings.overridden_attrs("radiance") == {"units": "1"}
    # a plain name is matched exactly (not as a substring)
    assert settings.overridden_attrs("radiance_bis") == {}
    assert settings.overridden_attrs("toa_radiance") == {}


def test_input_metadata_settings_override_attrs_pattern() -> None:
    settings = InputMetadataSettings(
        override_attrs={"b0[1-9]|b8a": {"units": "1", "long_name": "reflectance"}}
    )

    assert settings.overridden_attrs("b01") == {
        "units": "1",
        "long_name": "reflectance",
    }
    assert settings.overridden_attrs("b8a") == {
        "units": "1",
        "long_name": "reflectance",
    }
    assert settings.overridden_attrs("quality_mask") == {}


def test_input_metadata_settings_override_attrs_merged() -> None:
    # several patterns may match the same variable: the last one wins
    settings = InputMetadataSettings(
        override_attrs={
            "b.*": {"units": "digital_counts", "source": "eopf"},
            "b01": {"units": "1"},
        }
    )

    assert settings.overridden_attrs("b01") == {"units": "1", "source": "eopf"}
    assert settings.overridden_attrs("b02") == {
        "units": "digital_counts",
        "source": "eopf",
    }


def test_input_metadata_settings_override_attrs_beats_exclude() -> None:
    # the attribute is wrong on the input and needs a value, not a removal
    settings = InputMetadataSettings(
        exclude_attrs=["units"], override_attrs={"radiance": {"units": "1"}}
    )

    assert settings.filter_attrs({"units": "digital_counts"}) == {}
    assert settings.overridden_attrs("radiance") == {"units": "1"}


def test_input_metadata_settings_invalid_pattern() -> None:
    with pytest.raises(ValidationError, match=".*invalid regular expression.*"):
        InputMetadataSettings(exclude_attrs=["[unbalanced"])

    with pytest.raises(ValidationError, match=".*invalid regular expression.*"):
        InputMetadataSettings(override_attrs={"[unbalanced": {"units": "1"}})


def test_input_metadata_settings_extra_forbidden() -> None:
    with pytest.raises(ValidationError):
        InputMetadataSettings(exclude_attr=["history"])


def test_convert_group_settings_metadata() -> None:
    settings_dict = {
        "healpix": {"refinement_level": 10},
        "chunk": {"method": "no_chunk"},
        "resampler": {"name": "nearest"},
    }
    group_settings = HealpixGroupSettings.model_validate(settings_dict)

    assert group_settings.metadata == InputMetadataSettings()

    group_settings = HealpixGroupSettings.model_validate(
        settings_dict | {"metadata": {"exclude_attrs": ["history"]}}
    )

    assert group_settings.metadata.exclude_attrs == ["history"]


def test_convert_settings_metadata_roundtrip() -> None:
    settings = ConvertSettings.model_validate(
        {
            "group_settings": {
                "/root/group": {
                    "healpix": {"refinement_level": 10},
                    "chunk": {"method": "no_chunk"},
                    "resampler": {"name": "nearest"},
                    "metadata": {
                        "exclude_attrs": ["_.*"],
                        "override_attrs": {"radiance": {"units": "1"}},
                    },
                }
            },
        }
    )

    roundtripped = ConvertSettings.from_json(settings.to_json())

    group_settings = roundtripped.group_settings[PurePath("/root/group")]
    assert group_settings.metadata.exclude_attrs == ["_.*"]
    assert group_settings.metadata.override_attrs == {"radiance": {"units": "1"}}


def _group_settings(codecs):
    return {
        "healpix": {"refinement_level": 10},
        "chunk": {"method": "no_chunk"},
        "resampler": {"name": "nearest"},
        "codecs": codecs,
    }


def test_convert_group_settings_codecs() -> None:
    codecs = [{"name": "bytes", "configuration": {}}]

    actual = HealpixGroupSettings.model_validate(_group_settings(codecs))

    assert actual.codecs == codecs


def test_convert_group_settings_unknown_codec() -> None:
    with pytest.raises(ValidationError, match="'no-such-codec' is not available"):
        HealpixGroupSettings.model_validate(
            _group_settings([{"name": "no-such-codec", "configuration": {}}])
        )
