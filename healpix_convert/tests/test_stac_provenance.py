import importlib.metadata
import json
import socket
import subprocess
from datetime import UTC, datetime

import pytest

from healpix_convert.core import stac
from healpix_convert.core.json_schema import convert_settings_schema_url
from healpix_convert.core.stac import (
    COMPUTE_PROPERTY,
    FACILITY_ENV,
    STAC_PROCESSING_SCHEMA_URL,
    format_stac_derived_links,
    format_stac_processing,
)
from healpix_convert.settings.common import get_settings
from healpix_convert.settings.stac import STACSettings

SHA = "0123456789abcdef0123456789abcdef01234567"
FROM_GIT = {
    "url": "https://example.org/a.git",
    "vcs_info": {"vcs": "git", "commit_id": SHA},
}


def test_settings_are_recorded_verbatim() -> None:
    settings = get_settings("era5")
    properties = format_stac_processing(settings)

    expression = properties["processing:expression"]
    assert expression["format"] == convert_settings_schema_url()
    assert expression["expression"] == settings.to_dict()


def test_settings_can_be_left_out() -> None:
    properties = format_stac_processing(
        get_settings("era5"), include_conversion_settings=False
    )

    assert "processing:expression" not in properties
    # the rest of the provenance does not depend on it
    assert "processing:datetime" in properties
    assert "processing:software" in properties


def test_resampler_and_its_parameters_are_recoverable() -> None:
    # the point of this metadata: two outputs resampled from one input by
    # different methods must be tellable apart
    settings = get_settings("era5")
    properties = format_stac_processing(settings)

    recorded = properties["processing:expression"]["expression"]
    resamplers = {
        group["resampler"]["name"] for group in recorded["group_settings"].values()
    }

    assert resamplers == {
        group.resampler.name for group in settings.group_settings.values()
    }


def test_software_versions_recorded() -> None:
    properties = format_stac_processing(get_settings("era5"))

    software = properties["processing:software"]
    assert "healpix-convert" in software
    assert all(isinstance(version, str) for version in software.values())


def test_processing_datetime_is_utc_and_injectable() -> None:
    stamp = datetime(2026, 8, 18, 12, 0, tzinfo=UTC)
    properties = format_stac_processing(get_settings("era5"), processing_datetime=stamp)

    assert properties["processing:datetime"] == "2026-08-18T12:00:00+00:00"

    # defaults to now, in UTC
    default = format_stac_processing(get_settings("era5"))
    assert datetime.fromisoformat(default["processing:datetime"]).tzinfo is not None


def test_lineage_names_the_inputs() -> None:
    properties = format_stac_processing(
        get_settings("era5"), input_ids=["item-b", "item-a"]
    )

    assert "item-a" in properties["processing:lineage"]
    assert "item-b" in properties["processing:lineage"]


def test_no_lineage_without_inputs() -> None:
    assert "processing:lineage" not in format_stac_processing(get_settings("era5"))


def test_extension_url_is_versioned() -> None:
    assert STAC_PROCESSING_SCHEMA_URL.startswith(
        "https://stac-extensions.github.io/processing/v"
    )
    assert STAC_PROCESSING_SCHEMA_URL.endswith("/schema.json")


def test_lineage_names_the_input_locations() -> None:
    # the minutes ask for the input data's object-store location, not just ids
    properties = format_stac_processing(
        get_settings("era5"),
        input_ids=["item-a"],
        input_paths=["s3://grid4earth/public/era5/in.zarr"],
    )

    assert "s3://grid4earth/public/era5/in.zarr" in properties["processing:lineage"]


def test_derived_from_links() -> None:
    links = format_stac_derived_links(["s3://bucket/a.zarr", "s3://bucket/b.zarr"])

    assert [link["rel"] for link in links] == ["derived_from", "derived_from"]
    assert links[0]["href"] == "s3://bucket/a.zarr"


def test_no_links_without_inputs() -> None:
    assert format_stac_derived_links([]) == []


def test_more_software_versions_recorded() -> None:
    software = format_stac_processing(get_settings("era5"))["processing:software"]

    # installed in any environment that runs these tests
    assert {"healpix-convert", "zarr", "xarray", "numpy"} <= software.keys()


def test_provenance_packages_setting() -> None:
    settings = get_settings("era5")
    settings = settings.model_copy(
        update={
            "metadata": settings.metadata.model_copy(
                update={
                    "stac": STACSettings(
                        provenance_packages=["pydantic", "not-an-installed-package"]
                    )
                }
            )
        }
    )
    software = format_stac_processing(settings)["processing:software"]

    assert software["pydantic"] == importlib.metadata.version("pydantic")
    # not installed: left out rather than recorded wrong
    assert "not-an-installed-package" not in software


class _Dist:
    def __init__(self, version: str, direct_url: dict | None) -> None:
        self.version = version
        self._direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
        return None if self._direct_url is None else json.dumps(self._direct_url)


@pytest.mark.parametrize(
    ("version", "direct_url", "expected"),
    [
        # from an index: the version alone
        ("1.2.0", None, "1.2.0"),
        # from a git repository: the commit too
        (
            "1.2.0",
            FROM_GIT,
            f"1.2.0+git.{SHA[:12]}",
        ),
        # a local version label already: one more segment
        (
            "1.2.1.dev3+gdeadbee",
            FROM_GIT,
            f"1.2.1.dev3+gdeadbee.git.{SHA[:12]}",
        ),
        # the version names that commit already (e.g. setuptools-scm)
        (
            f"1.2.1.dev3+g{SHA[:7]}",
            FROM_GIT,
            f"1.2.1.dev3+g{SHA[:7]}",
        ),
        # an editable install from a directory: no commit known
        ("1.2.0", {"url": "file:///src/a", "dir_info": {"editable": True}}, "1.2.0"),
    ],
)
def test_git_commit_of_installed_packages(
    monkeypatch, version: str, direct_url: dict | None, expected: str
) -> None:
    monkeypatch.setattr(
        stac.importlib.metadata,
        "distribution",
        lambda name: _Dist(version, direct_url),
    )

    assert stac._package_version("a") == expected


def test_compute_recorded_on_cpu(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    compute = format_stac_processing(get_settings("era5"))[COMPUTE_PROPERTY]

    assert compute == {"device": "cpu"}


def test_compute_recorded_on_gpu(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "12.9")
    monkeypatch.setattr(stac, "_gpu_names", lambda: ["NVIDIA A100-SXM4-40GB"])

    compute = format_stac_processing(get_settings("era5"))[COMPUTE_PROPERTY]

    assert compute == {
        "device": "cuda",
        "gpus": ["NVIDIA A100-SXM4-40GB"],
        "cuda": "12.9",
    }


def test_gpu_names_follow_the_visible_devices(monkeypatch) -> None:
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="GPU A\nGPU B\n")

    monkeypatch.setattr(stac.subprocess, "run", run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")

    assert stac._gpu_names() == ["GPU A", "GPU B"]
    assert calls[0][-2:] == ["-i", "1,3"]


def test_gpu_names_without_nvidia_smi(monkeypatch) -> None:
    def run(command, **kwargs):
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(stac.subprocess, "run", run)

    assert stac._gpu_names() == []


def test_facility_from_the_environment(monkeypatch) -> None:
    monkeypatch.delenv(FACILITY_ENV, raising=False)
    assert "processing:facility" not in format_stac_processing(get_settings("era5"))

    monkeypatch.setenv(FACILITY_ENV, "some-cluster")
    properties = format_stac_processing(get_settings("era5"))

    assert properties["processing:facility"] == "some-cluster"


def test_no_host_name_recorded(monkeypatch) -> None:
    # the output may be published: the host the conversion ran on stays out
    monkeypatch.delenv(FACILITY_ENV, raising=False)
    properties = format_stac_processing(get_settings("era5"))

    assert socket.gethostname() not in json.dumps(
        {key: properties[key] for key in properties if key != "processing:expression"}
    )
