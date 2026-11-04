"""Reuse of the resamplers between groups (``healpix_convert.resampler_cache``)."""

from __future__ import annotations

import importlib.metadata

import healpix_resample
import numpy as np
import pytest

from healpix_convert import resampler_cache
from healpix_convert.resampler_cache import (
    CACHE_DIR_ENV,
    CACHE_METHODS_ENV,
    build_resampler,
)

LEVEL = 12


@pytest.fixture
def points():
    lon, lat = np.meshgrid(np.linspace(0.0, 0.2, 40), np.linspace(45.0, 45.2, 40))
    lon, lat = lon.ravel(), lat.ravel()
    val = np.sin(lon * 50.0) + lat
    return lon, lat, val


def _kwargs(lon, lat, out_cell_ids=None):
    return dict(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False, out_cell_ids=out_cell_ids,
                nest=True, ellipsoid="WGS84")


def test_second_group_loads_the_resampler(points, monkeypatch, tmp_path):
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))
    monkeypatch.delenv(CACHE_METHODS_ENV, raising=False)
    lon, lat, val = points
    cls = healpix_resample.NearestResampler
    built = []
    real_init = cls.__init__
    monkeypatch.setattr(cls, "__init__", lambda self, *a, **k: (built.append(1), real_init(self, *a, **k))[1])

    first = build_resampler(cls, "nearest", **_kwargs(lon, lat))
    second = build_resampler(cls, "nearest", **_kwargs(lon.copy(), lat.copy()))
    assert len(built) == 1
    assert len(list(tmp_path.glob("*.pkl"))) == 1
    a, b = first.resample(val), second.resample(val)
    np.testing.assert_array_equal(a.cell_ids, b.cell_ids)
    np.testing.assert_array_equal(a.cell_data, b.cell_data)


def test_other_inputs_other_key(points):
    lon, lat, _ = points
    cls = healpix_resample.NearestResampler
    key = resampler_cache.resampler_key(cls, _kwargs(lon, lat))
    assert key == resampler_cache.resampler_key(cls, _kwargs(lon.copy(), lat.copy()))
    moved = lon.copy()
    moved[0] += 1e-9
    assert key != resampler_cache.resampler_key(cls, _kwargs(moved, lat))
    assert key != resampler_cache.resampler_key(cls, _kwargs(lon, lat, out_cell_ids=np.arange(4, dtype=np.uint64)))
    assert key != resampler_cache.resampler_key(cls, {**_kwargs(lon, lat), "level": LEVEL + 1})
    assert key != resampler_cache.resampler_key(healpix_resample.PSFResampler, _kwargs(lon, lat))


def test_other_healpix_resample_code_other_key(points, monkeypatch):
    # same version string, other code behind it (e.g. installed from a source archive)
    lon, lat, _ = points
    cls = healpix_resample.NearestResampler
    key = resampler_cache.resampler_key(cls, _kwargs(lon, lat))
    monkeypatch.setattr(resampler_cache, "_healpix_resample_digest", lambda: b"other code")
    assert key != resampler_cache.resampler_key(cls, _kwargs(lon, lat))


def test_other_torch_other_key(points, monkeypatch):
    import torch

    lon, lat, _ = points
    cls = healpix_resample.NearestResampler
    key = resampler_cache.resampler_key(cls, _kwargs(lon, lat))
    monkeypatch.setattr(torch, "__version__", "0.0.0")
    assert key != resampler_cache.resampler_key(cls, _kwargs(lon, lat))


def test_other_healpix_geo_other_key(points, monkeypatch):
    lon, lat, _ = points
    cls = healpix_resample.NearestResampler
    key = resampler_cache.resampler_key(cls, _kwargs(lon, lat))
    real = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version",
                        lambda name: "0.0.0" if name == "healpix-geo" else real(name))
    assert key != resampler_cache.resampler_key(cls, _kwargs(lon, lat))


def test_package_digest_follows_the_code(tmp_path):
    pkg = tmp_path / "pkg"
    (pkg / "__pycache__").mkdir(parents=True)
    (pkg / "__init__.py").write_text("__version__ = '1.0'\n")
    (pkg / "knn.py").write_text("K = 1\n")
    digest = resampler_cache._package_digest(pkg)
    (pkg / "__pycache__" / "knn.cpython-313.pyc").write_bytes(b"bytecode")
    assert resampler_cache._package_digest(pkg) == digest
    (pkg / "knn.py").write_text("K = 2\n")
    assert resampler_cache._package_digest(pkg) != digest


def test_only_the_listed_methods(points, monkeypatch, tmp_path):
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))
    monkeypatch.delenv(CACHE_METHODS_ENV, raising=False)
    lon, lat, _ = points
    build_resampler(healpix_resample.BilinearResampler, "bilinear", **_kwargs(lon, lat))
    assert not list(tmp_path.glob("*.pkl"))
    monkeypatch.setenv(CACHE_METHODS_ENV, "nearest, bilinear")
    build_resampler(healpix_resample.BilinearResampler, "bilinear", **_kwargs(lon, lat))
    assert len(list(tmp_path.glob("*.pkl"))) == 1


def test_unreadable_file_is_rebuilt(points, monkeypatch, tmp_path):
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))
    monkeypatch.delenv(CACHE_METHODS_ENV, raising=False)
    lon, lat, val = points
    cls = healpix_resample.NearestResampler
    key = resampler_cache.resampler_key(cls, _kwargs(lon, lat))
    (tmp_path / f"{key}.pkl").write_bytes(b"partial")
    r = build_resampler(cls, "nearest", **_kwargs(lon, lat))
    ref = cls(**_kwargs(lon, lat)).resample(val)
    np.testing.assert_array_equal(r.resample(val).cell_data, ref.cell_data)


def test_device_in_the_key_of_the_pickles_only(points):
    lon, lat, _ = points
    cls = healpix_resample.NearestResampler
    cpu, cuda = {**_kwargs(lon, lat), "device": "cpu"}, {**_kwargs(lon, lat), "device": "cuda"}
    # the neighbour links serve every device, a pickled resampler one only
    assert resampler_cache.resampler_key(cls, cpu) == resampler_cache.resampler_key(cls, cuda)
    assert resampler_cache.resampler_key(cls, cpu, with_device=True) != resampler_cache.resampler_key(
        cls, cuda, with_device=True
    )


def test_device_slots_are_exclusive(monkeypatch, tmp_path):
    import threading
    import time

    monkeypatch.setenv("HEALPIX_CONVERT_DEVICE", "cpu")
    monkeypatch.setenv("HEALPIX_CONVERT_GPU_SLOTS", "1")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(resampler_cache.tempfile, "tempdir", None)
    inside, peak, lock = [0], [0], threading.Lock()

    def work():
        with resampler_cache.device_slot():
            with lock:
                inside[0] += 1
                peak[0] = max(peak[0], inside[0])
            time.sleep(0.1)
            with lock:
                inside[0] -= 1

    threads = [threading.Thread(target=work) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1


@pytest.mark.parametrize("slots", ["0", "-1"])
def test_device_slots_at_least_one(monkeypatch, slots):
    monkeypatch.setenv("HEALPIX_CONVERT_DEVICE", "cpu")
    monkeypatch.setenv("HEALPIX_CONVERT_GPU_SLOTS", slots)
    with pytest.raises(ValueError, match="at least 1"):
        with resampler_cache.device_slot():
            pass
