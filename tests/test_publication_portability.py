from __future__ import annotations

import errno
from types import SimpleNamespace

import pandas as pd
import pytest

from prefscope.api._lens_publication import save_lens
from prefscope.api.feature_catalog_io import decode_feature_catalog


def test_lens_publication_uses_portable_catalog_codec(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    lens = SimpleNamespace(
        lens_dir=source,
        input_rep="individual",
        backend=SimpleNamespace(m_total=2),
        projector=None,
        feature_space_identity={
            "feature_space_id": None,
            "feature_space_status": "unbound",
        },
    )

    destination = save_lens(lens, tmp_path / "published")

    names = pd.read_csv(destination / "feature_names.csv")
    assert names["feature_id"].tolist() == [0, 1]
    assert names["concept"].isna().all()
    catalog = decode_feature_catalog(
        (destination / "feature_catalog.json").read_bytes()
    )
    assert catalog.feature_ids == (0, 1)


def test_native_publication_writes_every_declared_feature(tmp_path):
    feature_width = 100_001
    source = tmp_path / "wide-source"
    source.mkdir()
    lens = SimpleNamespace(
        lens_dir=source,
        input_rep="individual",
        backend=SimpleNamespace(m_total=feature_width),
        projector=None,
        feature_space_identity={
            "feature_space_id": None,
            "feature_space_status": "unbound",
        },
    )

    destination = save_lens(lens, tmp_path / "wide-published")
    catalog = decode_feature_catalog(
        (destination / "feature_catalog.json").read_bytes()
    )
    assert len(catalog) == feature_width


def test_windows_failed_lock_acquisition_does_not_unlock_unowned_byte(
    tmp_path, monkeypatch
):
    import prefscope.api._lens_publication as publication

    class FakeLocker:
        LK_NBLCK = 1
        LK_UNLCK = 2

        def __init__(self):
            self.calls = []

        def locking(self, descriptor, operation, count):
            del descriptor, count
            self.calls.append(operation)
            if operation == self.LK_NBLCK:
                raise OSError(errno.EACCES, "busy")

    locker = FakeLocker()
    monkeypatch.setattr(publication, "_locking_module", lambda: locker)
    monkeypatch.setattr(publication.os, "name", "nt")

    with pytest.raises(RuntimeError, match="another active publisher"):
        with publication._publication_lock(tmp_path / "lens"):
            pass

    assert locker.calls == [locker.LK_NBLCK]
