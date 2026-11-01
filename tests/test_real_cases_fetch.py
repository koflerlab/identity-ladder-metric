import hashlib
import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import real_cases_fetch as rf  # noqa: E402


class FakeResponse(io.BytesIO):
    def __init__(self, body, size=None, etag=None):
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body) if size is None else size),
                        "ETag": '"%s"' % (hashlib.md5(body).hexdigest() if etag is None else etag)}


def opener_for(body, **kw):
    calls = []

    def opener(url, timeout=None):
        calls.append(url)
        return FakeResponse(body, **kw)
    opener.calls = calls
    return opener


def test_download_verifies_and_renames(tmp_path):
    body = b"nifti bytes" * 100
    row = rf.fetch_one("sub-x/ses-01/anat/sub-x_ses-01_T1w.nii.gz", str(tmp_path), opener_for(body))
    dest = tmp_path / "sub-x" / "ses-01" / "anat" / "sub-x_ses-01_T1w.nii.gz"
    assert dest.read_bytes() == body and not os.path.exists(str(dest) + ".part")
    assert row["status"] == "downloaded" and row["md5_verified"] and row["sha256"] == hashlib.sha256(body).hexdigest()


def test_existing_scan_is_never_overwritten(tmp_path):
    dest = tmp_path / "sub-x" / "ses-01" / "anat" / "sub-x_ses-01_T1w.nii.gz"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"original")
    op = opener_for(b"new content")
    row = rf.fetch_one("sub-x/ses-01/anat/sub-x_ses-01_T1w.nii.gz", str(tmp_path), op)
    assert dest.read_bytes() == b"original" and row["status"] == "present" and op.calls == []


@pytest.mark.parametrize("kw", [{"size": 5}, {"etag": "0" * 32}])
def test_bad_size_or_md5_leaves_no_scan(tmp_path, kw):
    with pytest.raises(IOError):
        rf.fetch_one("a/b.nii.gz", str(tmp_path), opener_for(b"0123456789", **kw))
    assert not (tmp_path / "a" / "b.nii.gz").exists()


def test_multipart_etag_skips_md5_but_checks_size(tmp_path):
    row = rf.fetch_one("a/b.nii.gz", str(tmp_path), opener_for(b"0123456789", etag="abc-2"))
    assert row["status"] == "downloaded" and not row["md5_verified"]


def test_failures_are_counted_not_hidden(tmp_path):
    def broken(url, timeout=None):
        raise OSError("network down")
    rows, failed = rf.fetch_all([("sub-x", "ses-01"), ("sub-y", "ses-01")], 2, str(tmp_path), broken)
    assert failed == 2 and {r["status"] for r in rows} == {"failed"}


def test_lock_is_exclusive_and_released_only_by_owner(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "LOCK", str(tmp_path / "storage-maintenance.lock"))
    rf.acquire_lock("test")
    with pytest.raises(FileExistsError):
        rf.acquire_lock("second agent")
    rf.release_lock()
    assert not os.path.exists(rf.LOCK)
    with open(rf.LOCK, "w", encoding="utf-8") as f:
        json.dump({"owner": "someone else"}, f)
    with pytest.raises(SystemExit):
        rf.release_lock()
    assert os.path.exists(rf.LOCK)
