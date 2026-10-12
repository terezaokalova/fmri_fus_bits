"""Offline checks for verified member extraction, using a local archive."""

import hashlib
import zipfile
from pathlib import Path

from fmri_fus_bits.datasets.remote import extract_from_archive

PAYLOAD = bytes(range(256)) * 1000


def make_archive(tmp_path: Path) -> Path:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("folder with space/data.bin", PAYLOAD)
        zf.writestr("other.txt", b"other")
    return archive


def test_member_is_extracted_with_its_sha256(tmp_path: Path) -> None:
    dest = tmp_path / "raw" / "folder with space" / "data.bin"
    with zipfile.ZipFile(make_archive(tmp_path)) as zf:
        result = extract_from_archive(zf, "folder with space/data.bin", dest)
    assert dest.read_bytes() == PAYLOAD
    assert result.sha256 == hashlib.sha256(PAYLOAD).hexdigest()
    assert result.size_bytes == len(PAYLOAD) and result.downloaded
    assert not dest.with_name("data.bin.part").exists()


def test_verified_copy_is_reused(tmp_path: Path) -> None:
    dest = tmp_path / "data.bin"
    with zipfile.ZipFile(make_archive(tmp_path)) as zf:
        extract_from_archive(zf, "folder with space/data.bin", dest)
        again = extract_from_archive(zf, "folder with space/data.bin", dest)
    assert not again.downloaded
    assert again.sha256 == hashlib.sha256(PAYLOAD).hexdigest()


def test_corrupted_copy_of_equal_size_is_replaced(tmp_path: Path) -> None:
    dest = tmp_path / "data.bin"
    dest.write_bytes(b"\x00" * len(PAYLOAD))
    with zipfile.ZipFile(make_archive(tmp_path)) as zf:
        result = extract_from_archive(zf, "folder with space/data.bin", dest)
    assert result.downloaded
    assert dest.read_bytes() == PAYLOAD
