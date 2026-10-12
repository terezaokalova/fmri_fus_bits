"""Extract single members of remote zip archives into a local raw directory.

An archive is read over HTTP range requests; only the requested member is transferred.
The server's archive size must equal the pinned value, and zipfile verifies the member's
CRC32 when its stream is exhausted. A member already on disk is reused only if its size
and CRC32 match the archive entry.
"""

import hashlib
import time
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

from remotezip import RemoteZip

from fmri_fus_bits.sources import ZipSource

REQUEST_TIMEOUT_S = 60
CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class ExtractedMember:
    member: str
    path: Path
    size_bytes: int
    sha256: str
    downloaded: bool


def member_path(raw_root: Path, source: ZipSource, member: str) -> Path:
    """Local path of an archive member: raw_root / source key / member path."""
    return raw_root / source.key / member


def extract_member(source: ZipSource, member: str, raw_root: Path) -> ExtractedMember:
    """Extract one member of a remote archive, or reuse a verified local copy."""
    with RemoteZip(source.request_url(time.time_ns()), timeout=REQUEST_TIMEOUT_S) as zf:
        archive_bytes = int(zf.size())
        if archive_bytes != source.archive_bytes:
            raise ValueError(
                f"{source.key}: server reports {archive_bytes} bytes, "
                f"pinned archive_bytes is {source.archive_bytes}"
            )
        return extract_from_archive(zf, member, member_path(raw_root, source, member))


def extract_from_archive(zf: zipfile.ZipFile, member: str, dest: Path) -> ExtractedMember:
    """Extract `member` from an open archive to `dest`, verifying size and CRC32."""
    info = zf.getinfo(member)
    if dest.exists() and dest.stat().st_size == info.file_size:
        crc, sha256 = _file_digests(dest)
        if crc == info.CRC:
            return ExtractedMember(member, dest, info.file_size, sha256, downloaded=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    with zf.open(info) as stream, partial.open("wb") as out:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
            out.write(chunk)
    partial.replace(dest)
    return ExtractedMember(member, dest, info.file_size, digest.hexdigest(), downloaded=True)


def _file_digests(path: Path) -> tuple[int, str]:
    crc = 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            crc = zlib.crc32(chunk, crc)
            digest.update(chunk)
    return crc, digest.hexdigest()
