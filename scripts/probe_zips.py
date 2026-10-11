"""List every remote-zip source in configs/sources.toml without downloading it.

Each archive's central directory is read over HTTP range requests, so only a few
megabytes are transferred. The archive size reported by the server must equal the
pinned archive_bytes. One CSV per source is written to data/manifests/.

A server that does not honor range requests raises remotezip.RangeNotSupported here,
since selective extraction depends on it. compress_type follows the zip spec:
0 = stored, 8 = deflated.
"""

import csv
import time
from collections import Counter
from pathlib import Path

from remotezip import RemoteZip

from fmri_fus_bits.sources import ZipSource, load_sources

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = REPO_ROOT / "configs" / "sources.toml"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
REQUEST_TIMEOUT_S = 60
GIB = 1024**3


def probe(source: ZipSource, manifest_dir: Path) -> None:
    with RemoteZip(source.request_url(time.time_ns()), timeout=REQUEST_TIMEOUT_S) as zf:
        archive_bytes = int(zf.size())
        infos = list(zf.infolist())
    if archive_bytes != source.archive_bytes:
        raise ValueError(
            f"{source.key}: server reports {archive_bytes} bytes, "
            f"pinned archive_bytes is {source.archive_bytes}"
        )

    out = manifest_dir / f"{source.key}_zip_listing.csv"
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["filename", "file_size", "compress_size", "compress_type", "header_offset"]
        )
        for zi in infos:
            writer.writerow(
                [zi.filename, zi.file_size, zi.compress_size, zi.compress_type, zi.header_offset]
            )

    files = [zi for zi in infos if not zi.is_dir()]
    total = sum(zi.file_size for zi in files)
    types = Counter(zi.compress_type for zi in files)
    top = Counter(zi.filename.split("/")[0] for zi in files)
    print(
        f"{source.key}: {len(files)} files, {total / GIB:.2f} GiB uncompressed, "
        f"compress_type counts {dict(types)}"
    )
    for name, count in top.most_common():
        print(f"  {name}  ({count} files)")
    print(f"  listing written to {out}")


def main() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for source in load_sources(SOURCES_PATH).values():
        if isinstance(source, ZipSource):
            probe(source, MANIFEST_DIR)


if __name__ == "__main__":
    main()
