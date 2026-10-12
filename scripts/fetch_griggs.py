"""Extract Griggs et al. 2023 session files and metadata into data/raw, verified.

Usage: uv run python scripts/fetch_griggs.py 9 14 16
Each session is run 1 (the only run in the release). Files already on disk are reused
only if their size and CRC32 match the archive entry.
"""

import argparse
from pathlib import Path

from fmri_fus_bits.datasets.griggs import ARCHIVE_ROOT, PROJECT_RECORD_MEMBER, session_member
from fmri_fus_bits.datasets.remote import extract_member
from fmri_fus_bits.sources import ZipSource, load_sources

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = REPO_ROOT / "configs" / "sources.toml"
RAW_ROOT = REPO_ROOT / "data" / "raw"
SOURCE_KEY = "griggs2023_bmi"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Griggs et al. 2023 sessions.")
    parser.add_argument("sessions", type=int, nargs="+", help="session numbers")
    args = parser.parse_args()
    source = load_sources(SOURCES_PATH)[SOURCE_KEY]
    if not isinstance(source, ZipSource):
        raise TypeError(f"{SOURCE_KEY} must be a remote_zip source")
    members = [PROJECT_RECORD_MEMBER, f"{ARCHIVE_ROOT}/DescriptionOfFiles.pdf"]
    members += [session_member(session, 1) for session in args.sessions]
    for member in members:
        result = extract_member(source, member, RAW_ROOT)
        status = "downloaded" if result.downloaded else "verified"
        print(f"{status:10s} {result.size_bytes / 1e9:6.2f} GB  {result.sha256}  {member}")


if __name__ == "__main__":
    main()
