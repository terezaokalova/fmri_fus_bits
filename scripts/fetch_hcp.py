"""Fetch HCP Young Adult motor runs for the first N subjects with complete run files.

Usage: uv run python scripts/fetch_hcp.py --n-subjects 20
Requires the AWS profile named in configs/sources.toml (default `hcp`). Subjects are
taken in sorted order; subjects missing any required run file (e.g. a physiological
log) are skipped and listed.
"""

import argparse
from pathlib import Path

import boto3

from fmri_fus_bits.datasets.hcp_access import (
    PHASE_ENCODINGS,
    fetch_run,
    list_subjects,
    missing_run_files,
)
from fmri_fus_bits.sources import S3Source, load_sources

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = REPO_ROOT / "configs" / "sources.toml"
RAW_ROOT = REPO_ROOT / "data" / "raw"
SOURCE_KEY = "hcp_ya_motor"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch HCP motor runs.")
    parser.add_argument("--n-subjects", type=int, required=True)
    args = parser.parse_args()
    source = load_sources(SOURCES_PATH)[SOURCE_KEY]
    if not isinstance(source, S3Source):
        raise TypeError(f"{SOURCE_KEY} must be an s3 source")
    client = boto3.Session(profile_name=source.aws_profile, region_name=source.region).client("s3")
    selected: list[str] = []
    for subject in list_subjects(client, source):
        if len(selected) == args.n_subjects:
            break
        missing = missing_run_files(client, source, subject)
        if missing:
            print(f"skipped {subject}: missing {', '.join(missing)}")
            continue
        selected.append(subject)
        for phase in PHASE_ENCODINGS:
            for item in fetch_run(client, source, subject, phase, RAW_ROOT):
                status = "downloaded" if item.downloaded else "verified"
                check = "md5" if item.md5_verified else "size"
                print(f"{status:10s} {item.size_bytes / 1e6:8.1f} MB ({check}) {item.key}")
    print(f"subjects: {' '.join(selected)}")


if __name__ == "__main__":
    main()
