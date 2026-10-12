"""S3 access to the HCP Young Adult motor task files.

Per run (phase encodings RL and LR) only the files the analyses use are fetched: the
MSMAll grayordinate time series, the task EVs, motion regressors and the physiological
log. Local copies are reused only if their size matches and, for single-part uploads
(ETag without '-'), their MD5 matches the ETag; multipart uploads are size-checked.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fmri_fus_bits.sources import S3Source

# RL was acquired first: its folder holds E-Prime MOTOR_run1_TAB.txt.
PHASE_ENCODINGS = ("RL", "LR")
EV_NAMES = ("cue", "lf", "lh", "rf", "rh", "t")
CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class FetchedFile:
    key: str
    path: Path
    size_bytes: int
    md5_verified: bool
    downloaded: bool


def run_files(phase: str) -> tuple[str, ...]:
    """Files fetched per run, relative to the run's Results folder."""
    return (
        f"tfMRI_MOTOR_{phase}_Atlas_MSMAll.dtseries.nii",
        "Movement_Regressors.txt",
        f"tfMRI_MOTOR_{phase}_Physio_log.txt",
        *(f"EVs/{name}.txt" for name in EV_NAMES),
    )


def run_prefix(source: S3Source, subject: str, phase: str) -> str:
    return f"{source.prefix}/{subject}/MNINonLinear/Results/tfMRI_MOTOR_{phase}"


def local_run_dir(raw_root: Path, source: S3Source, subject: str, phase: str) -> Path:
    return raw_root / source.key / subject / f"tfMRI_MOTOR_{phase}"


def list_subjects(client: Any, source: S3Source) -> list[str]:
    """All subject folder names under the source prefix, sorted."""
    subjects: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=source.bucket, Prefix=f"{source.prefix}/", Delimiter="/"):
        for entry in page.get("CommonPrefixes", []):
            subjects.append(entry["Prefix"].rstrip("/").split("/")[-1])
    return sorted(subjects)


def missing_run_files(client: Any, source: S3Source, subject: str) -> list[str]:
    """Keys of required motor-run files that do not exist for the subject."""
    missing: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for phase in PHASE_ENCODINGS:
        prefix = f"{run_prefix(source, subject, phase)}/"
        present = {
            entry["Key"][len(prefix) :]
            for page in paginator.paginate(Bucket=source.bucket, Prefix=prefix)
            for entry in page.get("Contents", [])
        }
        missing += [f"{prefix}{name}" for name in run_files(phase) if name not in present]
    return missing


def fetch_run(
    client: Any, source: S3Source, subject: str, phase: str, raw_root: Path
) -> list[FetchedFile]:
    """Download (or verify) every file of one motor run."""
    prefix = run_prefix(source, subject, phase)
    out_dir = local_run_dir(raw_root, source, subject, phase)
    return [
        _fetch(client, source.bucket, f"{prefix}/{name}", out_dir / name)
        for name in run_files(phase)
    ]


def _fetch(client: Any, bucket: str, key: str, dest: Path) -> FetchedFile:
    head = client.head_object(Bucket=bucket, Key=key)
    size = int(head["ContentLength"])
    etag = str(head["ETag"]).strip('"')
    single_part = "-" not in etag
    if dest.exists() and dest.stat().st_size == size and (not single_part or _md5(dest) == etag):
        return FetchedFile(key, dest, size, single_part, downloaded=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    client.download_file(bucket, key, str(partial))
    if partial.stat().st_size != size:
        raise OSError(f"{key}: downloaded {partial.stat().st_size} bytes, expected {size}")
    if single_part and _md5(partial) != etag:
        raise OSError(f"{key}: MD5 does not match ETag {etag}")
    partial.replace(dest)
    return FetchedFile(key, dest, size, single_part, downloaded=True)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
