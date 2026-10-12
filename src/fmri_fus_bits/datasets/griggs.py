"""Load one Griggs et al. 2023 fUS-BMI session at native frame times.

Each session is a MATLAB v7.3 (HDF5) file; HDF5 axes are reversed relative to MATLAB.
`dop` is (n_frames, width, depth) power Doppler. Per-frame `timestamps` are time of
day in seconds; `state_num`, `actual_labels`, `predicted_labels` and `memory_length`
annotate frames. `behavior` is a struct array whose fields reference char
('HH:MM:SS,mmm' clock times on the same time-of-day clock), double or logical
datasets, or MATLAB empties. Pixels are 0.1 mm (dataset DescriptionOfFiles.pdf).

Times are zeroed at min(first trial start, first frame), as upstream
clean_behavior_and_timestamps.m does. Trials are ordered by trialID. Class labels index
the distinct target positions in order of angle. Nothing is resampled, normalized or
excluded.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from fmri_fus_bits.recording import ImagingRecording, ProvenanceValue, Trials
from fmri_fus_bits.sources import ZipSource

ARCHIVE_ROOT = "fUS BMI Data"
PROJECT_RECORD_MEMBER = f"{ARCHIVE_ROOT}/project_record.json"
# Task states in state_num order 1..9 (upstream load_doppler_data.m).
STATE_NAMES = (
    "trialstart",
    "initialfixation",
    "fixationhold",
    "cue",
    "memory",
    "target_acquire",
    "target_hold",
    "reward",
    "iti",
)
MEMORY_STATE_NUM = STATE_NAMES.index("memory") + 1
PIXEL_SIZE_MM = (0.1, 0.1)
FRAME_BLOCK = 256
_CLOCK = re.compile(r"(-?)(\d{1,2}):(\d{2}):(\d{2}),(\d{3})")


def session_member(session: int, run: int) -> str:
    """Archive member path of one session file."""
    return f"{ARCHIVE_ROOT}/doppler data/rt_fUS_data_S{session}_R{run}.mat"


def parse_clock_time(text: str) -> float:
    """Convert 'HH:MM:SS,mmm', optionally with a leading '-', to seconds."""
    match = _CLOCK.fullmatch(text)
    if match is None:
        raise ValueError(f"Unrecognized clock time {text!r}")
    sign = -1.0 if match.group(1) else 1.0
    hours, minutes, seconds, millis = (int(g) for g in match.groups()[1:])
    return sign * (hours * 3600 + minutes * 60 + seconds + millis / 1000)


def load_session(mat_path: Path, project_record_path: Path, source: ZipSource) -> ImagingRecording:
    """Read one session file and its project_record entry."""
    with mat_path.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    with h5py.File(mat_path, "r") as f:
        session = int(_scalar(f["session"]))
        run = int(_scalar(f["run"]))
        record = _project_record_row(project_record_path, session, run)
        timestamps = _vector(f["timestamps"])
        annotations = {
            "state_num": _vector(f["state_num"]),
            "actual_label": _vector(f["actual_labels"]),
            "predicted_label": _vector(f["predicted_labels"]),
            "memory_length_s": _vector(f["memory_length"]),
        }
        behavior = _read_behavior(f)
        frames = _read_frames(f["dop"])
        uf = f["UF"]
        acquisition: dict[str, ProvenanceValue] = {
            "nominal_frame_rate_hz": float(_scalar(f["coreParams"]["framerate"])),
            "svd_threshold": float(_scalar(uf["SVDthreshold"])),
            "n_plane_wave_angles": int(_scalar(uf["N_FlatAngles"])),
            "compound_rate_hz": float(_scalar(uf["FrameRate"])),
            "compounded_images_per_doppler": int(_scalar(uf["NbOfFrames"])),
            "probe": _char(uf["Probe"]),
        }

    raw_id = np.array([_numeric(v, "trialID", 1)[0] for v in behavior["trialID"]])
    order = np.argsort(raw_id, kind="stable")
    trial_id = raw_id[order]
    if not np.array_equal(trial_id, np.round(trial_id)):
        raise ValueError(f"{mat_path.name}: non-integer trialID")
    events = {
        name: np.array([_clock_or_nan(v, name) for v in behavior[name]])[order]
        for name in STATE_NAMES
    }
    starts = events["trialstart"]
    if not np.all(np.isfinite(starts)) or np.any(np.diff(starts) <= 0):
        raise ValueError(f"{mat_path.name}: trial starts must be finite and increase with trialID")
    t0 = min(float(starts[0]), float(timestamps.min()))
    events = {name: times - t0 for name, times in events.items()}
    frame_times = timestamps - t0

    targets = np.array([_numeric(v, "targetPos", 2) for v in behavior["targetPos"]])[order]
    label, angles = _target_classes(targets)
    if angles.size != int(record["nTargets"]):
        raise ValueError(
            f"{mat_path.name}: {angles.size} distinct targets, project_record says "
            f"{record['nTargets']}"
        )
    _check_label_correspondence(
        label, events["trialstart"], events["memory"], frame_times, annotations, mat_path.name
    )

    trials = Trials(
        trial_id=trial_id.astype(np.int64),
        label=label,
        event_times_s=events,
        attributes={
            "target_x": targets[:, 0],
            "target_y": targets[:, 1],
            "success": np.array([_numeric(v, "success", 1)[0] for v in behavior["success"]])[
                order
            ].astype(bool),
            "effector": np.array([_text(v, "effector") for v in behavior["effector"]])[order],
            "constrained_effector": np.array(
                [_text(v, "constrainedEffector") for v in behavior["constrainedEffector"]]
            )[order],
        },
    )
    provenance: dict[str, ProvenanceValue] = {
        "dataset": source.title,
        "doi": source.doi,
        "version": source.version,
        "license": source.license,
        "member": session_member(session, run),
        "source_sha256": sha256,
        "session": session,
        "run": run,
        "monkey": str(record["Monkey"]),
        "imaging_plane": str(record["Slot"]),
        "task": str(record["Task"]),
        "n_targets": int(record["nTargets"]),
        "pretraining": bool(record["pretraining"]),
        "retraining": bool(record["retraining"]),
        "recording_date": str(record["Date"]),
        "time_zero_clock_s": t0,
        **acquisition,
    }
    return ImagingRecording(
        frames=frames,
        frame_times_s=frame_times,
        pixel_size_mm=PIXEL_SIZE_MM,
        frame_annotations=annotations,
        trials=trials,
        provenance=provenance,
    )


def _project_record_row(path: Path, session: int, run: int) -> dict[str, Any]:
    rows = [r for r in json.loads(path.read_text()) if r["Session"] == session and r["Run"] == run]
    if len(rows) != 1:
        raise ValueError(f"Expected one project_record row for S{session}R{run}, got {len(rows)}")
    row: dict[str, Any] = rows[0]
    return row


def _read_frames(dop: Any) -> NDArray[np.float32]:
    n_frames, width, depth = dop.shape
    frames = np.empty((n_frames, depth, width), dtype=np.float32)
    for start in range(0, n_frames, FRAME_BLOCK):
        stop = min(start + FRAME_BLOCK, n_frames)
        frames[start:stop] = np.transpose(dop[start:stop], (0, 2, 1))
    return frames


def _read_behavior(f: Any) -> dict[str, list[Any]]:
    group = f["behavior"]
    return {
        name: [_dereference(f, ref) for ref in group[name][()].ravel()] for name in group.keys()
    }


def _dereference(f: Any, ref: Any) -> str | NDArray[np.float64] | None:
    obj = f[ref]
    if "MATLAB_empty" in obj.attrs:
        return None
    matlab_class = obj.attrs["MATLAB_class"]
    if matlab_class == b"char":
        return _char(obj)
    if matlab_class in (b"double", b"logical"):
        return np.asarray(obj[()], dtype=np.float64).ravel()
    raise ValueError(f"Unsupported MATLAB class {matlab_class!r} at {obj.name}")


def _char(dataset: Any) -> str:
    return "".join(chr(c) for c in np.asarray(dataset[()]).ravel())


def _scalar(dataset: Any) -> float:
    values = np.asarray(dataset[()], dtype=np.float64).ravel()
    if values.size != 1:
        raise ValueError(f"Expected a scalar at {dataset.name}, got {values.size} values")
    return float(values[0])


def _vector(dataset: Any) -> NDArray[np.float64]:
    return np.asarray(dataset[()], dtype=np.float64).ravel()


def _clock_or_nan(value: Any, field: str) -> float:
    if value is None:
        return float("nan")
    if not isinstance(value, str):
        raise ValueError(f"behavior.{field}: expected a clock-time string, got {value!r}")
    return parse_clock_time(value)


def _numeric(value: Any, field: str, size: int) -> NDArray[np.float64]:
    if not isinstance(value, np.ndarray) or value.size != size:
        raise ValueError(f"behavior.{field}: expected {size} numeric values, got {value!r}")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"behavior.{field}: expected text, got {value!r}")
    return value


def _target_classes(
    targets: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Class index of each trial's target, ordering distinct targets by angle in [0, 2 pi)."""
    distinct, inverse = np.unique(targets, axis=0, return_inverse=True)
    angles = np.mod(np.arctan2(distinct[:, 1], distinct[:, 0]), 2 * np.pi)
    rank = np.argsort(np.argsort(angles, kind="stable"), kind="stable")
    return rank[inverse.ravel()].astype(np.int64), np.sort(angles)


def _check_label_correspondence(
    label: NDArray[np.int64],
    trial_starts: NDArray[np.float64],
    memory_onsets: NDArray[np.float64],
    frame_times: NDArray[np.float64],
    annotations: dict[str, NDArray[np.float64]],
    name: str,
) -> None:
    """Require a one-to-one map between target classes and upstream actual_labels.

    A memory-state frame is attributed to the last trial started at or before it, and
    only if that trial's memory period had begun. Frames from trials missing from the
    behavior table (e.g. after the last recorded trial) are thereby not attributed.
    """
    memory = annotations["state_num"] == MEMORY_STATE_NUM
    times = frame_times[memory]
    trial_index = np.searchsorted(trial_starts, times, side="right") - 1
    valid = trial_index >= 0
    valid[valid] = times[valid] >= memory_onsets[trial_index[valid]]
    pairs = set(
        zip(
            label[trial_index[valid]].tolist(),
            annotations["actual_label"][memory][valid].tolist(),
        )
    )
    classes = {c for c, _ in pairs}
    upstream = {u for _, u in pairs}
    if not (len(pairs) == len(classes) == len(upstream)):
        raise ValueError(f"{name}: target classes and actual_labels disagree: {sorted(pairs)}")
