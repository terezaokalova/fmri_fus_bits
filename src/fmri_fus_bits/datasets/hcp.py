"""Load one HCP Young Adult motor run (MSMAll grayordinates) at its native TR.

Blocks come from the run's EV files: one file per movement (lf, lh, rf, rh, t), each row
onset (s), duration (s), amplitude, with onsets relative to the first frame. Each block is
preceded by a visual cue (EVs/cue.txt); the cue onset is attached to its block. Motion
estimates (Movement_Regressors.txt, 12 columns) are attached per frame. Nothing is
filtered, normalized or excluded.
"""

import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.cifti2 import cifti2_axes
from numpy.typing import NDArray

from fmri_fus_bits.recording import GrayordinateRecording, ProvenanceValue, Trials

MOVEMENTS = ("lf", "lh", "rf", "rh", "t")
MOTION_COLUMNS = (
    "trans_x_mm",
    "trans_y_mm",
    "trans_z_mm",
    "rot_x_deg",
    "rot_y_deg",
    "rot_z_deg",
    "d_trans_x_mm",
    "d_trans_y_mm",
    "d_trans_z_mm",
    "d_rot_x_deg",
    "d_rot_y_deg",
    "d_rot_z_deg",
)


def load_run(run_dir: Path, subject: str, phase: str) -> GrayordinateRecording:
    """Read one run folder as written by scripts/fetch_hcp.py."""
    path = run_dir / f"tfMRI_MOTOR_{phase}_Atlas_MSMAll.dtseries.nii"
    with path.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    image = nib.load(path)
    series = image.header.get_axis(0)
    models = image.header.get_axis(1)
    if not isinstance(series, cifti2_axes.SeriesAxis) or series.unit != "SECOND":
        raise ValueError(f"{path.name}: first axis must be a series in seconds")
    if not isinstance(models, cifti2_axes.BrainModelAxis):
        raise ValueError(f"{path.name}: second axis must be brain models")
    frames = np.asarray(image.get_fdata(dtype=np.float32))
    # The last structure's slice is open-ended (stop is None), so index a full range.
    all_indices = np.arange(frames.shape[1], dtype=np.int64)
    structures = {str(name): all_indices[s] for name, s, _ in models.iter_structures()}

    motion = np.loadtxt(run_dir / "Movement_Regressors.txt", ndmin=2)
    if motion.shape != (frames.shape[0], len(MOTION_COLUMNS)):
        raise ValueError(f"Movement_Regressors.txt has shape {motion.shape}")
    annotations = {name: motion[:, i] for i, name in enumerate(MOTION_COLUMNS)}

    onsets, durations, labels = [], [], []
    for label, movement in enumerate(MOVEMENTS):
        ev = _read_ev(run_dir / "EVs" / f"{movement}.txt")
        onsets.append(ev[:, 0])
        durations.append(ev[:, 1])
        labels.append(np.full(ev.shape[0], label, dtype=np.int64))
    onset = np.concatenate(onsets)
    order = np.argsort(onset, kind="stable")
    onset = onset[order]
    cue = _read_ev(run_dir / "EVs" / "cue.txt")[:, 0]
    if cue.size != onset.size:
        raise ValueError(f"{run_dir}: {cue.size} cues for {onset.size} movement blocks")
    cue = np.sort(cue)
    if np.any(cue > onset) or np.any(cue[1:] <= onset[:-1]):
        raise ValueError(f"{run_dir}: cues do not interleave with movement blocks")

    blocks = Trials(
        trial_id=np.arange(onset.size, dtype=np.int64),
        label=np.concatenate(labels)[order],
        event_times_s={"cue": cue, "onset": onset},
        attributes={
            "duration_s": np.concatenate(durations)[order],
            "movement": np.array(MOVEMENTS)[np.concatenate(labels)[order]],
        },
    )
    provenance: dict[str, ProvenanceValue] = {
        "dataset": "HCP Young Adult 1200, tfMRI MOTOR",
        "subject": subject,
        "phase_encoding": phase,
        "source_file": path.name,
        "source_sha256": sha256,
        "tr_s": float(series.step),
        "series_start_s": float(series.start),
        "n_grayordinates": int(frames.shape[1]),
    }
    return GrayordinateRecording(
        frames=frames,
        tr_s=float(series.step),
        structures=structures,
        blocks=blocks,
        frame_annotations=annotations,
        provenance=provenance,
    )


def _read_ev(path: Path) -> NDArray[np.float64]:
    ev = np.loadtxt(path, ndmin=2)
    if ev.ndim != 2 or ev.shape[1] != 3:
        raise ValueError(f"{path}: expected rows of onset, duration, amplitude")
    return ev
