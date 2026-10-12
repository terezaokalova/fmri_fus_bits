"""Offline checks of the Griggs loader against a synthetic MATLAB v7.3 layout."""

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from fmri_fus_bits.datasets.griggs import load_session, parse_clock_time
from fmri_fus_bits.sources import ZipSource

SOURCE = ZipSource(
    key="griggs2023_bmi",
    title="synthetic",
    doi="10.0000/synthetic",
    version="1",
    license="CC BY-NC 4.0",
    url="https://example.org/a.zip",
    archive_bytes=1,
    archive_md5="0" * 32,
    cdn_caches_signed_redirect=False,
)
N_FRAMES, WIDTH, DEPTH = 12, 3, 4
CLOCK0 = 12 * 3600.0  # 12:00:00,000
# Two trials, listed out of trialID order; trial 7 is aborted before the cue.
TRIALS = [
    {
        "trialID": 8.0,
        "trialstart": "12:00:03,000",
        "cue": "12:00:03,500",
        "memory": "12:00:04,000",
        "target": [0.0, 0.75],
        "success": 1,
        "effector": "eye_only",
    },
    {
        "trialID": 7.0,
        "trialstart": "12:00:00,500",
        "cue": None,
        "memory": None,
        "target": [0.75, 0.0],
        "success": 0,
        "effector": "BMI_only",
    },
]
FIELDS = [
    "trialID",
    "trialstart",
    "initialfixation",
    "fixationhold",
    "cue",
    "memory",
    "target_acquire",
    "target_hold",
    "reward",
    "iti",
    "targetPos",
    "success",
    "effector",
    "constrainedEffector",
]


def _ref(f: Any, value: Any) -> Any:
    refs = f.require_group("#refs#")
    name = f"r{len(refs)}"
    if value is None:
        ds = refs.create_dataset(name, data=np.zeros(2, dtype=np.uint64))
        ds.attrs["MATLAB_class"] = np.bytes_(b"double")
        ds.attrs["MATLAB_empty"] = np.uint8(1)
    elif isinstance(value, str):
        ds = refs.create_dataset(name, data=np.array([[ord(c)] for c in value], dtype=np.uint16))
        ds.attrs["MATLAB_class"] = np.bytes_(b"char")
    else:
        ds = refs.create_dataset(name, data=np.asarray(value, dtype=np.float64).reshape(-1, 1))
        ds.attrs["MATLAB_class"] = np.bytes_(b"double")
    return ds.ref


def _row(f: Any, values: list[Any]) -> Any:
    return np.array([[_ref(f, v)] for v in values], dtype=h5py.ref_dtype)


def write_session(tmp_path: Path, memory_labels: tuple[float, float] = (2.0, 1.0)) -> Path:
    path = tmp_path / "rt_fUS_data_S5_R1.mat"
    timestamps = CLOCK0 + 0.25 + 0.5 * np.arange(N_FRAMES)
    timestamps[3] += 0.2  # irregular frame interval, as in the real data
    state = np.full(N_FRAMES, 9.0)
    actual = np.full(N_FRAMES, np.nan)
    # Frames at 4.25 s and 4.75 s fall in trial 8's memory period (memory onset 4.0 s).
    state[8:10], actual[8:10] = 5.0, memory_labels[0]
    with h5py.File(path, "w") as f:
        dop = np.arange(N_FRAMES * WIDTH * DEPTH, dtype=np.float64).reshape(N_FRAMES, WIDTH, DEPTH)
        f["dop"] = dop
        for name, column in {
            "timestamps": timestamps,
            "state_num": state,
            "actual_labels": actual,
            "predicted_labels": actual,
            "memory_length": np.full(N_FRAMES, 3.0),
        }.items():
            f[name] = column.reshape(1, -1)
        f["session"], f["run"] = np.array([[5.0]]), np.array([[1.0]])
        f.create_group("coreParams")["framerate"] = np.array([[2.0]])
        uf = f.create_group("UF")
        for name, value in {
            "SVDthreshold": 38.0,
            "N_FlatAngles": 11.0,
            "FrameRate": 500.0,
            "NbOfFrames": 200.0,
        }.items():
            uf[name] = np.array([[value]])
        uf["Probe"] = np.array([[ord(c)] for c in "DOMINO"], dtype=np.uint16)
        behavior = f.create_group("behavior")
        for name in FIELDS:
            values: list[Any] = []
            for trial in TRIALS:
                if name == "targetPos":
                    values.append(trial["target"])
                elif name in ("effector", "constrainedEffector"):
                    values.append(trial["effector"])
                elif name in ("trialID", "success"):
                    values.append(trial[name])
                else:
                    values.append(trial.get(name))
            behavior[name] = _row(f, values)
    return path


def write_record(tmp_path: Path, n_targets: int = 2) -> Path:
    path = tmp_path / "project_record.json"
    path.write_text(
        json.dumps(
            [
                {
                    "Session": 5,
                    "Run": 1,
                    "Date": "23-Mar-2022",
                    "Monkey": "P",
                    "Slot": "cor_9",
                    "Task": "saccade",
                    "nTrials": 2,
                    "nTargets": n_targets,
                    "pretraining": 1,
                    "retraining": 1,
                    "pretraining_sessionrun": "S2R1",
                    "pretraining_file": "N/A",
                    "simulated_results": False,
                    "Notes": [],
                }
            ]
        )
    )
    return path


def test_parse_clock_time() -> None:
    assert parse_clock_time("12:15:05,167") == pytest.approx(44105.167)
    assert parse_clock_time("-00:00:01,500") == pytest.approx(-1.5)
    with pytest.raises(ValueError):
        parse_clock_time("12:15:05.167")


def test_frames_are_depth_by_width_at_native_times(tmp_path: Path) -> None:
    rec = load_session(write_session(tmp_path), write_record(tmp_path), SOURCE)
    assert rec.frames.shape == (N_FRAMES, DEPTH, WIDTH) and rec.frames.dtype == np.float32
    dop = np.arange(N_FRAMES * WIDTH * DEPTH, dtype=np.float64).reshape(N_FRAMES, WIDTH, DEPTH)
    np.testing.assert_array_equal(rec.frames, np.transpose(dop, (0, 2, 1)))
    # Time zero is min(first trial start at 0.5 s, first frame at 0.25 s).
    assert rec.frame_times_s[0] == pytest.approx(0.0)
    assert rec.frame_times_s[4] - rec.frame_times_s[3] == pytest.approx(0.3)


def test_trials_sorted_with_nan_for_missing_events(tmp_path: Path) -> None:
    rec = load_session(write_session(tmp_path), write_record(tmp_path), SOURCE)
    np.testing.assert_array_equal(rec.trials.trial_id, [7, 8])
    np.testing.assert_allclose(rec.trials.event_times_s["trialstart"], [0.25, 2.75])
    assert np.isnan(rec.trials.event_times_s["cue"][0])
    assert rec.trials.event_times_s["memory"][1] == pytest.approx(3.75)
    # Targets at angles 0 and pi/2 become classes 0 and 1.
    np.testing.assert_array_equal(rec.trials.label, [0, 1])
    assert rec.trials.attributes["effector"].tolist() == ["BMI_only", "eye_only"]
    assert rec.provenance["n_targets"] == 2 and rec.provenance["probe"] == "DOMINO"


def test_inconsistent_upstream_labels_are_rejected(tmp_path: Path) -> None:
    session = write_session(tmp_path, memory_labels=(2.0, 1.0))
    with h5py.File(session, "a") as f:
        actual = f["actual_labels"][()]
        actual[0, 9] = 3.0
        f["actual_labels"][...] = actual
    with pytest.raises(ValueError, match="disagree"):
        load_session(session, write_record(tmp_path), SOURCE)


def test_memory_frames_outside_a_memory_period_are_not_attributed(tmp_path: Path) -> None:
    session = write_session(tmp_path)
    with h5py.File(session, "a") as f:
        # Frame 2 (1.0 s) falls in trial 7, which never reached memory: a stray frame
        # like those after the last recorded trial in the real data.
        state, actual = f["state_num"][()], f["actual_labels"][()]
        state[0, 2], actual[0, 2] = 5.0, 9.0
        f["state_num"][...], f["actual_labels"][...] = state, actual
    rec = load_session(session, write_record(tmp_path), SOURCE)
    assert rec.frame_annotations["actual_label"][2] == 9.0


def test_target_count_must_match_project_record(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="distinct targets"):
        load_session(write_session(tmp_path), write_record(tmp_path, n_targets=8), SOURCE)
