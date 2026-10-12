"""Offline checks of the HCP motor loader against a synthetic CIFTI run."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from nibabel.cifti2 import cifti2_axes

from fmri_fus_bits.datasets.hcp import MOVEMENTS, load_run

N_FRAMES, TR = 40, 0.72
# One block per movement, each preceded by a 3 s cue, listed out of time order.
BLOCK_ONSETS = {"lf": 20.0, "lh": 5.0, "rf": 12.5, "rh": 27.5, "t": 35.0}


def write_run(tmp_path: Path, cue_onsets: list[float] | None = None) -> Path:
    run = tmp_path / "tfMRI_MOTOR_RL"
    (run / "EVs").mkdir(parents=True)
    left = cifti2_axes.BrainModelAxis.from_mask(np.ones(6, dtype=bool), name="CortexLeft")
    right = cifti2_axes.BrainModelAxis.from_mask(np.ones(4, dtype=bool), name="CortexRight")
    series = cifti2_axes.SeriesAxis(start=0.0, step=TR, size=N_FRAMES)
    data = np.arange(N_FRAMES * 10, dtype=np.float32).reshape(N_FRAMES, 10)
    nib.save(
        nib.Cifti2Image(data, header=(series, left + right)),
        run / "tfMRI_MOTOR_RL_Atlas_MSMAll.dtseries.nii",
    )
    np.savetxt(run / "Movement_Regressors.txt", np.zeros((N_FRAMES, 12)))
    for movement, onset in BLOCK_ONSETS.items():
        (run / "EVs" / f"{movement}.txt").write_text(f"{onset}\t2\t1\n")
    cues = cue_onsets if cue_onsets is not None else [t - 3.0 for t in BLOCK_ONSETS.values()]
    (run / "EVs" / "cue.txt").write_text("".join(f"{c}\t3\t1\n" for c in cues))
    return run


def test_frames_structures_and_blocks(tmp_path: Path) -> None:
    rec = load_run(write_run(tmp_path), "000000", "RL")
    assert rec.frames.shape == (N_FRAMES, 10) and rec.tr_s == pytest.approx(TR)
    np.testing.assert_array_equal(rec.structures["CIFTI_STRUCTURE_CORTEX_LEFT"], np.arange(6))
    np.testing.assert_array_equal(rec.structures["CIFTI_STRUCTURE_CORTEX_RIGHT"], np.arange(6, 10))
    order = sorted(BLOCK_ONSETS, key=BLOCK_ONSETS.__getitem__)
    np.testing.assert_allclose(rec.blocks.event_times_s["onset"], sorted(BLOCK_ONSETS.values()))
    assert rec.blocks.attributes["movement"].tolist() == order
    np.testing.assert_array_equal(rec.blocks.label, [MOVEMENTS.index(m) for m in order])
    np.testing.assert_allclose(
        rec.blocks.event_times_s["onset"] - rec.blocks.event_times_s["cue"], 3.0
    )
    assert rec.frame_times_s[-1] == pytest.approx((N_FRAMES - 1) * TR)


def test_cues_must_interleave_with_blocks(tmp_path: Path) -> None:
    cues = [t - 3.0 for t in BLOCK_ONSETS.values()]
    cues[0] = 30.0  # a cue after its block
    with pytest.raises(ValueError, match="interleave"):
        load_run(write_run(tmp_path, cues), "000000", "RL")
