"""Core recording containers; independent of data access and analysis."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

ProvenanceValue = str | int | float | bool


@dataclass(frozen=True)
class Trials:
    """Per-trial table.

    `label` holds task class indices. `event_times_s` holds event onsets in seconds from
    recording start, NaN where the event did not occur. `attributes` holds
    dataset-specific per-trial columns.
    """

    trial_id: NDArray[np.int64]
    label: NDArray[np.int64]
    event_times_s: dict[str, NDArray[np.float64]]
    attributes: dict[str, NDArray[Any]]

    def __post_init__(self) -> None:
        n = self.trial_id.shape[0]
        if self.trial_id.ndim != 1 or self.label.shape != (n,):
            raise ValueError("trial_id and label must be 1-D arrays of equal length")
        if np.unique(self.trial_id).size != n:
            raise ValueError("trial_id values must be unique")
        for name, column in {**self.event_times_s, **self.attributes}.items():
            if column.shape[0] != n:
                raise ValueError(f"Trial column {name!r} has {column.shape[0]} rows, expected {n}")

    def __len__(self) -> int:
        return int(self.trial_id.shape[0])


@dataclass(frozen=True)
class ImagingRecording:
    """A 2-D imaging time series at native, possibly irregular, frame times.

    `frames` is (n_frames, n_depth, n_width). `frame_times_s` is strictly increasing,
    in seconds from recording start. `frame_annotations` holds per-frame columns.
    """

    frames: NDArray[np.float32]
    frame_times_s: NDArray[np.float64]
    pixel_size_mm: tuple[float, float]
    frame_annotations: dict[str, NDArray[np.float64]]
    trials: Trials
    provenance: dict[str, ProvenanceValue]

    def __post_init__(self) -> None:
        if self.frames.ndim != 3:
            raise ValueError(f"frames must be 3-D (time, depth, width), got {self.frames.shape}")
        n = self.frames.shape[0]
        if self.frame_times_s.shape != (n,):
            raise ValueError(f"frame_times_s has shape {self.frame_times_s.shape}, expected ({n},)")
        if not np.all(np.isfinite(self.frame_times_s)):
            raise ValueError("frame_times_s must be finite")
        if np.any(np.diff(self.frame_times_s) <= 0):
            raise ValueError("frame_times_s must be strictly increasing")
        for name, column in self.frame_annotations.items():
            if column.shape != (n,):
                raise ValueError(
                    f"Frame annotation {name!r} has shape {column.shape}, expected ({n},)"
                )


@dataclass(frozen=True)
class GrayordinateRecording:
    """An fMRI time series on CIFTI grayordinates at a fixed repetition time.

    `frames` is (n_frames, n_grayordinates); frame k is acquired at k * tr_s seconds.
    `structures` maps CIFTI structure names to grayordinate indices. `blocks` holds the
    task blocks; `frame_annotations` holds per-frame columns such as motion estimates.
    """

    frames: NDArray[np.float32]
    tr_s: float
    structures: dict[str, NDArray[np.int64]]
    blocks: Trials
    frame_annotations: dict[str, NDArray[np.float64]]
    provenance: dict[str, ProvenanceValue]

    def __post_init__(self) -> None:
        if self.frames.ndim != 2:
            raise ValueError(f"frames must be 2-D (time, grayordinate), got {self.frames.shape}")
        if not self.tr_s > 0:
            raise ValueError(f"tr_s must be positive, got {self.tr_s}")
        n_frames, n_grayordinates = self.frames.shape
        for name, indices in self.structures.items():
            if indices.size and (indices.min() < 0 or indices.max() >= n_grayordinates):
                raise ValueError(
                    f"Structure {name!r} indexes outside {n_grayordinates} grayordinates"
                )
        for name, column in self.frame_annotations.items():
            if column.shape != (n_frames,):
                raise ValueError(
                    f"Frame annotation {name!r} has shape {column.shape}, expected ({n_frames},)"
                )

    @property
    def frame_times_s(self) -> NDArray[np.float64]:
        return np.arange(self.frames.shape[0], dtype=np.float64) * self.tr_s
