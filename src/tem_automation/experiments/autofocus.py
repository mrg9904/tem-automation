from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from tem_automation.core.models import AcquisitionSettings
from tem_automation.core.protocols import Microscope


FocusMetric = Callable[[npt.NDArray[np.float32]], float]


@dataclass(frozen=True)
class FocusMeasurement:
    defocus_m: float
    score: float


@dataclass(frozen=True)
class AutofocusResult:
    original_defocus_m: float
    best_defocus_m: float
    best_score: float
    measurements: tuple[FocusMeasurement, ...]


def _box_blur_3x3(image: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    padded = np.pad(image, 1, mode="reflect")
    blurred = np.zeros_like(image)
    for row_offset in range(3):
        for column_offset in range(3):
            blurred += padded[
                row_offset:row_offset + image.shape[0],
                column_offset:column_offset + image.shape[1],
            ]
    return blurred / 9.0


def tenengrad_score(image: npt.NDArray[np.float32]) -> float:
    """Return a noise-resistant, intensity-normalized gradient focus score."""
    data = np.asarray(image, dtype=np.float64)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return float("-inf")

    low, high = np.percentile(finite, (1.0, 99.0))
    dynamic_range = high - low
    if dynamic_range <= np.finfo(np.float64).eps:
        return 0.0

    normalized = np.clip((data - low) / dynamic_range, 0.0, 1.0)
    smoothed = _box_blur_3x3(normalized)
    gradient_y, gradient_x = np.gradient(smoothed)
    gradient_energy = gradient_x * gradient_x + gradient_y * gradient_y

    threshold = np.percentile(gradient_energy, 75.0)
    strong_edges = gradient_energy[gradient_energy >= threshold]
    if strong_edges.size == 0:
        return 0.0
    return float(np.mean(strong_edges))


def autofocus(
    microscope: Microscope,
    acquisition: AcquisitionSettings,
    *,
    search_half_range_m: float = 200e-9,
    coarse_points: int = 9,
    fine_points: int = 7,
    settle_time_s: float = 0.0,
    frames_per_position: int = 1,
    metric: FocusMetric = tenengrad_score,
) -> AutofocusResult:
    """Run coarse-to-fine HAADF autofocus and leave the best focus applied."""
    if search_half_range_m <= 0:
        raise ValueError("search_half_range_m must be positive")
    if coarse_points < 3 or coarse_points % 2 == 0:
        raise ValueError("coarse_points must be an odd integer >= 3")
    if fine_points < 3 or fine_points % 2 == 0:
        raise ValueError("fine_points must be an odd integer >= 3")
    if frames_per_position < 1:
        raise ValueError("frames_per_position must be at least one")

    original_defocus_m = microscope.get_state().defocus_m
    measurements: list[FocusMeasurement] = []

    def evaluate(defocus_m: float) -> FocusMeasurement:
        microscope.set_defocus(float(defocus_m))
        if settle_time_s > 0:
            time.sleep(settle_time_s)
        scores = [
            metric(microscope.acquire(acquisition).data)
            for _ in range(frames_per_position)
        ]
        measurement = FocusMeasurement(
            defocus_m=float(defocus_m),
            score=float(np.median(scores)),
        )
        measurements.append(measurement)
        return measurement

    try:
        coarse_positions = np.linspace(
            original_defocus_m - search_half_range_m,
            original_defocus_m + search_half_range_m,
            coarse_points,
        )
        coarse_measurements = [evaluate(value) for value in coarse_positions]
        coarse_best = max(coarse_measurements, key=lambda item: item.score)

        coarse_step_m = float(coarse_positions[1] - coarse_positions[0])
        fine_positions = np.linspace(
            coarse_best.defocus_m - coarse_step_m,
            coarse_best.defocus_m + coarse_step_m,
            fine_points,
        )

        tested_positions = {
            round(item.defocus_m, 18) for item in measurements
        }
        fine_measurements = []
        for value in fine_positions:
            if round(float(value), 18) not in tested_positions:
                fine_measurements.append(evaluate(float(value)))

        best = max(
            coarse_measurements + fine_measurements,
            key=lambda item: item.score,
        )
        microscope.set_defocus(best.defocus_m)
    except Exception:
        microscope.set_defocus(original_defocus_m)
        raise

    return AutofocusResult(
        original_defocus_m=original_defocus_m,
        best_defocus_m=best.defocus_m,
        best_score=best.score,
        measurements=tuple(measurements),
    )

