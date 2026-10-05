from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from tem_automation.adapters.base import Microscope


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


@dataclass(frozen=True)
class AutofocusConfig:
    """Algorithm parameters kept inside the autofocus layer."""

    search_half_range_m: float = 200e-9
    coarse_points: int = 9
    fine_points: int = 7
    settle_time_s: float = 0.0
    frames_per_position: int = 1


DEFAULT_AUTOFOCUS_CONFIG = AutofocusConfig()


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
    """Return a noise-resistant, intensity-normalized focus score."""
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
    return float(np.mean(strong_edges)) if strong_edges.size else 0.0


def autofocus(
    microscope: Microscope,
    *,
    config: AutofocusConfig = DEFAULT_AUTOFOCUS_CONFIG,
    metric: FocusMetric = tenengrad_score,
) -> AutofocusResult:
    """Run coarse-to-fine HAADF autofocus and apply the best focus."""
    if config.search_half_range_m <= 0:
        raise ValueError("search_half_range_m must be positive")
    if config.coarse_points < 3 or config.coarse_points % 2 == 0:
        raise ValueError("coarse_points must be an odd integer >= 3")
    if config.fine_points < 3 or config.fine_points % 2 == 0:
        raise ValueError("fine_points must be an odd integer >= 3")
    if config.frames_per_position < 1:
        raise ValueError("frames_per_position must be at least one")

    original_defocus_m = microscope.get_defocus()
    measurements: list[FocusMeasurement] = []

    def evaluate(defocus_m: float) -> FocusMeasurement:
        microscope.set_defocus(float(defocus_m))
        if config.settle_time_s > 0:
            time.sleep(config.settle_time_s)
        scores = [
            metric(microscope.acquire_haadf())
            for _ in range(config.frames_per_position)
        ]
        measurement = FocusMeasurement(
            defocus_m=float(defocus_m),
            score=float(np.median(scores)),
        )
        measurements.append(measurement)
        return measurement

    try:
        coarse_positions = np.linspace(
            original_defocus_m - config.search_half_range_m,
            original_defocus_m + config.search_half_range_m,
            config.coarse_points,
        )
        coarse_results = [evaluate(value) for value in coarse_positions]
        coarse_best = max(coarse_results, key=lambda item: item.score)

        coarse_step_m = float(coarse_positions[1] - coarse_positions[0])
        fine_positions = np.linspace(
            coarse_best.defocus_m - coarse_step_m,
            coarse_best.defocus_m + coarse_step_m,
            config.fine_points,
        )
        tested = {round(item.defocus_m, 18) for item in measurements}
        fine_results = [
            evaluate(float(value))
            for value in fine_positions
            if round(float(value), 18) not in tested
        ]

        best = max(coarse_results + fine_results, key=lambda item: item.score)
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
