from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from adapters.base import Microscope


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
    target_precision_m: float
    final_step_m: float
    rounds: int
    converged: bool
    measurements: tuple[FocusMeasurement, ...]


@dataclass(frozen=True)
class AutofocusConfig:
    """Algorithm parameters kept inside the autofocus layer."""

    precision_fov_fraction: float = 1.0 / 100.0
    # Total initial search width is twice the active FoV: center +/- FoV.
    initial_half_range_fov_fraction: float = 1.0
    points_per_round: int = 7
    max_rounds: int = 12
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
    """Run adaptive HAADF autofocus using the active profile FoV."""
    if config.precision_fov_fraction <= 0:
        raise ValueError("precision_fov_fraction must be positive")
    if config.initial_half_range_fov_fraction <= 0:
        raise ValueError(
            "initial_half_range_fov_fraction must be positive"
        )
    if config.points_per_round < 3 or config.points_per_round % 2 == 0:
        raise ValueError("points_per_round must be an odd integer >= 3")
    if config.max_rounds < 1:
        raise ValueError("max_rounds must be at least one")
    if config.frames_per_position < 1:
        raise ValueError("frames_per_position must be at least one")

    original_defocus_m = microscope.get_defocus()
    fov_m = microscope.get_fov()
    if not np.isfinite(fov_m) or fov_m <= 0:
        raise RuntimeError(f"Microscope returned an invalid FoV: {fov_m!r}")

    target_precision_m = fov_m * config.precision_fov_fraction
    initial_half_range_m = (
        fov_m * config.initial_half_range_fov_fraction
    )

    measurements: list[FocusMeasurement] = []
    measurement_cache: dict[float, FocusMeasurement] = {}

    def evaluate(defocus_m: float) -> FocusMeasurement:
        defocus_m = float(defocus_m)
        cache_key = round(defocus_m, 18)
        cached = measurement_cache.get(cache_key)
        if cached is not None:
            return cached

        microscope.set_defocus(defocus_m)
        if config.settle_time_s > 0:
            time.sleep(config.settle_time_s)
        scores = [
            metric(microscope.acquire_haadf())
            for _ in range(config.frames_per_position)
        ]
        measurement = FocusMeasurement(
            defocus_m=defocus_m,
            score=float(np.median(scores)),
        )
        measurement_cache[cache_key] = measurement
        measurements.append(measurement)
        return measurement

    try:
        center_m = original_defocus_m
        half_range_m = initial_half_range_m
        final_step_m = float("inf")
        rounds = 0
        converged = False
        best: FocusMeasurement | None = None

        for round_index in range(config.max_rounds):
            rounds = round_index + 1
            positions = np.linspace(
                center_m - half_range_m,
                center_m + half_range_m,
                config.points_per_round,
            )
            final_step_m = float(positions[1] - positions[0])
            round_results = [
                evaluate(float(position))
                for position in positions
            ]
            best_index = max(
                range(len(round_results)),
                key=lambda index: round_results[index].score,
            )
            best = round_results[best_index]
            best_is_at_boundary = (
                best_index == 0
                or best_index == len(round_results) - 1
            )

            if (
                final_step_m <= target_precision_m
                and not best_is_at_boundary
            ):
                converged = True
                break

            center_m = best.defocus_m
            if best_is_at_boundary:
                # A boundary maximum does not bracket the focus peak.
                # Expand towards it before attempting finer sampling.
                half_range_m *= 2.0
            else:
                half_range_m = final_step_m

        if best is None:
            raise RuntimeError("Autofocus acquired no measurements")

        best = max(measurements, key=lambda item: item.score)
        microscope.set_defocus(best.defocus_m)
    except Exception:
        microscope.set_defocus(original_defocus_m)
        raise

    return AutofocusResult(
        original_defocus_m=original_defocus_m,
        best_defocus_m=best.defocus_m,
        best_score=best.score,
        target_precision_m=target_precision_m,
        final_step_m=final_step_m,
        rounds=rounds,
        converged=converged,
        measurements=tuple(measurements),
    )
