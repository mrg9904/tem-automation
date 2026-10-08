from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from adapters.base import Microscope
from adapters.cancellation import check_cancelled, suspend_cancellation


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
    stop_reason: str = ""


@dataclass(frozen=True)
class AutofocusConfig:
    """Algorithm parameters kept inside the autofocus layer."""

    precision_fov_fraction: float = 1.0 / 100.0
    # Total initial search width is twice the active FoV: center +/- FoV.
    initial_half_range_fov_fraction: float = 1.0
    points_per_round: int = 7
    max_rounds: int = 12
    settle_time_s: float = 0.0
    frames_per_position: int = 3
    max_expansion_rounds: int = 4
    max_search_offset_fov_fraction: float = 16.0
    score_relative_tolerance: float = 0.01
    minimum_precision_m: float = 0.0


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

    # Mean intensity is approximately conserved under defocus. Per-frame
    # contrast normalization incorrectly boosts blurred edges back to full contrast.
    scale = max(abs(float(np.mean(finite))), np.finfo(np.float64).eps)
    normalized = np.clip(data, low, high) / scale
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

    if config.max_expansion_rounds < 0:
        raise ValueError("max_expansion_rounds must be nonnegative")
    if not np.isfinite(config.max_search_offset_fov_fraction) or config.max_search_offset_fov_fraction <= 0:
        raise ValueError("max_search_offset_fov_fraction must be positive")
    if not np.isfinite(config.score_relative_tolerance) or config.score_relative_tolerance < 0:
        raise ValueError("score_relative_tolerance must be nonnegative")
    if not np.isfinite(config.minimum_precision_m) or config.minimum_precision_m < 0:
        raise ValueError("minimum_precision_m must be finite and nonnegative")
    check_cancelled()
    original_defocus_m = microscope.get_defocus()
    fov_m = microscope.get_fov()
    if not np.isfinite(fov_m) or fov_m <= 0:
        raise RuntimeError(f"Microscope returned an invalid FoV: {fov_m!r}")

    target_precision_m = max(fov_m * config.precision_fov_fraction, config.minimum_precision_m)
    initial_half_range_m = (
        fov_m * config.initial_half_range_fov_fraction
    )

    measurements: list[FocusMeasurement] = []
    measurement_cache: dict[float, FocusMeasurement] = {}

    def evaluate(defocus_m: float) -> FocusMeasurement:
        check_cancelled()
        defocus_m = float(defocus_m)
        cache_key = round(defocus_m, 18)
        cached = measurement_cache.get(cache_key)
        if cached is not None:
            return cached

        microscope.set_defocus(defocus_m)
        if config.settle_time_s > 0:
            deadline = time.monotonic() + config.settle_time_s
            while time.monotonic() < deadline:
                check_cancelled()
                time.sleep(min(.05, max(0, deadline - time.monotonic())))
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
        bracket = None
        expansions = 0
        stop_reason = "round_budget"
        limit = fov_m * config.max_search_offset_fov_fraction

        for round_index in range(config.max_rounds):
            rounds = round_index + 1
            check_cancelled()
            lower = max(center_m - half_range_m, original_defocus_m - limit)
            upper = min(center_m + half_range_m, original_defocus_m + limit)
            if bracket is not None:
                lower, upper = max(lower, bracket[0]), min(upper, bracket[1])
            positions = np.linspace(lower, upper, config.points_per_round)
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

            if not best_is_at_boundary and bracket is None:
                bracket = (float(positions[best_index - 1]), float(positions[best_index + 1]))
            if final_step_m <= target_precision_m and bracket is not None:
                converged = True
                stop_reason = "precision_reached"
                break

            scores = np.asarray([item.score for item in round_results])
            if not np.all(np.isfinite(scores)):
                raise RuntimeError("Autofocus metric returned a nonfinite score")
            scale = max(float(np.max(np.abs(scores))), np.finfo(float).eps)
            if bracket is None and np.ptp(scores) <= config.score_relative_tolerance * scale:
                stop_reason = "score_plateau"
                break
            center_m = best.defocus_m
            if best_is_at_boundary and bracket is None:
                if expansions >= config.max_expansion_rounds or lower <= original_defocus_m - limit or upper >= original_defocus_m + limit:
                    stop_reason = "expansion_limit"
                    break
                expansions += 1
                # A boundary maximum does not bracket the focus peak.
                # Expand towards it before attempting finer sampling.
                half_range_m *= 2.0
            else:
                half_range_m = final_step_m

        if best is None:
            raise RuntimeError("Autofocus acquired no measurements")

        best = max(measurements, key=lambda item: (item.score, -abs(item.defocus_m - original_defocus_m)))
        microscope.set_defocus(best.defocus_m)
    except BaseException:
        with suspend_cancellation():
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
        stop_reason=stop_reason,
    )
