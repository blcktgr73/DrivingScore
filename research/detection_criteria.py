#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detection criteria utilities for driving events (Phase 4).

This module codifies the updated thresholds:
- Rapid Accel/Decel: Δspeed ≥ 10 km/h/s sustained for 3 seconds
- Sharp Turn: Centrifugal Acceleration Jump ≥ 400 degree m/s^2

Functions here operate on simple 1D time series where applicable so that
real sensor pipelines can plug them in. Phase 4-A/B may simulate series.
"""

from __future__ import annotations

from typing import Iterable, Tuple


# Thresholds (spec-confirmed)
ACCEL_DECEL_THRESHOLD_KMH_PER_S: float = 10.0
SUSTAIN_SECONDS: int = 3
SHARP_TURN_JUMP_THRESHOLD_DEG_MPS2: float = 400.0


def _count_sustained_runs(mask: Iterable[bool], sustain: int) -> int:
    count = 0
    run = 0
    for flag in mask:
        run = run + 1 if flag else 0
        if run >= sustain:
            count += 1
            run = 0  # reset to avoid over-counting overlapping windows
    return count


def count_rapid_accel_events_kmh(speed_series_kmh: Iterable[float], sampling_hz: float) -> int:
    """Count rapid acceleration events using Δspeed ≥ 10 km/h/s sustained for 3s.

    Args:
        speed_series_kmh: sequence of speeds in km/h (uniform sampling).
        sampling_hz: samples per second.
    """
    speeds = list(speed_series_kmh)
    if len(speeds) < 2 or sampling_hz <= 0:
        return 0

    # Per-second slope approximation: Δ(km/h) per second
    # Compute simple finite difference and up-sample boolean mask.
    diffs = [speeds[i+1] - speeds[i] for i in range(len(speeds) - 1)]
    per_sec = [d * sampling_hz for d in diffs]
    mask = [v >= ACCEL_DECEL_THRESHOLD_KMH_PER_S for v in per_sec]
    return _count_sustained_runs(mask, sustain=int(SUSTAIN_SECONDS * sampling_hz))


def count_rapid_decel_events_kmh(speed_series_kmh: Iterable[float], sampling_hz: float) -> int:
    """Count rapid deceleration events using Δspeed ≤ -10 km/h/s sustained for 3s."""
    speeds = list(speed_series_kmh)
    if len(speeds) < 2 or sampling_hz <= 0:
        return 0
    diffs = [speeds[i+1] - speeds[i] for i in range(len(speeds) - 1)]
    per_sec = [d * sampling_hz for d in diffs]
    mask = [v <= -ACCEL_DECEL_THRESHOLD_KMH_PER_S for v in per_sec]
    return _count_sustained_runs(mask, sustain=int(SUSTAIN_SECONDS * sampling_hz))


def count_sharp_turn_events_jump(gyro_deg_per_s: Iterable[float], accel_mps2: Iterable[float], sampling_hz: float,
                                 min_separation_s: float = 1.0) -> int:
    """Count sharp-turn events by Centrifugal Acceleration Jump threshold.

    Metric definition (operational): jump_t = |gyro_deg_per_s[t]| * |accel_mps2[t]|.
    An event is detected when jump_t ≥ 400 (degree m/s^2). Events are debounced by
    min_separation_s to avoid multiple counts in contiguous exceedance regions.
    """
    g = list(gyro_deg_per_s)
    a = list(accel_mps2)
    n = min(len(g), len(a))
    if n == 0 or sampling_hz <= 0:
        return 0
    threshold = SHARP_TURN_JUMP_THRESHOLD_DEG_MPS2
    debound = int(max(1, round(min_separation_s * sampling_hz)))
    events = 0
    cool = 0
    for i in range(n):
        if cool > 0:
            cool -= 1
            continue
        jump_val = abs(g[i]) * abs(a[i])
        if jump_val >= threshold:
            events += 1
            cool = debound
    return events

