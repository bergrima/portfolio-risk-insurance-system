"""Causal ZigZag calibration and drawdown-gated protection policies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import sqrt
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np
import pandas as pd


class ProtectionState(StrEnum):
    NORMAL = "normal"
    PROTECTED = "protected"


@dataclass(frozen=True, slots=True)
class RegimePolicy:
    """Per-asset thresholds for entering and leaving the VPPI protection state."""

    entry_drawdown: float
    exit_recovery: float

    def __post_init__(self) -> None:
        if not 0 < self.entry_drawdown < 1:
            raise ValueError("entry_drawdown must be between zero and one")
        if not 0 < self.exit_recovery < 1:
            raise ValueError("exit_recovery must be between zero and one")


@dataclass(frozen=True, slots=True)
class CalibrationSettings:
    reversal_threshold: float = 0.05
    continuation_step: float = 0.02
    grid_step: float = 0.01
    minimum_samples: int = 8
    target_probability: float = 0.60

    def __post_init__(self) -> None:
        for name in ("reversal_threshold", "continuation_step", "grid_step"):
            value = float(getattr(self, name))
            if not 0 < value < 1:
                raise ValueError(f"{name} must be between zero and one")
        if self.minimum_samples < 2:
            raise ValueError("minimum_samples must be at least two")
        if not 0.5 <= self.target_probability < 1:
            raise ValueError("target_probability must be at least 50% and below 100%")


@dataclass(frozen=True, slots=True)
class RegimeCalibration:
    asset: str
    policy: RegimePolicy
    training_start: str
    training_end: str
    observations: int
    source_sha256: str
    down_leg_count: int
    up_leg_count: int
    entry_selection_rule: str
    exit_selection_rule: str
    drawdown_distribution: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class CalibrationBundle:
    metadata: Mapping[str, object]
    calibrations: Mapping[str, RegimeCalibration]
    pivots: pd.DataFrame
    legs: pd.DataFrame
    profiles: pd.DataFrame

    @property
    def policies(self) -> Mapping[str, RegimePolicy]:
        return MappingProxyType(
            {asset: calibration.policy for asset, calibration in self.calibrations.items()}
        )


def _validated_levels(levels: pd.Series) -> pd.Series:
    values = levels.astype(float).dropna().sort_index()
    if len(values) < 3:
        raise ValueError("at least three positive level observations are required")
    if not isinstance(values.index, pd.DatetimeIndex) or values.index.has_duplicates:
        raise ValueError("levels require a unique DatetimeIndex")
    if not np.isfinite(values.to_numpy()).all() or (values <= 0).any():
        raise ValueError("levels must be finite and positive")
    return values


def confirmed_zigzag(levels: pd.Series, reversal_threshold: float) -> pd.DataFrame:
    """Return pivots only when the opposite move has confirmed them.

    ``pivot_at`` is the historical extreme. ``confirmed_at`` is the later timestamp
    at which that extreme became observable as a ZigZag pivot. A trading rule may use
    the pivot only from ``confirmed_at`` onward.
    """
    if not 0 < reversal_threshold < 1:
        raise ValueError("reversal_threshold must be between zero and one")
    values = _validated_levels(levels)
    first_at = values.index[0]
    first_value = float(values.iloc[0])
    low_at = high_at = first_at
    low_value = high_value = first_value
    trend: str | None = None
    extreme_at = first_at
    extreme_value = first_value
    rows: list[dict[str, object]] = []

    def append_pivot(kind: str, pivot_at, pivot_value: float, confirmed_at, confirmed: float):
        rows.append(
            {
                "pivot_at": pivot_at,
                "confirmed_at": confirmed_at,
                "pivot_kind": kind,
                "pivot_value": pivot_value,
                "confirmation_value": confirmed,
                "confirmation_move": confirmed / pivot_value - 1,
            }
        )

    for timestamp, raw_value in values.iloc[1:].items():
        value = float(raw_value)
        if trend is None:
            if value < low_value:
                low_at, low_value = timestamp, value
            if value > high_value:
                high_at, high_value = timestamp, value
            up_confirmed = value >= low_value * (1 + reversal_threshold)
            down_confirmed = value <= high_value * (1 - reversal_threshold)
            if up_confirmed and (not down_confirmed or low_at >= high_at):
                append_pivot("low", low_at, low_value, timestamp, value)
                trend, extreme_at, extreme_value = "up", timestamp, value
            elif down_confirmed:
                append_pivot("high", high_at, high_value, timestamp, value)
                trend, extreme_at, extreme_value = "down", timestamp, value
            continue

        if trend == "up":
            if value > extreme_value:
                extreme_at, extreme_value = timestamp, value
            elif value <= extreme_value * (1 - reversal_threshold):
                append_pivot("high", extreme_at, extreme_value, timestamp, value)
                trend, extreme_at, extreme_value = "down", timestamp, value
        else:
            if value < extreme_value:
                extreme_at, extreme_value = timestamp, value
            elif value >= extreme_value * (1 + reversal_threshold):
                append_pivot("low", extreme_at, extreme_value, timestamp, value)
                trend, extreme_at, extreme_value = "up", timestamp, value

    columns = [
        "pivot_at",
        "confirmed_at",
        "pivot_kind",
        "pivot_value",
        "confirmation_value",
        "confirmation_move",
    ]
    return pd.DataFrame(rows, columns=columns)


def zigzag_legs(pivots: pd.DataFrame, levels: pd.Series) -> pd.DataFrame:
    """Build completed peak-to-trough and trough-to-peak legs from confirmed pivots."""
    columns = [
        "start_at",
        "end_at",
        "known_at",
        "direction",
        "start_value",
        "end_value",
        "return",
        "magnitude",
        "duration_bars",
    ]
    if len(pivots) < 2:
        return pd.DataFrame(columns=columns)
    values = _validated_levels(levels)
    positions = {timestamp: number for number, timestamp in enumerate(values.index)}
    rows: list[dict[str, object]] = []
    ordered = pivots.sort_values("pivot_at").reset_index(drop=True)
    for index in range(1, len(ordered)):
        start = ordered.iloc[index - 1]
        end = ordered.iloc[index]
        raw_return = float(end["pivot_value"] / start["pivot_value"] - 1)
        direction = "up" if raw_return > 0 else "down"
        magnitude = raw_return if direction == "up" else -raw_return
        rows.append(
            {
                "start_at": start["pivot_at"],
                "end_at": end["pivot_at"],
                "known_at": end["confirmed_at"],
                "direction": direction,
                "start_value": float(start["pivot_value"]),
                "end_value": float(end["pivot_value"]),
                "return": raw_return,
                "magnitude": magnitude,
                "duration_bars": positions[end["pivot_at"]] - positions[start["pivot_at"]],
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _wilson_interval(successes: int, samples: int, z: float = 1.6448536269514722):
    if samples == 0:
        return float("nan"), float("nan")
    probability = successes / samples
    denominator = 1 + z * z / samples
    centre = (probability + z * z / (2 * samples)) / denominator
    half_width = (
        z
        * sqrt(probability * (1 - probability) / samples + z * z / (4 * samples * samples))
        / denominator
    )
    return centre - half_width, centre + half_width


def continuation_profile(
    legs: pd.DataFrame,
    direction: str,
    settings: CalibrationSettings,
) -> pd.DataFrame:
    """Estimate P(leg extends by continuation_step | it has reached trigger)."""
    if direction not in {"up", "down"}:
        raise ValueError("direction must be up or down")
    magnitudes = legs.loc[legs["direction"] == direction, "magnitude"].astype(float)
    columns = [
        "direction",
        "trigger",
        "additional_move",
        "samples",
        "continued",
        "probability",
        "wilson_low_90",
        "wilson_high_90",
    ]
    if magnitudes.empty:
        return pd.DataFrame(columns=columns)
    largest = float(magnitudes.max()) - settings.continuation_step
    if largest + 1e-12 < settings.reversal_threshold:
        return pd.DataFrame(columns=columns)
    grid_count = int(
        np.floor((largest - settings.reversal_threshold + 1e-12) / settings.grid_step)
    )
    triggers = settings.reversal_threshold + np.arange(grid_count + 1) * settings.grid_step
    rows: list[dict[str, object]] = []
    for trigger in triggers:
        samples = int((magnitudes + 1e-12 >= trigger).sum())
        continued = int(
            (magnitudes + 1e-12 >= trigger + settings.continuation_step).sum()
        )
        low, high = _wilson_interval(continued, samples)
        rows.append(
            {
                "direction": direction,
                "trigger": float(trigger),
                "additional_move": settings.continuation_step,
                "samples": samples,
                "continued": continued,
                "probability": continued / samples if samples else float("nan"),
                "wilson_low_90": low,
                "wilson_high_90": high,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _select_threshold(profile: pd.DataFrame, settings: CalibrationSettings) -> tuple[float, str]:
    supported = profile[profile["samples"] >= settings.minimum_samples].copy()
    if supported.empty:
        raise ValueError(
            "insufficient completed ZigZag legs for the requested minimum_samples"
        )
    qualified = supported[
        (supported["probability"] >= settings.target_probability)
        & (supported["wilson_low_90"] >= 0.50)
    ]
    if not qualified.empty:
        selected = qualified.sort_values(["trigger", "samples"], ascending=[True, False]).iloc[0]
        return float(selected["trigger"]), "target_probability_and_wilson_lower_bound"
    selected = supported.sort_values(
        ["wilson_low_90", "samples", "trigger"], ascending=[False, False, True]
    ).iloc[0]
    return float(selected["trigger"]), "best_supported_wilson_fallback"


def drawdown_distribution(levels: pd.Series) -> Mapping[str, float]:
    values = _validated_levels(levels)
    losses = (1 - values / values.cummax()).clip(lower=0)
    return MappingProxyType(
        {
            "median": float(losses.quantile(0.50)),
            "q75": float(losses.quantile(0.75)),
            "q90": float(losses.quantile(0.90)),
            "q95": float(losses.quantile(0.95)),
            "q99": float(losses.quantile(0.99)),
            "maximum": float(losses.max()),
        }
    )


def calibrate_asset_regime(
    asset: str,
    levels: pd.Series,
    settings: CalibrationSettings | None = None,
) -> tuple[RegimeCalibration, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    settings = settings or CalibrationSettings()
    values = _validated_levels(levels)
    pivots = confirmed_zigzag(values, settings.reversal_threshold)
    legs = zigzag_legs(pivots, values)
    down_profile = continuation_profile(legs, "down", settings)
    up_profile = continuation_profile(legs, "up", settings)
    entry, entry_rule = _select_threshold(down_profile, settings)
    exit_threshold, exit_rule = _select_threshold(up_profile, settings)
    canonical = values.to_csv(date_format="%Y-%m-%dT%H:%M:%S.%f%z")
    calibration = RegimeCalibration(
        asset=asset,
        policy=RegimePolicy(entry, exit_threshold),
        training_start=values.index[0].isoformat(),
        training_end=values.index[-1].isoformat(),
        observations=len(values),
        source_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
        down_leg_count=int((legs["direction"] == "down").sum()),
        up_leg_count=int((legs["direction"] == "up").sum()),
        entry_selection_rule=entry_rule,
        exit_selection_rule=exit_rule,
        drawdown_distribution=drawdown_distribution(values),
    )
    pivots = pivots.assign(asset=asset)
    legs = legs.assign(asset=asset)
    profiles = pd.concat([down_profile, up_profile], ignore_index=True).assign(asset=asset)
    return calibration, pivots, legs, profiles


def calibrate_regimes(
    levels_by_asset: Mapping[str, pd.Series],
    *,
    valid_after: pd.Timestamp,
    settings: CalibrationSettings | None = None,
) -> CalibrationBundle:
    settings = settings or CalibrationSettings()
    boundary = pd.Timestamp(valid_after)
    calibrations: dict[str, RegimeCalibration] = {}
    pivot_frames: list[pd.DataFrame] = []
    leg_frames: list[pd.DataFrame] = []
    profile_frames: list[pd.DataFrame] = []
    for asset, raw_levels in sorted(levels_by_asset.items()):
        levels = raw_levels[raw_levels.index < boundary]
        if levels.empty:
            raise ValueError(f"{asset} has no observations before valid_after")
        calibration, pivots, legs, profiles = calibrate_asset_regime(
            asset, levels, settings
        )
        calibrations[asset] = calibration
        pivot_frames.append(pivots)
        leg_frames.append(legs)
        profile_frames.append(profiles)
    payload = {
        "version": 1,
        "valid_after": boundary.isoformat(),
        "settings": asdict(settings),
        "assets": {
            asset: {
                "asset": calibration.asset,
                "training_start": calibration.training_start,
                "training_end": calibration.training_end,
                "observations": calibration.observations,
                "source_sha256": calibration.source_sha256,
                "down_leg_count": calibration.down_leg_count,
                "up_leg_count": calibration.up_leg_count,
                "entry_selection_rule": calibration.entry_selection_rule,
                "exit_selection_rule": calibration.exit_selection_rule,
                "policy": asdict(calibration.policy),
                "drawdown_distribution": dict(calibration.drawdown_distribution),
            }
            for asset, calibration in calibrations.items()
        },
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    metadata = MappingProxyType(
        {**payload, "calibration_id": hashlib.sha256(canonical.encode()).hexdigest()}
    )
    return CalibrationBundle(
        metadata,
        MappingProxyType(calibrations),
        pd.concat(pivot_frames, ignore_index=True),
        pd.concat(leg_frames, ignore_index=True),
        pd.concat(profile_frames, ignore_index=True),
    )


def export_regime_calibration(bundle: CalibrationBundle, directory: str | Path) -> Path:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    (target / "calibration.json").write_text(
        json.dumps(dict(bundle.metadata), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    bundle.pivots.to_csv(target / "pivots.csv", index=False)
    bundle.legs.to_csv(target / "legs.csv", index=False)
    bundle.profiles.to_csv(target / "continuation_profiles.csv", index=False)
    return target


def load_regime_calibration(path: str | Path) -> CalibrationBundle:
    source = Path(path)
    if source.is_dir():
        source = source / "calibration.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError("unsupported regime calibration version")
    recorded_id = payload.pop("calibration_id", None)
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    actual_id = hashlib.sha256(canonical.encode()).hexdigest()
    if recorded_id != actual_id:
        raise ValueError("regime calibration ID does not match its contents")
    calibrations: dict[str, RegimeCalibration] = {}
    for asset, values in payload["assets"].items():
        item = dict(values)
        item["policy"] = RegimePolicy(**item["policy"])
        item["drawdown_distribution"] = MappingProxyType(item["drawdown_distribution"])
        calibrations[asset] = RegimeCalibration(**item)
    metadata = MappingProxyType({**payload, "calibration_id": actual_id})
    directory = source.parent
    frames = []
    for name in ("pivots.csv", "legs.csv", "continuation_profiles.csv"):
        csv_path = directory / name
        frames.append(pd.read_csv(csv_path) if csv_path.is_file() else pd.DataFrame())
    return CalibrationBundle(
        metadata,
        MappingProxyType(calibrations),
        frames[0],
        frames[1],
        frames[2],
    )
