"""Reproducible Stage 3 stress testing and Monte Carlo scenario analysis."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import partial
from itertools import pairwise, product
from math import sqrt
from pathlib import Path
from statistics import NormalDist
from types import MappingProxyType
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pandas as pd

from .backtest import (
    AssetTransactionCost,
    BacktestPolicy,
    ExecutionModel,
    ReviewFrequency,
    Strategy,
    run_backtest,
)
from .metrics import (
    drawdown_series_from_levels,
    maximum_drawdown_from_levels,
    maximum_underwater_duration,
)
from .regime import RegimePolicy

_INFLATION_COLUMN = "__inflation__"
_FX_COLUMN = "__fx__"


class SimulationMethod(StrEnum):
    MOVING_BLOCK_BOOTSTRAP = "moving_block_bootstrap"
    STATIONARY_BOOTSTRAP = "stationary_bootstrap"
    REGIME_SWITCHING = "regime_switching"


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    method: SimulationMethod = SimulationMethod.MOVING_BLOCK_BOOTSTRAP
    paths: int = 1_000
    horizon_bars: int = 126
    seed: int = 20260824
    block_size: int = 10
    student_t_df: float = 5
    confidence: float = 0.90
    tail_confidence: float = 0.95

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", SimulationMethod(self.method))
        if self.paths <= 0 or self.horizon_bars <= 0:
            raise ValueError("paths and horizon_bars must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.student_t_df <= 2:
            raise ValueError("student_t_df must exceed two")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be between zero and one")
        if not 0 < self.tail_confidence < 1:
            raise ValueError("tail_confidence must be between zero and one")


@dataclass(frozen=True, slots=True)
class HistoricalStressWindow:
    name: str
    start: str | pd.Timestamp
    end: str | pd.Timestamp

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("historical window name cannot be empty")
        if pd.Timestamp(self.start) > pd.Timestamp(self.end):
            raise ValueError("historical window start must not follow its end")


@dataclass(frozen=True, slots=True)
class StressScenario:
    """A deterministic shock applied to every generated or historical path.

    Asset and FX shocks are simple returns composed with the path return at ``shock_bar``.
    FX betas translate an exchange-rate jump into local asset returns. Liquidity settings
    apply over the full scenario horizon because the Stage 2 execution model is path-wide.
    """

    name: str
    shock_bar: int = 1
    asset_shocks: Mapping[str, float] | None = None
    fx_jump: float = 0
    fx_asset_betas: Mapping[str, float] | None = None
    liquidity_cost_multiplier: float = 1
    extra_slippage_bps: float = 0
    latency_bars_addition: int = 0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("stress scenario name cannot be empty")
        if self.shock_bar < 1:
            raise ValueError("shock_bar must be at least one")
        if self.fx_jump <= -1:
            raise ValueError("fx_jump must be greater than -100%")
        if self.liquidity_cost_multiplier < 1 or self.extra_slippage_bps < 0:
            raise ValueError("liquidity stress cannot reduce costs or slippage")
        if self.latency_bars_addition < 0:
            raise ValueError("latency_bars_addition cannot be negative")
        shocks = dict(self.asset_shocks or {})
        if any(value <= -1 for value in shocks.values()):
            raise ValueError("asset shocks must be greater than -100%")
        object.__setattr__(self, "asset_shocks", MappingProxyType(shocks))
        object.__setattr__(
            self, "fx_asset_betas", MappingProxyType(dict(self.fx_asset_betas or {}))
        )


@dataclass(frozen=True, slots=True)
class ScenarioPath:
    path_id: str
    generator: str
    scenario: str
    prices: pd.DataFrame
    inflation_index: pd.Series | None = None
    fx_reference: pd.Series | None = None
    seed: int | None = None
    tradable: pd.DataFrame | None = None


@dataclass(frozen=True, slots=True)
class ScenarioAnalysisResult:
    surface: pd.DataFrame
    path_metrics: pd.DataFrame
    metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _RegimeModel:
    means: np.ndarray
    covariances: tuple[np.ndarray, np.ndarray]
    transition: np.ndarray
    initial_probabilities: np.ndarray


def _validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.astype(float)
    if len(frame) < 3:
        raise ValueError("scenario generation requires at least three price observations")
    if (
        not isinstance(frame.index, pd.DatetimeIndex)
        or not frame.index.is_monotonic_increasing
        or frame.index.has_duplicates
    ):
        raise ValueError("prices require a sorted, unique DatetimeIndex")
    if frame.columns.has_duplicates or frame.isna().any().any():
        raise ValueError("scenario prices require unique columns and no missing values")
    if not np.isfinite(frame.to_numpy()).all() or (frame <= 0).any().any():
        raise ValueError("scenario prices must be finite and positive")
    return frame


def _align_tradability(tradable: pd.DataFrame | None, prices: pd.DataFrame) -> pd.DataFrame:
    if tradable is None:
        return pd.DataFrame(True, index=prices.index, columns=prices.columns)
    if not tradable.index.equals(prices.index) or set(tradable.columns) != set(prices.columns):
        raise ValueError("tradability index and assets must match scenario prices")
    aligned = tradable.reindex(columns=prices.columns)
    if aligned.isna().any().any():
        raise ValueError("scenario tradability cannot contain missing values")
    return aligned.astype(bool)


def _align_level_series(
    values: pd.Series | None, index: pd.DatetimeIndex, label: str
) -> pd.Series | None:
    if values is None:
        return None
    series = values.astype(float).sort_index()
    if not isinstance(series.index, pd.DatetimeIndex) or series.index.has_duplicates:
        raise ValueError(f"{label} requires a unique DatetimeIndex")
    aligned = series.reindex(index).ffill()
    if aligned.isna().any():
        raise ValueError(f"{label} does not cover the beginning of the price history")
    if not np.isfinite(aligned.to_numpy()).all() or (aligned <= 0).any():
        raise ValueError(f"{label} must be finite and positive")
    return aligned.rename(label)


def _joint_returns(
    prices: pd.DataFrame,
    inflation_index: pd.Series | None,
    fx_reference: pd.Series | None,
) -> tuple[pd.DataFrame, pd.Series]:
    levels = prices.copy()
    inflation = _align_level_series(inflation_index, prices.index, _INFLATION_COLUMN)
    fx = _align_level_series(fx_reference, prices.index, _FX_COLUMN)
    if inflation is not None:
        levels[_INFLATION_COLUMN] = inflation
    if fx is not None:
        levels[_FX_COLUMN] = fx
    returns = levels.pct_change(fill_method=None).iloc[1:]
    if (returns <= -1).any().any() or not np.isfinite(returns.to_numpy()).all():
        raise ValueError("historical returns must be finite and greater than -100%")
    return returns, levels.iloc[-1]


def _future_index(index: pd.DatetimeIndex, horizon_bars: int) -> pd.DatetimeIndex:
    active_weekdays = set(index.weekday)
    timestamps = [index[-1]]
    if len(active_weekdays) < 7 and len(index) >= 7:
        candidate = index[-1]
        while len(timestamps) <= horizon_bars:
            candidate += pd.Timedelta(days=1)
            if candidate.weekday() in active_weekdays:
                timestamps.append(candidate)
    else:
        deltas = np.diff(index.asi8)
        step_ns = int(np.median(deltas))
        if step_ns <= 0:
            raise ValueError("cannot infer a positive scenario time step")
        for _ in range(horizon_bars):
            timestamps.append(timestamps[-1] + pd.Timedelta(step_ns, unit="ns"))
    return pd.DatetimeIndex(timestamps, name=index.name)


def _moving_block_indices(
    observations: int, horizon_bars: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    if block_size > observations:
        raise ValueError("block_size cannot exceed the available return observations")
    sampled: list[int] = []
    last_start = observations - block_size
    while len(sampled) < horizon_bars:
        start = int(rng.integers(0, last_start + 1))
        sampled.extend(range(start, start + block_size))
    return np.asarray(sampled[:horizon_bars], dtype=int)


def _stationary_indices(
    observations: int, horizon_bars: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    indices = np.empty(horizon_bars, dtype=int)
    current = int(rng.integers(0, observations))
    restart_probability = 1 / block_size
    for position in range(horizon_bars):
        if position and rng.random() < restart_probability:
            current = int(rng.integers(0, observations))
        elif position:
            current = (current + 1) % observations
        indices[position] = current
    return indices


def _fit_regime_model(returns: pd.DataFrame) -> _RegimeModel:
    values = returns.to_numpy(dtype=float)
    centered = values - np.median(values, axis=0)
    scale = np.std(values, axis=0, ddof=1)
    scale = np.where(scale > 1e-12, scale, 1.0)
    volatility_score = np.mean(np.square(centered / scale), axis=1)
    labels = (volatility_score > np.median(volatility_score)).astype(int)
    if min(np.bincount(labels, minlength=2)) < 2:
        order = np.argsort(volatility_score)
        labels = np.zeros(len(values), dtype=int)
        labels[order[len(values) // 2 :]] = 1

    overall_covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    dimension = values.shape[1]
    overall_covariance = overall_covariance.reshape(dimension, dimension)
    diagonal_scale = max(float(np.trace(overall_covariance)) / max(dimension, 1), 1e-10)
    jitter = np.eye(dimension) * diagonal_scale * 1e-8
    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    for state in (0, 1):
        members = values[labels == state]
        means.append(members.mean(axis=0))
        covariance = (
            np.atleast_2d(np.cov(members, rowvar=False, ddof=1)).reshape(dimension, dimension)
            if len(members) > 1
            else overall_covariance
        )
        covariances.append(covariance + jitter)

    transition_counts = np.ones((2, 2), dtype=float)
    for previous, current in pairwise(labels):
        transition_counts[previous, current] += 1
    transition = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    initial_counts = np.bincount(labels, minlength=2).astype(float) + 1
    return _RegimeModel(
        np.vstack(means),
        (covariances[0], covariances[1]),
        transition,
        initial_counts / initial_counts.sum(),
    )


def _regime_switching_returns(
    model: _RegimeModel,
    horizon_bars: int,
    student_t_df: float,
    rng: np.random.Generator,
) -> np.ndarray:
    dimension = model.means.shape[1]
    simulated = np.empty((horizon_bars, dimension), dtype=float)
    state = int(rng.choice(2, p=model.initial_probabilities))
    for bar in range(horizon_bars):
        if bar:
            state = int(rng.choice(2, p=model.transition[state]))
        normal = rng.multivariate_normal(np.zeros(dimension), model.covariances[state])
        tail_scale = sqrt(float(rng.chisquare(student_t_df)) / student_t_df)
        simulated[bar] = model.means[state] + normal / tail_scale
    return np.clip(simulated, -0.95, 3.0)


def _compound_joint_path(
    sampled_returns: pd.DataFrame,
    initial_levels: pd.Series,
    index: pd.DatetimeIndex,
    asset_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None]:
    factors = (1 + sampled_returns).cumprod()
    levels = factors.mul(initial_levels, axis=1)
    levels.index = index[1:]
    initial = initial_levels.to_frame().T
    initial.index = index[:1]
    levels = pd.concat([initial, levels])
    prices = levels[list(asset_columns)].astype(float)
    inflation = (
        levels[_INFLATION_COLUMN].rename("inflation") if _INFLATION_COLUMN in levels else None
    )
    fx = levels[_FX_COLUMN].rename("fx") if _FX_COLUMN in levels else None
    return prices, inflation, fx


def generate_scenario_paths(
    prices: pd.DataFrame,
    config: SimulationConfig,
    inflation_index: pd.Series | None = None,
    fx_reference: pd.Series | None = None,
    tradable: pd.DataFrame | None = None,
) -> tuple[ScenarioPath, ...]:
    """Generate joint asset/macro paths with independent deterministic child seeds."""
    prices = _validate_prices(prices)
    source_tradability = _align_tradability(tradable, prices).iloc[1:]
    returns, initial_levels = _joint_returns(prices, inflation_index, fx_reference)
    if config.method == SimulationMethod.MOVING_BLOCK_BOOTSTRAP and config.block_size > len(
        returns
    ):
        raise ValueError("block_size cannot exceed the available return observations")
    model = (
        _fit_regime_model(returns) if config.method == SimulationMethod.REGIME_SWITCHING else None
    )
    output_index = _future_index(prices.index, config.horizon_bars)
    child_sequences = np.random.SeedSequence(config.seed).spawn(config.paths)
    paths: list[ScenarioPath] = []
    for number, child_sequence in enumerate(child_sequences):
        rng = np.random.default_rng(child_sequence)
        if config.method == SimulationMethod.MOVING_BLOCK_BOOTSTRAP:
            sampled_indices = _moving_block_indices(
                len(returns), config.horizon_bars, config.block_size, rng
            )
            sampled = returns.iloc[sampled_indices].reset_index(drop=True)
            sampled_tradability = source_tradability.iloc[sampled_indices].reset_index(drop=True)
        elif config.method == SimulationMethod.STATIONARY_BOOTSTRAP:
            sampled_indices = _stationary_indices(
                len(returns), config.horizon_bars, config.block_size, rng
            )
            sampled = returns.iloc[sampled_indices].reset_index(drop=True)
            sampled_tradability = source_tradability.iloc[sampled_indices].reset_index(drop=True)
        else:
            assert model is not None
            sampled = pd.DataFrame(
                _regime_switching_returns(model, config.horizon_bars, config.student_t_df, rng),
                columns=returns.columns,
            )
            sampled_tradability = pd.DataFrame(
                True, index=range(config.horizon_bars), columns=prices.columns
            )
        path_prices, path_inflation, path_fx = _compound_joint_path(
            sampled, initial_levels, output_index, list(prices.columns)
        )
        seed_fingerprint = int(child_sequence.generate_state(1, dtype=np.uint32)[0])
        sampled_tradability.index = output_index[1:]
        initial_tradability = pd.DataFrame(True, index=output_index[:1], columns=prices.columns)
        path_tradability = pd.concat([initial_tradability, sampled_tradability])
        paths.append(
            ScenarioPath(
                path_id=f"{config.method.value}-{number:06d}",
                generator=config.method.value,
                scenario="baseline",
                prices=path_prices,
                inflation_index=path_inflation,
                fx_reference=path_fx,
                seed=seed_fingerprint,
                tradable=path_tradability,
            )
        )
    return tuple(paths)


def historical_stress_paths(
    prices: pd.DataFrame,
    windows: Sequence[HistoricalStressWindow],
    inflation_index: pd.Series | None = None,
    fx_reference: pd.Series | None = None,
    tradable: pd.DataFrame | None = None,
) -> tuple[ScenarioPath, ...]:
    """Extract named, realized windows without changing their original timestamps."""
    prices = _validate_prices(prices)
    inflation = _align_level_series(inflation_index, prices.index, "inflation")
    fx = _align_level_series(fx_reference, prices.index, "fx")
    source_tradability = _align_tradability(tradable, prices)
    paths: list[ScenarioPath] = []
    for window in windows:
        start = pd.Timestamp(window.start)
        end = pd.Timestamp(window.end)
        if prices.index.tz is not None:
            start = (
                start.tz_localize(prices.index.tz)
                if start.tzinfo is None
                else start.tz_convert(prices.index.tz)
            )
            end = (
                end.tz_localize(prices.index.tz)
                if end.tzinfo is None
                else end.tz_convert(prices.index.tz)
            )
        elif start.tzinfo is not None or end.tzinfo is not None:
            start = start.tz_localize(None)
            end = end.tz_localize(None)
        selected = prices.loc[start:end]
        if len(selected) < 2:
            raise ValueError(f"historical window {window.name!r} contains fewer than two bars")
        paths.append(
            ScenarioPath(
                path_id=f"historical-{window.name}",
                generator="historical",
                scenario=window.name,
                prices=selected,
                inflation_index=None if inflation is None else inflation.reindex(selected.index),
                fx_reference=None if fx is None else fx.reindex(selected.index),
                tradable=source_tradability.reindex(selected.index),
            )
        )
    return tuple(paths)


def apply_stress(path: ScenarioPath, stress: StressScenario) -> ScenarioPath:
    """Apply a deterministic market/FX shock to a path without changing prior bars."""
    if stress.shock_bar >= len(path.prices):
        raise ValueError(f"stress bar {stress.shock_bar} lies outside path {path.path_id}")
    unknown = (set(stress.asset_shocks) | set(stress.fx_asset_betas)) - set(path.prices.columns)
    if unknown:
        raise ValueError(f"stress references unknown assets: {sorted(unknown)}")

    returns = path.prices.pct_change(fill_method=None).fillna(0.0)
    for asset in path.prices.columns:
        direct_shock = stress.asset_shocks.get(asset, 0.0)
        fx_shock = (1 + stress.fx_jump) ** stress.fx_asset_betas.get(asset, 0.0) - 1
        combined_shock = (1 + direct_shock) * (1 + fx_shock) - 1
        returns.iloc[stress.shock_bar, returns.columns.get_loc(asset)] = (
            1 + returns.iloc[stress.shock_bar][asset]
        ) * (1 + combined_shock) - 1
    stressed_prices = (1 + returns).cumprod().mul(path.prices.iloc[0], axis=1)

    stressed_fx = path.fx_reference
    if stressed_fx is not None and stress.fx_jump:
        fx_returns = stressed_fx.pct_change(fill_method=None).fillna(0.0)
        fx_returns.iloc[stress.shock_bar] = (1 + fx_returns.iloc[stress.shock_bar]) * (
            1 + stress.fx_jump
        ) - 1
        stressed_fx = stressed_fx.iloc[0] * (1 + fx_returns).cumprod()
        stressed_fx.name = path.fx_reference.name

    scenario = f"{path.scenario}+{stress.name}" if path.generator == "historical" else stress.name
    return ScenarioPath(
        path_id=path.path_id,
        generator=path.generator,
        scenario=scenario,
        prices=stressed_prices,
        inflation_index=path.inflation_index,
        fx_reference=stressed_fx,
        seed=path.seed,
        tradable=path.tradable,
    )


def standard_stress_scenarios(
    assets: Sequence[str], reserve_asset: str = "fixed_income"
) -> tuple[StressScenario, ...]:
    """Return transparent Stage 3 FX, joint-decline, and liquidity stress presets."""
    risky_assets = [asset for asset in assets if asset != reserve_asset]
    decline_defaults = {"gold": -0.15, "equity": -0.25}
    fx_beta_defaults = {"gold": 1.0, "equity": 0.25}
    return (
        StressScenario(
            name="exchange_rate_jump",
            fx_jump=0.25,
            fx_asset_betas={asset: fx_beta_defaults.get(asset, 0.5) for asset in risky_assets},
        ),
        StressScenario(
            name="simultaneous_asset_decline",
            asset_shocks={asset: decline_defaults.get(asset, -0.20) for asset in risky_assets},
        ),
        StressScenario(
            name="reduced_liquidity",
            liquidity_cost_multiplier=3,
            extra_slippage_bps=25,
            latency_bars_addition=1,
        ),
    )


def constant_inflation_index(
    index: pd.DatetimeIndex, annual_rate: float, periods_per_year: int = 252
) -> pd.Series:
    if annual_rate <= -1:
        raise ValueError("annual inflation rate must be greater than -100%")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    growth = (1 + annual_rate) ** (np.arange(len(index)) / periods_per_year)
    return pd.Series(100 * growth, index=index, name="inflation")


def _stress_execution(base: ExecutionModel, stress: StressScenario | None) -> ExecutionModel:
    if stress is None:
        return base
    multiplier = stress.liquidity_cost_multiplier
    return ExecutionModel(
        transaction_cost_bps=base.transaction_cost_bps * multiplier,
        slippage_bps=base.slippage_bps * multiplier + stress.extra_slippage_bps,
        latency_bars=base.latency_bars + stress.latency_bars_addition,
        asset_costs=tuple(
            AssetTransactionCost(
                asset=item.asset,
                buy_bps=item.buy_bps * multiplier,
                sell_bps=item.sell_bps * multiplier,
            )
            for item in base.asset_costs
        ),
    )


def _total_change(levels: pd.Series | None) -> float:
    if levels is None:
        return float("nan")
    return float(levels.iloc[-1] / levels.iloc[0] - 1)


def _quantile_summary(values: pd.Series, confidence: float, prefix: str) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_lower": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_upper": float("nan"),
        }
    alpha = (1 - confidence) / 2
    return {
        f"{prefix}_mean": float(clean.mean()),
        f"{prefix}_lower": float(clean.quantile(alpha)),
        f"{prefix}_median": float(clean.median()),
        f"{prefix}_upper": float(clean.quantile(1 - alpha)),
    }


def _tail_loss_summary(values: pd.Series, confidence: float, prefix: str) -> dict[str, float]:
    """Summarize a non-negative loss distribution at the requested tail confidence."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            f"{prefix}_tail_quantile": float("nan"),
            f"{prefix}_expected_shortfall": float("nan"),
        }
    if (clean < 0).any():
        raise ValueError(f"{prefix} must contain non-negative losses")
    threshold = float(clean.quantile(confidence))
    return {
        f"{prefix}_tail_quantile": threshold,
        f"{prefix}_expected_shortfall": float(clean[clean >= threshold].mean()),
    }


def _maximum_consecutive_true(values: pd.Series) -> int:
    longest = current = 0
    for value in values.astype(bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _floor_path_metrics(nav: pd.Series, floor: pd.Series, prefix: str) -> dict[str, float | bool]:
    aligned_nav, aligned_floor = nav.astype(float).align(floor.astype(float), join="inner")
    if aligned_nav.empty or (aligned_floor <= 0).any():
        raise ValueError("floor metrics require aligned positive floor observations")
    shortfall = (1 - aligned_nav / aligned_floor).clip(lower=0)
    breached = shortfall > 0
    return {
        f"{prefix}ever_breached": bool(breached.any()),
        f"{prefix}breach_bar_fraction": float(breached.mean()),
        f"{prefix}maximum_shortfall": float(shortfall.max()),
        f"{prefix}conditional_shortfall": (
            float(shortfall[breached].mean()) if breached.any() else 0.0
        ),
        f"{prefix}maximum_breach_duration_bars": float(_maximum_consecutive_true(breached)),
        f"{prefix}terminal_shortfall": float(shortfall.iloc[-1]),
    }


def _relative_level_path(nav: pd.Series, reference: pd.Series | None) -> pd.Series | None:
    if reference is None:
        return None
    aligned = reference.astype(float).reindex(nav.index).ffill()
    if aligned.isna().any() or (aligned <= 0).any():
        raise ValueError("relative-level benchmark must cover the full NAV path and stay positive")
    return nav.astype(float) / (aligned / aligned.iloc[0])


def _wilson_interval(successes: int, observations: int, confidence: float) -> tuple[float, float]:
    if observations <= 0:
        return float("nan"), float("nan")
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    probability = successes / observations
    denominator = 1 + z * z / observations
    center = (probability + z * z / (2 * observations)) / denominator
    margin = (
        z
        / denominator
        * sqrt(probability * (1 - probability) / observations + z * z / (4 * observations**2))
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _aggregate_surface(
    path_metrics: pd.DataFrame, confidence: float, tail_confidence: float
) -> pd.DataFrame:
    keys = [
        "generator",
        "scenario",
        "multiplier",
        "floor_fraction",
        "review_frequency",
        "drift_band",
    ]
    rows: list[dict[str, object]] = []
    for group_values, group in path_metrics.groupby(keys, sort=True, dropna=False):
        row = dict(zip(keys, group_values, strict=True))
        row["n_paths"] = len(group)
        for metric in (
            "nominal_return",
            "real_return",
            "usd_relative_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "maximum_drawdown",
            "drawdown_loss",
            "maximum_drawdown_duration_bars",
            "real_maximum_drawdown",
            "real_drawdown_loss",
            "real_maximum_drawdown_duration_bars",
            "usd_relative_maximum_drawdown",
            "usd_relative_drawdown_loss",
            "usd_relative_maximum_drawdown_duration_bars",
            "maximum_floor_shortfall",
            "conditional_floor_shortfall",
            "maximum_floor_breach_duration_bars",
            "terminal_floor_shortfall",
            "real_floor_maximum_shortfall",
            "real_floor_conditional_shortfall",
            "real_floor_maximum_breach_duration_bars",
            "real_floor_terminal_shortfall",
            "turnover",
            "total_cost",
            "total_transaction_cost",
            "total_slippage_cost",
            "market_closed_bar_fraction",
            "deferred_execution_bars",
        ):
            row.update(_quantile_summary(group[metric], confidence, metric))
        for metric in (
            "drawdown_loss",
            "real_drawdown_loss",
            "usd_relative_drawdown_loss",
            "maximum_floor_shortfall",
            "real_floor_maximum_shortfall",
        ):
            row.update(_tail_loss_summary(group[metric], tail_confidence, metric))
        breaches = int(group["ever_breached"].sum())
        lower, upper = _wilson_interval(breaches, len(group), confidence)
        row["floor_breach_probability"] = breaches / len(group)
        row["floor_breach_probability_lower"] = lower
        row["floor_breach_probability_upper"] = upper
        real_floor_observations = group["real_floor_ever_breached"].notna()
        real_floor_breaches = int(
            group.loc[real_floor_observations, "real_floor_ever_breached"].sum()
        )
        real_floor_lower, real_floor_upper = _wilson_interval(
            real_floor_breaches, int(real_floor_observations.sum()), confidence
        )
        row["real_floor_breach_probability"] = (
            real_floor_breaches / int(real_floor_observations.sum())
            if real_floor_observations.any()
            else float("nan")
        )
        row["real_floor_breach_probability_lower"] = real_floor_lower
        row["real_floor_breach_probability_upper"] = real_floor_upper
        real_observations = group["real_return"].notna()
        real_losses = int((group.loc[real_observations, "real_return"] < 0).sum())
        loss_lower, loss_upper = _wilson_interval(
            real_losses, int(real_observations.sum()), confidence
        )
        row["real_loss_probability"] = (
            real_losses / int(real_observations.sum()) if real_observations.any() else float("nan")
        )
        row["real_loss_probability_lower"] = loss_lower
        row["real_loss_probability_upper"] = loss_upper
        nominal_losses = int((group["nominal_return"] < 0).sum())
        nominal_loss_lower, nominal_loss_upper = _wilson_interval(
            nominal_losses, len(group), confidence
        )
        row["nominal_loss_probability"] = nominal_losses / len(group)
        row["nominal_loss_probability_lower"] = nominal_loss_lower
        row["nominal_loss_probability_upper"] = nominal_loss_upper
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def _serializable_stress(stress: StressScenario) -> dict[str, object]:
    return {
        "name": stress.name,
        "shock_bar": stress.shock_bar,
        "asset_shocks": dict(stress.asset_shocks),
        "fx_jump": stress.fx_jump,
        "fx_asset_betas": dict(stress.fx_asset_betas),
        "liquidity_cost_multiplier": stress.liquidity_cost_multiplier,
        "extra_slippage_bps": stress.extra_slippage_bps,
        "latency_bars_addition": stress.latency_bars_addition,
    }


def _series_hash(values: pd.Series | None) -> str | None:
    if values is None:
        return None
    canonical = values.astype(float).to_csv(date_format="%Y-%m-%dT%H:%M:%S.%f%z")
    return hashlib.sha256(canonical.encode()).hexdigest()


def _frame_hash(values: pd.DataFrame | None) -> str | None:
    if values is None:
        return None
    canonical = values.astype(int).to_csv(date_format="%Y-%m-%dT%H:%M:%S.%f%z")
    return hashlib.sha256(canonical.encode()).hexdigest()


def _evaluate_base_path(
    base_path: ScenarioPath,
    *,
    serialized_stresses: Sequence[Mapping[str, object]],
    weights: Mapping[str, float],
    multipliers: Sequence[float],
    floors: Sequence[float],
    frequencies: Sequence[ReviewFrequency],
    drift_bands: Sequence[float],
    execution: ExecutionModel,
    reserve_daily_return: float | None,
    reserve_calendar_day_accrual: bool,
    regime_policies: Mapping[str, RegimePolicy] | None,
    regime_calibration_id: str | None,
    initial_nav: float,
    inflation_benchmark_name: str,
) -> list[dict[str, object]]:
    stresses = tuple(StressScenario(**dict(values)) for values in serialized_stresses)
    variants: list[tuple[ScenarioPath, StressScenario | None]] = [(base_path, None)]
    variants.extend((apply_stress(base_path, stress), stress) for stress in stresses)
    rows: list[dict[str, object]] = []
    for path, stress in variants:
        path_execution = _stress_execution(execution, stress)
        inflation_return = _total_change(path.inflation_index)
        fx_return = _total_change(path.fx_reference)
        for multiplier, floor, frequency, drift_band in product(
            multipliers, floors, frequencies, drift_bands
        ):
            policy = BacktestPolicy(
                strategy=Strategy.VPPI,
                multiplier=multiplier,
                floor_fraction=floor,
                review_frequency=frequency,
                drift_band=drift_band,
                execution=path_execution,
                reserve_daily_return=reserve_daily_return,
                reserve_calendar_day_accrual=reserve_calendar_day_accrual,
                regime_policies=regime_policies,
                regime_calibration_id=regime_calibration_id,
            )
            result = run_backtest(
                path.prices,
                weights,
                policy,
                initial_nav,
                tradable=path.tradable,
            )
            nominal_return = float(result.summary["total_return"])
            nav = result.ledger["nav"].astype(float)
            nominal_drawdowns = drawdown_series_from_levels(nav)
            real_nav = _relative_level_path(nav, path.inflation_index)
            fx_relative_nav = _relative_level_path(nav, path.fx_reference)
            real_maximum_drawdown = (
                maximum_drawdown_from_levels(real_nav) if real_nav is not None else float("nan")
            )
            usd_relative_maximum_drawdown = (
                maximum_drawdown_from_levels(fx_relative_nav)
                if fx_relative_nav is not None
                else float("nan")
            )
            nominal_floor = result.ledger["floor_value"].astype(float)
            nominal_floor_metrics = _floor_path_metrics(nav, nominal_floor, "")
            if path.inflation_index is not None:
                benchmark = path.inflation_index.astype(float).reindex(nav.index).ffill()
                real_floor = nominal_floor * (benchmark / benchmark.iloc[0])
                real_floor_metrics: dict[str, float | bool] = _floor_path_metrics(
                    nav, real_floor, "real_floor_"
                )
            else:
                real_floor_metrics = {
                    "real_floor_ever_breached": float("nan"),
                    "real_floor_breach_bar_fraction": float("nan"),
                    "real_floor_maximum_shortfall": float("nan"),
                    "real_floor_conditional_shortfall": float("nan"),
                    "real_floor_maximum_breach_duration_bars": float("nan"),
                    "real_floor_terminal_shortfall": float("nan"),
                }
            real_return = (
                (1 + nominal_return) / (1 + inflation_return) - 1
                if np.isfinite(inflation_return)
                else float("nan")
            )
            usd_relative_return = (
                (1 + nominal_return) / (1 + fx_return) - 1
                if np.isfinite(fx_return)
                else float("nan")
            )
            rows.append(
                {
                    "path_id": path.path_id,
                    "path_seed": path.seed,
                    "generator": path.generator,
                    "scenario": path.scenario,
                    "run_id": result.metadata["run_id"],
                    "multiplier": multiplier,
                    "floor_fraction": floor,
                    "review_frequency": frequency.value,
                    "drift_band": drift_band,
                    "nominal_return": nominal_return,
                    "inflation_return": inflation_return,
                    "inflation_benchmark": inflation_benchmark_name,
                    "real_return": real_return,
                    "fx_return": fx_return,
                    "usd_relative_return": usd_relative_return,
                    "annualized_volatility": result.summary["annualized_volatility"],
                    "sharpe_ratio": result.summary["sharpe_ratio"],
                    "sortino_ratio": result.summary["sortino_ratio"],
                    "maximum_drawdown": result.summary["maximum_drawdown"],
                    "drawdown_loss": max(0.0, -float(result.summary["maximum_drawdown"])),
                    "maximum_drawdown_duration_bars": float(
                        maximum_underwater_duration(nominal_drawdowns)
                    ),
                    "real_maximum_drawdown": real_maximum_drawdown,
                    "real_drawdown_loss": (
                        max(0.0, -real_maximum_drawdown)
                        if np.isfinite(real_maximum_drawdown)
                        else float("nan")
                    ),
                    "real_maximum_drawdown_duration_bars": (
                        float(maximum_underwater_duration(drawdown_series_from_levels(real_nav)))
                        if real_nav is not None
                        else float("nan")
                    ),
                    "usd_relative_maximum_drawdown": usd_relative_maximum_drawdown,
                    "usd_relative_drawdown_loss": (
                        max(0.0, -usd_relative_maximum_drawdown)
                        if np.isfinite(usd_relative_maximum_drawdown)
                        else float("nan")
                    ),
                    "usd_relative_maximum_drawdown_duration_bars": (
                        float(
                            maximum_underwater_duration(
                                drawdown_series_from_levels(fx_relative_nav)
                            )
                        )
                        if fx_relative_nav is not None
                        else float("nan")
                    ),
                    "ever_breached": bool(result.summary["ever_breached"]),
                    "floor_breach_bar_fraction": result.summary["floor_breach_bar_fraction"],
                    "floor_breach_severity": result.summary["floor_breach_severity"],
                    "maximum_floor_shortfall": nominal_floor_metrics["maximum_shortfall"],
                    "conditional_floor_shortfall": nominal_floor_metrics["conditional_shortfall"],
                    "maximum_floor_breach_duration_bars": nominal_floor_metrics[
                        "maximum_breach_duration_bars"
                    ],
                    "terminal_floor_shortfall": nominal_floor_metrics["terminal_shortfall"],
                    **real_floor_metrics,
                    "gap_breach_count": result.summary["gap_breach_count"],
                    "turnover": result.summary["turnover"],
                    "total_cost": result.summary["total_cost"],
                    "total_transaction_cost": result.summary["total_transaction_cost"],
                    "total_slippage_cost": result.summary["total_slippage_cost"],
                    "market_closed_bar_fraction": result.summary["market_closed_bar_fraction"],
                    "deferred_execution_bars": result.summary["deferred_execution_bars"],
                }
            )
    return rows


def run_scenario_analysis(
    prices: pd.DataFrame,
    weights: Mapping[str, float],
    multipliers: Sequence[float],
    floors: Sequence[float],
    frequencies: Sequence[ReviewFrequency],
    drift_bands: Sequence[float],
    simulation: SimulationConfig,
    execution: ExecutionModel | None = None,
    reserve_daily_return: float | None = None,
    reserve_calendar_day_accrual: bool = False,
    inflation_index: pd.Series | None = None,
    fx_reference: pd.Series | None = None,
    tradable: pd.DataFrame | None = None,
    inflation_benchmark_name: str = "cpi",
    stresses: Sequence[StressScenario] = (),
    historical_windows: Sequence[HistoricalStressWindow] = (),
    initial_nav: float = 1_000_000,
    max_workers: int = 1,
    regime_policies: Mapping[str, RegimePolicy] | None = None,
    regime_calibration_id: str | None = None,
) -> ScenarioAnalysisResult:
    """Run every Stage 3 path through the Stage 2 engine and aggregate the policy surface."""
    prices = _validate_prices(prices)
    if not multipliers or not floors or not frequencies or not drift_bands:
        raise ValueError("scenario parameter collections cannot be empty")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    execution = execution or ExecutionModel()
    simulated = generate_scenario_paths(prices, simulation, inflation_index, fx_reference, tradable)
    historical = historical_stress_paths(
        prices, historical_windows, inflation_index, fx_reference, tradable
    )
    base_paths = (*simulated, *historical)
    evaluator = partial(
        _evaluate_base_path,
        serialized_stresses=tuple(_serializable_stress(stress) for stress in stresses),
        weights=dict(weights),
        multipliers=tuple(multipliers),
        floors=tuple(floors),
        frequencies=tuple(frequencies),
        drift_bands=tuple(drift_bands),
        execution=execution,
        reserve_daily_return=reserve_daily_return,
        reserve_calendar_day_accrual=reserve_calendar_day_accrual,
        regime_policies=(None if regime_policies is None else dict(regime_policies)),
        regime_calibration_id=regime_calibration_id,
        initial_nav=initial_nav,
        inflation_benchmark_name=inflation_benchmark_name,
    )
    rows: list[dict[str, object]] = []
    if max_workers == 1:
        for base_path in base_paths:
            rows.extend(evaluator(base_path))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor_pool:
            for path_rows in executor_pool.map(evaluator, base_paths):
                rows.extend(path_rows)
    path_metrics = pd.DataFrame(rows)
    surface = _aggregate_surface(path_metrics, simulation.confidence, simulation.tail_confidence)

    price_hash = hashlib.sha256(
        prices.to_csv(date_format="%Y-%m-%dT%H:%M:%S.%f%z").encode()
    ).hexdigest()
    simulation_payload = asdict(simulation)
    simulation_payload["method"] = simulation.method.value
    payload = {
        "data_sha256": price_hash,
        "inflation_sha256": _series_hash(inflation_index),
        "fx_sha256": _series_hash(fx_reference),
        "tradability_sha256": _frame_hash(tradable),
        "inflation_benchmark": inflation_benchmark_name,
        "weights": dict(weights),
        "simulation": simulation_payload,
        "execution": asdict(execution),
        "reserve_daily_return": reserve_daily_return,
        "reserve_calendar_day_accrual": reserve_calendar_day_accrual,
        "regime_policies": {
            asset: asdict(policy) for asset, policy in (regime_policies or {}).items()
        },
        "regime_calibration_id": regime_calibration_id,
        "multipliers": list(multipliers),
        "floors": list(floors),
        "frequencies": [frequency.value for frequency in frequencies],
        "drift_bands": list(drift_bands),
        "stresses": [_serializable_stress(stress) for stress in stresses],
        "historical_windows": [
            {
                "name": window.name,
                "start": pd.Timestamp(window.start).isoformat(),
                "end": pd.Timestamp(window.end).isoformat(),
            }
            for window in historical_windows
        ],
        "initial_nav": initial_nav,
        "inflation_included": inflation_index is not None,
        "fx_included": fx_reference is not None,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    metadata: Mapping[str, object] = MappingProxyType(
        {
            "analysis_id": str(uuid5(NAMESPACE_URL, canonical)),
            **payload,
            "generated_path_count": len(simulated),
            "historical_path_count": len(historical),
            "path_metric_rows": len(path_metrics),
            "max_workers": max_workers,
            "interval_interpretation": (
                "empirical outcome quantiles; Wilson score intervals for event probabilities"
            ),
        }
    )
    return ScenarioAnalysisResult(surface, path_metrics, metadata)


def export_scenario_analysis(result: ScenarioAnalysisResult, directory: str | Path) -> Path:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    result.surface.to_csv(target / "surface.csv", index=False)
    result.path_metrics.to_csv(target / "path_metrics.csv", index=False)
    (target / "metadata.json").write_text(
        json.dumps(dict(result.metadata), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target
