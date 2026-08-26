"""Stage 4 Pareto, stability-plateau, and walk-forward policy selection."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from types import MappingProxyType
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pandas as pd

from .backtest import BacktestPolicy, ExecutionModel, ReviewFrequency, Strategy, run_backtest
from .regime import RegimePolicy

POLICY_COLUMNS = (
    "multiplier",
    "floor_fraction",
    "review_frequency",
    "drift_band",
)
_FREQUENCY_ORDER = {
    ReviewFrequency.DAILY.value: 0,
    ReviewFrequency.WEEKLY.value: 1,
    ReviewFrequency.BIWEEKLY.value: 2,
    ReviewFrequency.MONTHLY.value: 3,
}


@dataclass(frozen=True, slots=True)
class SelectionCriteria:
    """Transparent preferences and optional hard gates for one portfolio mandate."""

    return_weight: float
    drawdown_weight: float
    breach_weight: float
    cost_weight: float
    tail_probability: float = 0.10
    minimum_neighbors: int = 2
    minimum_feasible_neighbor_fraction: float = 2 / 3
    maximum_neighbor_utility_gap: float = 0.20
    minimum_real_return: float | None = None
    strict_minimum_real_return: bool = False
    maximum_drawdown: float | None = None
    maximum_floor_breach_probability: float | None = None
    maximum_cost_rate: float | None = None

    def __post_init__(self) -> None:
        weights = (
            self.return_weight,
            self.drawdown_weight,
            self.breach_weight,
            self.cost_weight,
        )
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("selection weights must be non-negative and have a positive sum")
        if not 0 < self.tail_probability <= 0.5:
            raise ValueError("tail_probability must be in (0, 0.5]")
        if self.minimum_neighbors < 0:
            raise ValueError("minimum_neighbors cannot be negative")
        if not 0 <= self.minimum_feasible_neighbor_fraction <= 1:
            raise ValueError("minimum_feasible_neighbor_fraction must be between zero and one")
        if self.maximum_neighbor_utility_gap < 0:
            raise ValueError("maximum_neighbor_utility_gap cannot be negative")
        if not isinstance(self.strict_minimum_real_return, bool):
            raise TypeError("strict_minimum_real_return must be boolean")
        if self.maximum_drawdown is not None and not 0 <= self.maximum_drawdown <= 1:
            raise ValueError("maximum_drawdown must be a positive loss fraction")
        if self.maximum_floor_breach_probability is not None and not (
            0 <= self.maximum_floor_breach_probability <= 1
        ):
            raise ValueError("maximum_floor_breach_probability must be between zero and one")
        if self.maximum_cost_rate is not None and self.maximum_cost_rate < 0:
            raise ValueError("maximum_cost_rate cannot be negative")


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    folds: pd.DataFrame
    policy_metrics: pd.DataFrame
    candidate_metrics: pd.DataFrame


@dataclass(frozen=True, slots=True)
class PolicySelectionResult:
    portfolio: str
    status: str
    validation_status: str
    selected_policy: Mapping[str, object] | None
    candidates: pd.DataFrame
    context_metrics: pd.DataFrame
    walk_forward: WalkForwardResult
    metadata: Mapping[str, object]


def load_selection_criteria(path: str | Path, portfolio: str) -> SelectionCriteria:
    """Load a portfolio-specific selection profile from version-controlled TOML."""
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    try:
        values = {**raw.get("common", {}), **raw["portfolios"][portfolio]}
    except KeyError as exc:
        raise ValueError(f"no selection profile is configured for {portfolio!r}") from exc
    return SelectionCriteria(**values)


def _validate_path_metrics(path_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "generator",
        "scenario",
        *POLICY_COLUMNS,
        "real_return",
        "maximum_drawdown",
        "ever_breached",
        "total_cost",
    }
    missing = required - set(path_metrics.columns)
    if missing:
        raise ValueError(f"path metrics are missing required columns: {sorted(missing)}")
    if path_metrics.empty:
        raise ValueError("path metrics cannot be empty")
    frame = path_metrics.copy()
    for column in ("multiplier", "floor_fraction", "drift_band"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    for column in ("real_return", "maximum_drawdown", "total_cost"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["review_frequency"] = frame["review_frequency"].astype(str)
    unknown = set(frame["review_frequency"]) - set(_FREQUENCY_ORDER)
    if unknown:
        raise ValueError(f"unknown review frequencies: {sorted(unknown)}")
    if frame["ever_breached"].dtype == object:
        normalized = frame["ever_breached"].astype(str).str.strip().str.lower()
        if not normalized.isin({"true", "false"}).all():
            raise ValueError("ever_breached must contain booleans")
        frame["ever_breached"] = normalized.map({"true": True, "false": False})
    else:
        frame["ever_breached"] = frame["ever_breached"].astype(bool)
    if frame["real_return"].notna().sum() == 0:
        raise ValueError("policy selection requires inflation-adjusted path returns")
    return frame


def _context_table(
    path_metrics: pd.DataFrame,
    criteria: SelectionCriteria,
    initial_nav: float,
) -> pd.DataFrame:
    if initial_nav <= 0:
        raise ValueError("initial_nav must be positive")
    keys = [*POLICY_COLUMNS, "generator", "scenario"]
    rows: list[dict[str, object]] = []
    for values, group in path_metrics.groupby(keys, sort=True, dropna=False):
        real_returns = group["real_return"].dropna()
        drawdowns = group["maximum_drawdown"].dropna()
        costs = group["total_cost"].dropna()
        row = dict(zip(keys, values, strict=True))
        row.update(
            {
                "n_paths": len(group),
                "tail_real_return": float(real_returns.quantile(criteria.tail_probability)),
                "drawdown_loss": (
                    max(0.0, -float(drawdowns.quantile(criteria.tail_probability)))
                    if not drawdowns.empty
                    else float("nan")
                ),
                "floor_breach_probability": float(group["ever_breached"].mean()),
                "real_loss_probability": float((real_returns < 0).mean()),
                "median_cost_rate": (
                    float(costs.median()) / initial_nav if not costs.empty else float("nan")
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def _candidate_table(contexts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for values, group in contexts.groupby(list(POLICY_COLUMNS), sort=True, dropna=False):
        row = dict(zip(POLICY_COLUMNS, values, strict=True))
        row.update(
            {
                "context_count": len(group),
                "robust_real_return": float(group["tail_real_return"].min()),
                "drawdown_loss": float(group["drawdown_loss"].max()),
                "floor_breach_probability": float(group["floor_breach_probability"].max()),
                "real_loss_probability": float(group["real_loss_probability"].max()),
                "cost_rate": float(group["median_cost_rate"].max()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(POLICY_COLUMNS)).reset_index(drop=True)


def _desirability(values: pd.Series, higher_is_better: bool) -> pd.Series:
    numeric = values.astype(float)
    low = float(numeric.min())
    high = float(numeric.max())
    if np.isclose(low, high):
        return pd.Series(1.0, index=values.index)
    scaled = (numeric - low) / (high - low)
    return scaled if higher_is_better else 1 - scaled


def _pareto_flags(candidates: pd.DataFrame) -> np.ndarray:
    objectives = np.column_stack(
        [
            -candidates["robust_real_return"].to_numpy(float),
            candidates["drawdown_loss"].to_numpy(float),
            candidates["floor_breach_probability"].to_numpy(float),
            candidates["cost_rate"].to_numpy(float),
        ]
    )
    flags = np.ones(len(candidates), dtype=bool)
    for index, point in enumerate(objectives):
        dominates = np.all(objectives <= point + 1e-12, axis=1) & np.any(
            objectives < point - 1e-12, axis=1
        )
        dominates[index] = False
        if dominates.any():
            flags[index] = False
    return flags


def _constraint_flags(candidates: pd.DataFrame, criteria: SelectionCriteria) -> pd.Series:
    feasible = pd.Series(True, index=candidates.index)
    if criteria.minimum_real_return is not None:
        if criteria.strict_minimum_real_return:
            feasible &= candidates["robust_real_return"] > criteria.minimum_real_return
        else:
            feasible &= candidates["robust_real_return"] >= criteria.minimum_real_return
    if criteria.maximum_drawdown is not None:
        feasible &= candidates["drawdown_loss"] <= criteria.maximum_drawdown
    if criteria.maximum_floor_breach_probability is not None:
        feasible &= (
            candidates["floor_breach_probability"] <= criteria.maximum_floor_breach_probability
        )
    if criteria.maximum_cost_rate is not None:
        feasible &= candidates["cost_rate"] <= criteria.maximum_cost_rate
    return feasible


def _coordinates(candidates: pd.DataFrame) -> list[tuple[int, int, int, int]]:
    numeric_values = {
        column: sorted(candidates[column].astype(float).unique())
        for column in ("multiplier", "floor_fraction", "drift_band")
    }
    numeric_positions = {
        column: {value: index for index, value in enumerate(values)}
        for column, values in numeric_values.items()
    }
    frequencies = sorted(candidates["review_frequency"].unique(), key=_FREQUENCY_ORDER.__getitem__)
    frequency_positions = {value: index for index, value in enumerate(frequencies)}
    return [
        (
            numeric_positions["multiplier"][float(row.multiplier)],
            numeric_positions["floor_fraction"][float(row.floor_fraction)],
            frequency_positions[str(row.review_frequency)],
            numeric_positions["drift_band"][float(row.drift_band)],
        )
        for row in candidates.itertuples(index=False)
    ]


def _add_scores(candidates: pd.DataFrame, criteria: SelectionCriteria) -> pd.DataFrame:
    table = candidates.copy()
    weights = np.asarray(
        [
            criteria.return_weight,
            criteria.drawdown_weight,
            criteria.breach_weight,
            criteria.cost_weight,
        ],
        dtype=float,
    )
    weights /= weights.sum()
    desirability = np.column_stack(
        [
            _desirability(table["robust_real_return"], True),
            _desirability(table["drawdown_loss"], False),
            _desirability(table["floor_breach_probability"], False),
            _desirability(table["cost_rate"], False),
        ]
    )
    table["utility_score"] = desirability @ weights
    table["is_pareto"] = _pareto_flags(table)
    table["constraint_feasible"] = _constraint_flags(table, criteria)

    coordinates = _coordinates(table)
    coordinate_index = {coordinate: index for index, coordinate in enumerate(coordinates)}
    neighbor_counts: list[int] = []
    neighbor_means: list[float] = []
    neighbor_worsts: list[float] = []
    neighbor_stds: list[float] = []
    feasible_fractions: list[float] = []
    utility_gaps: list[float] = []
    neighbor_indices_by_row: list[list[int]] = []
    for row_index, coordinate in enumerate(coordinates):
        neighbors: list[int] = []
        for dimension in range(4):
            for direction in (-1, 1):
                candidate = list(coordinate)
                candidate[dimension] += direction
                neighbor = coordinate_index.get(tuple(candidate))
                if neighbor is not None:
                    neighbors.append(neighbor)
        neighbor_indices_by_row.append(neighbors)
        utilities = table.loc[neighbors, "utility_score"] if neighbors else pd.Series(dtype=float)
        neighbor_counts.append(len(neighbors))
        neighbor_means.append(float(utilities.mean()) if neighbors else float("nan"))
        neighbor_worsts.append(float(utilities.min()) if neighbors else float("nan"))
        neighbor_stds.append(float(utilities.std(ddof=0)) if neighbors else float("nan"))
        feasible_fractions.append(
            float(table.loc[neighbors, "constraint_feasible"].mean()) if neighbors else 0.0
        )
        utility_gaps.append(
            max(0.0, float(table.loc[row_index, "utility_score"] - utilities.min()))
            if neighbors
            else float("inf")
        )

    table["neighbor_count"] = neighbor_counts
    table["neighbor_mean_utility"] = neighbor_means
    table["neighbor_worst_utility"] = neighbor_worsts
    table["neighbor_utility_std"] = neighbor_stds
    table["feasible_neighbor_fraction"] = feasible_fractions
    table["neighbor_utility_gap"] = utility_gaps
    table["is_stable_plateau"] = (
        (table["neighbor_count"] >= criteria.minimum_neighbors)
        & (table["feasible_neighbor_fraction"] >= criteria.minimum_feasible_neighbor_fraction)
        & (table["neighbor_utility_gap"] <= criteria.maximum_neighbor_utility_gap)
    )
    table["plateau_score"] = (
        table["utility_score"]
        + table["neighbor_mean_utility"].fillna(table["utility_score"])
        + table["neighbor_worst_utility"].fillna(table["utility_score"])
    ) / 3 - table["neighbor_utility_std"].fillna(0)
    table["is_selected"] = False
    return table


def _select_evidence(
    path_metrics: pd.DataFrame,
    criteria: SelectionCriteria,
    initial_nav: float,
) -> tuple[str, Mapping[str, object] | None, pd.DataFrame, pd.DataFrame]:
    frame = _validate_path_metrics(path_metrics)
    contexts = _context_table(frame, criteria, initial_nav)
    candidates = _add_scores(_candidate_table(contexts), criteria)
    feasible = candidates[candidates["constraint_feasible"]]
    if feasible.empty:
        return "no_feasible_policy", None, candidates, contexts
    pool = feasible[feasible["is_pareto"] & feasible["is_stable_plateau"]]
    if pool.empty:
        return "no_stable_pareto_policy", None, candidates, contexts
    ranking = pool.sort_values(
        [
            "plateau_score",
            "utility_score",
            "robust_real_return",
            "drawdown_loss",
            "floor_breach_probability",
            "cost_rate",
        ],
        ascending=[False, False, False, True, True, True],
        kind="stable",
    )
    selected_index = int(ranking.index[0])
    candidates.loc[selected_index, "is_selected"] = True
    selected = {
        column: candidates.loc[selected_index, column]
        for column in (
            *POLICY_COLUMNS,
            "robust_real_return",
            "drawdown_loss",
            "floor_breach_probability",
            "cost_rate",
            "utility_score",
            "plateau_score",
        )
    }
    return "selected", MappingProxyType(selected), candidates, contexts


def _level_change(levels: pd.Series | None, index: pd.DatetimeIndex) -> float:
    if levels is None:
        return float("nan")
    aligned = levels.astype(float).sort_index().reindex(index).ffill()
    if aligned.isna().any() or (aligned <= 0).any():
        raise ValueError("inflation index does not cover a walk-forward window")
    return float(aligned.iloc[-1] / aligned.iloc[0] - 1)


def _evaluate_historical_window(
    prices: pd.DataFrame,
    inflation_index: pd.Series | None,
    tradable: pd.DataFrame | None,
    weights: Mapping[str, float],
    policies: Sequence[tuple[float, float, ReviewFrequency, float]],
    execution: ExecutionModel,
    reserve_daily_return: float | None,
    reserve_calendar_day_accrual: bool,
    regime_policies: Mapping[str, RegimePolicy] | None,
    regime_calibration_id: str | None,
    initial_nav: float,
    generator: str,
    scenario: str,
) -> list[dict[str, object]]:
    inflation_return = _level_change(inflation_index, prices.index)
    rows: list[dict[str, object]] = []
    for multiplier, floor, frequency, drift_band in policies:
        result = run_backtest(
            prices,
            weights,
            BacktestPolicy(
                strategy=Strategy.VPPI,
                multiplier=multiplier,
                floor_fraction=floor,
                review_frequency=frequency,
                drift_band=drift_band,
                execution=execution,
                reserve_daily_return=reserve_daily_return,
                reserve_calendar_day_accrual=reserve_calendar_day_accrual,
                regime_policies=regime_policies,
                regime_calibration_id=regime_calibration_id,
            ),
            initial_nav,
            tradable=tradable,
        )
        nominal_return = float(result.summary["total_return"])
        rows.append(
            {
                "path_id": scenario,
                "generator": generator,
                "scenario": scenario,
                "run_id": result.metadata["run_id"],
                "multiplier": multiplier,
                "floor_fraction": floor,
                "review_frequency": frequency.value,
                "drift_band": drift_band,
                "nominal_return": nominal_return,
                "real_return": (
                    (1 + nominal_return) / (1 + inflation_return) - 1
                    if np.isfinite(inflation_return)
                    else float("nan")
                ),
                "maximum_drawdown": float(result.summary["maximum_drawdown"]),
                "ever_breached": bool(result.summary["ever_breached"]),
                "turnover": float(result.summary["turnover"]),
                "total_cost": float(result.summary["total_cost"]),
            }
        )
    return rows


def walk_forward_validate(
    prices: pd.DataFrame,
    inflation_index: pd.Series,
    weights: Mapping[str, float],
    multipliers: Sequence[float],
    floors: Sequence[float],
    frequencies: Sequence[ReviewFrequency],
    drift_bands: Sequence[float],
    criteria: SelectionCriteria,
    *,
    test_bars: int,
    folds: int = 3,
    minimum_train_bars: int | None = None,
    execution: ExecutionModel | None = None,
    reserve_daily_return: float | None = None,
    reserve_calendar_day_accrual: bool = False,
    tradable: pd.DataFrame | None = None,
    initial_nav: float = 1_000_000,
    regime_policies: Mapping[str, RegimePolicy] | None = None,
    regime_calibration_id: str | None = None,
) -> WalkForwardResult:
    """Select on expanding training windows and score only later observations as OOS."""
    if test_bars < 2 or folds <= 0:
        raise ValueError("test_bars must be at least two and folds must be positive")
    if not multipliers or not floors or not frequencies or not drift_bands:
        raise ValueError("walk-forward parameter collections cannot be empty")
    if (
        not isinstance(prices.index, pd.DatetimeIndex)
        or not prices.index.is_monotonic_increasing
        or prices.index.has_duplicates
    ):
        raise ValueError("walk-forward prices require a sorted, unique DatetimeIndex")
    execution = execution or ExecutionModel()
    if tradable is not None:
        if not tradable.index.equals(prices.index) or set(tradable.columns) != set(prices.columns):
            raise ValueError("walk-forward tradability must match prices")
        tradable = tradable.reindex(columns=prices.columns).astype(bool)
    minimum_train_bars = minimum_train_bars or test_bars
    if minimum_train_bars < test_bars:
        raise ValueError("minimum_train_bars must cover at least one full test horizon")
    maximum_split = len(prices) - test_bars - 1
    if maximum_split < minimum_train_bars:
        raise ValueError("price history is too short for the requested walk-forward design")
    split_positions = np.unique(
        np.rint(np.linspace(minimum_train_bars, maximum_split, folds)).astype(int)
    )
    if len(split_positions) < folds:
        raise ValueError("history cannot provide the requested number of distinct folds")

    policies = [
        (float(multiplier), float(floor), ReviewFrequency(frequency), float(drift_band))
        for multiplier, floor, frequency, drift_band in product(
            multipliers, floors, frequencies, drift_bands
        )
    ]
    fold_rows: list[dict[str, object]] = []
    oos_rows: list[dict[str, object]] = []
    for fold_number, split in enumerate(split_positions, start=1):
        training_rows: list[dict[str, object]] = []
        training_windows: list[tuple[int, int]] = []
        end = int(split)
        while end - test_bars >= 0:
            start = end - test_bars
            training_windows.append((start, end))
            end = start
        training_windows.reverse()
        for window_number, (start, end) in enumerate(training_windows, start=1):
            window_index = prices.index[start : end + 1]
            training_rows.extend(
                _evaluate_historical_window(
                    prices.iloc[start : end + 1],
                    inflation_index.reindex(window_index).ffill(),
                    None if tradable is None else tradable.reindex(window_index),
                    weights,
                    policies,
                    execution,
                    reserve_daily_return,
                    reserve_calendar_day_accrual,
                    regime_policies,
                    regime_calibration_id,
                    initial_nav,
                    "walk_forward_train",
                    f"fold_{fold_number:03d}_train_{window_number:03d}",
                )
            )
        fold_status, fold_policy, _, _ = _select_evidence(
            pd.DataFrame(training_rows), criteria, initial_nav
        )
        test_start = int(split)
        test_end = test_start + test_bars
        oos_index = prices.index[test_start : test_end + 1]
        fold_oos = _evaluate_historical_window(
            prices.iloc[test_start : test_end + 1],
            inflation_index.reindex(oos_index).ffill(),
            None if tradable is None else tradable.reindex(oos_index),
            weights,
            policies,
            execution,
            reserve_daily_return,
            reserve_calendar_day_accrual,
            regime_policies,
            regime_calibration_id,
            initial_nav,
            "walk_forward_oos",
            f"fold_{fold_number:03d}",
        )
        for row in fold_oos:
            row["fold"] = fold_number
            row["selected_in_training"] = bool(
                fold_policy is not None
                and all(row[column] == fold_policy[column] for column in POLICY_COLUMNS)
            )
        oos_rows.extend(fold_oos)
        selected_oos = next((row for row in fold_oos if row["selected_in_training"]), None)
        fold_row: dict[str, object] = {
            "fold": fold_number,
            "train_start": prices.index[training_windows[0][0]],
            "train_end": prices.index[test_start],
            "test_start": prices.index[test_start + 1],
            "test_end": prices.index[test_end],
            "training_window_count": len(training_windows),
            "selection_status": fold_status,
        }
        if fold_policy is not None:
            fold_row.update({column: fold_policy[column] for column in POLICY_COLUMNS})
        if selected_oos is not None:
            fold_row.update(
                {
                    "oos_real_return": selected_oos["real_return"],
                    "oos_maximum_drawdown": selected_oos["maximum_drawdown"],
                    "oos_ever_breached": selected_oos["ever_breached"],
                    "oos_turnover": selected_oos["turnover"],
                    "oos_total_cost": selected_oos["total_cost"],
                }
            )
        fold_rows.append(fold_row)

    oos = pd.DataFrame(oos_rows)
    _, _, oos_candidates, _ = _select_evidence(oos, criteria, initial_nav)
    oos_candidates = oos_candidates.drop(columns="is_selected")
    oos_candidates["diagnostic_only"] = True
    return WalkForwardResult(pd.DataFrame(fold_rows), oos, oos_candidates)


def run_policy_selection(
    path_metrics: pd.DataFrame,
    prices: pd.DataFrame,
    inflation_index: pd.Series,
    weights: Mapping[str, float],
    portfolio: str,
    criteria: SelectionCriteria,
    *,
    test_bars: int,
    folds: int = 3,
    minimum_train_bars: int | None = None,
    execution: ExecutionModel | None = None,
    reserve_daily_return: float | None = None,
    reserve_calendar_day_accrual: bool = False,
    tradable: pd.DataFrame | None = None,
    inflation_benchmark_name: str = "usd_irr",
    initial_nav: float = 1_000_000,
    source_analysis_id: str | None = None,
    regime_policies: Mapping[str, RegimePolicy] | None = None,
    regime_calibration_id: str | None = None,
) -> PolicySelectionResult:
    """Select from Stage 3 evidence, then validate without using OOS data to choose it."""
    if inflation_benchmark_name != "usd_irr":
        raise ValueError("policy selection mandate requires the USD/IRR benchmark")
    execution = execution or ExecutionModel()
    status, selected, candidates, contexts = _select_evidence(path_metrics, criteria, initial_nav)
    grid = candidates[list(POLICY_COLUMNS)]
    effective_minimum_train_bars = minimum_train_bars or test_bars
    walk_forward = walk_forward_validate(
        prices,
        inflation_index,
        weights,
        multipliers=sorted(grid["multiplier"].unique()),
        floors=sorted(grid["floor_fraction"].unique()),
        frequencies=[
            ReviewFrequency(value)
            for value in sorted(grid["review_frequency"].unique(), key=_FREQUENCY_ORDER.__getitem__)
        ],
        drift_bands=sorted(grid["drift_band"].unique()),
        criteria=criteria,
        test_bars=test_bars,
        folds=folds,
        minimum_train_bars=effective_minimum_train_bars,
        execution=execution,
        reserve_daily_return=reserve_daily_return,
        reserve_calendar_day_accrual=reserve_calendar_day_accrual,
        tradable=tradable,
        initial_nav=initial_nav,
        regime_policies=regime_policies,
        regime_calibration_id=regime_calibration_id,
    )

    validation_status = "not_selected"
    if selected is not None:
        if len(walk_forward.folds) < 2:
            validation_status = "insufficient_folds"
        else:
            deployed = walk_forward.policy_metrics[
                walk_forward.policy_metrics["selected_in_training"]
            ]
            if len(deployed) != len(walk_forward.folds):
                validation_status = "incomplete_fold_selection"
            else:
                gate_results: list[bool] = []
                if criteria.minimum_real_return is not None:
                    minimum_oos_return = float(deployed["real_return"].min())
                    gate_results.append(
                        minimum_oos_return > criteria.minimum_real_return
                        if criteria.strict_minimum_real_return
                        else minimum_oos_return >= criteria.minimum_real_return
                    )
                if criteria.maximum_drawdown is not None:
                    gate_results.append(
                        float((-deployed["maximum_drawdown"]).max()) <= criteria.maximum_drawdown
                    )
                if criteria.maximum_floor_breach_probability is not None:
                    gate_results.append(
                        float(deployed["ever_breached"].mean())
                        <= criteria.maximum_floor_breach_probability
                    )
                if criteria.maximum_cost_rate is not None:
                    gate_results.append(
                        float((deployed["total_cost"] / initial_nav).max())
                        <= criteria.maximum_cost_rate
                    )
                validation_status = (
                    "completed_no_mandate_gates"
                    if not gate_results
                    else "passed"
                    if all(gate_results)
                    else "failed"
                )

    canonical_metrics = _validate_path_metrics(path_metrics)
    identity_columns = ["generator", "scenario", *POLICY_COLUMNS]
    identity_columns.extend(
        column for column in ("path_id", "run_id") if column in canonical_metrics
    )
    path_hash = hashlib.sha256(
        canonical_metrics.sort_values(identity_columns).to_csv(index=False).encode()
    ).hexdigest()
    price_hash = hashlib.sha256(
        prices.to_csv(date_format="%Y-%m-%dT%H:%M:%S.%f%z").encode()
    ).hexdigest()
    inflation_hash = hashlib.sha256(
        inflation_index.astype(float).to_csv(date_format="%Y-%m-%dT%H:%M:%S.%f%z").encode()
    ).hexdigest()
    tradability_hash = (
        hashlib.sha256(
            tradable.astype(int).to_csv(date_format="%Y-%m-%dT%H:%M:%S.%f%z").encode()
        ).hexdigest()
        if tradable is not None
        else None
    )
    payload = {
        "portfolio": portfolio,
        "source_analysis_id": source_analysis_id,
        "criteria": asdict(criteria),
        "weights": dict(weights),
        "execution": asdict(execution),
        "reserve_daily_return": reserve_daily_return,
        "reserve_calendar_day_accrual": reserve_calendar_day_accrual,
        "regime_policies": {
            asset: asdict(policy) for asset, policy in (regime_policies or {}).items()
        },
        "regime_calibration_id": regime_calibration_id,
        "initial_nav": initial_nav,
        "test_bars": test_bars,
        "folds": folds,
        "minimum_train_bars": effective_minimum_train_bars,
        "path_metrics_sha256": path_hash,
        "prices_sha256": price_hash,
        "inflation_sha256": inflation_hash,
        "tradability_sha256": tradability_hash,
        "inflation_benchmark": inflation_benchmark_name,
    }
    selection_id = str(uuid5(NAMESPACE_URL, json.dumps(payload, sort_keys=True)))
    metadata = MappingProxyType(
        {
            "selection_id": selection_id,
            **payload,
            "selection_status": status,
            "validation_status": validation_status,
            "selection_rule": (
                "highest plateau score among constraint-feasible Pareto policies; "
                "walk-forward observations are validation-only"
            ),
        }
    )
    selected_policy = selected
    if selected is not None and regime_policies is not None:
        selected_policy = MappingProxyType(
            {
                **dict(selected),
                "regime_policies": {
                    asset: asdict(policy) for asset, policy in regime_policies.items()
                },
                "regime_calibration_id": regime_calibration_id,
            }
        )
    return PolicySelectionResult(
        portfolio,
        status,
        validation_status,
        selected_policy,
        candidates,
        contexts,
        walk_forward,
        metadata,
    )


def with_criteria_overrides(
    criteria: SelectionCriteria,
    *,
    minimum_real_return: float | None = None,
    maximum_drawdown: float | None = None,
    maximum_floor_breach_probability: float | None = None,
    maximum_cost_rate: float | None = None,
) -> SelectionCriteria:
    """Apply only explicitly supplied mandate gates to a configured profile."""
    overrides = {
        key: value
        for key, value in {
            "minimum_real_return": minimum_real_return,
            "maximum_drawdown": maximum_drawdown,
            "maximum_floor_breach_probability": maximum_floor_breach_probability,
            "maximum_cost_rate": maximum_cost_rate,
        }.items()
        if value is not None
    }
    return replace(criteria, **overrides)


def export_policy_selection(result: PolicySelectionResult, directory: str | Path) -> Path:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    result.candidates.to_csv(target / "candidate_policies.csv", index=False)
    result.context_metrics.to_csv(target / "context_metrics.csv", index=False)
    result.walk_forward.folds.to_csv(target / "walk_forward_folds.csv", index=False)
    result.walk_forward.policy_metrics.to_csv(
        target / "walk_forward_policy_metrics.csv", index=False
    )
    result.walk_forward.candidate_metrics.to_csv(
        target / "walk_forward_candidates.csv", index=False
    )
    decision = {
        "portfolio": result.portfolio,
        "status": result.status,
        "validation_status": result.validation_status,
        "selected_policy": None if result.selected_policy is None else dict(result.selected_policy),
    }
    (target / "selected_policy.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (target / "metadata.json").write_text(
        json.dumps(dict(result.metadata), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target
