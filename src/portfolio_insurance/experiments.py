"""Deterministic Stage 2 parameter-grid execution and artifact export."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import pandas as pd

from .backtest import (
    BacktestPolicy,
    BacktestResult,
    ExecutionModel,
    ReviewFrequency,
    Strategy,
    attribute_effects,
    relative_prices,
    run_backtest,
    run_baselines,
    with_daily_accrual_prices,
)
from .regime import RegimePolicy


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    table: pd.DataFrame
    runs: Mapping[str, BacktestResult]
    baselines: Mapping[str, BacktestResult]
    attributions: Mapping[str, pd.DataFrame]


def run_parameter_grid(
    prices: pd.DataFrame,
    weights: Mapping[str, float],
    multipliers: Sequence[float],
    floors: Sequence[float],
    frequencies: Sequence[ReviewFrequency],
    drift_bands: Sequence[float],
    execution: ExecutionModel | None = None,
    reference_prices: pd.Series | None = None,
    reserve_daily_return: float | None = None,
    reserve_calendar_day_accrual: bool = False,
    initial_nav: float = 1_000_000,
    tradable: pd.DataFrame | None = None,
    regime_policies: Mapping[str, RegimePolicy] | None = None,
    regime_calibration_id: str | None = None,
    protection_drift_band: float | None = None,
) -> ExperimentResult:
    """Evaluate the full Stage 2 surface in local and optional reference currency."""
    execution = execution or ExecutionModel()
    if not multipliers or not floors or not frequencies or not drift_bands:
        raise ValueError("experiment parameter collections cannot be empty")
    local_prices = with_daily_accrual_prices(
        prices,
        "fixed_income",
        reserve_daily_return,
        reserve_calendar_day_accrual,
    )
    scenarios = {"local": local_prices}
    if reference_prices is not None:
        scenarios["usd_relative"] = relative_prices(local_prices, reference_prices)
    rows: list[dict[str, object]] = []
    runs: dict[str, BacktestResult] = {}
    baselines: dict[str, BacktestResult] = {}
    attributions: dict[str, pd.DataFrame] = {}
    baseline_cache: dict[tuple[str, ReviewFrequency, float], dict[str, BacktestResult]] = {}
    for scenario, scenario_prices in scenarios.items():
        scenario_reserve_return = reserve_daily_return if scenario == "local" else None
        for multiplier, floor, frequency, drift_band in product(
            multipliers, floors, frequencies, drift_bands
        ):
            cache_key = (scenario, frequency, drift_band)
            if cache_key not in baseline_cache:
                comparison = run_baselines(
                    scenario_prices,
                    weights,
                    review_frequency=frequency,
                    drift_band=drift_band,
                    execution=execution,
                    reserve_daily_return=scenario_reserve_return,
                    reserve_calendar_day_accrual=(
                        reserve_calendar_day_accrual if scenario_reserve_return is not None else False
                    ),
                    initial_nav=initial_nav,
                    tradable=tradable,
                )
                baseline_cache[cache_key] = comparison
                for name, baseline in comparison.items():
                    baselines[f"{scenario}:{frequency.value}:{drift_band}:{name}"] = baseline
            comparison = baseline_cache[cache_key]
            policy = BacktestPolicy(
                strategy=Strategy.VPPI,
                multiplier=multiplier,
                floor_fraction=floor,
                review_frequency=frequency,
                drift_band=drift_band,
                protection_drift_band=protection_drift_band,
                execution=execution,
                reserve_daily_return=scenario_reserve_return,
                reserve_calendar_day_accrual=(
                    reserve_calendar_day_accrual if scenario_reserve_return is not None else False
                ),
                regime_policies=regime_policies,
                regime_calibration_id=regime_calibration_id,
            )
            result = run_backtest(scenario_prices, weights, policy, initial_nav, tradable=tradable)
            run_id = str(result.metadata["run_id"])
            runs[run_id] = result
            attributions[run_id] = attribute_effects(
                comparison[Strategy.BUY_AND_HOLD.value],
                comparison[Strategy.CALENDAR_REBALANCED.value],
                result,
            )
            rows.append(
                {
                    "run_id": run_id,
                    "scenario": scenario,
                    "multiplier": multiplier,
                    "floor_fraction": floor,
                    "review_frequency": frequency.value,
                    "drift_band": drift_band,
                    "protection_drift_band": protection_drift_band,
                    "buy_hold_total_return": comparison[Strategy.BUY_AND_HOLD.value].summary[
                        "total_return"
                    ],
                    "calendar_total_return": comparison[Strategy.CALENDAR_REBALANCED.value].summary[
                        "total_return"
                    ],
                    **result.summary,
                }
            )
    table = (
        pd.DataFrame(rows)
        .sort_values(["scenario", "floor_fraction", "multiplier", "review_frequency", "drift_band"])
        .reset_index(drop=True)
    )
    return ExperimentResult(table, runs, baselines, attributions)


def export_experiment(result: ExperimentResult, directory: str | Path) -> Path:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    result.table.to_csv(target / "summary.csv", index=False)
    baseline_directory = target / "baselines"
    baseline_directory.mkdir(exist_ok=True)
    for key, baseline in result.baselines.items():
        safe_key = key.replace(":", "_")
        baseline.ledger.to_csv(baseline_directory / f"{safe_key}_ledger.csv")
    for run_id, run in result.runs.items():
        run_directory = target / run_id
        run_directory.mkdir(exist_ok=True)
        run.ledger.to_csv(run_directory / "ledger.csv")
        result.attributions[run_id].to_csv(run_directory / "attribution.csv")
        (run_directory / "metadata.json").write_text(
            json.dumps(dict(run.metadata), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (run_directory / "summary.json").write_text(
            json.dumps(dict(run.summary), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return target
