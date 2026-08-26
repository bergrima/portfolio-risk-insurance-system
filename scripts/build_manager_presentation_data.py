"""Build compact, auditable evidence tables for the asset-manager presentation."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from portfolio_insurance.backtest import (  # noqa: E402
    BacktestPolicy,
    ExecutionModel,
    ReviewFrequency,
    Strategy,
    run_backtest,
)
from portfolio_insurance.config import load_project_config  # noqa: E402
from portfolio_insurance.cost_config import load_asset_transaction_costs  # noqa: E402
from portfolio_insurance.data_contract import load_data_contract  # noqa: E402
from portfolio_insurance.data_sources import (  # noqa: E402
    load_snapshot,
    point_in_time_prices,
    point_in_time_tradability,
)
from portfolio_insurance.regime import confirmed_zigzag, zigzag_legs  # noqa: E402
from portfolio_insurance.scenarios import (  # noqa: E402
    SimulationConfig,
    generate_scenario_paths,
)


def _snapshot_series(data: pd.DataFrame, key: str, index: pd.DatetimeIndex) -> pd.Series:
    selected = data[data["series"] == key].copy()
    selected["valuation_at"] = pd.to_datetime(selected["valuation_at"], utc=True)
    values = (
        selected.sort_values("available_at")
        .drop_duplicates("valuation_at", keep="last")
        .set_index("valuation_at")["value"]
        .astype(float)
        .sort_index()
    )
    result = values.reindex(index).ffill()
    if result.isna().any():
        raise ValueError(f"{key} does not cover the requested index")
    return result


def _asset_risk_row(name: str, levels: pd.Series, horizon_bars: int = 10) -> dict[str, object]:
    clean = levels.astype(float).dropna()
    rolling = clean.pct_change(horizon_bars).dropna()
    drawdown_loss = (1 - clean / clean.cummax()).clip(lower=0)
    underwater = drawdown_loss[drawdown_loss > 0]
    reversal = float(rolling.std(ddof=1))
    pivots = confirmed_zigzag(clean, reversal)
    legs = zigzag_legs(pivots, clean)
    magnitudes = legs["magnitude"].astype(float)
    return {
        "series": name,
        "start": clean.index[0].isoformat(),
        "end": clean.index[-1].isoformat(),
        "observations": len(clean),
        "two_week_return_q05": float(rolling.quantile(0.05)),
        "two_week_return_median": float(rolling.median()),
        "two_week_return_q95": float(rolling.quantile(0.95)),
        "two_week_loss_probability": float((rolling < 0).mean()),
        "maximum_drawdown": -float(drawdown_loss.max()),
        "underwater_drawdown_median": -float(underwater.median()),
        "underwater_drawdown_q90": -float(underwater.quantile(0.90)),
        "zigzag_diagnostic_reversal": reversal,
        "zigzag_leg_count": len(legs),
        "zigzag_leg_magnitude_median": float(magnitudes.median()),
        "zigzag_leg_magnitude_q75": float(magnitudes.quantile(0.75)),
        "zigzag_leg_magnitude_q90": float(magnitudes.quantile(0.90)),
    }


def _summary_row(portfolio: str, policy: str, frame: pd.DataFrame) -> dict[str, object]:
    drawdown_loss = (-frame["maximum_drawdown"].astype(float)).clip(lower=0)
    threshold = float(drawdown_loss.quantile(0.95))
    return {
        "portfolio": portfolio,
        "policy": policy,
        "paths": len(frame),
        "nominal_return_median": float(frame["nominal_return"].median()),
        "nominal_loss_probability": float((frame["nominal_return"] < 0).mean()),
        "real_return_median": float(frame["real_return"].median()),
        "real_loss_probability": float((frame["real_return"] < 0).mean()),
        "sharpe_ratio_q05": float(frame["sharpe_ratio"].quantile(0.05)),
        "sharpe_ratio_median": float(frame["sharpe_ratio"].median()),
        "drawdown_loss_q95": threshold,
        "drawdown_loss_expected_shortfall_95": float(
            drawdown_loss[drawdown_loss >= threshold].mean()
        ),
        "turnover_median": float(frame["turnover"].median()),
        "total_cost_median": float(frame["total_cost"].median()),
    }


def _baseline_paths(
    portfolio: str,
    paths: int,
    horizon_bars: int,
    prices: pd.DataFrame,
    tradable: pd.DataFrame,
    usd: pd.Series,
    weights: Mapping[str, float],
    execution: ExecutionModel,
) -> pd.DataFrame:
    simulation = SimulationConfig(
        paths=paths,
        horizon_bars=horizon_bars,
        block_size=10,
        seed=20260825,
    )
    generated = generate_scenario_paths(prices, simulation, usd, usd, tradable)
    rows: list[dict[str, object]] = []
    for path in generated:
        fx_return = float(path.fx_reference.iloc[-1] / path.fx_reference.iloc[0] - 1)
        for strategy in (Strategy.BUY_AND_HOLD, Strategy.CALENDAR_REBALANCED):
            result = run_backtest(
                path.prices,
                weights,
                BacktestPolicy(
                    strategy=strategy,
                    review_frequency=ReviewFrequency.BIWEEKLY,
                    drift_band=0.05,
                    execution=execution,
                ),
                tradable=path.tradable,
            )
            nominal_return = float(result.summary["total_return"])
            rows.append(
                {
                    "portfolio": portfolio,
                    "path_id": path.path_id,
                    "policy": strategy.value,
                    "nominal_return": nominal_return,
                    "real_return": (1 + nominal_return) / (1 + fx_return) - 1,
                    "annualized_volatility": result.summary["annualized_volatility"],
                    "sharpe_ratio": result.summary["sharpe_ratio"],
                    "maximum_drawdown": result.summary["maximum_drawdown"],
                    "turnover": result.summary["turnover"],
                    "total_cost": result.summary["total_cost"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    snapshot = load_snapshot(PROJECT / "data" / "snapshots" / "full_history")
    contract = load_data_contract(PROJECT / "configs" / "data_contract.toml")
    config = load_project_config(PROJECT / "configs" / "portfolios.toml")
    assets = ["fixed_income", "gold", "equity"]
    prices = point_in_time_prices(snapshot.data, contract, assets)
    tradable = point_in_time_tradability(snapshot.data, contract, assets, prices.index)
    usd = _snapshot_series(snapshot.data, "usd_irr", prices.index)
    execution = ExecutionModel(
        asset_costs=load_asset_transaction_costs(
            PROJECT / "configs" / "transaction_costs.toml"
        )
    )

    gold = point_in_time_prices(snapshot.data, contract, ["gold"])["gold"]
    equity = point_in_time_prices(snapshot.data, contract, ["equity"])["equity"]
    common = gold.to_frame().join(equity, how="inner", lsuffix="_gold", rsuffix="_equity")
    common.columns = ["gold", "equity"]
    usd_start = pd.to_datetime(
        snapshot.data.loc[snapshot.data["series"] == "usd_irr", "valuation_at"], utc=True
    ).min()
    common = common.loc[usd_start:]
    common_usd = _snapshot_series(snapshot.data, "usd_irr", common.index)
    normalized = common / common.iloc[0]
    medium_weights = config.portfolios["medium_term"].weights
    risky_total = medium_weights["gold"] + medium_weights["equity"]
    risky_basket = (
        normalized["gold"] * medium_weights["gold"] / risky_total
        + normalized["equity"] * medium_weights["equity"] / risky_total
    )
    risky_basket_usd = risky_basket / (common_usd / common_usd.iloc[0])
    risk_rows = [
        _asset_risk_row("gold_local", gold),
        _asset_risk_row("equity_local", equity),
        _asset_risk_row(
            "gold_usd_relative",
            gold.loc[common_usd.index] / (common_usd / common_usd.iloc[0]),
        ),
        _asset_risk_row(
            "equity_usd_relative",
            equity.loc[common_usd.index] / (common_usd / common_usd.iloc[0]),
        ),
        _asset_risk_row("medium_risky_basket_usd_relative", risky_basket_usd),
    ]
    output = PROJECT / "runs" / "manager_surface"
    pd.DataFrame(risk_rows).to_csv(output / "asset_risk_summary.csv", index=False)

    comparison_rows: list[dict[str, object]] = []
    frequency_rows: list[dict[str, object]] = []
    for portfolio, path_count, horizon in (
        ("medium_term", 200, 126),
        ("long_term", 100, 252),
    ):
        surface = pd.read_csv(output / portfolio / "surface.csv")
        for frequency, group in surface.groupby("review_frequency"):
            frequency_rows.append(
                {
                    "portfolio": portfolio,
                    "review_frequency": frequency,
                    "robust_sharpe_q05_mean": float(group["sharpe_ratio_lower"].mean()),
                    "real_loss_probability_mean": float(group["real_loss_probability"].mean()),
                    "drawdown_es_95_mean": float(
                        group["drawdown_loss_expected_shortfall"].mean()
                    ),
                    "turnover_median_mean": float(group["turnover_median"].mean()),
                }
            )
        path_metrics = pd.read_csv(output / portfolio / "path_metrics.csv")
        candidate = path_metrics[
            np.isclose(path_metrics["multiplier"], 4)
            & np.isclose(path_metrics["floor_fraction"], 0.7)
            & (path_metrics["review_frequency"] == "biweekly")
            & np.isclose(path_metrics["drift_band"], 0.05)
        ]
        comparison_rows.append(_summary_row(portfolio, "vppi_m4_floor70_biweekly", candidate))
        baselines = _baseline_paths(
            portfolio,
            path_count,
            horizon,
            prices,
            tradable,
            usd,
            config.portfolios[portfolio].weights,
            execution,
        )
        for policy, group in baselines.groupby("policy"):
            comparison_rows.append(_summary_row(portfolio, policy, group))

    comparison = pd.DataFrame(comparison_rows)
    frequency = pd.DataFrame(frequency_rows)
    comparison.to_csv(output / "policy_comparison.csv", index=False)
    frequency.to_csv(output / "frequency_comparison.csv", index=False)
    payload = {
        "method": {
            "simulation": "joint 10-bar moving-block bootstrap",
            "seed": 20260825,
            "medium_paths": 200,
            "long_paths": 100,
            "medium_horizon_bars": 126,
            "long_horizon_bars": 252,
            "candidate_policy": {
                "multiplier": 4,
                "floor_fraction": 0.7,
                "review_frequency": "biweekly",
                "drift_band": 0.05,
            },
        },
        "asset_risk": risk_rows,
        "policy_comparison": comparison.to_dict(orient="records"),
        "frequency_comparison": frequency.to_dict(orient="records"),
    }
    (output / "presentation_data.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote presentation evidence to {output}")


if __name__ == "__main__":
    main()
