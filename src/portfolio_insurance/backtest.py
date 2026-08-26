"""Auditable event-driven portfolio backtesting with explicit execution timing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isclose
from types import MappingProxyType
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pandas as pd

from .metrics import (
    annualized_volatility,
    expected_shortfall,
    maximum_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from .overlay import apply_risk_overlays
from .rebalancing import RebalancePolicy, needs_rebalance
from .regime import ProtectionState, RegimePolicy
from .vppi import VppiPolicy, allocate_vppi


class ReviewFrequency(StrEnum):
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"


class Strategy(StrEnum):
    BUY_AND_HOLD = "buy_and_hold"
    CALENDAR_REBALANCED = "calendar_rebalanced"
    VPPI = "vppi"


@dataclass(frozen=True, slots=True)
class AssetTransactionCost:
    asset: str
    buy_bps: float
    sell_bps: float

    def __post_init__(self) -> None:
        if not self.asset.strip():
            raise ValueError("transaction-cost asset cannot be empty")
        if self.buy_bps < 0 or self.sell_bps < 0:
            raise ValueError("asset transaction costs cannot be negative")


@dataclass(frozen=True, slots=True)
class ExecutionModel:
    transaction_cost_bps: float = 0
    slippage_bps: float = 0
    latency_bars: int = 1
    asset_costs: tuple[AssetTransactionCost, ...] = ()

    def __post_init__(self) -> None:
        if self.transaction_cost_bps < 0 or self.slippage_bps < 0:
            raise ValueError("cost and slippage cannot be negative")
        if self.latency_bars < 1:
            raise ValueError("latency_bars must be at least one")
        normalized = tuple(
            item if isinstance(item, AssetTransactionCost) else AssetTransactionCost(**item)
            for item in self.asset_costs
        )
        assets = [item.asset for item in normalized]
        if len(assets) != len(set(assets)):
            raise ValueError("asset transaction-cost entries cannot be duplicated")
        object.__setattr__(self, "asset_costs", normalized)

    def transaction_bps(self, asset: str, trade_value: float) -> float:
        configured = next((item for item in self.asset_costs if item.asset == asset), None)
        if configured is None:
            return self.transaction_cost_bps
        return configured.buy_bps if trade_value >= 0 else configured.sell_bps


def calculate_trade_costs(
    trades: Mapping[str, float] | pd.Series,
    execution: ExecutionModel,
) -> tuple[float, float, dict[str, float]]:
    """Return explicit commissions, slippage, and total cost allocated by asset."""
    transaction_cost = 0.0
    slippage_cost = 0.0
    by_asset: dict[str, float] = {}
    for asset, raw_value in trades.items():
        value = float(raw_value)
        notional = abs(value)
        commission = notional * execution.transaction_bps(str(asset), value) / 10_000
        slippage = notional * execution.slippage_bps / 10_000
        transaction_cost += commission
        slippage_cost += slippage
        by_asset[str(asset)] = commission + slippage
    return transaction_cost, slippage_cost, by_asset


@dataclass(frozen=True, slots=True)
class SleevePolicy:
    multiplier: float = 3
    floor_fraction: float = 0.8

    def __post_init__(self) -> None:
        if self.multiplier < 0 or not 0 <= self.floor_fraction <= 1:
            raise ValueError("invalid sleeve multiplier or floor")


@dataclass(frozen=True, slots=True)
class BacktestPolicy:
    strategy: Strategy = Strategy.CALENDAR_REBALANCED
    review_frequency: ReviewFrequency = ReviewFrequency.MONTHLY
    drift_band: float = 0
    execution: ExecutionModel = ExecutionModel()
    floor_fraction: float = 0.8
    multiplier: float = 3
    insured_assets: tuple[str, ...] = ("gold", "equity")
    reserve_asset: str = "fixed_income"
    reserve_daily_return: float | None = None
    reserve_calendar_day_accrual: bool = False
    sleeve_policies: Mapping[str, SleevePolicy] | None = None
    regime_policies: Mapping[str, RegimePolicy] | None = None
    regime_calibration_id: str | None = None
    protection_drift_band: float | None = None

    def __post_init__(self) -> None:
        SleevePolicy(self.multiplier, self.floor_fraction)
        RebalancePolicy(self.drift_band)
        if self.protection_drift_band is not None:
            RebalancePolicy(self.protection_drift_band)
        if self.reserve_daily_return is not None and self.reserve_daily_return <= -1:
            raise ValueError("reserve_daily_return must be greater than -100%")
        if self.reserve_calendar_day_accrual and self.reserve_daily_return is None:
            raise ValueError("calendar-day reserve accrual requires reserve_daily_return")
        if len(set(self.insured_assets)) != len(self.insured_assets):
            raise ValueError("insured_assets cannot contain duplicates")
        if self.sleeve_policies is not None:
            unknown = set(self.sleeve_policies) - set(self.insured_assets)
            if unknown:
                raise ValueError(f"sleeve policies reference uninsured assets: {sorted(unknown)}")
            object.__setattr__(
                self, "sleeve_policies", MappingProxyType(dict(self.sleeve_policies))
            )
        if self.regime_policies is not None:
            normalized = {
                asset: item if isinstance(item, RegimePolicy) else RegimePolicy(**item)
                for asset, item in self.regime_policies.items()
            }
            if set(normalized) != set(self.insured_assets):
                raise ValueError("regime policies must cover every insured asset exactly")
            if self.strategy != Strategy.VPPI:
                raise ValueError("regime policies apply only to the VPPI strategy")
            object.__setattr__(self, "regime_policies", MappingProxyType(normalized))
        if self.regime_calibration_id is not None and self.regime_policies is None:
            raise ValueError("regime_calibration_id requires regime policies")

    def sleeve_policy(self, asset: str) -> SleevePolicy:
        if self.sleeve_policies and asset in self.sleeve_policies:
            return self.sleeve_policies[asset]
        return SleevePolicy(self.multiplier, self.floor_fraction)

    @property
    def effective_protection_drift_band(self) -> float:
        """Use the legacy band unless a separate protection band is configured."""
        return (
            self.drift_band
            if self.protection_drift_band is None
            else self.protection_drift_band
        )


@dataclass(frozen=True, slots=True)
class BacktestResult:
    ledger: pd.DataFrame
    summary: Mapping[str, float]
    metadata: Mapping[str, object]

    @property
    def returns(self) -> pd.Series:
        return self.ledger["net_return"].rename("return")

    def verify_ledger(self, tolerance: float = 1e-8) -> None:
        holding_columns = [column for column in self.ledger if column.startswith("holding_")]
        weight_columns = [column for column in self.ledger if column.startswith("weight_")]
        if not np.allclose(
            self.ledger[holding_columns].sum(axis=1), self.ledger["nav"], atol=tolerance
        ):
            raise ValueError("asset holdings do not reconcile to NAV")
        if not np.allclose(self.ledger[weight_columns].sum(axis=1), 1, atol=tolerance):
            raise ValueError("asset weights do not sum to one")
        if not np.allclose(
            self.ledger["gross_nav"] - self.ledger["cost"], self.ledger["nav"], atol=tolerance
        ):
            raise ValueError("gross NAV less costs does not reconcile to net NAV")
        executions = self.ledger[self.ledger["order_executed"]]
        if (
            not executions.empty
            and not (executions.index.to_series().array > executions["decision_at"].array).all()
        ):
            raise ValueError("ledger contains same-bar or future-informed execution")
        for asset in (
            column.removeprefix("trade_") for column in self.ledger if column.startswith("trade_")
        ):
            traded = executions[f"trade_{asset}"].abs() > tolerance
            if traded.any() and not executions.loc[traded, f"tradable_{asset}"].all():
                raise ValueError(f"ledger traded {asset} while its market was closed")


@dataclass(slots=True)
class _SleeveState:
    nav: float
    floor: float
    risky_value: float
    reserve_value: float

    @property
    def factor(self) -> float:
        return self.risky_value / self.nav


@dataclass(slots=True)
class _ProtectionRegimeState:
    state: ProtectionState
    peak: float
    trough: float
    drawdown: float = 0
    recovery: float = 0
    transition: str | None = None

    def update(self, signal: float, policy: RegimePolicy) -> bool:
        self.transition = None
        if self.state == ProtectionState.NORMAL:
            self.peak = max(self.peak, signal)
            self.drawdown = signal / self.peak - 1
            self.recovery = 0
            if self.drawdown <= -policy.entry_drawdown:
                self.state = ProtectionState.PROTECTED
                self.trough = signal
                self.transition = "enter_protection"
        else:
            self.trough = min(self.trough, signal)
            self.drawdown = signal / self.peak - 1
            self.recovery = signal / self.trough - 1
            if self.recovery >= policy.exit_recovery:
                self.state = ProtectionState.NORMAL
                self.peak = signal
                self.trough = signal
                self.drawdown = 0
                self.transition = "exit_protection"
        return self.transition is not None


@dataclass(frozen=True, slots=True)
class _PendingOrder:
    order_id: str
    decision_at: pd.Timestamp
    execute_bar: int
    execute_at: pd.Timestamp
    target: pd.Series
    sleeve_factors: Mapping[str, float]


def _reviews(index: pd.DatetimeIndex, frequency: ReviewFrequency) -> np.ndarray:
    interval_bars = {
        ReviewFrequency.DAILY: 1,
        ReviewFrequency.WEEKLY: 5,
        ReviewFrequency.BIWEEKLY: 10,
        ReviewFrequency.MONTHLY: 21,
    }[frequency]
    positions = np.arange(len(index))
    return (positions > 0) & (positions % interval_bars == 0)


def relative_prices(prices: pd.DataFrame, reference: pd.Series) -> pd.DataFrame:
    """Express prices in a reference unit, such as USD/IRR."""
    frame, ref = prices.align(reference.astype(float), axis=0, join="inner")
    if (ref <= 0).any():
        raise ValueError("reference prices must be positive")
    return frame.div(ref, axis=0)


def with_daily_accrual_prices(
    prices: pd.DataFrame,
    asset: str,
    daily_return: float | None,
    calendar_day_accrual: bool,
) -> pd.DataFrame:
    """Replace one analytical price series with a deterministic daily accrual index."""
    if daily_return is None:
        return prices.copy()
    if asset not in prices:
        raise ValueError(f"daily-accrual asset is missing from prices: {asset}")
    if daily_return <= -1:
        raise ValueError("daily accrual return must be greater than -100%")
    if calendar_day_accrual:
        elapsed = (prices.index - prices.index[0]).total_seconds() / 86_400
    else:
        elapsed = np.arange(len(prices), dtype=float)
    result = prices.copy()
    result[asset] = float(prices[asset].iloc[0]) * (1 + daily_return) ** elapsed
    return result


def _asset_returns(prices: pd.DataFrame, policy: BacktestPolicy) -> pd.DataFrame:
    returns = prices.pct_change().fillna(0)
    if policy.reserve_daily_return is None:
        return returns
    if policy.reserve_calendar_day_accrual:
        elapsed = prices.index.to_series().diff().dt.total_seconds().div(86_400).fillna(0)
    else:
        elapsed = pd.Series(np.r_[0.0, np.ones(len(prices) - 1)], index=prices.index)
    returns[policy.reserve_asset] = (1 + policy.reserve_daily_return) ** elapsed - 1
    return returns


def _validate_inputs(
    prices: pd.DataFrame, weights: Mapping[str, float], policy: BacktestPolicy, initial_nav: float
) -> pd.DataFrame:
    if initial_nav <= 0 or len(prices) < 2:
        raise ValueError("positive initial_nav and at least two prices are required")
    if (
        not isinstance(prices.index, pd.DatetimeIndex)
        or not prices.index.is_monotonic_increasing
        or prices.index.has_duplicates
    ):
        raise ValueError("prices require a sorted, unique DatetimeIndex")
    if set(prices.columns) != set(weights) or not isclose(sum(weights.values()), 1):
        raise ValueError("price assets must match weights that sum to one")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("weights cannot be negative")
    frame = prices.astype(float)
    if (
        frame.isna().any().any()
        or not np.isfinite(frame.to_numpy()).all()
        or (frame <= 0).any().any()
    ):
        raise ValueError("prices must be finite and positive")
    assets = set(frame.columns)
    if policy.reserve_asset not in assets or set(policy.insured_assets) - assets:
        raise ValueError("reserve and insured assets must exist")
    if policy.reserve_asset in policy.insured_assets:
        raise ValueError("reserve asset cannot be insured")
    return frame


def _validate_tradability(prices: pd.DataFrame, tradable: pd.DataFrame | None) -> pd.DataFrame:
    if tradable is None:
        return pd.DataFrame(True, index=prices.index, columns=prices.columns)
    if not isinstance(tradable, pd.DataFrame):
        raise TypeError("tradable must be a DataFrame")
    if not tradable.index.equals(prices.index) or set(tradable.columns) != set(prices.columns):
        raise ValueError("tradability index and assets must match prices")
    aligned = tradable.reindex(columns=prices.columns)
    if aligned.isna().any().any():
        raise ValueError("tradability cannot contain missing values")
    return aligned.astype(bool)


def _serializable_policy(policy: BacktestPolicy) -> dict[str, object]:
    return {
        "strategy": policy.strategy.value,
        "review_frequency": policy.review_frequency.value,
        "drift_band": policy.drift_band,
        "protection_drift_band": policy.protection_drift_band,
        "execution": asdict(policy.execution),
        "floor_fraction": policy.floor_fraction,
        "multiplier": policy.multiplier,
        "insured_assets": list(policy.insured_assets),
        "reserve_asset": policy.reserve_asset,
        "reserve_daily_return": policy.reserve_daily_return,
        "reserve_calendar_day_accrual": policy.reserve_calendar_day_accrual,
        "sleeve_policies": {
            asset: asdict(item) for asset, item in (policy.sleeve_policies or {}).items()
        },
        "regime_policies": {
            asset: asdict(item) for asset, item in (policy.regime_policies or {}).items()
        },
        "regime_calibration_id": policy.regime_calibration_id,
    }


def _run_identity(
    prices: pd.DataFrame,
    tradable: pd.DataFrame,
    weights: Mapping[str, float],
    policy: BacktestPolicy,
    initial_nav: float,
) -> tuple[str, str, str]:
    canonical_prices = prices.to_csv(date_format="%Y-%m-%dT%H:%M:%S.%f%z")
    data_hash = hashlib.sha256(canonical_prices.encode()).hexdigest()
    canonical_tradability = tradable.astype(int).to_csv(date_format="%Y-%m-%dT%H:%M:%S.%f%z")
    availability_hash = hashlib.sha256(canonical_tradability.encode()).hexdigest()
    payload = json.dumps(
        {
            "data_sha256": data_hash,
            "tradability_sha256": availability_hash,
            "weights": dict(weights),
            "policy": _serializable_policy(policy),
            "initial_nav": initial_nav,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return str(uuid5(NAMESPACE_URL, payload)), data_hash, availability_hash


def run_backtest(
    prices: pd.DataFrame,
    strategic_weights: Mapping[str, float],
    policy: BacktestPolicy | None = None,
    initial_nav: float = 1_000_000,
    tradable: pd.DataFrame | None = None,
) -> BacktestResult:
    """Simulate close-to-close; observations at t can only create trades after t."""
    policy = policy or BacktestPolicy()
    prices = _validate_inputs(prices, strategic_weights, policy, initial_nav)
    tradability = _validate_tradability(prices, tradable)
    assets = list(prices.columns)
    returns = _asset_returns(prices, policy)
    review = _reviews(prices.index, policy.review_frequency)
    holdings = pd.Series(strategic_weights, dtype=float)[assets] * initial_nav
    portfolio_floor = initial_nav * policy.floor_fraction
    sleeves = {
        asset: _SleeveState(
            nav=initial_nav * strategic_weights[asset],
            floor=initial_nav
            * strategic_weights[asset]
            * policy.sleeve_policy(asset).floor_fraction,
            risky_value=initial_nav * strategic_weights[asset],
            reserve_value=0,
        )
        for asset in policy.insured_assets
    }
    regimes = (
        {
            asset: _ProtectionRegimeState(
                ProtectionState.NORMAL,
                float(prices[asset].iloc[0]),
                float(prices[asset].iloc[0]),
            )
            for asset in policy.insured_assets
        }
        if policy.regime_policies is not None
        else {}
    )
    pending: _PendingOrder | None = None
    rows: list[dict[str, object]] = []
    cumulative_cost = cumulative_transaction_cost = cumulative_slippage_cost = 0.0
    cumulative_turnover = 0.0
    deferred_execution_bars = 0
    next_order_number = 1

    for bar, timestamp in enumerate(prices.index):
        bar_returns = returns.loc[timestamp]
        market_open = tradability.loc[timestamp]
        opening_nav = float(holdings.sum())
        holdings *= 1 + bar_returns
        gross_nav = float(holdings.sum())
        for asset, sleeve in sleeves.items():
            sleeve.risky_value *= 1 + bar_returns[asset]
            sleeve.reserve_value *= 1 + bar_returns[policy.reserve_asset]
            sleeve.nav = sleeve.risky_value + sleeve.reserve_value

        trade = pd.Series(0.0, index=assets)
        execution_target = pd.Series(np.nan, index=assets)
        decision_at, execution_order_id, executed = pd.NaT, None, False
        execution_deferred = False
        deferred_assets: list[str] = []
        scheduled_execution_at = pending.execute_at if pending is not None else pd.NaT
        if pending is not None and pending.execute_bar <= bar:
            candidate_trade = pending.target * gross_nav - holdings
            required_assets = candidate_trade.index[
                candidate_trade.abs() > max(gross_nav * 1e-12, 1e-8)
            ]
            deferred_assets = [asset for asset in required_assets if not market_open[asset]]
        if pending is not None and pending.execute_bar <= bar and not deferred_assets:
            execution_target = pending.target
            decision_at = pending.decision_at
            execution_order_id = pending.order_id
            trade = execution_target * gross_nav - holdings
            turnover = float(trade.abs().sum() / gross_nav)
            transaction_cost, slippage_cost, cost_by_asset = calculate_trade_costs(
                trade, policy.execution
            )
            cost = transaction_cost + slippage_cost
            holdings = execution_target * (gross_nav - cost)
            for asset, factor in pending.sleeve_factors.items():
                sleeves[asset].risky_value = sleeves[asset].nav * factor
                sleeves[asset].reserve_value = sleeves[asset].nav * (1 - factor)
            cumulative_cost += cost
            cumulative_transaction_cost += transaction_cost
            cumulative_slippage_cost += slippage_cost
            cumulative_turnover += turnover
            pending, executed = None, True
        elif pending is not None and pending.execute_bar <= bar:
            turnover = cost = transaction_cost = slippage_cost = 0.0
            cost_by_asset = {asset: 0.0 for asset in assets}
            execution_deferred = True
            deferred_execution_bars += 1
        else:
            turnover = cost = transaction_cost = slippage_cost = 0.0
            cost_by_asset = {asset: 0.0 for asset in assets}

        nav = float(holdings.sum())
        current = holdings / nav
        regime_changed = False
        for asset, regime in regimes.items():
            regime_changed = (
                regime.update(float(prices.loc[timestamp, asset]), policy.regime_policies[asset])
                or regime_changed
            )
        calendar_reviewed = bool(review[bar])
        reviewed = (
            (calendar_reviewed or regime_changed)
            and bar < len(prices) - 1
            and policy.strategy != Strategy.BUY_AND_HOLD
        )
        desired = pd.Series(strategic_weights, dtype=float)
        desired_factors = {asset: sleeves[asset].factor for asset in sleeves}
        if policy.strategy == Strategy.VPPI:
            for asset, sleeve in sleeves.items():
                protected = not regimes or regimes[asset].state == ProtectionState.PROTECTED
                if protected:
                    allocation = allocate_vppi(
                        sleeve.nav,
                        sleeve.floor,
                        VppiPolicy(policy.sleeve_policy(asset).multiplier),
                    )
                    desired_factors[asset] = allocation.risky_weight
                else:
                    desired_factors[asset] = 1.0
            desired = pd.Series(
                apply_risk_overlays(strategic_weights, desired_factors, policy.reserve_asset)
            )

        scheduled, decision_order_id = False, None
        blocked_by_pending_order_id: str | None = None
        execute_at = pd.NaT
        decision_target = pd.Series(np.nan, index=assets)
        if reviewed:
            decision_target = desired[assets]
            protection_review = policy.strategy == Strategy.VPPI and (
                not regimes
                or regime_changed
                or any(
                    regime.state == ProtectionState.PROTECTED
                    for regime in regimes.values()
                )
            )
            active_drift_band = (
                policy.effective_protection_drift_band
                if protection_review
                else policy.drift_band
            )
            if needs_rebalance(
                current.to_dict(), desired.to_dict(), RebalancePolicy(active_drift_band)
            ):
                execute_bar = bar + policy.execution.latency_bars
                if execute_bar < len(prices) and pending is None:
                    decision_order_id = f"order-{next_order_number:06d}"
                    next_order_number += 1
                    execute_at = prices.index[execute_bar]
                    pending = _PendingOrder(
                        decision_order_id,
                        timestamp,
                        execute_bar,
                        execute_at,
                        desired[assets],
                        MappingProxyType(dict(desired_factors)),
                    )
                    scheduled = True
                elif execute_bar < len(prices) and pending is not None:
                    blocked_by_pending_order_id = pending.order_id

        gap_breach = opening_nav >= portfolio_floor and gross_nav < portfolio_floor
        row: dict[str, object] = {
            "timestamp": timestamp,
            "opening_nav": opening_nav,
            "gross_nav": gross_nav,
            "nav": nav,
            "gross_return": gross_nav / opening_nav - 1,
            "net_return": nav / opening_nav - 1,
            "floor_value": portfolio_floor,
            "floor_breach": nav < portfolio_floor,
            "gap_breach": gap_breach,
            "gap_breach_amount": max(portfolio_floor - gross_nav, 0) if gap_breach else 0,
            "calendar_reviewed": calendar_reviewed,
            "reviewed": reviewed,
            "regime_changed": regime_changed,
            "order_scheduled": scheduled,
            "order_executed": executed,
            "order_execution_deferred": execution_deferred,
            "decision_order_id": decision_order_id,
            "execution_order_id": execution_order_id,
            "decision_at": decision_at,
            "execute_at": execute_at,
            "scheduled_execution_at": scheduled_execution_at,
            "actual_execution_at": timestamp if executed else pd.NaT,
            "pending_order_id": pending.order_id if pending is not None else None,
            "deferred_assets": ",".join(deferred_assets) if deferred_assets else None,
            "blocked_by_pending_order_id": blocked_by_pending_order_id,
            "market_fully_open": bool(market_open.all()),
            "turnover": turnover,
            "cost": cost,
            "transaction_cost": transaction_cost,
            "slippage_cost": slippage_cost,
        }
        for asset in assets:
            row[f"tradable_{asset}"] = bool(market_open[asset])
            row[f"holding_{asset}"] = float(holdings[asset])
            row[f"weight_{asset}"] = float(holdings[asset] / nav)
            row[f"decision_target_{asset}"] = float(decision_target[asset])
            row[f"execution_target_{asset}"] = float(execution_target[asset])
            row[f"target_{asset}"] = float(execution_target[asset])
            row[f"trade_{asset}"] = float(trade[asset])
            row[f"cost_{asset}"] = float(cost_by_asset[asset])
        for asset, sleeve in sleeves.items():
            row[f"sleeve_nav_{asset}"] = sleeve.nav
            row[f"sleeve_floor_{asset}"] = sleeve.floor
            row[f"exposure_factor_{asset}"] = sleeve.factor
            row[f"sleeve_breach_{asset}"] = sleeve.nav < sleeve.floor
            if asset in regimes:
                regime = regimes[asset]
                row[f"regime_{asset}"] = regime.state.value
                row[f"regime_signal_{asset}"] = float(prices.loc[timestamp, asset])
                row[f"regime_peak_{asset}"] = regime.peak
                row[f"regime_trough_{asset}"] = regime.trough
                row[f"regime_drawdown_{asset}"] = regime.drawdown
                row[f"regime_recovery_{asset}"] = regime.recovery
                row[f"regime_transition_{asset}"] = regime.transition
        rows.append(row)

    ledger = pd.DataFrame(rows).set_index("timestamp")
    net_returns = ledger["net_return"]
    run_id, data_hash, availability_hash = _run_identity(
        prices, tradability, strategic_weights, policy, initial_nav
    )
    summary = {
        "ending_nav": float(ledger["nav"].iloc[-1]),
        "total_return": float(ledger["nav"].iloc[-1] / initial_nav - 1),
        "annualized_volatility": annualized_volatility(net_returns, 252),
        "sharpe_ratio": sharpe_ratio(net_returns, 252),
        "sortino_ratio": sortino_ratio(net_returns, 252),
        "maximum_drawdown": maximum_drawdown(net_returns),
        "expected_shortfall_95": expected_shortfall(net_returns, 0.95),
        "turnover": cumulative_turnover,
        "total_cost": cumulative_cost,
        "total_transaction_cost": cumulative_transaction_cost,
        "total_slippage_cost": cumulative_slippage_cost,
        "cost_drag": cumulative_cost / initial_nav,
        "floor_breach_bar_fraction": float(ledger["floor_breach"].mean()),
        "floor_breach_probability": float(ledger["floor_breach"].mean()),
        "ever_breached": float(ledger["floor_breach"].any()),
        "floor_breach_severity": float(((ledger["nav"] / portfolio_floor - 1).clip(upper=0)).min()),
        "gap_breach_count": float(ledger["gap_breach"].sum()),
        "market_closed_bar_fraction": float((~ledger["market_fully_open"]).mean()),
        "deferred_execution_bars": float(deferred_execution_bars),
    }
    for asset in regimes:
        summary[f"protection_activation_count_{asset}"] = float(
            (ledger[f"regime_transition_{asset}"] == "enter_protection").sum()
        )
        summary[f"protection_bar_fraction_{asset}"] = float(
            (ledger[f"regime_{asset}"] == ProtectionState.PROTECTED.value).mean()
        )
    metadata: Mapping[str, object] = MappingProxyType(
        {
            "run_id": run_id,
            "data_sha256": data_hash,
            "tradability_sha256": availability_hash,
            "policy": _serializable_policy(policy),
            "strategic_weights": dict(strategic_weights),
            "initial_nav": initial_nav,
            "first_timestamp": prices.index[0].isoformat(),
            "last_timestamp": prices.index[-1].isoformat(),
            "observations": len(prices),
            "timing": "close-to-close return, then prior order at close, then new decision",
        }
    )
    result = BacktestResult(ledger, MappingProxyType(summary), metadata)
    result.verify_ledger()
    return result


def run_baselines(
    prices: pd.DataFrame,
    weights: Mapping[str, float],
    review_frequency: ReviewFrequency = ReviewFrequency.MONTHLY,
    drift_band: float = 0,
    execution: ExecutionModel | None = None,
    reserve_daily_return: float | None = None,
    reserve_calendar_day_accrual: bool = False,
    tradable: pd.DataFrame | None = None,
    **kwargs: object,
) -> dict[str, BacktestResult]:
    execution = execution or ExecutionModel()
    return {
        strategy.value: run_backtest(
            prices,
            weights,
            BacktestPolicy(
                strategy=strategy,
                review_frequency=review_frequency,
                drift_band=drift_band,
                execution=execution,
                reserve_daily_return=reserve_daily_return,
                reserve_calendar_day_accrual=reserve_calendar_day_accrual,
            ),
            tradable=tradable,
            **kwargs,
        )
        for strategy in (Strategy.BUY_AND_HOLD, Strategy.CALENDAR_REBALANCED)
    }


def attribute_effects(
    buy_hold: BacktestResult, calendar: BacktestResult, insured: BacktestResult
) -> pd.DataFrame:
    """Geometrically attribute allocation, rebalancing, insurance, and insured costs."""
    frame = pd.concat(
        {
            "allocation": buy_hold.returns,
            "calendar": calendar.returns,
            "insured_gross": insured.ledger["gross_return"],
            "insured_net": insured.returns,
        },
        axis=1,
    ).dropna()
    out = pd.DataFrame(index=frame.index)
    out["allocation"] = frame["allocation"]
    out["rebalancing"] = (1 + frame["calendar"]) / (1 + frame["allocation"]) - 1
    out["insurance"] = (1 + frame["insured_gross"]) / (1 + frame["calendar"]) - 1
    out["cost"] = (1 + frame["insured_net"]) / (1 + frame["insured_gross"]) - 1
    out["reconstructed"] = (1 + out["allocation"]) * (1 + out["rebalancing"]) * (
        1 + out["insurance"]
    ) * (1 + out["cost"]) - 1
    return out
