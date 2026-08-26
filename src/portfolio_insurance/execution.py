"""Controlled paper execution with approval gates and tamper-evident audit records."""

from __future__ import annotations

import hashlib
import html
import json
import sqlite3
import tomllib
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from math import isclose
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

import pandas as pd

from .backtest import ExecutionModel, ReviewFrequency, calculate_trade_costs
from .overlay import apply_risk_overlays
from .rebalancing import RebalancePolicy, needs_rebalance
from .vppi import VppiPolicy, allocate_vppi

SCHEMA_VERSION = 1
ACTIVE_ORDER_STATUSES = ("pending_approval", "approved")


class OrderStatus(StrEnum):
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    EXPIRED = "expired"
    CANCELED = "canceled"
    CONTROL_BLOCKED = "control_blocked"


@dataclass(frozen=True, slots=True)
class ExecutionControls:
    """Human-approved operational boundaries for a paper portfolio."""

    max_daily_turnover: float = 0.20
    max_data_age_hours: float = 36
    floor_alert_buffer_fraction: float = 0.05
    reconciliation_tolerance: float = 0.01
    approval_expiry_hours: float = 24

    def __post_init__(self) -> None:
        if not 0 < self.max_daily_turnover <= 2:
            raise ValueError("max_daily_turnover must be in (0, 2]")
        if self.max_data_age_hours <= 0:
            raise ValueError("max_data_age_hours must be positive")
        if not 0 <= self.floor_alert_buffer_fraction <= 1:
            raise ValueError("floor_alert_buffer_fraction must be between zero and one")
        if self.reconciliation_tolerance < 0:
            raise ValueError("reconciliation_tolerance cannot be negative")
        if self.approval_expiry_hours <= 0:
            raise ValueError("approval_expiry_hours must be positive")


@dataclass(frozen=True, slots=True)
class PaperDayResult:
    valuation_at: pd.Timestamp
    nav: float
    floor_value: float
    distance_to_floor: float
    reconciliation_status: str
    order_id: str | None
    order_status: str | None
    alerts: tuple[Mapping[str, object], ...]
    idempotent: bool = False


def load_execution_controls(path: str | Path, portfolio: str) -> ExecutionControls:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    values = dict(raw.get("common", {}))
    values.update(raw.get("portfolios", {}).get(portfolio, {}))
    return ExecutionControls(**values)


def with_control_overrides(
    controls: ExecutionControls,
    *,
    max_daily_turnover: float | None = None,
    max_data_age_hours: float | None = None,
    floor_alert_buffer_fraction: float | None = None,
    reconciliation_tolerance: float | None = None,
    approval_expiry_hours: float | None = None,
) -> ExecutionControls:
    values = {
        key: value
        for key, value in {
            "max_daily_turnover": max_daily_turnover,
            "max_data_age_hours": max_data_age_hours,
            "floor_alert_buffer_fraction": floor_alert_buffer_fraction,
            "reconciliation_tolerance": reconciliation_tolerance,
            "approval_expiry_hours": approval_expiry_hours,
        }.items()
        if value is not None
    }
    return replace(controls, **values)


def _utc_timestamp(value: object | None = None) -> pd.Timestamp:
    timestamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp


def _iso(value: object | None = None) -> str:
    return _utc_timestamp(value).isoformat()


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _review_due(
    previous_review: pd.Timestamp | None,
    valuation_at: pd.Timestamp,
    frequency: ReviewFrequency,
) -> bool:
    if previous_review is None:
        return True
    previous = previous_review.tz_convert("UTC").tz_localize(None)
    current = valuation_at.tz_convert("UTC").tz_localize(None)
    if frequency == ReviewFrequency.DAILY:
        return current.date() > previous.date()
    if frequency == ReviewFrequency.MONTHLY:
        return current.to_period("M") != previous.to_period("M")
    previous_week = previous.to_period("W").ordinal
    current_week = current.to_period("W").ordinal
    divisor = 2 if frequency == ReviewFrequency.BIWEEKLY else 1
    return current_week // divisor != previous_week // divisor


class PaperPortfolio:
    """SQLite-backed paper portfolio whose orders require explicit approval."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"paper portfolio does not exist: {self.path}")
        with self._connect() as connection:
            version = self._metadata(connection, "schema_version")
        if int(version) != SCHEMA_VERSION:
            raise ValueError(f"unsupported paper portfolio schema: {version}")

    @classmethod
    def initialize(
        cls,
        path: str | Path,
        *,
        portfolio: str,
        strategic_weights: Mapping[str, float],
        selected_policy: Mapping[str, object],
        selection_metadata: Mapping[str, object],
        prices: Mapping[str, float],
        valuation_at: object,
        approved_by: str,
        controls: ExecutionControls,
        execution: ExecutionModel | None = None,
        initial_nav: float = 1_000_000,
        reserve_asset: str = "fixed_income",
        insured_assets: tuple[str, ...] = ("gold", "equity"),
        approved_at: object | None = None,
    ) -> PaperPortfolio:
        target = Path(path)
        if target.exists():
            raise FileExistsError(f"paper portfolio already exists: {target}")
        if not approved_by.strip():
            raise ValueError("approved_by is required")
        if initial_nav <= 0:
            raise ValueError("initial_nav must be positive")
        if selection_metadata.get("selection_status") != "selected":
            raise ValueError("Stage 4 must have selected a policy")
        if selection_metadata.get("validation_status") in {
            "failed",
            "insufficient_folds",
            "incomplete_fold_selection",
            "not_selected",
        }:
            raise ValueError("Stage 4 validation does not permit paper deployment")
        if selection_metadata.get("portfolio") != portfolio:
            raise ValueError("Stage 4 selection portfolio does not match")
        if set(strategic_weights) != set(prices):
            raise ValueError("opening prices must match strategic assets")
        if reserve_asset not in strategic_weights:
            raise ValueError("reserve asset must be a strategic asset")
        effective_insured = tuple(asset for asset in insured_assets if asset in strategic_weights)
        if not effective_insured:
            raise ValueError("paper portfolio requires at least one insured asset")
        if any(float(price) <= 0 for price in prices.values()):
            raise ValueError("opening prices must be positive")
        if not isclose(sum(strategic_weights.values()), 1, abs_tol=1e-10):
            raise ValueError("strategic weights must sum to one")

        policy = {
            "multiplier": float(selected_policy["multiplier"]),
            "floor_fraction": float(selected_policy["floor_fraction"]),
            "review_frequency": ReviewFrequency(str(selected_policy["review_frequency"])).value,
            "drift_band": float(selected_policy["drift_band"]),
            "reserve_asset": reserve_asset,
            "insured_assets": list(effective_insured),
        }
        VppiPolicy(policy["multiplier"])
        RebalancePolicy(policy["drift_band"])
        if not 0 <= policy["floor_fraction"] <= 1:
            raise ValueError("floor_fraction must be between zero and one")
        execution = execution or ExecutionModel()
        timestamp = _utc_timestamp(valuation_at)
        approval_timestamp = _utc_timestamp(approved_at)
        if approval_timestamp < timestamp:
            raise ValueError("policy approval cannot precede the opening valuation")
        opening_data_age = (approval_timestamp - timestamp).total_seconds() / 3600
        if opening_data_age > controls.max_data_age_hours:
            raise ValueError(
                f"opening valuation is {opening_data_age:.1f} hours old; "
                f"limit is {controls.max_data_age_hours:g} hours"
            )
        target.parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(target)
        try:
            connection.row_factory = sqlite3.Row
            cls._create_schema(connection)
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "portfolio": portfolio,
                "strategic_weights": dict(strategic_weights),
                "policy": policy,
                "controls": asdict(controls),
                "execution": asdict(execution),
                "selection_id": selection_metadata.get("selection_id"),
                "source_analysis_id": selection_metadata.get("source_analysis_id"),
                "selection_validation_status": selection_metadata.get("validation_status"),
                "policy_approved_by": approved_by,
                "policy_approved_at": approval_timestamp.isoformat(),
                "initial_nav": initial_nav,
                "portfolio_floor": initial_nav * policy["floor_fraction"],
                "kill_switch": False,
                "created_at": approval_timestamp.isoformat(),
                "last_review_at": timestamp.isoformat(),
            }
            for key, value in metadata.items():
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)", (key, _canonical(value))
                )
            for asset, weight in strategic_weights.items():
                connection.execute(
                    "INSERT INTO positions(asset, units) VALUES (?, ?)",
                    (asset, initial_nav * float(weight) / float(prices[asset])),
                )
                connection.execute(
                    "INSERT INTO price_marks(valuation_at, asset, price, tradable) "
                    "VALUES (?, ?, ?, 1)",
                    (timestamp.isoformat(), asset, float(prices[asset])),
                )
            for asset in effective_insured:
                sleeve_nav = initial_nav * float(strategic_weights[asset])
                connection.execute(
                    "INSERT INTO sleeves(asset, risky_value, reserve_value, floor_value) "
                    "VALUES (?, ?, 0, ?)",
                    (asset, sleeve_nav, sleeve_nav * policy["floor_fraction"]),
                )
            connection.execute(
                "INSERT INTO valuations(valuation_at, observed_at, nav, floor_value, "
                "distance_to_floor, distance_to_floor_fraction, reconciliation_status, "
                "expected_value, observed_value, order_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    timestamp.isoformat(),
                    approval_timestamp.isoformat(),
                    initial_nav,
                    metadata["portfolio_floor"],
                    initial_nav - metadata["portfolio_floor"],
                    (initial_nav - metadata["portfolio_floor"]) / initial_nav,
                    "matched",
                    initial_nav,
                    initial_nav,
                    None,
                ),
            )
            cls._append_audit(
                connection,
                "paper_portfolio_initialized",
                approved_by,
                "portfolio",
                portfolio,
                {
                    "valuation_at": timestamp.isoformat(),
                    "opening_positions": "strategic_weights",
                    "initial_nav": initial_nav,
                    "selection_id": metadata["selection_id"],
                },
                approval_timestamp,
            )
            cls._append_audit(
                connection,
                "policy_approved",
                approved_by,
                "policy",
                str(metadata["selection_id"]),
                {
                    "policy": policy,
                    "controls": asdict(controls),
                    "validation_status": metadata["selection_validation_status"],
                },
                approval_timestamp,
            )
            connection.commit()
        except Exception:
            connection.close()
            if target.exists():
                target.unlink()
            raise
        finally:
            connection.close()
        account = cls(target)
        account.export_dashboard()
        return account

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE positions (asset TEXT PRIMARY KEY, units REAL NOT NULL);
            CREATE TABLE sleeves (
                asset TEXT PRIMARY KEY, risky_value REAL NOT NULL,
                reserve_value REAL NOT NULL, floor_value REAL NOT NULL
            );
            CREATE TABLE price_marks (
                valuation_at TEXT NOT NULL, asset TEXT NOT NULL,
                price REAL NOT NULL CHECK(price > 0),
                tradable INTEGER NOT NULL CHECK(tradable IN (0, 1)),
                PRIMARY KEY (valuation_at, asset)
            );
            CREATE TABLE orders (
                order_id TEXT PRIMARY KEY, decision_at TEXT NOT NULL, expires_at TEXT NOT NULL,
                status TEXT NOT NULL, target_json TEXT NOT NULL, factors_json TEXT NOT NULL,
                rationale_json TEXT NOT NULL, proposed_turnover REAL NOT NULL,
                approved_by TEXT, approved_at TEXT, rejected_by TEXT, rejected_at TEXT,
                rejection_reason TEXT, executed_at TEXT, total_cost REAL
            );
            CREATE TABLE fills (
                fill_id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL REFERENCES orders(order_id), asset TEXT NOT NULL,
                units_delta REAL NOT NULL, price REAL NOT NULL,
                gross_notional REAL NOT NULL, allocated_cost REAL NOT NULL
            );
            CREATE TABLE valuations (
                valuation_at TEXT PRIMARY KEY, observed_at TEXT NOT NULL, nav REAL NOT NULL,
                floor_value REAL NOT NULL, distance_to_floor REAL NOT NULL,
                distance_to_floor_fraction REAL NOT NULL, reconciliation_status TEXT NOT NULL,
                expected_value REAL NOT NULL, observed_value REAL NOT NULL, order_id TEXT
            );
            CREATE TABLE alerts (
                alert_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, valuation_at TEXT,
                severity TEXT NOT NULL, code TEXT NOT NULL, message TEXT NOT NULL
            );
            CREATE TABLE audit_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
                event_type TEXT NOT NULL, actor TEXT NOT NULL, entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL, payload_json TEXT NOT NULL,
                previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL UNIQUE
            );
            CREATE TRIGGER audit_events_no_update
            BEFORE UPDATE ON audit_events BEGIN
                SELECT RAISE(ABORT, 'audit events are immutable');
            END;
            CREATE TRIGGER audit_events_no_delete
            BEFORE DELETE ON audit_events BEGIN
                SELECT RAISE(ABORT, 'audit events are immutable');
            END;
            """
        )

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _metadata(connection: sqlite3.Connection, key: str) -> Any:
        row = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        if row is None:
            raise ValueError(f"paper portfolio metadata is missing {key}")
        return json.loads(row["value"])

    @staticmethod
    def _set_metadata(connection: sqlite3.Connection, key: str, value: object) -> None:
        connection.execute("UPDATE metadata SET value = ? WHERE key = ?", (_canonical(value), key))

    @staticmethod
    def _append_audit(
        connection: sqlite3.Connection,
        event_type: str,
        actor: str,
        entity_type: str,
        entity_id: str,
        payload: Mapping[str, object],
        created_at: object | None = None,
    ) -> str:
        if not actor.strip():
            raise ValueError("audit actor is required")
        last = connection.execute(
            "SELECT event_hash FROM audit_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        previous_hash = "0" * 64 if last is None else str(last["event_hash"])
        event = {
            "created_at": _iso(created_at),
            "event_type": event_type,
            "actor": actor,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "payload": dict(payload),
            "previous_hash": previous_hash,
        }
        event_hash = hashlib.sha256(_canonical(event).encode()).hexdigest()
        connection.execute(
            "INSERT INTO audit_events(created_at, event_type, actor, entity_type, entity_id, "
            "payload_json, previous_hash, event_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event["created_at"],
                event_type,
                actor,
                entity_type,
                entity_id,
                _canonical(payload),
                previous_hash,
                event_hash,
            ),
        )
        return event_hash

    def verify_audit_chain(self) -> bool:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM audit_events ORDER BY sequence").fetchall()
        previous_hash = "0" * 64
        for row in rows:
            if row["previous_hash"] != previous_hash:
                return False
            event = {
                "created_at": row["created_at"],
                "event_type": row["event_type"],
                "actor": row["actor"],
                "entity_type": row["entity_type"],
                "entity_id": row["entity_id"],
                "payload": json.loads(row["payload_json"]),
                "previous_hash": row["previous_hash"],
            }
            if hashlib.sha256(_canonical(event).encode()).hexdigest() != row["event_hash"]:
                return False
            previous_hash = row["event_hash"]
        return bool(rows)

    @staticmethod
    def _add_alert(
        connection: sqlite3.Connection,
        *,
        valuation_at: pd.Timestamp | None,
        severity: str,
        code: str,
        message: str,
        created_at: pd.Timestamp,
    ) -> dict[str, object]:
        alert_id = str(uuid4())
        record = {
            "alert_id": alert_id,
            "created_at": created_at.isoformat(),
            "valuation_at": None if valuation_at is None else valuation_at.isoformat(),
            "severity": severity,
            "code": code,
            "message": message,
        }
        connection.execute(
            "INSERT INTO alerts(alert_id, created_at, valuation_at, severity, code, message) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            tuple(record.values()),
        )
        PaperPortfolio._append_audit(
            connection,
            "alert_raised",
            "system",
            "alert",
            alert_id,
            {key: value for key, value in record.items() if key != "alert_id"},
            created_at,
        )
        return record

    @staticmethod
    def _marks(connection: sqlite3.Connection, valuation_at: str) -> dict[str, float]:
        rows = connection.execute(
            "SELECT asset, price FROM price_marks WHERE valuation_at = ?", (valuation_at,)
        ).fetchall()
        return {row["asset"]: float(row["price"]) for row in rows}

    @staticmethod
    def _positions(connection: sqlite3.Connection) -> dict[str, float]:
        return {
            row["asset"]: float(row["units"])
            for row in connection.execute("SELECT asset, units FROM positions")
        }

    @staticmethod
    def _active_order(connection: sqlite3.Connection) -> sqlite3.Row | None:
        return connection.execute(
            "SELECT * FROM orders WHERE status IN (?, ?) ORDER BY decision_at LIMIT 1",
            ACTIVE_ORDER_STATUSES,
        ).fetchone()

    @staticmethod
    def _update_sleeves_for_returns(
        connection: sqlite3.Connection,
        last_prices: Mapping[str, float],
        prices: Mapping[str, float],
        reserve_asset: str,
    ) -> None:
        reserve_return = float(prices[reserve_asset]) / float(last_prices[reserve_asset])
        for row in connection.execute("SELECT * FROM sleeves").fetchall():
            asset_return = float(prices[row["asset"]]) / float(last_prices[row["asset"]])
            connection.execute(
                "UPDATE sleeves SET risky_value = ?, reserve_value = ? WHERE asset = ?",
                (
                    float(row["risky_value"]) * asset_return,
                    float(row["reserve_value"]) * reserve_return,
                    row["asset"],
                ),
            )

    @staticmethod
    def _target(
        connection: sqlite3.Connection,
        weights: Mapping[str, float],
        policy: Mapping[str, object],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, object]]:
        factors: dict[str, float] = {}
        sleeves: dict[str, object] = {}
        for row in connection.execute("SELECT * FROM sleeves ORDER BY asset").fetchall():
            nav = float(row["risky_value"]) + float(row["reserve_value"])
            allocation = allocate_vppi(
                nav, float(row["floor_value"]), VppiPolicy(float(policy["multiplier"]))
            )
            factors[row["asset"]] = allocation.risky_weight
            sleeves[row["asset"]] = {
                "nav": nav,
                "floor": float(row["floor_value"]),
                "cushion": allocation.cushion,
                "exposure_factor": allocation.risky_weight,
            }
        target = apply_risk_overlays(weights, factors, str(policy["reserve_asset"]))
        return target, factors, sleeves

    @staticmethod
    def _execute_order(
        connection: sqlite3.Connection,
        order: sqlite3.Row,
        prices: Mapping[str, float],
        tradable: Mapping[str, bool],
        execution: ExecutionModel,
        valuation_at: pd.Timestamp,
        observed_at: pd.Timestamp,
    ) -> tuple[bool, Mapping[str, object] | None]:
        positions = PaperPortfolio._positions(connection)
        values = {asset: units * prices[asset] for asset, units in positions.items()}
        gross_nav = sum(values.values())
        target = json.loads(order["target_json"])
        proposed = {asset: target[asset] * gross_nav - values[asset] for asset in values}
        required = [
            asset for asset, value in proposed.items() if abs(value) > max(1e-8, gross_nav * 1e-12)
        ]
        closed = [asset for asset in required if not tradable[asset]]
        if closed:
            alert = PaperPortfolio._add_alert(
                connection,
                valuation_at=valuation_at,
                severity="warning",
                code="execution_deferred_market_closed",
                message=f"Order {order['order_id']} waits for: {', '.join(sorted(closed))}",
                created_at=observed_at,
            )
            PaperPortfolio._append_audit(
                connection,
                "order_execution_deferred",
                "system",
                "order",
                order["order_id"],
                {"closed_assets": sorted(closed), "valuation_at": valuation_at.isoformat()},
                observed_at,
            )
            return False, alert
        transaction_cost, slippage_cost, cost_by_asset = calculate_trade_costs(
            proposed, execution
        )
        cost = transaction_cost + slippage_cost
        net_nav = gross_nav - cost
        if net_nav <= 0:
            raise ValueError("execution costs exhaust paper NAV")
        actual_values = {asset: float(target[asset]) * net_nav for asset in values}
        actual_trades = {asset: actual_values[asset] - values[asset] for asset in values}
        for asset in sorted(values):
            new_units = actual_values[asset] / prices[asset]
            units_delta = new_units - positions[asset]
            allocated_cost = cost_by_asset[asset]
            connection.execute("UPDATE positions SET units = ? WHERE asset = ?", (new_units, asset))
            connection.execute(
                "INSERT INTO fills(order_id, asset, units_delta, price, gross_notional, "
                "allocated_cost) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    order["order_id"],
                    asset,
                    units_delta,
                    prices[asset],
                    abs(actual_trades[asset]),
                    allocated_cost,
                ),
            )
        factors = json.loads(order["factors_json"])
        sleeve_scale = net_nav / gross_nav
        for row in connection.execute("SELECT * FROM sleeves").fetchall():
            sleeve_nav = (float(row["risky_value"]) + float(row["reserve_value"])) * sleeve_scale
            factor = float(factors[row["asset"]])
            connection.execute(
                "UPDATE sleeves SET risky_value = ?, reserve_value = ? WHERE asset = ?",
                (sleeve_nav * factor, sleeve_nav * (1 - factor), row["asset"]),
            )
        connection.execute(
            "UPDATE orders SET status = ?, executed_at = ?, total_cost = ? WHERE order_id = ?",
            (OrderStatus.EXECUTED.value, valuation_at.isoformat(), cost, order["order_id"]),
        )
        PaperPortfolio._append_audit(
            connection,
            "paper_order_executed",
            "system",
            "order",
            order["order_id"],
            {
                "valuation_at": valuation_at.isoformat(),
                "gross_nav": gross_nav,
                "net_nav": net_nav,
                "cost": cost,
                "transaction_cost": transaction_cost,
                "slippage_cost": slippage_cost,
                "target": target,
            },
            observed_at,
        )
        return True, None

    def run_day(
        self,
        *,
        prices: Mapping[str, float],
        valuation_at: object,
        tradable: Mapping[str, bool] | None = None,
        observed_at: object | None = None,
        observed_units: Mapping[str, float] | None = None,
    ) -> PaperDayResult:
        if not self.verify_audit_chain():
            raise ValueError("audit chain verification failed")
        valuation = _utc_timestamp(valuation_at)
        observed = _utc_timestamp(observed_at)
        if observed < valuation:
            raise ValueError("observed_at cannot precede valuation_at")
        with self._connect() as connection:
            weights = self._metadata(connection, "strategic_weights")
            policy = self._metadata(connection, "policy")
            controls = ExecutionControls(**self._metadata(connection, "controls"))
            execution = ExecutionModel(**self._metadata(connection, "execution"))
            kill_switch = bool(self._metadata(connection, "kill_switch"))
            if set(prices) != set(weights) or any(float(value) <= 0 for value in prices.values()):
                raise ValueError("daily prices must be positive and match portfolio assets")
            daily_prices = {asset: float(prices[asset]) for asset in weights}
            daily_tradable = (
                {asset: True for asset in weights}
                if tradable is None
                else {asset: bool(tradable[asset]) for asset in weights}
            )
            if set(daily_tradable) != set(weights):
                raise ValueError("tradability must match portfolio assets")
            latest = connection.execute(
                "SELECT * FROM valuations ORDER BY valuation_at DESC LIMIT 1"
            ).fetchone()
            latest_timestamp = _utc_timestamp(latest["valuation_at"])
            if valuation < latest_timestamp:
                raise ValueError("paper valuations must be strictly chronological")
            if valuation == latest_timestamp:
                return self._day_result(connection, latest, idempotent=True)
            last_prices = self._marks(connection, latest["valuation_at"])
            for asset in weights:
                connection.execute(
                    "INSERT INTO price_marks(valuation_at, asset, price, tradable) VALUES (?, ?, ?, ?)",
                    (valuation.isoformat(), asset, daily_prices[asset], int(daily_tradable[asset])),
                )
            self._update_sleeves_for_returns(
                connection, last_prices, daily_prices, str(policy["reserve_asset"])
            )
            alerts: list[Mapping[str, object]] = []
            blocked = False
            data_age_hours = (observed - valuation).total_seconds() / 3600
            if data_age_hours > controls.max_data_age_hours:
                alerts.append(
                    self._add_alert(
                        connection,
                        valuation_at=valuation,
                        severity="critical",
                        code="stale_data",
                        message=(
                            f"Valuation is {data_age_hours:.1f} hours old; limit is "
                            f"{controls.max_data_age_hours:g} hours"
                        ),
                        created_at=observed,
                    )
                )
                blocked = True
            active = self._active_order(connection)
            if (
                active is not None
                and active["status"] == OrderStatus.PENDING_APPROVAL.value
                and observed > _utc_timestamp(active["expires_at"])
            ):
                connection.execute(
                    "UPDATE orders SET status = ? WHERE order_id = ?",
                    (OrderStatus.EXPIRED.value, active["order_id"]),
                )
                self._append_audit(
                    connection,
                    "paper_order_expired",
                    "system",
                    "order",
                    active["order_id"],
                    {"expired_at": observed.isoformat()},
                    observed,
                )
                alerts.append(
                    self._add_alert(
                        connection,
                        valuation_at=valuation,
                        severity="warning",
                        code="approval_expired",
                        message=f"Order {active['order_id']} expired without approval",
                        created_at=observed,
                    )
                )
                active = None
            positions = self._positions(connection)
            values = {asset: positions[asset] * daily_prices[asset] for asset in weights}
            reconciled_nav = sum(values.values())
            observed_positions = positions if observed_units is None else dict(observed_units)
            if set(observed_positions) != set(weights):
                raise ValueError("observed units must match portfolio assets")
            observed_value = sum(
                float(observed_positions[asset]) * daily_prices[asset] for asset in weights
            )
            asset_break = max(
                abs(float(observed_positions[asset]) - positions[asset]) * daily_prices[asset]
                for asset in weights
            )
            reconciliation_gap = abs(observed_value - reconciled_nav)
            reconciliation_status = (
                "matched"
                if max(asset_break, reconciliation_gap) <= controls.reconciliation_tolerance
                else "break"
            )
            if reconciliation_status == "break":
                alerts.append(
                    self._add_alert(
                        connection,
                        valuation_at=valuation,
                        severity="critical",
                        code="reconciliation_break",
                        message=(
                            f"Position reconciliation differs by up to "
                            f"{max(asset_break, reconciliation_gap):,.2f}"
                        ),
                        created_at=observed,
                    )
                )
                blocked = True
            self._append_audit(
                connection,
                "daily_reconciliation",
                "system",
                "portfolio",
                str(self._metadata(connection, "portfolio")),
                {
                    "valuation_at": valuation.isoformat(),
                    "status": reconciliation_status,
                    "expected_value": reconciled_nav,
                    "observed_value": observed_value,
                    "maximum_asset_break": asset_break,
                },
                observed,
            )
            floor_value = float(self._metadata(connection, "portfolio_floor"))
            pre_trade_distance = reconciled_nav - floor_value
            if reconciled_nav < floor_value:
                alerts.append(
                    self._add_alert(
                        connection,
                        valuation_at=valuation,
                        severity="critical",
                        code="floor_breach",
                        message=(
                            f"NAV is {abs(pre_trade_distance):,.2f} below the synthetic floor"
                        ),
                        created_at=observed,
                    )
                )
                blocked = True

            executed_order_id: str | None = None
            if active is not None and active["status"] == OrderStatus.APPROVED.value:
                elapsed_bars = int(
                    connection.execute(
                        "SELECT COUNT(DISTINCT valuation_at) FROM price_marks "
                        "WHERE valuation_at > ? AND valuation_at <= ?",
                        (active["decision_at"], valuation.isoformat()),
                    ).fetchone()[0]
                )
                timing_ready = (
                    valuation > _utc_timestamp(active["approved_at"])
                    and elapsed_bars >= execution.latency_bars
                )
                if not timing_ready:
                    self._append_audit(
                        connection,
                        "order_execution_deferred_timing",
                        "system",
                        "order",
                        active["order_id"],
                        {
                            "valuation_at": valuation.isoformat(),
                            "approved_at": active["approved_at"],
                            "elapsed_bars": elapsed_bars,
                            "required_bars": execution.latency_bars,
                        },
                        observed,
                    )
                elif kill_switch or blocked:
                    reason = "kill switch" if kill_switch else "a pre-trade control"
                    alerts.append(
                        self._add_alert(
                            connection,
                            valuation_at=valuation,
                            severity="critical",
                            code="execution_blocked",
                            message=f"Approved order {active['order_id']} blocked by {reason}",
                            created_at=observed,
                        )
                    )
                else:
                    executed, execution_alert = self._execute_order(
                        connection,
                        active,
                        daily_prices,
                        daily_tradable,
                        execution,
                        valuation,
                        observed,
                    )
                    if execution_alert is not None:
                        alerts.append(execution_alert)
                    if executed:
                        executed_order_id = active["order_id"]

            positions = self._positions(connection)
            values = {asset: positions[asset] * daily_prices[asset] for asset in weights}
            nav = sum(values.values())
            distance = nav - floor_value
            distance_fraction = distance / nav
            if nav < floor_value and reconciled_nav >= floor_value:
                alerts.append(
                    self._add_alert(
                        connection,
                        valuation_at=valuation,
                        severity="critical",
                        code="floor_breach",
                        message=f"NAV is {abs(distance):,.2f} below the synthetic floor",
                        created_at=observed,
                    )
                )
                blocked = True
            elif nav >= floor_value and nav <= floor_value * (
                1 + controls.floor_alert_buffer_fraction
            ):
                alerts.append(
                    self._add_alert(
                        connection,
                        valuation_at=valuation,
                        severity="warning",
                        code="floor_buffer",
                        message=f"NAV is only {distance:,.2f} above the synthetic floor",
                        created_at=observed,
                    )
                )
            new_order_id: str | None = None
            new_order_status: str | None = None
            previous_review_raw = self._metadata(connection, "last_review_at")
            due = _review_due(
                _utc_timestamp(previous_review_raw) if previous_review_raw else None,
                valuation,
                ReviewFrequency(str(policy["review_frequency"])),
            )
            if due:
                self._set_metadata(connection, "last_review_at", valuation.isoformat())
                target, factors, sleeve_evidence = self._target(connection, weights, policy)
                current = {asset: values[asset] / nav for asset in weights}
                if needs_rebalance(current, target, RebalancePolicy(float(policy["drift_band"]))):
                    active = self._active_order(connection)
                    proposed_turnover = (
                        sum(abs(target[asset] * nav - values[asset]) for asset in weights) / nav
                    )
                    rationale: dict[str, object] = {
                        "rule": "VPPI target crossed the configured absolute drift band",
                        "review_frequency": policy["review_frequency"],
                        "drift_band": policy["drift_band"],
                        "current_weights": current,
                        "target_weights": target,
                        "sleeves": sleeve_evidence,
                        "proposed_turnover": proposed_turnover,
                    }
                    if active is not None:
                        self._append_audit(
                            connection,
                            "decision_blocked_by_open_order",
                            "system",
                            "order",
                            active["order_id"],
                            {"valuation_at": valuation.isoformat()},
                            observed,
                        )
                    else:
                        payload = {
                            "portfolio": self._metadata(connection, "portfolio"),
                            "decision_at": valuation.isoformat(),
                            "target": target,
                        }
                        new_order_id = str(uuid5(NAMESPACE_URL, _canonical(payload)))
                        expires = observed + pd.Timedelta(hours=controls.approval_expiry_hours)
                        control_reason: str | None = None
                        if kill_switch:
                            control_reason = "kill_switch"
                        elif blocked:
                            control_reason = "pre_trade_control"
                        elif proposed_turnover > controls.max_daily_turnover + 1e-12:
                            control_reason = "turnover_limit"
                        new_order_status = (
                            OrderStatus.CONTROL_BLOCKED.value
                            if control_reason
                            else OrderStatus.PENDING_APPROVAL.value
                        )
                        rationale["control_reason"] = control_reason
                        connection.execute(
                            "INSERT INTO orders(order_id, decision_at, expires_at, status, "
                            "target_json, factors_json, rationale_json, proposed_turnover) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                new_order_id,
                                valuation.isoformat(),
                                expires.isoformat(),
                                new_order_status,
                                _canonical(target),
                                _canonical(factors),
                                _canonical(rationale),
                                proposed_turnover,
                            ),
                        )
                        self._append_audit(
                            connection,
                            "paper_order_control_blocked"
                            if control_reason
                            else "paper_order_proposed",
                            "system",
                            "order",
                            new_order_id,
                            {
                                "status": new_order_status,
                                "expires_at": expires.isoformat(),
                                "rationale": rationale,
                            },
                            observed,
                        )
                        if control_reason:
                            message = (
                                f"Rebalance {new_order_id} blocked; proposed turnover "
                                f"is {proposed_turnover:.2%}"
                                if control_reason == "turnover_limit"
                                else f"Rebalance {new_order_id} blocked by {control_reason}"
                            )
                            alerts.append(
                                self._add_alert(
                                    connection,
                                    valuation_at=valuation,
                                    severity="critical",
                                    code=control_reason,
                                    message=message,
                                    created_at=observed,
                                )
                            )
            display_order_id = new_order_id or executed_order_id
            connection.execute(
                "INSERT INTO valuations(valuation_at, observed_at, nav, floor_value, "
                "distance_to_floor, distance_to_floor_fraction, reconciliation_status, "
                "expected_value, observed_value, order_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    valuation.isoformat(),
                    observed.isoformat(),
                    nav,
                    floor_value,
                    distance,
                    distance_fraction,
                    reconciliation_status,
                    reconciled_nav,
                    observed_value,
                    display_order_id,
                ),
            )
            self._append_audit(
                connection,
                "paper_valuation_completed",
                "system",
                "portfolio",
                str(self._metadata(connection, "portfolio")),
                {
                    "valuation_at": valuation.isoformat(),
                    "nav": nav,
                    "floor": floor_value,
                    "distance_to_floor": distance,
                    "reconciliation_status": reconciliation_status,
                    "order_id": display_order_id,
                },
                observed,
            )
            connection.commit()
            row = connection.execute(
                "SELECT * FROM valuations WHERE valuation_at = ?", (valuation.isoformat(),)
            ).fetchone()
            result = self._day_result(connection, row, alerts=alerts)
        self.export_dashboard()
        return result

    @staticmethod
    def _day_result(
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        alerts: list[Mapping[str, object]] | None = None,
        idempotent: bool = False,
    ) -> PaperDayResult:
        order_id = row["order_id"]
        order_status = None
        if order_id:
            order = connection.execute(
                "SELECT status FROM orders WHERE order_id = ?", (order_id,)
            ).fetchone()
            order_status = None if order is None else order["status"]
        return PaperDayResult(
            valuation_at=_utc_timestamp(row["valuation_at"]),
            nav=float(row["nav"]),
            floor_value=float(row["floor_value"]),
            distance_to_floor=float(row["distance_to_floor"]),
            reconciliation_status=row["reconciliation_status"],
            order_id=order_id,
            order_status=order_status,
            alerts=tuple(alerts or ()),
            idempotent=idempotent,
        )

    def approve_order(
        self, order_id: str, *, approved_by: str, approved_at: object | None = None
    ) -> None:
        if not approved_by.strip():
            raise ValueError("approved_by is required")
        timestamp = _utc_timestamp(approved_at)
        if not self.verify_audit_chain():
            raise ValueError("audit chain verification failed")
        expired = False
        with self._connect() as connection:
            if self._metadata(connection, "kill_switch"):
                raise ValueError("kill switch is active")
            order = connection.execute(
                "SELECT * FROM orders WHERE order_id = ?", (order_id,)
            ).fetchone()
            if order is None:
                raise ValueError("unknown order")
            if order["status"] != OrderStatus.PENDING_APPROVAL.value:
                raise ValueError(f"order is not pending approval: {order['status']}")
            if timestamp > _utc_timestamp(order["expires_at"]):
                connection.execute(
                    "UPDATE orders SET status = ? WHERE order_id = ?",
                    (OrderStatus.EXPIRED.value, order_id),
                )
                self._append_audit(
                    connection,
                    "paper_order_expired",
                    approved_by,
                    "order",
                    order_id,
                    {"approval_attempt_at": timestamp.isoformat()},
                    timestamp,
                )
                connection.commit()
                expired = True
            else:
                connection.execute(
                    "UPDATE orders SET status = ?, approved_by = ?, approved_at = ? "
                    "WHERE order_id = ?",
                    (OrderStatus.APPROVED.value, approved_by, timestamp.isoformat(), order_id),
                )
                self._append_audit(
                    connection,
                    "paper_order_approved",
                    approved_by,
                    "order",
                    order_id,
                    {
                        "approved_at": timestamp.isoformat(),
                        "target": json.loads(order["target_json"]),
                    },
                    timestamp,
                )
                connection.commit()
        self.export_dashboard()
        if expired:
            raise ValueError("order approval window has expired")

    def reject_order(
        self,
        order_id: str,
        *,
        rejected_by: str,
        reason: str,
        rejected_at: object | None = None,
    ) -> None:
        if not rejected_by.strip() or not reason.strip():
            raise ValueError("rejected_by and reason are required")
        timestamp = _utc_timestamp(rejected_at)
        if not self.verify_audit_chain():
            raise ValueError("audit chain verification failed")
        with self._connect() as connection:
            order = connection.execute(
                "SELECT * FROM orders WHERE order_id = ?", (order_id,)
            ).fetchone()
            if order is None:
                raise ValueError("unknown order")
            if order["status"] not in ACTIVE_ORDER_STATUSES:
                raise ValueError(f"order cannot be rejected: {order['status']}")
            connection.execute(
                "UPDATE orders SET status = ?, rejected_by = ?, rejected_at = ?, "
                "rejection_reason = ? WHERE order_id = ?",
                (OrderStatus.REJECTED.value, rejected_by, timestamp.isoformat(), reason, order_id),
            )
            self._append_audit(
                connection,
                "paper_order_rejected",
                rejected_by,
                "order",
                order_id,
                {"reason": reason},
                timestamp,
            )
            connection.commit()
        self.export_dashboard()

    def set_kill_switch(
        self, active: bool, *, actor: str, reason: str, changed_at: object | None = None
    ) -> None:
        if not actor.strip() or not reason.strip():
            raise ValueError("actor and reason are required")
        timestamp = _utc_timestamp(changed_at)
        if not self.verify_audit_chain():
            raise ValueError("audit chain verification failed")
        with self._connect() as connection:
            current = bool(self._metadata(connection, "kill_switch"))
            if current == active:
                return
            canceled: list[str] = []
            if active:
                open_orders = connection.execute(
                    "SELECT order_id FROM orders WHERE status IN (?, ?)", ACTIVE_ORDER_STATUSES
                ).fetchall()
                canceled = [row["order_id"] for row in open_orders]
                connection.execute(
                    "UPDATE orders SET status = ?, rejection_reason = ? WHERE status IN (?, ?)",
                    (OrderStatus.CANCELED.value, f"kill switch: {reason}", *ACTIVE_ORDER_STATUSES),
                )
            self._set_metadata(connection, "kill_switch", active)
            self._append_audit(
                connection,
                "kill_switch_activated" if active else "kill_switch_deactivated",
                actor,
                "portfolio",
                str(self._metadata(connection, "portfolio")),
                {"reason": reason, "canceled_orders": canceled},
                timestamp,
            )
            connection.commit()
        self.export_dashboard()

    def state(self) -> dict[str, object]:
        with self._connect() as connection:
            latest = connection.execute(
                "SELECT * FROM valuations ORDER BY valuation_at DESC LIMIT 1"
            ).fetchone()
            latest_prices = self._marks(connection, latest["valuation_at"])
            positions = self._positions(connection)
            values = {asset: positions[asset] * latest_prices[asset] for asset in positions}
            nav = sum(values.values())
            orders = [
                {
                    **dict(row),
                    "target": json.loads(row["target_json"]),
                    "rationale": json.loads(row["rationale_json"]),
                }
                for row in connection.execute(
                    "SELECT * FROM orders ORDER BY decision_at DESC LIMIT 20"
                )
            ]
            for order in orders:
                order.pop("target_json")
                order.pop("factors_json")
                order.pop("rationale_json")
            alerts = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM alerts ORDER BY created_at DESC LIMIT 20"
                )
            ]
            audit_head = connection.execute(
                "SELECT sequence, event_hash FROM audit_events ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            return {
                "portfolio": self._metadata(connection, "portfolio"),
                "selection_id": self._metadata(connection, "selection_id"),
                "selection_validation_status": self._metadata(
                    connection, "selection_validation_status"
                ),
                "policy": self._metadata(connection, "policy"),
                "controls": self._metadata(connection, "controls"),
                "kill_switch": bool(self._metadata(connection, "kill_switch")),
                "valuation": dict(latest),
                "positions": {
                    asset: {
                        "units": positions[asset],
                        "price": latest_prices[asset],
                        "value": values[asset],
                        "weight": values[asset] / nav,
                    }
                    for asset in sorted(positions)
                },
                "orders": orders,
                "alerts": alerts,
                "audit": {
                    "valid": self.verify_audit_chain(),
                    "events": int(audit_head["sequence"]),
                    "head": audit_head["event_hash"],
                },
            }

    def export_dashboard(self, path: str | Path | None = None) -> Path:
        state = self.state()
        target = Path(path) if path is not None else self.path.with_name("dashboard.html")
        target.parent.mkdir(parents=True, exist_ok=True)
        valuation = state["valuation"]
        nav = float(valuation["nav"])
        floor = float(valuation["floor_value"])
        distance = float(valuation["distance_to_floor"])
        status_class = "danger" if state["kill_switch"] or distance < 0 else "good"
        active_orders = [
            order for order in state["orders"] if order["status"] in ACTIVE_ORDER_STATUSES
        ]
        latest_order = state["orders"][0] if state["orders"] else None
        rationale = latest_order["rationale"] if latest_order else {}
        rows = "".join(
            "<tr>"
            f"<td>{html.escape(asset.replace('_', ' ').title())}</td>"
            f"<td>{values['weight']:.1%}</td><td>{values['value']:,.2f}</td>"
            f"<td>{values['units']:,.6f}</td></tr>"
            for asset, values in state["positions"].items()
        )
        alerts = (
            "".join(
                f"<li class='{html.escape(str(alert['severity']))}'>"
                f"<strong>{html.escape(str(alert['code']).replace('_', ' ').title())}</strong>"
                f"<span>{html.escape(str(alert['message']))}</span></li>"
                for alert in state["alerts"][:8]
            )
            or "<li class='quiet'>No alerts recorded.</li>"
        )
        target_weights = rationale.get("target_weights", {})
        current_weights = rationale.get("current_weights", {})
        rationale_rows = (
            "".join(
                "<tr>"
                f"<td>{html.escape(str(asset).replace('_', ' ').title())}</td>"
                f"<td>{float(current_weights.get(asset, 0)):.1%}</td>"
                f"<td>{float(weight):.1%}</td>"
                f"<td>{float(weight) - float(current_weights.get(asset, 0)):+.1%}</td></tr>"
                for asset, weight in target_weights.items()
            )
            or "<tr><td colspan='4'>No rebalance rationale is currently available.</td></tr>"
        )
        order_status = (
            f"{len(active_orders)} awaiting action"
            if active_orders
            else (latest_order["status"].replace("_", " ") if latest_order else "none")
        )
        document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Controlled Execution — {html.escape(str(state["portfolio"]))}</title><style>
:root{{--ink:#17211d;--paper:#f5f2e9;--panel:#fffdf7;--line:#d9d2c2;--green:#12634b;--red:#a43b2b;--amber:#9a6414;--muted:#657068}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);background:var(--paper);font:15px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:auto;padding:34px 22px 56px}}header{{display:flex;justify-content:space-between;gap:24px;align-items:end;border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:20px}}
h1{{font:600 32px/1.05 Georgia,serif;margin:4px 0}}h2{{font:600 19px Georgia,serif;margin:0 0 14px}}.eyebrow{{color:var(--green);text-transform:uppercase;letter-spacing:.12em;font-size:11px;font-weight:700}}.stamp{{color:var(--muted);text-align:right}}
.grid{{display:grid;grid-template-columns:repeat(12,1fr);gap:16px}}.card{{grid-column:span 3;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px}}.wide{{grid-column:span 7}}.side{{grid-column:span 5}}.label{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.07em}}
.value{{font:600 29px Georgia,serif;margin-top:7px}}.sub{{color:var(--muted);margin-top:5px}}.pill{{display:inline-block;border-radius:999px;padding:6px 10px;font-size:12px;font-weight:700;background:#e5eee9;color:var(--green)}}.pill.danger{{background:#f4dfd9;color:var(--red)}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:10px 8px;text-align:right;border-bottom:1px solid #ebe5d8}}th:first-child,td:first-child{{text-align:left}}th{{color:var(--muted);font-size:11px;text-transform:uppercase}}ul{{list-style:none;padding:0;margin:0}}li{{display:flex;flex-direction:column;gap:2px;padding:10px 12px;border-left:3px solid var(--amber);background:#fbf3df;margin:8px 0}}li.critical{{border-color:var(--red);background:#f9e8e3}}li.quiet{{border-color:var(--line);background:transparent;color:var(--muted)}}.audit{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px;overflow-wrap:anywhere}}
@media(max-width:800px){{.card,.wide,.side{{grid-column:1/-1}}header{{align-items:start;flex-direction:column}}.stamp{{text-align:left}}}}</style></head><body><main>
<header><div><div class="eyebrow">Stage 5 · Paper portfolio</div><h1>{html.escape(str(state["portfolio"]).replace("_", " ").title())}</h1><div>Policy {html.escape(str(state["selection_id"]))}</div></div>
<div class="stamp"><span class="pill {status_class}">{"KILL SWITCH ACTIVE" if state["kill_switch"] else "CONTROLS ACTIVE"}</span><div>Valued {html.escape(str(valuation["valuation_at"]))}</div></div></header><section class="grid">
<article class="card"><div class="label">Paper NAV</div><div class="value">{nav:,.2f}</div><div class="sub">Reconciliation: {html.escape(str(valuation["reconciliation_status"]))}</div></article>
<article class="card"><div class="label">Synthetic floor</div><div class="value">{floor:,.2f}</div><div class="sub">Static initial-release floor</div></article>
<article class="card"><div class="label">Distance to floor</div><div class="value">{distance:,.2f}</div><div class="sub">{distance / nav:.2%} of current NAV</div></article>
<article class="card"><div class="label">Order status</div><div class="value" style="font-size:22px">{html.escape(str(order_status).title())}</div><div class="sub">Human approval required</div></article>
<article class="card wide"><h2>Current allocation</h2><table><thead><tr><th>Asset</th><th>Weight</th><th>Value</th><th>Paper units</th></tr></thead><tbody>{rows}</tbody></table></article>
<article class="card side"><h2>Alerts</h2><ul>{alerts}</ul></article>
<article class="card wide"><h2>Latest order rationale</h2><div class="sub">{html.escape(str(rationale.get("rule", "No order proposed.")))}</div><table><thead><tr><th>Asset</th><th>Current</th><th>Target</th><th>Change</th></tr></thead><tbody>{rationale_rows}</tbody></table></article>
<article class="card side"><h2>Control record</h2><p><span class="label">Approval window</span><br>{float(state["controls"]["approval_expiry_hours"]):g} hours</p><p><span class="label">Daily turnover ceiling</span><br>{float(state["controls"]["max_daily_turnover"]):.1%}</p><p><span class="label">Audit chain</span><br>{"Verified" if state["audit"]["valid"] else "FAILED"} · {state["audit"]["events"]} events</p><div class="audit">Head {html.escape(str(state["audit"]["head"]))}</div></article>
</section></main></body></html>"""
        target.write_text(document, encoding="utf-8")
        target.with_name("state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target
