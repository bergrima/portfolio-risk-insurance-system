from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .backtest import ExecutionModel, ReviewFrequency
from .config import load_project_config
from .cost_config import load_asset_transaction_costs
from .data_contract import load_data_contract
from .data_quality import validate_market_data
from .data_sources import (
    fetch_market_data,
    load_snapshot,
    point_in_time_prices,
    point_in_time_tradability,
    save_snapshot,
)
from .execution import (
    PaperPortfolio,
    load_execution_controls,
    with_control_overrides,
)
from .experiments import export_experiment, run_parameter_grid
from .policy_selection import (
    export_policy_selection,
    load_selection_criteria,
    run_policy_selection,
    with_criteria_overrides,
)
from .regime import (
    CalibrationSettings,
    calibrate_regimes,
    export_regime_calibration,
    load_regime_calibration,
)
from .scenarios import (
    HistoricalStressWindow,
    SimulationConfig,
    SimulationMethod,
    constant_inflation_index,
    export_scenario_analysis,
    run_scenario_analysis,
    standard_stress_scenarios,
)


def _project_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[2].joinpath(*parts)


def _floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _frequencies(value: str) -> list[ReviewFrequency]:
    return [ReviewFrequency(item.strip()) for item in value.split(",") if item.strip()]


def _historical_window(value: str) -> HistoricalStressWindow:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("use NAME:YYYY-MM-DD:YYYY-MM-DD")
    try:
        return HistoricalStressWindow(parts[0], parts[1], parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _add_shared_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=_project_path("configs", "portfolios.toml"))
    parser.add_argument(
        "--contract", type=Path, default=_project_path("configs", "data_contract.toml")
    )


def _add_regime_calibration(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--regime-calibration",
        type=Path,
        help="pre-evaluation ZigZag calibration directory or calibration.json",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Portfolio risk insurance research CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    show = subparsers.add_parser("show-portfolios", help="show configured strategic weights")
    _add_shared_paths(show)

    fetch = subparsers.add_parser("fetch-data", help="fetch TSETMC and Wallex history")
    _add_shared_paths(fetch)
    fetch.add_argument("--start", required=True)
    fetch.add_argument("--end", required=True)
    fetch.add_argument("--output", type=Path)
    fetch.add_argument("--allow-market-closures", action="store_true")

    validate = subparsers.add_parser(
        "validate-data", help="validate a saved point-in-time snapshot"
    )
    _add_shared_paths(validate)
    validate.add_argument("--snapshot", type=Path, required=True)
    validate.add_argument("--portfolio", choices=["medium_term", "long_term"])
    validate.add_argument("--allow-market-closures", action="store_true")

    calibrate = subparsers.add_parser(
        "calibrate-regime",
        help="derive causal drawdown/ZigZag protection thresholds before evaluation",
    )
    _add_shared_paths(calibrate)
    calibrate.add_argument("--snapshot", type=Path, required=True)
    calibrate.add_argument(
        "--portfolio", choices=["medium_term", "long_term"], required=True
    )
    calibrate.add_argument(
        "--output", type=Path, default=_project_path("runs", "regime_calibration")
    )
    calibrate.add_argument(
        "--evaluation-start",
        help="exclusive training boundary; defaults to common portfolio inception",
    )
    calibrate.add_argument("--reversal-threshold", type=float, default=0.05)
    calibrate.add_argument("--continuation-step", type=float, default=0.02)
    calibrate.add_argument("--grid-step", type=float, default=0.01)
    calibrate.add_argument("--minimum-samples", type=int, default=8)
    calibrate.add_argument("--target-probability", type=float, default=0.60)
    calibrate.add_argument("--allow-market-closures", action="store_true")

    run = subparsers.add_parser("run-backtest", help="run the Stage 2 parameter surface")
    _add_shared_paths(run)
    run.add_argument("--snapshot", type=Path, required=True)
    run.add_argument("--portfolio", choices=["medium_term", "long_term"], required=True)
    run.add_argument("--output", type=Path, default=_project_path("runs"))
    run.add_argument("--multipliers", default="3")
    run.add_argument("--floors", default="0.8")
    run.add_argument("--frequencies", default="monthly")
    run.add_argument("--drift-bands", default="0.05")
    run.add_argument(
        "--protection-drift-band",
        type=float,
        help=(
            "separate VPPI protection band used on regime changes and while protected; "
            "defaults to each ordinary drift band"
        ),
    )
    run.add_argument(
        "--transaction-cost-bps",
        type=float,
        help="uniform commission override; defaults to the per-asset cost config",
    )
    run.add_argument(
        "--transaction-costs-config",
        type=Path,
        default=_project_path("configs", "transaction_costs.toml"),
    )
    run.add_argument("--slippage-bps", type=float, default=0)
    run.add_argument("--latency-bars", type=int, default=1)
    run.add_argument("--initial-nav", type=float, default=1_000_000)
    run.add_argument("--include-usd-relative", action="store_true")
    run.add_argument("--allow-market-closures", action="store_true")
    _add_regime_calibration(run)

    scenarios = subparsers.add_parser(
        "run-scenarios", help="run the Stage 3 stress and Monte Carlo surface"
    )
    _add_shared_paths(scenarios)
    scenarios.add_argument("--snapshot", type=Path, required=True)
    scenarios.add_argument("--portfolio", choices=["medium_term", "long_term"], required=True)
    scenarios.add_argument("--output", type=Path, default=_project_path("runs", "stage3"))
    scenarios.add_argument("--multipliers", default="3")
    scenarios.add_argument("--floors", default="0.8")
    scenarios.add_argument("--frequencies", default="monthly")
    scenarios.add_argument("--drift-bands", default="0.05")
    scenarios.add_argument(
        "--transaction-cost-bps",
        type=float,
        help="uniform commission override; defaults to the per-asset cost config",
    )
    scenarios.add_argument(
        "--transaction-costs-config",
        type=Path,
        default=_project_path("configs", "transaction_costs.toml"),
    )
    scenarios.add_argument("--slippage-bps", type=float, default=0)
    scenarios.add_argument("--latency-bars", type=int, default=1)
    scenarios.add_argument("--initial-nav", type=float, default=1_000_000)
    scenarios.add_argument(
        "--method",
        choices=[method.value for method in SimulationMethod],
        default=SimulationMethod.MOVING_BLOCK_BOOTSTRAP.value,
    )
    scenarios.add_argument("--paths", type=int, default=1_000)
    scenarios.add_argument("--workers", type=int, default=1, help="parallel path workers")
    scenarios.add_argument(
        "--horizon-bars",
        type=int,
        help="valuation bars per path; defaults to 21 per portfolio horizon month",
    )
    scenarios.add_argument("--block-size", type=int, default=10)
    scenarios.add_argument("--seed", type=int, default=20260824)
    scenarios.add_argument("--student-t-df", type=float, default=5)
    scenarios.add_argument("--confidence", type=float, default=0.90)
    scenarios.add_argument(
        "--tail-confidence",
        type=float,
        default=0.95,
        help="tail level for drawdown and floor-shortfall quantiles and expected shortfall",
    )
    scenarios.add_argument(
        "--annual-inflation-rate",
        type=float,
        help="decimal annual assumption used when the snapshot has no CPI series",
    )
    scenarios.add_argument(
        "--inflation-proxy",
        choices=["cpi", "usd_irr"],
        default="usd_irr",
        help="benchmark used for real-return measurement; mandate default is USD/IRR",
    )
    scenarios.add_argument("--allow-market-closures", action="store_true")
    _add_regime_calibration(scenarios)
    scenarios.add_argument("--standard-stresses", action="store_true")
    scenarios.add_argument(
        "--historical-window",
        type=_historical_window,
        action="append",
        default=[],
        metavar="NAME:START:END",
    )

    selection = subparsers.add_parser(
        "select-policy", help="run Stage 4 Pareto selection and walk-forward validation"
    )
    _add_shared_paths(selection)
    selection.add_argument(
        "--selection-config",
        type=Path,
        default=_project_path("configs", "policy_selection.toml"),
    )
    selection.add_argument(
        "--analysis",
        type=Path,
        required=True,
        help="Stage 3 export directory containing path_metrics.csv and metadata.json",
    )
    selection.add_argument("--snapshot", type=Path, required=True)
    selection.add_argument("--portfolio", choices=["medium_term", "long_term"], required=True)
    selection.add_argument("--output", type=Path, default=_project_path("runs", "stage4"))
    selection.add_argument("--folds", type=int, default=3)
    selection.add_argument(
        "--test-bars",
        type=int,
        help="returns per unseen fold; defaults to 21 per portfolio horizon month",
    )
    selection.add_argument("--minimum-train-bars", type=int)
    selection.add_argument(
        "--annual-inflation-rate",
        type=float,
        help="decimal annual assumption used when the snapshot has no CPI series",
    )
    selection.add_argument(
        "--inflation-proxy",
        choices=["cpi", "usd_irr"],
        default="usd_irr",
        help="must match Stage 3; the configured mandate requires USD/IRR",
    )
    selection.add_argument("--allow-market-closures", action="store_true")
    selection.add_argument("--minimum-real-return", type=float)
    selection.add_argument(
        "--maximum-drawdown",
        type=float,
        help="positive maximum acceptable loss fraction",
    )
    selection.add_argument("--maximum-breach-probability", type=float)
    selection.add_argument("--maximum-cost-rate", type=float)
    _add_regime_calibration(selection)

    paper_init = subparsers.add_parser(
        "paper-init", help="initialize an explicitly approved Stage 5 paper portfolio"
    )
    _add_shared_paths(paper_init)
    paper_init.add_argument(
        "--controls-config",
        type=Path,
        default=_project_path("configs", "execution_controls.toml"),
    )
    paper_init.add_argument(
        "--selection",
        type=Path,
        required=True,
        help="Stage 4 export containing selected_policy.json and metadata.json",
    )
    paper_init.add_argument("--snapshot", type=Path, required=True)
    paper_init.add_argument("--portfolio", choices=["medium_term", "long_term"], required=True)
    paper_init.add_argument("--state", type=Path)
    paper_init.add_argument("--approved-by", required=True)
    paper_init.add_argument(
        "--approve-policy",
        action="store_true",
        help="explicitly acknowledge the selected policy and operational controls",
    )
    paper_init.add_argument("--as-of", help="opening valuation timestamp; defaults to latest")
    paper_init.add_argument("--initial-nav", type=float)
    paper_init.add_argument("--max-daily-turnover", type=float)
    paper_init.add_argument("--max-data-age-hours", type=float)
    paper_init.add_argument("--floor-alert-buffer", type=float)
    paper_init.add_argument("--reconciliation-tolerance", type=float)
    paper_init.add_argument("--approval-expiry-hours", type=float)
    paper_init.add_argument("--allow-market-closures", action="store_true")

    paper_run = subparsers.add_parser(
        "paper-run", help="reconcile and evaluate one Stage 5 paper valuation"
    )
    _add_shared_paths(paper_run)
    paper_run.add_argument("--state", type=Path, required=True)
    paper_run.add_argument("--snapshot", type=Path, required=True)
    paper_run.add_argument("--as-of", help="valuation timestamp; defaults to latest")
    paper_run.add_argument("--observed-at", help="when the valuation was observed")
    paper_run.add_argument(
        "--positions",
        type=Path,
        help="optional reconciliation CSV with asset and units columns",
    )
    paper_run.add_argument("--allow-market-closures", action="store_true")

    paper_approve = subparsers.add_parser("paper-approve", help="approve one pending paper order")
    paper_approve.add_argument("--state", type=Path, required=True)
    paper_approve.add_argument("--order", required=True)
    paper_approve.add_argument("--approved-by", required=True)
    paper_approve.add_argument("--approved-at")

    paper_reject = subparsers.add_parser(
        "paper-reject", help="reject one pending or approved paper order"
    )
    paper_reject.add_argument("--state", type=Path, required=True)
    paper_reject.add_argument("--order", required=True)
    paper_reject.add_argument("--rejected-by", required=True)
    paper_reject.add_argument("--reason", required=True)
    paper_reject.add_argument("--rejected-at")

    paper_kill = subparsers.add_parser(
        "paper-kill-switch", help="activate or deactivate the Stage 5 kill switch"
    )
    paper_kill.add_argument("--state", type=Path, required=True)
    switch = paper_kill.add_mutually_exclusive_group(required=True)
    switch.add_argument("--activate", action="store_true")
    switch.add_argument("--deactivate", action="store_true")
    paper_kill.add_argument("--actor", required=True)
    paper_kill.add_argument("--reason", required=True)
    paper_kill.add_argument("--changed-at")

    paper_status = subparsers.add_parser(
        "paper-status", help="show the current Stage 5 paper state"
    )
    paper_status.add_argument("--state", type=Path, required=True)
    return parser


def _quality_report(snapshot_path: Path, contract_path: Path, series_keys: set[str] | None = None):
    snapshot = load_snapshot(snapshot_path)
    contract = load_data_contract(contract_path)
    effective_keys = (
        series_keys
        if series_keys is not None
        else set(snapshot.data["series"]).intersection(contract.series)
    )
    if not effective_keys:
        raise ValueError("snapshot contains no series from the active data contract")
    report = validate_market_data(
        snapshot.data,
        contract,
        snapshot.manifest.start,
        snapshot.manifest.end,
        series_keys=effective_keys,
        series_start_dates=snapshot.manifest.series_start_dates,
    )
    return snapshot, contract, report


def _snapshot_level_series(
    data: pd.DataFrame, key: str, index: pd.DatetimeIndex
) -> pd.Series | None:
    selected = data[data["series"] == key].copy()
    if selected.empty:
        return None
    selected["valuation_at"] = pd.to_datetime(selected["valuation_at"], utc=True)
    selected["available_at"] = pd.to_datetime(selected["available_at"], utc=True)
    if (selected["available_at"] > selected["valuation_at"]).any():
        raise ValueError(f"cannot construct {key} from unavailable observations")
    values = (
        selected.sort_values("available_at")
        .drop_duplicates("valuation_at", keep="last")
        .set_index("valuation_at")["value"]
        .astype(float)
        .sort_index()
    )
    return values.reindex(index).ffill().rename(key)


def _series_keys_for_assets(contract, assets: list[str], include_usd: bool) -> set[str]:
    selected = {
        key
        for key, definition in contract.series.items()
        if definition.investable and definition.asset in assets
    }
    if include_usd:
        selected.add("usd_irr")
    return selected


def _execution_model_from_args(args: argparse.Namespace) -> ExecutionModel:
    uniform = args.transaction_cost_bps
    return ExecutionModel(
        transaction_cost_bps=0.0 if uniform is None else float(uniform),
        slippage_bps=args.slippage_bps,
        latency_bars=args.latency_bars,
        asset_costs=(
            load_asset_transaction_costs(args.transaction_costs_config)
            if uniform is None
            else ()
        ),
    )


def _inflation_benchmark(args, snapshot, prices: pd.DataFrame) -> tuple[pd.Series, str]:
    if args.annual_inflation_rate is not None:
        return (
            constant_inflation_index(prices.index, args.annual_inflation_rate),
            f"annual_assumption:{args.annual_inflation_rate:g}",
        )
    key = args.inflation_proxy
    benchmark = _snapshot_level_series(snapshot.data, key, prices.index)
    if benchmark is None:
        raise ValueError(
            f"the snapshot has no {key} series; provide --annual-inflation-rate or "
            "select an available --inflation-proxy"
        )
    return benchmark, key


def _valuation_row(
    prices: pd.DataFrame,
    tradable: pd.DataFrame,
    as_of: str | None,
) -> tuple[pd.Timestamp, dict[str, float], dict[str, bool]]:
    if as_of is None:
        timestamp = prices.index[-1]
    else:
        requested = pd.Timestamp(as_of)
        if requested.tzinfo is None:
            requested = requested.tz_localize("UTC")
        else:
            requested = requested.tz_convert("UTC")
        eligible = prices.index[prices.index <= requested]
        if eligible.empty:
            raise ValueError("the snapshot has no valuation at or before --as-of")
        timestamp = eligible[-1]
    return (
        timestamp,
        prices.loc[timestamp].astype(float).to_dict(),
        tradable.loc[timestamp].astype(bool).to_dict(),
    )


def _load_observed_units(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    frame = pd.read_csv(path)
    if set(frame.columns) != {"asset", "units"} or frame["asset"].duplicated().any():
        raise ValueError("positions CSV must have unique asset and units columns")
    return dict(zip(frame["asset"], frame["units"].astype(float), strict=True))


def _paper_data(
    snapshot_path: Path,
    contract_path: Path,
    config_path: Path,
    portfolio_key: str,
    allow_market_closures: bool,
):
    config = load_project_config(config_path)
    portfolio = config.portfolios[portfolio_key]
    contract = load_data_contract(contract_path)
    required = _series_keys_for_assets(contract, list(portfolio.weights), include_usd=False)
    snapshot, contract, report = _quality_report(snapshot_path, contract_path, required)
    allowed = {"stale"} if allow_market_closures else set()
    report.raise_for_errors(allowed)
    prices = point_in_time_prices(snapshot.data, contract, list(portfolio.weights))
    tradable = point_in_time_tradability(
        snapshot.data, contract, list(portfolio.weights), prices.index
    )
    return config, portfolio, prices, tradable


def _print_paper_status(account: PaperPortfolio) -> None:
    state = account.state()
    valuation = state["valuation"]
    active = [
        order for order in state["orders"] if order["status"] in {"pending_approval", "approved"}
    ]
    print(
        f"{state['portfolio']}: NAV={valuation['nav']:,.2f}, "
        f"floor={valuation['floor_value']:,.2f}, "
        f"distance={valuation['distance_to_floor']:,.2f}, "
        f"reconciliation={valuation['reconciliation_status']}"
    )
    print(
        f"kill_switch={'active' if state['kill_switch'] else 'inactive'}, "
        f"open_orders={len(active)}, alerts={len(state['alerts'])}, "
        f"audit={'verified' if state['audit']['valid'] else 'FAILED'}"
    )
    for order in active:
        print(
            f"order {order['order_id']}: {order['status']}, "
            f"turnover={order['proposed_turnover']:.2%}, expires={order['expires_at']}"
        )
    print(f"dashboard: {account.path.with_name('dashboard.html')}")


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "paper-status":
        _print_paper_status(PaperPortfolio(args.state))
        return

    if args.command == "paper-approve":
        account = PaperPortfolio(args.state)
        account.approve_order(
            args.order, approved_by=args.approved_by, approved_at=args.approved_at
        )
        print(f"approved paper order {args.order}; execution remains deferred to a later valuation")
        _print_paper_status(account)
        return

    if args.command == "paper-reject":
        account = PaperPortfolio(args.state)
        account.reject_order(
            args.order,
            rejected_by=args.rejected_by,
            reason=args.reason,
            rejected_at=args.rejected_at,
        )
        print(f"rejected paper order {args.order}")
        _print_paper_status(account)
        return

    if args.command == "paper-kill-switch":
        account = PaperPortfolio(args.state)
        account.set_kill_switch(
            args.activate,
            actor=args.actor,
            reason=args.reason,
            changed_at=args.changed_at,
        )
        print(f"kill switch {'activated' if args.activate else 'deactivated'}")
        _print_paper_status(account)
        return

    if args.command == "paper-init":
        if not args.approve_policy:
            raise ValueError("paper-init requires explicit --approve-policy acknowledgement")
        _, portfolio, prices, tradable = _paper_data(
            args.snapshot,
            args.contract,
            args.config,
            args.portfolio,
            args.allow_market_closures,
        )
        decision_path = args.selection / "selected_policy.json"
        metadata_path = args.selection / "metadata.json"
        candidates_path = args.selection / "candidate_policies.csv"
        if not all(path.is_file() for path in (decision_path, metadata_path, candidates_path)):
            raise ValueError(
                "selection must contain selected_policy.json, metadata.json, and "
                "candidate_policies.csv"
            )
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        selection_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        candidates = pd.read_csv(candidates_path)
        selected_policy = decision.get("selected_policy")
        if decision.get("status") != "selected" or selected_policy is None:
            raise ValueError("Stage 4 did not select a deployable policy")
        if selected_policy.get("regime_policies"):
            raise ValueError(
                "drawdown/ZigZag-gated policies are research-only until the paper state "
                "machine stores and audits their peak/trough regime state"
            )
        if decision.get("portfolio") != args.portfolio:
            raise ValueError("selected policy portfolio does not match --portfolio")
        if decision.get("validation_status") != selection_metadata.get("validation_status"):
            raise ValueError("Stage 4 decision and metadata validation statuses differ")
        stage4_weights = selection_metadata.get("weights")
        if (
            stage4_weights is None
            or set(stage4_weights) != set(portfolio.weights)
            or any(
                abs(float(stage4_weights[asset]) - weight) > 1e-12
                for asset, weight in portfolio.weights.items()
            )
        ):
            raise ValueError("Stage 4 selection weights do not match the configured portfolio")
        if "is_selected" not in candidates:
            raise ValueError("candidate_policies.csv has no is_selected decision column")
        selected_rows = candidates[
            candidates["is_selected"].astype(str).str.lower().isin({"true", "1"})
        ]
        if len(selected_rows) != 1:
            raise ValueError("candidate_policies.csv must identify exactly one selected policy")
        selected_row = selected_rows.iloc[0]
        for field in ("multiplier", "floor_fraction", "drift_band"):
            if abs(float(selected_row[field]) - float(selected_policy[field])) > 1e-12:
                raise ValueError(f"selected policy differs from candidate evidence: {field}")
        if str(selected_row["review_frequency"]) != str(selected_policy["review_frequency"]):
            raise ValueError("selected policy differs from candidate evidence: review_frequency")
        price_hash = hashlib.sha256(
            prices.to_csv(date_format="%Y-%m-%dT%H:%M:%S.%f%z").encode()
        ).hexdigest()
        if selection_metadata.get("prices_sha256") != price_hash:
            raise ValueError("opening snapshot differs from the Stage 4 selection snapshot")
        controls = with_control_overrides(
            load_execution_controls(args.controls_config, args.portfolio),
            max_daily_turnover=args.max_daily_turnover,
            max_data_age_hours=args.max_data_age_hours,
            floor_alert_buffer_fraction=args.floor_alert_buffer,
            reconciliation_tolerance=args.reconciliation_tolerance,
            approval_expiry_hours=args.approval_expiry_hours,
        )
        execution_values = selection_metadata.get("execution", {})
        execution = ExecutionModel(**execution_values)
        timestamp, opening_prices, _ = _valuation_row(prices, tradable, args.as_of)
        if timestamp != prices.index[-1]:
            raise ValueError("paper operation must open at the latest Stage 4 valuation")
        initial_nav = (
            args.initial_nav
            if args.initial_nav is not None
            else float(selection_metadata.get("initial_nav", 1_000_000))
        )
        state_path = args.state or _project_path("runs", "stage5", args.portfolio, "paper.db")
        config = load_project_config(args.config)
        insured_assets = tuple(
            asset
            for asset in portfolio.weights
            if config.assets[asset].risky and asset != "fixed_income"
        )
        account = PaperPortfolio.initialize(
            state_path,
            portfolio=args.portfolio,
            strategic_weights=portfolio.weights,
            selected_policy=selected_policy,
            selection_metadata=selection_metadata,
            prices=opening_prices,
            valuation_at=timestamp,
            approved_by=args.approved_by,
            controls=controls,
            execution=execution,
            initial_nav=initial_nav,
            insured_assets=insured_assets,
        )
        print(
            f"initialized approved Stage 5 paper portfolio from selection "
            f"{selection_metadata['selection_id']}"
        )
        _print_paper_status(account)
        return

    if args.command == "paper-run":
        account = PaperPortfolio(args.state)
        account_state = account.state()
        portfolio_key = str(account_state["portfolio"])
        _, portfolio, prices, tradable = _paper_data(
            args.snapshot,
            args.contract,
            args.config,
            portfolio_key,
            args.allow_market_closures,
        )
        if set(portfolio.weights) != set(account_state["positions"]):
            raise ValueError("paper state assets differ from the configured portfolio")
        timestamp, daily_prices, daily_tradable = _valuation_row(prices, tradable, args.as_of)
        result = account.run_day(
            prices=daily_prices,
            tradable=daily_tradable,
            valuation_at=timestamp,
            observed_at=args.observed_at,
            observed_units=_load_observed_units(args.positions),
        )
        prefix = "already processed" if result.idempotent else "processed"
        print(
            f"{prefix} {result.valuation_at.isoformat()}: NAV={result.nav:,.2f}, "
            f"distance_to_floor={result.distance_to_floor:,.2f}, "
            f"reconciliation={result.reconciliation_status}"
        )
        if result.order_id:
            print(f"order {result.order_id}: {result.order_status}")
        for alert in result.alerts:
            print(f"{alert['severity']}: {alert['code']}: {alert['message']}")
        _print_paper_status(account)
        return

    if args.command == "show-portfolios":
        config = load_project_config(args.config)
        for portfolio in config.portfolios.values():
            weights = ", ".join(
                f"{asset}={weight:.0%}" for asset, weight in portfolio.weights.items()
            )
            print(f"{portfolio.key}: {weights}")
        return

    if args.command == "fetch-data":
        contract = load_data_contract(args.contract)
        snapshot = fetch_market_data(contract, args.start, args.end)
        output = args.output or _project_path("data", "snapshots", f"{args.start}_{args.end}")
        save_snapshot(snapshot, output)
        report = validate_market_data(
            snapshot.data,
            contract,
            args.start,
            args.end,
            series_keys=set(snapshot.data["series"]),
            series_start_dates=snapshot.manifest.series_start_dates,
        )
        allowed = {"stale"} if args.allow_market_closures else set()
        report.raise_for_errors(allowed)
        print(f"saved {len(snapshot.data)} validated observations to {output}")
        return

    snapshot, contract, report = _quality_report(args.snapshot, args.contract)
    if args.command == "validate-data":
        if args.portfolio:
            config = load_project_config(args.config)
            assets = list(config.portfolios[args.portfolio].weights)
            required = _series_keys_for_assets(contract, assets, include_usd=True)
            snapshot, contract, report = _quality_report(args.snapshot, args.contract, required)
        allowed = {"stale"} if args.allow_market_closures else set()
        for key, coverage in sorted(report.coverage.items()):
            print(f"{key}: coverage={coverage:.2%}")
        for issue in report.issues:
            severity = "accepted_market_closure" if issue.code in allowed else issue.severity
            print(f"{severity}: {issue.code}: {issue.message}")
        report.raise_for_errors(allowed)
        print("snapshot passed all configured checks")
        return

    config = load_project_config(args.config)
    portfolio = config.portfolios[args.portfolio]
    reserve_definition = config.assets["fixed_income"]
    reserve_daily_return = reserve_definition.daily_return
    reserve_calendar_day_accrual = reserve_definition.calendar_day_accrual
    assets = list(portfolio.weights)
    required = _series_keys_for_assets(
        contract,
        assets,
        include_usd=args.command in {"run-scenarios", "select-policy"}
        or bool(getattr(args, "include_usd_relative", False)),
    )
    snapshot, contract, report = _quality_report(args.snapshot, args.contract, required)
    allowed = {"stale"} if args.allow_market_closures else set()
    report.raise_for_errors(allowed)
    prices = point_in_time_prices(snapshot.data, contract, assets)
    tradable = point_in_time_tradability(snapshot.data, contract, assets, prices.index)
    insured_assets = tuple(
        asset for asset in portfolio.weights if config.assets[asset].risky
    )
    if args.command == "calibrate-regime":
        evaluation_start = (
            pd.Timestamp(args.evaluation_start) if args.evaluation_start else prices.index[0]
        )
        if evaluation_start.tzinfo is None:
            evaluation_start = evaluation_start.tz_localize(prices.index.tz)
        else:
            evaluation_start = evaluation_start.tz_convert(prices.index.tz)
        settings = CalibrationSettings(
            reversal_threshold=args.reversal_threshold,
            continuation_step=args.continuation_step,
            grid_step=args.grid_step,
            minimum_samples=args.minimum_samples,
            target_probability=args.target_probability,
        )
        levels_by_asset = {
            asset: point_in_time_prices(snapshot.data, contract, [asset])[asset]
            for asset in insured_assets
        }
        bundle = calibrate_regimes(
            levels_by_asset,
            valid_after=evaluation_start,
            settings=settings,
        )
        output = export_regime_calibration(bundle, args.output / args.portfolio)
        for asset, calibration in bundle.calibrations.items():
            print(
                f"{asset}: enter after {calibration.policy.entry_drawdown:.2%} drawdown, "
                f"exit after {calibration.policy.exit_recovery:.2%} recovery; "
                f"training={calibration.training_start}..{calibration.training_end}"
            )
        print(
            f"saved causal regime calibration {bundle.metadata['calibration_id']} to {output}"
        )
        return

    regime_policies = None
    regime_calibration_id = None
    regime_path = getattr(args, "regime_calibration", None)
    if regime_path is not None:
        regime_bundle = load_regime_calibration(regime_path)
        if set(regime_bundle.policies) != set(insured_assets):
            raise ValueError("regime calibration assets do not match insured portfolio assets")
        valid_after = pd.Timestamp(regime_bundle.metadata["valid_after"])
        if valid_after > prices.index[0]:
            raise ValueError("regime calibration uses observations from inside the backtest")
        regime_policies = regime_bundle.policies
        regime_calibration_id = str(regime_bundle.metadata["calibration_id"])
    if args.command == "select-policy":
        path_metrics_path = args.analysis / "path_metrics.csv"
        metadata_path = args.analysis / "metadata.json"
        if not path_metrics_path.is_file() or not metadata_path.is_file():
            raise ValueError("analysis must contain Stage 3 path_metrics.csv and metadata.json")
        path_metrics = pd.read_csv(path_metrics_path)
        stage3_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        stage3_weights = stage3_metadata.get("weights")
        if (
            stage3_weights is None
            or set(stage3_weights) != set(portfolio.weights)
            or any(
                abs(float(stage3_weights[asset]) - weight) > 1e-12
                for asset, weight in portfolio.weights.items()
            )
        ):
            raise ValueError("Stage 3 analysis weights do not match the selected portfolio")
        inflation, benchmark_name = _inflation_benchmark(args, snapshot, prices)
        price_hash = hashlib.sha256(
            prices.to_csv(date_format="%Y-%m-%dT%H:%M:%S.%f%z").encode()
        ).hexdigest()
        if stage3_metadata.get("data_sha256") != price_hash:
            raise ValueError("Stage 3 analysis was produced from a different price snapshot")
        inflation_hash = hashlib.sha256(
            inflation.astype(float).to_csv(date_format="%Y-%m-%dT%H:%M:%S.%f%z").encode()
        ).hexdigest()
        if stage3_metadata.get("inflation_sha256") != inflation_hash:
            raise ValueError("walk-forward inflation input differs from the Stage 3 analysis")
        tradability_hash = hashlib.sha256(
            tradable.astype(int).to_csv(date_format="%Y-%m-%dT%H:%M:%S.%f%z").encode()
        ).hexdigest()
        if stage3_metadata.get("tradability_sha256") != tradability_hash:
            raise ValueError("walk-forward tradability differs from the Stage 3 analysis")
        if stage3_metadata.get("inflation_benchmark") != benchmark_name:
            raise ValueError("walk-forward real-return benchmark differs from Stage 3")
        if stage3_metadata.get("regime_calibration_id") != regime_calibration_id:
            raise ValueError("Stage 3 and Stage 4 regime calibrations differ")
        if stage3_metadata.get("reserve_daily_return") != reserve_daily_return or bool(
            stage3_metadata.get("reserve_calendar_day_accrual", False)
        ) != bool(reserve_calendar_day_accrual):
            raise ValueError("Stage 3 reserve accrual assumptions differ from project config")
        criteria = with_criteria_overrides(
            load_selection_criteria(args.selection_config, args.portfolio),
            minimum_real_return=args.minimum_real_return,
            maximum_drawdown=args.maximum_drawdown,
            maximum_floor_breach_probability=args.maximum_breach_probability,
            maximum_cost_rate=args.maximum_cost_rate,
        )
        execution_values = stage3_metadata.get("execution", {})
        execution = ExecutionModel(**execution_values)
        result = run_policy_selection(
            path_metrics,
            prices,
            inflation,
            portfolio.weights,
            args.portfolio,
            criteria,
            test_bars=args.test_bars or portfolio.horizon_months * 21,
            folds=args.folds,
            minimum_train_bars=args.minimum_train_bars,
            execution=execution,
            reserve_daily_return=reserve_daily_return,
            reserve_calendar_day_accrual=reserve_calendar_day_accrual,
            tradable=tradable,
            inflation_benchmark_name=benchmark_name,
            initial_nav=float(stage3_metadata.get("initial_nav", 1_000_000)),
            source_analysis_id=stage3_metadata.get("analysis_id"),
            regime_policies=regime_policies,
            regime_calibration_id=regime_calibration_id,
        )
        output = export_policy_selection(result, args.output / args.portfolio)
        if result.selected_policy is None:
            print(f"no policy selected: {result.status}")
        else:
            policy = result.selected_policy
            print(
                "selected "
                f"multiplier={policy['multiplier']:g}, floor={policy['floor_fraction']:.0%}, "
                f"frequency={policy['review_frequency']}, drift_band={policy['drift_band']:.2%}"
            )
        print(
            f"walk-forward validation={result.validation_status}; "
            f"saved Stage 4 selection {result.metadata['selection_id']} to {output}"
        )
        return
    if args.command == "run-scenarios":
        inflation, benchmark_name = _inflation_benchmark(args, snapshot, prices)
        reference = _snapshot_level_series(snapshot.data, "usd_irr", prices.index)
        simulation = SimulationConfig(
            method=SimulationMethod(args.method),
            paths=args.paths,
            horizon_bars=args.horizon_bars or portfolio.horizon_months * 21,
            seed=args.seed,
            block_size=args.block_size,
            student_t_df=args.student_t_df,
            confidence=args.confidence,
            tail_confidence=args.tail_confidence,
        )
        stresses = standard_stress_scenarios(assets) if args.standard_stresses else ()
        result = run_scenario_analysis(
            prices,
            portfolio.weights,
            multipliers=_floats(args.multipliers),
            floors=_floats(args.floors),
            frequencies=_frequencies(args.frequencies),
            drift_bands=_floats(args.drift_bands),
            simulation=simulation,
            execution=_execution_model_from_args(args),
            reserve_daily_return=reserve_daily_return,
            reserve_calendar_day_accrual=reserve_calendar_day_accrual,
            inflation_index=inflation,
            fx_reference=reference,
            tradable=tradable,
            inflation_benchmark_name=benchmark_name,
            stresses=stresses,
            historical_windows=args.historical_window,
            initial_nav=args.initial_nav,
            max_workers=args.workers,
            regime_policies=regime_policies,
            regime_calibration_id=regime_calibration_id,
        )
        output = export_scenario_analysis(result, args.output / args.portfolio)
        print(
            f"evaluated {len(result.path_metrics)} path-policy outcomes across "
            f"{len(result.surface)} surface rows"
        )
        print(f"saved reproducible Stage 3 analysis {result.metadata['analysis_id']} to {output}")
        return

    reference = None
    if args.include_usd_relative:
        usd = snapshot.data[snapshot.data["series"] == "usd_irr"].copy()
        usd["valuation_at"] = pd.to_datetime(usd["valuation_at"], utc=True)
        reference = usd.set_index("valuation_at")["value"].sort_index()
    result = run_parameter_grid(
        prices,
        portfolio.weights,
        multipliers=_floats(args.multipliers),
        floors=_floats(args.floors),
        frequencies=_frequencies(args.frequencies),
        drift_bands=_floats(args.drift_bands),
        protection_drift_band=args.protection_drift_band,
        execution=_execution_model_from_args(args),
        reference_prices=reference,
        reserve_daily_return=reserve_daily_return,
        reserve_calendar_day_accrual=reserve_calendar_day_accrual,
        initial_nav=args.initial_nav,
        tradable=tradable,
        regime_policies=regime_policies,
        regime_calibration_id=regime_calibration_id,
    )
    output = export_experiment(result, args.output / args.portfolio)
    print(result.table.to_string(index=False))
    print(f"saved {len(result.runs)} auditable runs to {output}")


if __name__ == "__main__":
    main()
