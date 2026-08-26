"""Core building blocks for a prediction-independent portfolio insurance system."""

from .backtest import BacktestPolicy, BacktestResult, SleevePolicy, run_backtest
from .data_sources import fetch_market_data, point_in_time_prices, point_in_time_tradability
from .domain import AssetDefinition, PortfolioDefinition
from .execution import (
    ExecutionControls,
    OrderStatus,
    PaperDayResult,
    PaperPortfolio,
    load_execution_controls,
)
from .experiments import run_parameter_grid
from .overlay import apply_risk_overlays
from .policy_selection import (
    PolicySelectionResult,
    SelectionCriteria,
    WalkForwardResult,
    run_policy_selection,
    walk_forward_validate,
)
from .regime import (
    CalibrationBundle,
    CalibrationSettings,
    ProtectionState,
    RegimeCalibration,
    RegimePolicy,
    calibrate_regimes,
    confirmed_zigzag,
    export_regime_calibration,
    load_regime_calibration,
)
from .rebalancing import RebalancePolicy, needs_rebalance
from .scenarios import (
    HistoricalStressWindow,
    ScenarioAnalysisResult,
    SimulationConfig,
    SimulationMethod,
    StressScenario,
    run_scenario_analysis,
)
from .vppi import VppiAllocation, VppiPolicy, allocate_vppi

__all__ = [
    "AssetDefinition",
    "BacktestPolicy",
    "BacktestResult",
    "ExecutionControls",
    "HistoricalStressWindow",
    "OrderStatus",
    "PaperDayResult",
    "PaperPortfolio",
    "PolicySelectionResult",
    "PortfolioDefinition",
    "ProtectionState",
    "RebalancePolicy",
    "RegimeCalibration",
    "RegimePolicy",
    "CalibrationBundle",
    "CalibrationSettings",
    "ScenarioAnalysisResult",
    "SelectionCriteria",
    "SimulationConfig",
    "SimulationMethod",
    "SleevePolicy",
    "StressScenario",
    "VppiAllocation",
    "VppiPolicy",
    "WalkForwardResult",
    "allocate_vppi",
    "apply_risk_overlays",
    "calibrate_regimes",
    "confirmed_zigzag",
    "export_regime_calibration",
    "fetch_market_data",
    "load_execution_controls",
    "load_regime_calibration",
    "needs_rebalance",
    "point_in_time_prices",
    "point_in_time_tradability",
    "run_backtest",
    "run_parameter_grid",
    "run_policy_selection",
    "run_scenario_analysis",
    "walk_forward_validate",
]
