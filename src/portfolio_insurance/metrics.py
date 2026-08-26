from __future__ import annotations

import numpy as np
import pandas as pd


def wealth_index(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    if initial_value <= 0:
        raise ValueError("initial_value must be positive")
    return initial_value * (1.0 + returns.astype(float)).cumprod()


def relative_returns(asset_returns: pd.Series, reference_returns: pd.Series) -> pd.Series:
    """Convert asset returns to returns measured in a reference unit.

    Use USD/IRR returns as the reference for dollar-relative performance or CPI changes
    for inflation-adjusted performance. Inputs are aligned by index; missing pairs remain
    missing so the caller's data-quality policy stays explicit.
    """
    asset, reference = asset_returns.astype(float).align(
        reference_returns.astype(float), join="outer"
    )
    if (reference <= -1.0).any():
        raise ValueError("reference returns must be greater than -100%")
    return (1.0 + asset) / (1.0 + reference) - 1.0


def drawdown_series(returns: pd.Series) -> pd.Series:
    return drawdown_series_from_levels(wealth_index(returns))


def drawdown_series_from_levels(levels: pd.Series) -> pd.Series:
    """Return the causal high-water-mark drawdown of a positive level series."""
    clean = levels.astype(float)
    if clean.empty:
        return clean.copy()
    if clean.isna().any() or not np.isfinite(clean.to_numpy()).all() or (clean <= 0).any():
        raise ValueError("drawdown levels must be finite and positive")
    running_peak = clean.cummax()
    return clean / running_peak - 1.0


def maximum_drawdown(returns: pd.Series) -> float:
    drawdowns = drawdown_series(returns)
    return float(drawdowns.min()) if not drawdowns.empty else float("nan")


def maximum_drawdown_from_levels(levels: pd.Series) -> float:
    drawdowns = drawdown_series_from_levels(levels)
    return float(drawdowns.min()) if not drawdowns.empty else float("nan")


def maximum_underwater_duration(drawdowns: pd.Series) -> int:
    """Count the longest consecutive run below the high-water mark."""
    longest = current = 0
    for underwater in drawdowns.astype(float) < 0:
        current = current + 1 if underwater else 0
        longest = max(longest, current)
    return longest


def annualized_volatility(returns: pd.Series, periods_per_year: int) -> float:
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, periods_per_year: int, risk_free_per_period: float = 0.0
) -> float:
    excess = returns.astype(float) - risk_free_per_period
    volatility = excess.std(ddof=1)
    if volatility == 0 or np.isnan(volatility):
        return float("nan")
    return float(excess.mean() / volatility * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, periods_per_year: int, minimum_acceptable_return: float = 0.0
) -> float:
    excess = returns.astype(float) - minimum_acceptable_return
    downside = np.minimum(excess, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
    if downside_deviation == 0:
        return float("nan")
    return float(excess.mean() / downside_deviation * np.sqrt(periods_per_year))


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    clean = returns.dropna().astype(float)
    if clean.empty:
        return float("nan")
    cutoff = clean.quantile(1.0 - confidence)
    return float(clean[clean <= cutoff].mean())
