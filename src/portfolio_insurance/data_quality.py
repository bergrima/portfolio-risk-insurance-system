from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from collections.abc import Mapping

import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay

from .data_contract import DataContract

REQUIRED_COLUMNS = frozenset({"series", "valuation_at", "observed_at", "available_at", "value"})


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class QualityIssue:
    code: str
    severity: Severity
    message: str
    series: str | None = None


@dataclass(frozen=True, slots=True)
class QualityReport:
    issues: tuple[QualityIssue, ...]
    coverage: dict[str, float]

    @property
    def passed(self) -> bool:
        return not any(issue.severity == Severity.ERROR for issue in self.issues)

    def raise_for_errors(self, allowed_codes: set[str] | frozenset[str] = frozenset()) -> None:
        blocking = [
            issue
            for issue in self.issues
            if issue.severity == Severity.ERROR and issue.code not in allowed_codes
        ]
        if blocking:
            messages = "; ".join(issue.message for issue in blocking)
            raise ValueError(messages)


def validate_market_data(
    data: pd.DataFrame,
    contract: DataContract,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    series_keys: set[str] | None = None,
    series_start_dates: Mapping[str, str | pd.Timestamp] | None = None,
) -> QualityReport:
    """Validate long-form, point-in-time market data without modifying it.

    `available_at` is the timestamp at which a value could first have been used. A row
    whose value was unavailable at its `valuation_at` is a look-ahead violation.
    """
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    frame = data.copy()
    for column in ("valuation_at", "observed_at", "available_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")

    issues: list[QualityIssue] = []
    known = set(contract.series)
    selected = known if series_keys is None else set(series_keys)
    if unknown_selected := selected - known:
        raise ValueError(f"unknown requested series: {sorted(unknown_selected)}")
    if series_keys is None:
        for key in sorted(set(frame["series"].dropna()) - known):
            issues.append(
                QualityIssue("unknown_series", Severity.ERROR, f"unknown series: {key}", key)
            )
    else:
        frame = frame.loc[frame["series"].isin(selected)].copy()

    bad_time = frame[["valuation_at", "observed_at", "available_at"]].isna().any(axis=1)
    if bad_time.any():
        issues.append(
            QualityIssue(
                "invalid_timestamp",
                Severity.ERROR,
                f"{bad_time.sum()} rows have invalid timestamps",
            )
        )
    duplicate = frame.duplicated(["series", "valuation_at"], keep=False)
    if duplicate.any():
        issues.append(
            QualityIssue(
                "duplicate",
                Severity.ERROR,
                f"{duplicate.sum()} rows duplicate a series/valuation timestamp",
            )
        )
    numeric = pd.to_numeric(frame["value"], errors="coerce")
    bad_value = numeric.map(lambda value: not isfinite(value) if pd.notna(value) else True) | (
        numeric <= 0
    )
    if bad_value.any():
        issues.append(
            QualityIssue(
                "invalid_value",
                Severity.ERROR,
                f"{bad_value.sum()} rows have non-positive or non-finite values",
            )
        )
    lookahead = frame["available_at"] > frame["valuation_at"]
    if lookahead.any():
        issues.append(
            QualityIssue(
                "look_ahead",
                Severity.ERROR,
                f"{lookahead.sum()} rows were unavailable at valuation time",
            )
        )
    future_observation = frame["observed_at"] > frame["valuation_at"]
    if future_observation.any():
        issues.append(
            QualityIssue(
                "future_observation",
                Severity.ERROR,
                f"{future_observation.sum()} rows observe the future",
            )
        )

    if start is None:
        start = frame["valuation_at"].min()
    if end is None:
        end = frame["valuation_at"].max()
    coverage: dict[str, float] = {}
    for key, definition in contract.series.items():
        if key not in selected:
            continue
        rows = frame.loc[frame["series"] == key].sort_values("valuation_at")
        expected_start = pd.Timestamp(start).date()
        if series_start_dates and key in series_start_dates:
            expected_start = max(
                expected_start, pd.Timestamp(series_start_dates[key]).date()
            )
        expected = pd.date_range(
            start=expected_start,
            end=pd.Timestamp(end).date(),
            freq=CustomBusinessDay(weekmask=contract.policy.weekmask),
        )
        observed_dates = set(rows["valuation_at"].dt.tz_localize(None).dt.normalize().dropna())
        expected_dates = set(expected.normalize())
        ratio = (
            len(observed_dates & expected_dates) / len(expected_dates) if expected_dates else 1.0
        )
        coverage[key] = ratio
        if ratio < contract.policy.minimum_coverage:
            issues.append(
                QualityIssue(
                    "coverage",
                    Severity.ERROR,
                    f"{key} coverage {ratio:.1%} is below {contract.policy.minimum_coverage:.1%}",
                    key,
                )
            )
        if not rows.empty:
            ages = (rows["valuation_at"] - rows["observed_at"]).dt.total_seconds() / 86400
            stale = ages > definition.max_stale_days
            if stale.any():
                issues.append(
                    QualityIssue(
                        "stale", Severity.ERROR, f"{key} has {stale.sum()} stale observations", key
                    )
                )
            earliest_allowed = rows["observed_at"] + pd.to_timedelta(
                definition.publication_lag_days, unit="D"
            )
            early = rows["available_at"] < earliest_allowed
            if early.any():
                issues.append(
                    QualityIssue(
                        "publication_lag",
                        Severity.ERROR,
                        f"{key} has {early.sum()} values available before its declared publication lag",
                        key,
                    )
                )
    return QualityReport(tuple(issues), coverage)
