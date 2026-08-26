"""Concrete, point-in-time market-data adapters used by Stage 2 research runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay

from .data_contract import DataContract

DEFAULT_TSE_SYMBOLS: Mapping[str, str] = {
    "fixed_income_tr": "لبخند",
    "gold_tr": "طلا",
    "equity_tr": "آگاس",
}
WALLEX_HISTORY_URL = "https://api.wallex.ir/v1/udf/history"


class PytseClient(Protocol):
    def download(self, **kwargs: Any) -> Mapping[str, pd.DataFrame]: ...


@dataclass(frozen=True, slots=True)
class FetchManifest:
    fetched_at: str
    start: str
    end: str
    tse_symbols: Mapping[str, str]
    wallex_symbol: str
    wallex_resolution: int
    currency: str
    pytse_client_version: str
    row_count: int
    series_start_dates: Mapping[str, str] = field(default_factory=dict)
    series_end_dates: Mapping[str, str] = field(default_factory=dict)
    sha256: str = ""


@dataclass(frozen=True, slots=True)
class MarketDataSnapshot:
    data: pd.DataFrame
    manifest: FetchManifest


def valuation_index(
    start: str | pd.Timestamp, end: str | pd.Timestamp, contract: DataContract
) -> pd.DatetimeIndex:
    """Create the official local valuation calendar, represented in UTC."""
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()
    dates = pd.date_range(
        start_date, end_date, freq=CustomBusinessDay(weekmask=contract.policy.weekmask)
    )
    hour, minute = (int(part) for part in contract.policy.valuation_time.split(":"))
    return (
        (dates + pd.Timedelta(hours=hour, minutes=minute))
        .tz_localize(contract.policy.timezone, ambiguous="raise", nonexistent="raise")
        .tz_convert("UTC")
    )


def _asof_rows(key: str, observations: pd.Series, valuations: pd.DatetimeIndex) -> pd.DataFrame:
    observations = observations.sort_index()
    if observations.index.tz is None:
        raise ValueError("observation timestamps must be timezone-aware")
    source = pd.DataFrame({"value": observations.astype(float), "observed_at": observations.index})
    combined = (
        source.reindex(source.index.union(valuations)).sort_index().ffill().reindex(valuations)
    )
    combined.index.name = "valuation_at"
    combined = combined.dropna()
    combined["series"] = key
    combined["available_at"] = combined["observed_at"]
    return combined.reset_index()[
        ["series", "valuation_at", "observed_at", "available_at", "value"]
    ]


class PytseAdapter:
    def __init__(self, client: PytseClient | None = None) -> None:
        if client is None:
            try:
                import pytse_client as client_module
            except ImportError as exc:
                raise RuntimeError("install pytse-client to fetch TSETMC history") from exc
            client = client_module
        self._client = client

    def fetch(self, symbols: Mapping[str, str], valuations: pd.DatetimeIndex) -> pd.DataFrame:
        """Fetch adjusted TSETMC closes and expand them onto valuation dates."""
        histories = self._client.download(
            symbols=list(symbols.values()), adjust=True, write_to_csv=False
        )
        rows: list[pd.DataFrame] = []
        timezone = str(valuations.tz)
        local_time = valuations.tz_convert("Asia/Tehran")[0].time()
        for key, symbol in symbols.items():
            if symbol not in histories or histories[symbol].empty:
                raise ValueError(f"pytse-client returned no history for {symbol}")
            history = histories[symbol].copy()
            required = {"date", "adjClose"}
            if not required <= set(history.columns):
                raise ValueError(
                    f"{symbol} history is missing {sorted(required - set(history.columns))}"
                )
            dates = pd.to_datetime(history["date"], errors="raise")
            local = pd.DatetimeIndex(dates.dt.date) + pd.Timedelta(
                hours=local_time.hour, minutes=local_time.minute
            )
            observed = local.tz_localize("Asia/Tehran").tz_convert(timezone)
            values = pd.Series(
                pd.to_numeric(history["adjClose"], errors="raise").to_numpy(), index=observed
            )
            values = values[~values.index.duplicated(keep="last")]
            rows.append(_asof_rows(key, values, valuations))
        return pd.concat(rows, ignore_index=True)


class WallexAdapter:
    """Fetch hourly USDT/toman candles and select only closes known by valuation time."""

    def __init__(
        self,
        get_json: Callable[[str], Mapping[str, Any]] | None = None,
        timeout: float = 20,
        max_workers: int = 4,
    ) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self._get_json = get_json or self._request_json
        self.timeout = timeout
        self.max_workers = max_workers

    def _request_json(self, url: str) -> Mapping[str, Any]:
        request = Request(
            url, headers={"Accept": "application/json", "User-Agent": "portfolio-insurance/0.1"}
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def fetch(
        self, valuations: pd.DatetimeIndex, symbol: str = "USDTTMN", resolution: int = 60
    ) -> pd.DataFrame:
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        # Add one day on each side and chunk below typical UDF response limits.
        start_at = valuations.min() - pd.Timedelta(days=1)
        final_at = valuations.max() + pd.Timedelta(days=1)
        timestamps: list[object] = []
        closes: list[object] = []
        urls: list[str] = []
        while start_at < final_at:
            end_at = min(start_at + pd.Timedelta(days=30), final_at)
            query = urlencode(
                {
                    "symbol": symbol,
                    "resolution": resolution,
                    "from": int(start_at.timestamp()),
                    "to": int(end_at.timestamp()),
                }
            )
            urls.append(f"{WALLEX_HISTORY_URL}?{query}")
            start_at = end_at
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(urls))) as executor:
            payloads = executor.map(self._get_json, urls)
            for payload in payloads:
                status = payload.get("s")
                if status == "no_data":
                    continue
                if status != "ok":
                    raise ValueError(
                        f"Wallex returned an unsuccessful candle response: {status}"
                    )
                timestamps.extend(payload.get("t", []))
                closes.extend(payload.get("c", []))
        if not timestamps or len(timestamps) != len(closes):
            raise ValueError("Wallex returned empty or inconsistent candles")
        # API timestamps denote candle openings. Only the close is usable by the engine.
        observed = pd.to_datetime(timestamps, unit="s", utc=True) + pd.Timedelta(minutes=resolution)
        irr_values = pd.to_numeric(pd.Series(closes), errors="raise") * 10  # TMN -> IRR
        series = pd.Series(irr_values.to_numpy(), index=observed)
        series = series[~series.index.duplicated(keep="last")]
        return _asof_rows("usd_irr", series, valuations)


def fetch_market_data(
    contract: DataContract,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    tse_symbols: Mapping[str, str] = DEFAULT_TSE_SYMBOLS,
    wallex_symbol: str = "USDTTMN",
    wallex_resolution: int = 60,
    pytse: PytseAdapter | None = None,
    wallex: WallexAdapter | None = None,
) -> MarketDataSnapshot:
    valuations = valuation_index(start, end, contract)
    if valuations.empty:
        raise ValueError("requested range contains no valuation dates")
    pytse = pytse or PytseAdapter()
    wallex = wallex or WallexAdapter()
    data = pd.concat(
        [
            pytse.fetch(tse_symbols, valuations),
            wallex.fetch(valuations, wallex_symbol, wallex_resolution),
        ],
        ignore_index=True,
    ).sort_values(["valuation_at", "series"], ignore_index=True)
    ranges = data.groupby("series", sort=True)["valuation_at"].agg(["min", "max"])
    try:
        pytse_version = version("pytse-client")
    except PackageNotFoundError:
        pytse_version = "injected-test-client"
    manifest = FetchManifest(
        fetched_at=datetime.now(UTC).isoformat(),
        start=str(pd.Timestamp(start).date()),
        end=str(pd.Timestamp(end).date()),
        tse_symbols=dict(tse_symbols),
        wallex_symbol=wallex_symbol,
        wallex_resolution=wallex_resolution,
        currency="IRR",
        pytse_client_version=pytse_version,
        row_count=len(data),
        series_start_dates={key: value.isoformat() for key, value in ranges["min"].items()},
        series_end_dates={key: value.isoformat() for key, value in ranges["max"].items()},
    )
    return MarketDataSnapshot(data, manifest)


def point_in_time_prices(
    data: pd.DataFrame, contract: DataContract, assets: Sequence[str]
) -> pd.DataFrame:
    """Convert validated long-form observations into the wide price input for Stage 2."""
    required = {"series", "valuation_at", "available_at", "value"}
    if required - set(data.columns):
        raise ValueError(f"market data is missing {sorted(required - set(data.columns))}")
    series_for_asset = {
        definition.asset: key
        for key, definition in contract.series.items()
        if definition.investable and definition.asset in assets
    }
    missing = set(assets) - set(series_for_asset)
    if missing:
        raise ValueError(f"no investable series configured for assets: {sorted(missing)}")
    frame = data[data["series"].isin(series_for_asset.values())].copy()
    frame["valuation_at"] = pd.to_datetime(frame["valuation_at"], utc=True)
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    if (frame["available_at"] > frame["valuation_at"]).any():
        raise ValueError("cannot construct prices from unavailable observations")
    reverse = {series: asset for asset, series in series_for_asset.items()}
    frame["asset"] = frame["series"].map(reverse)
    prices = frame.pivot(index="valuation_at", columns="asset", values="value")
    prices = prices.reindex(columns=list(assets)).astype(float).sort_index()
    if prices.empty or any(prices[asset].first_valid_index() is None for asset in assets):
        raise ValueError("one or more investable assets have no price history")
    common_start = max(prices[asset].first_valid_index() for asset in assets)
    prices = prices.loc[common_start:]
    if prices.isna().any().any():
        missing = prices.columns[prices.isna().any()].tolist()
        raise ValueError(f"price history has internal gaps after common inception: {missing}")
    return prices


def point_in_time_tradability(
    data: pd.DataFrame,
    contract: DataContract,
    assets: Sequence[str],
    index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Mark bars where each investable asset has a newly observed executable close.

    Carried prices remain valid last-known valuations during a market closure, but they
    are not executable. Downstream engines use this mask to defer whole-portfolio orders
    until every asset required by the order is tradable again.
    """
    required = {"series", "valuation_at", "observed_at", "available_at"}
    if required - set(data.columns):
        raise ValueError(f"market data is missing {sorted(required - set(data.columns))}")
    series_for_asset = {
        definition.asset: key
        for key, definition in contract.series.items()
        if definition.investable and definition.asset in assets
    }
    missing = set(assets) - set(series_for_asset)
    if missing:
        raise ValueError(f"no investable series configured for assets: {sorted(missing)}")
    frame = data[data["series"].isin(series_for_asset.values())].copy()
    for column in ("valuation_at", "observed_at", "available_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    if (frame["available_at"] > frame["valuation_at"]).any():
        raise ValueError("cannot construct tradability from unavailable observations")
    reverse = {series: asset for asset, series in series_for_asset.items()}
    frame["asset"] = frame["series"].map(reverse)
    frame["tradable"] = frame["observed_at"] == frame["valuation_at"]
    mask = frame.pivot(index="valuation_at", columns="asset", values="tradable")
    if index is not None:
        mask = mask.reindex(index)
    return mask.reindex(columns=list(assets)).fillna(False).astype(bool).sort_index()


def save_snapshot(snapshot: MarketDataSnapshot, directory: str | Path) -> Path:
    """Persist an immutable CSV plus a content-addressed provenance manifest."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    csv_path = target / "market_data.csv"
    snapshot.data.to_csv(csv_path, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f%z")
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest = asdict(snapshot.manifest) | {"sha256": digest}
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return target


def load_snapshot(directory: str | Path) -> MarketDataSnapshot:
    source = Path(directory)
    csv_path = source / "market_data.csv"
    raw_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if digest != raw_manifest["sha256"]:
        raise ValueError("snapshot checksum does not match its manifest")
    data = pd.read_csv(csv_path)
    for column in ("valuation_at", "observed_at", "available_at"):
        data[column] = pd.to_datetime(data[column], utc=True)
    return MarketDataSnapshot(data, FetchManifest(**raw_manifest))
