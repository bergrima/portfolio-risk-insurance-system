# Stage 2 — Data and Backtest Contract

## Selected instruments

| Internal asset | Vendor | Symbol | Input |
|---|---|---|---|
| Fixed income | TSETMC through `pytse-client` | لبخند | Adjusted close |
| Gold | TSETMC through `pytse-client` | طلا | Adjusted close |
| Equity | TSETMC through `pytse-client` | آگاس | Adjusted close |
| USD reference | Wallex public UDF API | USDTTMN | Hourly close, multiplied by 10 to IRR |

USD/IRR is non-investable and is the mandate benchmark. The long-term allocation is 10%
fixed income, 50% gold, and 40% equity.

The versioned `full_history` snapshot preserves unequal series inceptions instead of
discarding older asset-level observations. Coverage is measured from each recorded
series inception, and multi-asset price inputs begin only when every required investable
asset has a valid observation. For the current instruments, the common portfolio history
therefore begins on 2022-08-29 even though equity, gold, and USD/IRR histories begin
earlier.

The fixed-income sleeve uses the fund's point-in-time adjusted total-return history.
Although the fund accrues economically on a calendar-day basis, a closure carries the
last published NAV. The accumulated multi-day return is recognized when the next NAV is
published, so the research ledger never invents an unpublished daily valuation.

Explicit transaction fees are versioned in `configs/transaction_costs.toml` and applied
to absolute traded notional by asset and side:

| Asset | Buy | Sell |
|---|---:|---:|
| Fixed income | 0.01875% | 0.01875% |
| Equity | 0.116% | 0.11875% |
| Gold | 0.125% | 0.125% |

Slippage is a separate execution assumption and is not included in these fee rates.

## Point-in-time rules

- The official valuation calendar is Saturday through Wednesday at 16:00 Asia/Tehran.
- TSETMC observations are carried forward across instrument holidays and checked against
  each series' maximum staleness.
- Wallex is a 24/7 market. The adapter fetches hourly candles in 30-day chunks and admits
  a close only after the candle has ended and before the portfolio valuation timestamp.
- Snapshots contain `valuation_at`, `observed_at`, and `available_at`. Any row with
  `available_at > valuation_at` fails validation.
- Every snapshot has a SHA-256 checksum and records its vendor symbols, package version,
  requested range, currency conversion, and fetch timestamp.

Adjusted histories may be revised by a vendor after distributions or corporate actions.
The snapshot checksum makes a research run reproducible from the data actually used, but
does not turn a later vendor restatement into an original point-in-time publication archive.

## Event order

For each bar the engine:

1. applies the close-to-close return to positions held before the bar;
2. executes an order decided on an earlier bar at the current close;
3. deducts side-specific transaction cost and separately modeled slippage from NAV;
4. updates each insured sleeve's independent risky/reserve state;
5. updates each configured causal drawdown/recovery state and reviews immediately on a
   state change or on the normal calendar;
6. records the complete decision target;
7. queues an order no earlier than `latency_bars` in the future; and
8. writes holdings, weights, targets, trades, costs, floors, regimes, gaps, and order IDs to
   the ledger.

Only one order may be pending. A review that occurs while an order is pending is still
recorded, but its new trade is blocked and linked to the pending order. This avoids hidden
overlapping orders when latency exceeds one bar.

Carried prices during an accepted market closure remain last-known valuations, not
executable quotes. The tradability mask is derived from each observation timestamp. If a
pending whole-portfolio order requires a closed asset, execution waits until every asset
required by that order is tradable. Deferred bars and affected assets are recorded in the
ledger and included in the deterministic run identity.

## Floor interpretation

Gold and equity have independent synthetic VPPI sleeves. When a pre-evaluation regime
calibration is supplied, every sleeve starts fully invested at its strategic weight and VPPI
is dormant until that asset's drawdown gate is crossed. Each sleeve earns its risky asset
return on its risky allocation and the fixed-income return on its released reserve.
Its exposure is `min(multiplier * cushion, sleeve NAV) / sleeve NAV`. The aggregate
portfolio also reports observed floor and gap breaches.

ZigZag pivots are used only in the offline calibration artifact, and a pivot is timestamped
by both its historical extreme and its later confirmation. Backtest state uses only the
causal running peak, protected-state trough, observed drawdown, and observed recovery. See
`docs/REGIME_GATE.md`.

For one historical path, `floor_breach_bar_fraction` is the fraction of valuation bars
below the floor and `ever_breached` indicates whether any breach occurred. True breach
probability across simulated paths belongs to Phase 3; the legacy
`floor_breach_probability` output currently aliases the historical bar fraction.

## Reproducibility and outputs

A run ID is a deterministic UUID derived from the price-table hash, strategic weights,
initial NAV, and complete policy. `BacktestResult.verify_ledger()` checks that holdings
reconcile to NAV, weights sum to one, gross NAV minus costs equals net NAV, and every
execution follows its decision.

An experiment export contains:

- `summary.csv` for the full parameter surface;
- buy-and-hold and calendar-rebalanced baseline ledgers;
- one directory per deterministic run ID;
- `ledger.csv`, `metadata.json`, `summary.json`, and `attribution.csv` per run.

Attribution geometrically decomposes each return into allocation, rebalancing, insurance,
and insured-strategy cost effects with an exact reconstructed return.
