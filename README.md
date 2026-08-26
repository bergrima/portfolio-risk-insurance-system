# Portfolio Risk Insurance System

A prediction-independent risk-control engine for medium- and long-term investment portfolios.

## Strategic portfolios

| Portfolio | Fixed income | Gold | Equity |
|---|---:|---:|---:|
| Medium-term | 30% | 40% | 30% |
| Long-term | 10% | 50% | 40% |

These are strategic allocation weights and remain the actual starting weights. With a
pre-evaluation drawdown/ZigZag calibration, the VPPI overlay may reduce one risky asset only
after its protection gate is triggered; released exposure moves to fixed income.

The 100% fixed-income short-term product is outside the risk-engine scope because it has no
risky sleeve to insure. USD is modeled as a non-investable reference asset for dollar-relative
performance; it does not receive a strategic portfolio weight.

## Design principles

1. The core decision engine is independent of forecasting models.
2. Strategic allocation, downside protection, and rebalancing are separate and testable mechanisms.
3. A constant multiplier is evaluated first. A dynamic multiplier is introduced only after the baseline is stable.
4. Simulation results do not imply a guarantee against loss. Floor breaches and gap risk must be reported explicitly.
5. Parameters are selected from a stable region and a Pareto frontier, not from a single in-sample optimum.

## Local development

```powershell
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
portfolio-insurance show-portfolios
```

`pytse-client` currently requires Python 3.11-3.13 because of its Windows `lxml`
dependency. Python 3.14 is intentionally excluded by `pyproject.toml`.

## Stage 2 workflow

The default investable series are `لبخند` (fixed income), `طلا` (gold), and `آگاس`
(equity). The mandate benchmark is Wallex `USDTTMN`
hourly candles converted from toman to IRR. Vendor data is research input, not a
representation that any price was executable at the recorded close.

Research uses the point-in-time adjusted total-return history of the fixed-income fund.
The fund is economically calendar-day accruing, while the portfolio carries the last
published NAV during closures and recognizes the accumulated return only when a new NAV
becomes available. Explicit fees default to the asset/side schedule in
`configs/transaction_costs.toml`; any configured slippage is charged separately.

The `full_history` snapshot retains each series from its own first available valuation:
equity from 2015-11-01, gold from 2017-06-10, USD/IRR from 2018-11-28, and fixed income
from 2022-08-29. Portfolio backtests start at the common three-asset inception on
2022-08-29, while the earlier single-asset observations remain available for research.

```powershell
portfolio-insurance fetch-data --start 2015-11-01 --end 2026-08-24 `
  --output data/snapshots/full_history --allow-market-closures
portfolio-insurance validate-data --snapshot data/snapshots/full_history `
  --portfolio medium_term --allow-market-closures
portfolio-insurance calibrate-regime --snapshot data/snapshots/full_history `
  --portfolio medium_term --allow-market-closures
portfolio-insurance run-backtest --snapshot data/snapshots/full_history --portfolio medium_term `
  --multipliers 2,3,4 --floors 0.75,0.8,0.9 `
  --frequencies daily,weekly,biweekly,monthly --drift-bands 0.02,0.05 `
  --protection-drift-band 0.025 `
  --slippage-bps 5 --latency-bars 1 `
  --include-usd-relative `
  --regime-calibration runs/regime_calibration/medium_term
```

Each run writes a deterministic run ID, input hash, policy metadata, reconciled ledger,
summary, baseline ledgers, and geometric effect attribution. See `docs/STAGE2.md` for
timing and data assumptions, and `docs/REGIME_GATE.md` for the financial logic and current
pre-evaluation calibration evidence.

## Interactive notebook

The executed `notebooks/stage2_playground.ipynb` opens with verified tables and charts
already populated from `data/snapshots/live_smoke`. Launch it with:

```powershell
uv sync --extra dev
uv run --extra dev jupyter lab notebooks/stage2_playground.ipynb
```

Change `SNAPSHOT_NAME` in the first setup cell to inspect another fetched snapshot. The
notebook includes validation, price charts, an editable VPPI policy, ledger reconciliation,
trade inspection, NAV/floor and allocation charts, baselines, attribution, local/USD-relative
parameter grids, optional artifact export, and a long-term portfolio run.

## Stage 3 workflow

Stage 3 sends joint asset, inflation, and FX paths through the audited Stage 2 execution
engine. It supports moving-block and stationary bootstrap paths, a fat-tailed two-state
regime model, named historical windows, and explicit FX, joint-decline, and reduced-liquidity
stresses. Fixed root and child seeds make every result reproducible.

```powershell
portfolio-insurance run-scenarios --snapshot data/snapshots/full_history `
  --portfolio medium_term --paths 1000 --workers 4 --seed 20260824 `
  --method moving_block_bootstrap --block-size 10 `
  --multipliers 2,3,4 --floors 0.75,0.8,0.9 `
  --frequencies daily,weekly,biweekly,monthly --drift-bands 0.02,0.05 `
  --slippage-bps 5 --latency-bars 1 `
  --inflation-proxy usd_irr --allow-market-closures --standard-stresses `
  --regime-calibration runs/regime_calibration/medium_term
```

The export contains the policy `surface.csv`, path-level `path_metrics.csv`, and complete
reproducibility metadata. USD/IRR is the default mandate benchmark; CPI remains available
for separate diagnostic research. See `docs/STAGE3.md` for model and interval semantics.

When market closures are explicitly accepted, carried prices remain last-known
valuations while orders wait for the required assets to reopen. The medium-term mandate
requires six-month return not below USD/IRR, and the long-term mandate requires 12-month
return strictly above USD/IRR.

## Stage 4 workflow

Stage 4 first enforces the portfolio's dollar-relative return mandate, then ranks only
feasible policies on the return, drawdown, floor-breach, and cost Pareto frontier and
requires stable behavior among adjacent parameter choices.
It repeats selection on expanding historical windows and scores the selected policy only
on the immediately later fold. Medium- and long-term portfolios use separate,
version-controlled objective weights.

```powershell
portfolio-insurance select-policy `
  --analysis runs/stage3/medium_term `
  --snapshot data/snapshots/full_history --portfolio medium_term `
  --folds 3 --inflation-proxy usd_irr --allow-market-closures
```

The export keeps the full Pareto and stability table, context-level evidence, the selected
policy or non-selection reason, all walk-forward folds, and reproducibility metadata.
Floor, drawdown, breach, and cost remain optimization objectives rather than predeclared
pass/fail limits. See `docs/STAGE4.md` for the exact rules.

## Stage 5 workflow

Stage 5 runs an explicitly approved Stage 4 candidate as a controlled paper portfolio.
It records daily reconciliation, tamper-evident audit events, alerts, approval-gated
orders, and later paper fills. Turnover, data age, floor proximity, and approval expiry
are versioned controls; the kill switch cancels every open order.

```powershell
portfolio-insurance paper-init `
  --selection runs/stage4/medium_term `
  --snapshot data/snapshots/full_history --portfolio medium_term `
  --approved-by "investment-committee" --approve-policy `
  --allow-market-closures

portfolio-insurance paper-run `
  --state runs/stage5/medium_term/paper.db `
  --snapshot data/snapshots/full_history --allow-market-closures

portfolio-insurance paper-status --state runs/stage5/medium_term/paper.db
```

Every proposed rebalance remains pending until `paper-approve` records a named approver;
approval never causes same-valuation execution. `paper-reject` and
`paper-kill-switch` provide the corresponding manual controls. The Stage 5 directory
contains the paper database, a machine-readable state export, and a responsive local
operator dashboard. See `docs/STAGE5.md` for the complete workflow.

Codex Desktop can use its bundled workspace Python runtime if Python is unavailable on the system path.

## Repository structure

- `src/portfolio_insurance`: domain logic and independent calculation engines
- `configs/portfolios.toml`: version-controlled portfolio definitions
- `configs/data_contract.toml`: point-in-time series definitions and valuation policy
- `configs/transaction_costs.toml`: asset- and side-specific transaction fees
- `configs/execution_controls.toml`: paper-execution boundaries and alert thresholds
- `tests`: tests for financial contracts and invariants
- `docs/ROADMAP.md`: development phases and acceptance criteria

## Current status

Phases 1–5 are implemented. The system now includes the point-in-time data contract and
quality controls, the event-driven backtest and attribution layer, reproducible scenario
analysis, Pareto/stability policy selection with walk-forward validation, and controlled
paper execution with human approval, reconciliation, alerts, and a local operator dashboard.
Live broker integration is intentionally not implemented.
