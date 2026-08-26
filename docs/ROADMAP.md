# Development Roadmap

## Product objective

Build a portfolio-management engine that turns two investment objectives into measurable constraints:

- Medium-term: do not underperform USD/IRR over a six-month horizon.
- Long-term: outperform USD/IRR over a 12-month horizon.

The 100% fixed-income short-term product is intentionally excluded because it has no risky
sleeve requiring a portfolio-insurance overlay.

USD/IRR is the explicit mandate benchmark. The configured lower-tail probability is the
statistical interpretation of the return objective; VPPI does not create a legal or
absolute guarantee by itself. Drawdown, floor behavior, turnover, and cost determine the
best stable policy among return-feasible candidates rather than acting as invented limits.

## Decision flow

`Strategic weights -> VPPI risk overlay -> Drift-band rebalance -> Orders -> Measurement`

Every stage has recorded inputs and outputs so that its effect can be attributed independently.

## Phase 0 — Contracts and baseline

Deliverables:

- Version-controlled definitions for the assets and two managed portfolios.
- A constant-multiplier VPPI allocation primitive.
- Calendar review combined with an allocation-drift band.
- Drawdown, Sharpe, Sortino, and Expected Shortfall metrics.
- Unit tests for financial invariants.

Acceptance criterion: all tests pass, and weights, floors, and exposures remain within their defined constraints.

## Phase 1 — Data contract and quality controls

Status: implemented. The enforceable contract is `configs/data_contract.toml`; automated
checks live in `portfolio_insurance.data_quality`.

Required data:

- Investable NAV or total-return series for representative gold, equity, and fixed-income instruments.
- Exchange-rate and inflation series with original publication timestamps.
- Distributions, splits, market holidays, trading costs, and trading restrictions.

Decisions required before ingestion:

- Select the investable instrument representing each asset class.
- Use USD/IRR as a non-investable reference asset for dollar-relative returns.
- Define local-currency and USD-relative returns; CPI-adjusted returns are diagnostic only.
- Define the official valuation calendar and missing-data policy.

Acceptance criterion: coverage, stale-price, data-quality, and look-ahead checks run automatically.

## Phase 2 — Event-driven backtest

Status: complete. `portfolio_insurance.data_sources` fetches and snapshots the selected
TSETMC and Wallex series; `portfolio_insurance.backtest` enforces delayed execution and
ledger reconciliation; and `portfolio_insurance.experiments` exports baselines, the full
parameter surface, run provenance, and effect attribution. Live smoke verification was
completed for both managed portfolios on 2026-08-23.

- Establish buy-and-hold and calendar-rebalanced baselines.
- Apply independent VPPI overlays to gold and equity sleeves and to USD-denominated scenarios.
- Test constant multipliers, multiple floors, and daily, weekly, biweekly, and monthly review frequencies.
- Model transaction costs, slippage, execution latency, and gap risk.
- Attribute the return and risk effects of allocation, insurance, and rebalancing independently.

Acceptance criterion: all calculations are free of look-ahead bias and reproducible from an auditable ledger.

## Phase 3 — Scenario analysis and Monte Carlo

Status: complete. `portfolio_insurance.scenarios` provides joint moving-block and
stationary bootstraps, a fat-tailed two-state regime model, named historical windows,
deterministic FX/asset/liquidity shocks, path-level audit records, and uncertainty-aware
policy surfaces. Distribution outputs include nominal and purchasing-power drawdown tails,
drawdown expected shortfall, duration, direct floor-shortfall severity, and Wilson event
intervals. See `docs/STAGE3.md` for assumptions and usage.

- Historical stress windows.
- Stationary or block bootstrap to preserve volatility clustering.
- Regime-switching or fat-tailed innovations.
- Exchange-rate jumps, simultaneous asset declines, and reduced-liquidity scenarios.

The primary result is a parameter surface:

`(multiplier, floor, review frequency, drift band) -> (real return, drawdown, breach probability, turnover)`

Acceptance criterion: results are reproducible with fixed seeds and include uncertainty intervals.

## Phase 4 — Policy selection

Status: implemented. `portfolio_insurance.policy_selection` builds an adverse-context
Pareto frontier, enforces optional mandate gates, requires a neighboring-parameter
stability plateau, and validates the repeated selection process on expanding-window,
strictly later test folds. Separate medium- and long-term preference profiles are
versioned in `configs/policy_selection.toml`. See `docs/STAGE4.md` for decision and
validation semantics.

- Build the Pareto frontier across return, drawdown, floor-breach probability, and cost.
- Identify a stability plateau instead of choosing an in-sample maximum.
- Perform walk-forward and out-of-sample validation.
- Select separate policies for the medium- and long-term portfolios.

Acceptance criterion: the policy is not fragile to small parameter changes across regimes and evaluation windows.

## Phase 5 — Controlled execution

Status: implemented for controlled paper operation. `portfolio_insurance.execution`
persists paper positions, valuations, approval-gated orders, fills, alerts, reconciliations,
and a tamper-evident audit chain. Versioned operational controls include daily turnover,
data age, floor proximity, reconciliation tolerance, approval expiry, and a kill switch.
The generated local dashboard shows current allocation, floor distance, alerts, and order
rationale. See `docs/STAGE5.md` for operating and safety semantics.

- Run paper portfolios with an audit log and daily reconciliation.
- Add turnover limits, a kill switch, and data/floor-breach alerts.
- Provide a dashboard showing current state, order rationale, and distance to the floor.
- Require human approval before every trade in the initial release.

## Phase 6 — Later extensions

- A bounded dynamic multiplier with transparent rules.
- Profit-taking and a ratcheting floor.
- Qualitative or predictive signals only as limited overlays that can be disabled.

## Standard reporting metrics

- CAGR and inflation-adjusted return
- Volatility, Sharpe ratio, and Sortino ratio
- Maximum drawdown, drawdown duration, and recovery time
- Value at Risk and Expected Shortfall
- Floor-breach probability and breach severity
- Upside and downside capture
- Turnover, costs, and return drag relative to the strategic baseline
- Rolling six-month loss probability for the medium-term portfolio
