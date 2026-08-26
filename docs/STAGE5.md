# Stage 5 — Controlled Paper Execution

## Purpose

Stage 5 turns one explicitly approved Stage 4 candidate into a controlled paper
portfolio. It does not connect to a broker and cannot place a live trade. Its purpose is
to exercise the complete operating process before any live integration exists:

`valuation -> reconciliation -> controls -> order rationale -> human approval -> later paper fill`

The initial paper positions are onboarded at the strategic weights. This opening state is
not represented as a trade. Every later rebalance requires a separately recorded human
approval.

## Deployment gate

`paper-init` accepts only a Stage 4 export whose status is `selected`. It refuses a
selection with failed, incomplete, or insufficient walk-forward validation, checks that
the portfolio and complete opening price history match the Stage 4 metadata, requires the
latest selected valuation to remain within the data-age limit, and requires both an
approver identity and the explicit `--approve-policy` acknowledgement.

A Stage 4 selection with `completed_no_mandate_gates` can enter paper operation because a
named human explicitly approves it at initialization. The audit record preserves that
the research candidate did not have externally supplied mandate gates; paper operation
does not silently turn it into an approved live mandate.

## Daily operating cycle

Each new valuation is processed exactly once and strictly in chronological order:

1. verify the hash-chained audit history;
2. store the point-in-time prices and market-open state;
3. mark the paper units and independently update each VPPI sleeve;
4. attempt any previously approved order at the later valuation, deferring it if a
   required market is closed;
5. reconcile expected paper units to an optional observed-position file;
6. raise stale-data, reconciliation, near-floor, or floor-breach alerts;
7. calculate the current VPPI target on a due review date; and
8. either create a pending order or record the control that blocked it.

Re-running the same valuation is idempotent. Approval never fills an order. A fill can
occur only on a later valuation, preserving the Stage 2 no-same-bar timing contract.

For independent reconciliation, pass a CSV containing exactly these columns:

```csv
asset,units
fixed_income,3000
gold,4000
equity,3000
```

Without that file, the engine performs an internal paper-ledger reconciliation. A unit or
value difference above the configured currency tolerance creates a critical alert and
blocks a new order.

## Controls and alerts

The versioned defaults are in `configs/execution_controls.toml`. A human accepts their
effective values during initialization. The controls are:

- maximum daily double-sided turnover;
- maximum age of the valuation when it is observed;
- warning buffer above the synthetic floor;
- position-reconciliation tolerance; and
- order-approval expiry.

An order above the turnover ceiling remains recorded as `control_blocked` and cannot be
approved. An active kill switch cancels all pending and approved orders and prevents new
approvals. Deactivating it does not resurrect canceled orders.

Alerts are durable records shown in the command output and dashboard. Critical stale-data,
reconciliation, and floor-breach conditions block new orders. A near-floor condition is a
warning. The floor is the static initial-release floor; ratcheting is deliberately reserved
for Phase 6.

## Human approval workflow

Initialize a paper portfolio from the exact Stage 4 evidence and opening snapshot:

```powershell
portfolio-insurance paper-init `
  --selection runs/stage4/medium_term `
  --snapshot data/snapshots/full_history --portfolio medium_term `
  --approved-by "investment-committee" --approve-policy `
  --allow-market-closures
```

Process the latest valuation and optionally reconcile an independent paper-position file:

```powershell
portfolio-insurance paper-run `
  --state runs/stage5/medium_term/paper.db `
  --snapshot data/snapshots/full_history `
  --positions paper_positions.csv --allow-market-closures
```

Approve or reject the displayed order ID:

```powershell
portfolio-insurance paper-approve `
  --state runs/stage5/medium_term/paper.db `
  --order ORDER_ID --approved-by "portfolio-manager"

portfolio-insurance paper-reject `
  --state runs/stage5/medium_term/paper.db `
  --order ORDER_ID --rejected-by "portfolio-manager" `
  --reason "price source under review"
```

The emergency control requires a named actor and reason:

```powershell
portfolio-insurance paper-kill-switch `
  --state runs/stage5/medium_term/paper.db `
  --activate --actor "risk-officer" --reason "reconciliation incident"
```

Use `--deactivate` with a new reason after the incident has been resolved. `paper-status`
shows NAV, floor distance, reconciliation, open approvals, alerts, and audit status.

## Audit and dashboard artifacts

The Stage 5 directory contains:

- `paper.db`: positions, sleeves, price marks, valuations, orders, fills, alerts, and the
  hash-chained audit events;
- `dashboard.html`: a responsive local operator view of NAV, floor distance, allocation,
  control state, alerts, and the latest order rationale; and
- `state.json`: the same current state in a machine-readable export.

Database triggers reject updates or deletions to audit events. Every operational action
also carries the prior event hash, so ordering and content changes are detectable. The
chain is verified before every mutation and displayed in the dashboard.

The dashboard is intentionally a local, read-only artifact: it contains sensitive
portfolio-operating state and is not published by this project.

## Acceptance criterion

The test suite demonstrates that paper orders cannot bypass approval, cannot execute on
their decision valuation, respect turnover and market-closure controls, are canceled by
the kill switch, and produce durable reconciliation, alert, fill, dashboard, and immutable
audit records.
