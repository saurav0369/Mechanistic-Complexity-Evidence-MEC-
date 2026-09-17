# MEC E2E v1.4 — Robust Set-Valued Commitment Repair

**Protocol ID:** `MEC_E2E_V1_4_ROBUST_SET_COMMIT_DEV_2026_08_29`  
**Status:** FROZEN DEVELOPMENT-ONLY BEFORE EXECUTION  
**Hidden status:** SEALED

## Why v1.4 exists

v1.3 R2 failed its preregistered primary gate despite strong pure-fault intervention and probe efficiency. The observed medians were: wrong intervention **0.87%**, coverage **95.83%**, correct intervention **95.0%**, composite/OOD pure commit **13.33%**, composite/OOD safe **86.67%**, and MEC/random probe cost **0.635x**. Therefore v1.4 is not a general architecture rewrite. It is restricted to the final pure-vs-composite/OOD commitment semantics.

## Frozen successes

The following are not targets for repair and are held fixed in scope: the eight R2 development world models and their nominal residual normalization, the Reacher mechanism/fault generator, the residual evidence representation, the static/dynamic/full probe interface and costs, the v1.1 theorem arm, and the hidden seeds. No world model is retrained.

## Prospective anti-overfit split

The old v1.3 train groups remain the only data used to fit base evidence heads and the VOI regressors. v1.4 creates **120 fresh development groups** from seed **764001**. A new split seed **54217** produces **60 fresh calibration groups** and **60 fresh qualification-test groups**. Fresh test labels are never used for fitting or threshold selection. Hidden seeds remain inaccessible.

## Repair: robust set-valued pure certificate

Each evidence stage fits four binary mechanism-bit heads and one seven-state semantic classifier using lightweight HistGradientBoosting on the old v1.3 train groups. For candidate pure mechanism `k`, define

`certificate_k = min(p_bit[k], 1 - second_bit_probability, p_state[k], 1 - p_state(nonpure))`.

A pure intervention is allowed only when the highest-scoring mechanism is a singleton certificate above the calibrated stage threshold. Otherwise the policy acquires more evidence (non-final stage) or escalates (full stage). This explicitly prevents the v1.3 failure mode in which a composite case could look like one dominant bit plus one sub-threshold bit.

## Distributionally robust calibration across world models

For each LOMO fold, calibration uses the **seven source WMs only** on the fresh calibration groups. A candidate threshold is accepted only if its one-sided **95% Clopper-Pearson upper bound** is <=5% for both (a) wrong pure action among committed pure cases and (b) pure-action commits on composite/OOD cases, **for every source WM separately and pooled**. Among safe thresholds, select the threshold maximizing the minimum source-WM pure coverage, then pooled coverage.

This is intentionally more conservative than v1.3's pooled empirical grid search.

## LOMO qualification

Each fold is evaluated on one held-out development WM and the **fresh v1.4 test groups**. The original scientific bars remain unchanged:

- wrong intervention risk <=5%;
- pure coverage >=70%;
- composite/OOD pure commit <=5%;
- composite/OOD safe >=95%;
- >=6/8 LOMO folds pass all four;
- median primary metrics also pass.

Efficiency remains secondary: median MEC/random probe-cost ratio <=0.85 with >=6/8 efficiency-passing folds. Efficiency failure cannot erase a primary scientific pass, but it removes the efficiency-superiority claim.

## Decision

- `PRIMARY_PASS_EFFICIENCY_PASS`: freeze one final v1.4 policy and proceed to build the untouched hidden evaluator.
- `PRIMARY_PASS_EFFICIENCY_FAIL`: freeze one final v1.4 policy and proceed to hidden, but prohibit an efficiency-superiority claim.
- `PRIMARY_FAIL`: hidden remains sealed. No threshold tuning against v1.4 test groups is permitted.

## Governance

v1.3's `PRIMARY_FAIL` is permanent historical evidence and is not overwritten. v1.4 is a prospective new development protocol.
