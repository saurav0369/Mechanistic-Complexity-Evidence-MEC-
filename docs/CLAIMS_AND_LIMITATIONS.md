# Claims and limitations

This file is the public claim boundary for the code release.

## Supported empirical claims

### Reacher-v5

The frozen one-shot hidden evaluation covered 8 untouched learned world models and 3 hidden scenario roots, for 33,600 evaluations in total.

Audited pooled results:

- wrong intervention risk among committed pure cases: **0.151%**;
- pure-case coverage: **79.318%**;
- correct-intervention rate: **79.198%**;
- declared-nonpure pure-commit rate: **2.299%**;
- declared-nonpure safe abstain/escalate rate: **97.701%**;
- MEC/random pure-case probe-cost ratio: **0.7222**.

Random probing had slightly lower wrong-intervention risk, so the result is not Pareto dominance. The MEC value-of-information chooser and active-discrimination baseline made the same commit/escalate and mechanism decisions on all 33,600 hidden evaluations; no chooser-superiority claim is supported.

The final dynamic nonfinal early-commit stage did not obtain a robust threshold. The frozen policy therefore used its predeclared conservative fallback: no early dynamic commitment and acquisition of full evidence.

### InvertedDoublePendulum-v5

The IDP work is a **development stress test**, not hidden cross-family confirmation.

- static learned-WM threshold transfer: supported across 8 development WMs;
- prospective control-slope dynamic threshold: supported across 8 development WMs;
- broad selective-intervention primary safety/coverage: 8/8 development folds passed;
- preregistered efficiency criterion: 0/8 folds passed;
- median MEC/random cost ratio: **0.8978** vs. frozen target **≤0.85**;
- bounded conditional-calibration diagnostic: did not rescue efficiency; median ratio **1.0836**;
- broad hidden IDP evaluation: **not run**.

## Theoretical scope

Under the declared exact global displacement-probe semantics, the project derives restricted representation-conditioned laws including

```text
kappa_affine = 0
kappa_polynomial = d
kappa_C1 = d + 1
```

and exact bounded-degree polynomial query laws for the stated oracles. These are restricted class-conditioned results. They are not a universal identifiability theorem, and their proofs use classical mathematical machinery.

## Prohibited extrapolations

Do not use this repository as evidence that MEC:

- solves general identifiability;
- certifies arbitrary world models;
- guarantees arbitrary/open-world OOD safety;
- beats active model discrimination;
- Pareto-dominates random probing;
- has hidden cross-family IDP confirmation;
- has been externally replicated;
- transfers to large latent/video/foundation world models or real robots without additional evidence.
