# Mechanism Evidence Complexity

**When is a predictive world model sufficiently evidenced to support a mechanism-specific intervention?**

Mechanism Evidence Complexity (MEC) studies a narrower question than world-model identification: given a declared mechanism family, residual representation class, and diagnostic interface, what evidence is needed before the intervention semantic is well defined across the models still compatible with what has been observed?

This repository contains the frozen implementation and development history behind the MEC experiments. The original frozen Reacher runtime is preserved byte-for-byte; it is not refactored into a cleaner package because doing so would break the pre-hidden SHA-256 provenance record.

## Main empirical record

| Evidence lane | Status | Result |
|---|---|---|
| Reacher-v5, one-shot hidden evaluation | **completed** | 8 held-out world models × 3 hidden scenario roots; 33,600 evaluations |
| Wrong intervention risk among committed pure cases | **0.151%** | 23 wrong actions / 15,229 committed pure cases |
| Pure-case coverage | **79.318%** | 15,229 / 19,200 pure cases |
| Declared-nonpure safe abstain/escalate | **97.701%** | two known compositions + outside-nominal-training-hull morphology |
| MEC / random probe-cost ratio | **0.7222** | ≈27.8% lower diagnostic probe cost than random |
| MEC vs. active discrimination | **negative result** | identical commit/escalate and mechanism decisions on all 33,600 evaluations |
| InvertedDoublePendulum-v5 development stress test | **primary pass / efficiency fail** | safety/coverage transferred; preregistered efficiency target did not |

The Reacher result is **not** Pareto dominance over random probing: random had slightly lower wrong-intervention risk. MEC also does **not** establish superiority over active model discrimination.

For IDP, the broad selective-intervention development policy passed the frozen primary safety/coverage criteria on all 8 held-out development world models, but the efficiency criterion failed on 0/8 folds (median MEC/random cost ratio 0.8978 vs. preregistered ≤0.85). A bounded post-failure calibration diagnostic did not rescue the result (median ratio 1.0836), so no broad hidden IDP evaluation was authorized.

## Repository map

```text
.
├── README.md
├── requirements.txt
├── docs/
│   ├── CLAIMS_AND_LIMITATIONS.md
│   ├── PROVENANCE.md
│   └── REPRODUCIBILITY.md
├── reproduction/
│   ├── reacher/
│   │   ├── frozen/       # byte-preserved pre-hidden runtime + configs + manifest
│   │   └── notebooks/    # final policy freeze + one-shot hidden evaluation wrappers
│   └── idp/
│       ├── code/         # extracted executable development scripts
│       ├── configs/      # public configs; unspent hidden seeds redacted
│       └── protocols/
├── scripts/
│   └── verify_frozen_artifacts.py
└── tests/
    └── test_frozen_artifacts.py
```

## Verify the frozen Reacher runtime

The pre-hidden R2 authority manifest records SHA-256 hashes for the exact runtime files used to authorize the final hidden evaluation.

```bash
python scripts/verify_frozen_artifacts.py
```

A successful run prints `frozen artifact verification: PASS`.

## Environment

The frozen notebooks explicitly used:

```text
gymnasium[mujoco]==1.3.0
mujoco==3.10.0
```

plus NumPy, SciPy, scikit-learn, and PyTorch. See `docs/REPRODUCIBILITY.md` before attempting a replay; the historical scripts were designed around checkpoint directories produced by the preceding development stages.

## Scope

MEC does **not** claim:

- generic world-model identifiability;
- arbitrary/open-world OOD safety;
- Pareto dominance over random probing;
- superiority to active model discrimination;
- hidden cross-family confirmation on IDP;
- external independent replication;
- direct empirical validation of every exact theoretical probe law.

The theoretical and learned-residual experiments are separate evidence lanes. The exact representation-conditioned probe/query results use classical periodicity, finite-difference, dense-subgroup, affine-normalizer, and finite-dimensional linear-algebra machinery; claims of mathematical priority remain deliberately narrow.

## Paper

The current submitted manuscript is intentionally **not mirrored in this public repository** because the submission copy is marked “Do not distribute.” Public manuscript material can be added after the relevant review/anonymity constraints are cleared.
