# Provenance

The Reacher runtime under `reproduction/reacher/frozen/` is intentionally preserved as a historical frozen artifact set.

Before hidden evaluation, an authority manifest recorded SHA-256 hashes for the runtime source and configuration files. The files in this repository were recovered from the final policy-freeze / hidden-evaluation notebooks and verified against that manifest before publication.

The verification script checks the files named by `MEC_PRE_HIDDEN_AUTHORITY_MANIFEST_R2.json`. A mismatch should be treated as a provenance failure, not silently repaired.

## Why the code is not refactored

Several scripts use historical flat-module imports such as `import mec_e2e_v1_runner as core`. Moving functions into a modern package would improve ergonomics but would change the exact source bytes that were frozen before hidden evaluation. For that reason:

1. the frozen Reacher lane remains byte-preserved;
2. helper/documentation code lives outside the frozen directory;
3. future refactors should be added as a separate implementation and compared against the frozen reference rather than replacing it.

## Evidence lanes

The repository keeps two distinct lanes:

- **Reacher:** frozen hidden evaluation with pre-hidden source/config hashes;
- **IDP:** development-only cross-family stress test, including failed and repaired mechanisms and the final efficiency failure.

Negative results are part of the record and should not be removed from later cleanups.
