# Reproducibility

## 1. Integrity check

Run this first:

```bash
python scripts/verify_frozen_artifacts.py
```

This verifies the byte-preserved Reacher runtime against the pre-hidden R2 authority manifest.

## 2. Python environment

The frozen notebooks explicitly install:

```bash
pip install 'gymnasium[mujoco]==1.3.0' mujoco==3.10.0 scikit-learn scipy
```

and use PyTorch and NumPy from the runtime. `requirements.txt` captures the dependency families while keeping the two simulator packages pinned exactly as in the frozen notebooks.

For an archival reproduction release, a container/lockfile should additionally freeze the exact CUDA/PyTorch/NumPy/scikit-learn/SciPy build used for the final execution. That environment lock is not reconstructed here, so this repository should not claim bitwise replay across arbitrary machines.

## 3. Reacher runtime

The frozen scripts are CLI programs rather than an installed package. Run them from the frozen directory (or add that directory to `PYTHONPATH`) so the original flat-module imports resolve.

Examples:

```bash
cd reproduction/reacher/frozen
python mec_e2e_v1_runner.py --help
python mec_e2e_v1_dev_full.py --help
python mec_v14_reacher_hidden_final_r2.py --help
```

The final hidden evaluator expects checkpoint directories and a final policy-freeze bundle produced by earlier stages. Large learned-model checkpoints/policy bundles are not included in this source reconstruction.

## 4. IDP development lane

The IDP scripts were extracted from the corresponding frozen notebooks without executing notebook cells. Public configs are stored under `reproduction/idp/configs/`. Unspent future hidden seed/root values are redacted; `ORIGINAL_CONFIG_SHA256.json` records the SHA-256 hashes of the original frozen configs.

The development sequence represented here is:

1. semantic preflight;
2. learned-WM threshold development (v0.2);
3. prospective control-slope redesign (v0.3B);
4. broad selective-intervention development (v0.4C);
5. bounded post-failure conditional-calibration diagnostic.

The broad efficiency gate failed and hidden IDP evaluation was not authorized.

## 5. Reproduction terminology

A clean internal runtime reconstruction is not independent external replication. Any external reproduction should record the code commit, environment, raw outputs, seeds/config hashes, and whether the implementer used the original implementation or only the specification.
